//! The denoiser: a rectified-flow transformer over the object latent (8192 tokens) and the input camera (1 token),
//! conditioned on the image tokens. Joint attention over `[latent, image, camera]` with learned rotary positions
//! (`LatentSeqMMFlowModel` in the reference, adapted there from diffusers' Wan transformer).

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::fused;
use crate::ops::{self, apply_rotary, layer_norm_plain, FeedForward, HeadRmsNorm, LayerNorm, Linear, Mlp, ScalarEmbedder};

#[derive(Clone, Debug, Deserialize)]
pub struct DenoiserConfig {
    pub q_token_length: usize,
    pub in_channels: usize,
    pub cam_channels: usize,
    pub model_channels: usize,
    pub cond_channels: usize,
    pub cond2_channels: usize,
    pub num_blocks: usize,
    #[serde(default = "two")]
    pub num_refiner_blocks: usize,
    #[serde(default = "sixteen")]
    pub num_heads: usize,
    #[serde(default = "sixty_four")]
    pub num_head_channels: usize,
    #[serde(default = "four")]
    pub mlp_ratio: f64,
}
fn two() -> usize {
    2
}
fn sixteen() -> usize {
    16
}
fn sixty_four() -> usize {
    64
}
fn four() -> f64 {
    4.0
}

/// `RePo3DRotaryEmbedding`: each token's features predict a 3D offset per head, turned into RoPE angles.
struct RePo {
    norm: LayerNorm,
    gate_map: Linear,
    content_map: Linear,
    final_map: Linear,
    freqs: [Tensor; 3],
    /// the three frequency tables back to back: `[32]`
    freqs_cat: Tensor,
    heads: usize,
    fused: bool,
    /// half-precision copies of the three projections, used by the fused path
    half: Option<[Linear; 3]>,
    /// the LayerNorm's affine as an adaLN-shaped `[1, 6C]` (shift = bias, scale = weight - 1) for `fused::mod_ln`
    ln_m: Option<Tensor>,
}

impl RePo {
    fn load(vb: &VarBuilder, c: usize, heads: usize, head_dim: usize, fused: bool, dev: &Device) -> Result<Self> {
        let hidden = (c as f64 * 0.125) as usize;
        let d0 = 2 * (head_dim / 6);
        let dims = [d0 / 2, d0 / 2, (head_dim - 2 * d0) / 2];
        let f = |i: usize| -> Result<Tensor> { Ok(vb.get(dims[i], &format!("freqs_{i}"))?.to_dtype(DType::F32)?) };
        let freqs = [f(0)?, f(1)?, f(2)?];
        let freqs_cat = Tensor::cat(&[&freqs[0], &freqs[1], &freqs[2]], 0)?;
        Ok(Self {
            norm: LayerNorm::load(&vb.pp("norm"), c, 1e-5, true, dev)?,
            gate_map: Linear::load(&vb.pp("gate_map"), c, hidden, false, DType::F32)?,
            content_map: Linear::load(&vb.pp("content_map"), c, hidden, false, DType::F32)?,
            final_map: Linear::load(&vb.pp("final_map"), hidden, 3 * heads, false, DType::F32)?,
            freqs,
            freqs_cat,
            heads,
            fused: fused && head_dim == 64 && dims == [10, 10, 12],
            ln_m: if fused {
                let w = vb.pp("norm").get(c, "weight")?.to_dtype(DType::F32)?;
                let b = vb.pp("norm").get(c, "bias")?.to_dtype(DType::F32)?;
                let z = Tensor::zeros(4 * c, DType::F32, w.device())?;
                Some(Tensor::cat(&[&b, &(w - 1.0)?, &z], 0)?.to_dtype(DType::F16)?.reshape((1, 6 * c))?)
            } else {
                None
            },
            half: if fused && head_dim == 64 && dims == [10, 10, 12] && std::env::var_os("IDOL_REPO_F32").is_none() {
                Some([
                    Linear::load(&vb.pp("gate_map"), c, hidden, false, DType::F16)?,
                    Linear::load(&vb.pp("content_map"), c, hidden, false, DType::F16)?,
                    Linear::load(&vb.pp("final_map"), hidden, 3 * heads, false, DType::F16)?,
                ])
            } else {
                None
            },
        })
    }

    /// `(cos, sin)` stacked: `[2, B, L, H, head_dim / 2]` f32.
    fn forward(&self, h: &Tensor) -> Result<Tensor> {
        let (b, l, _) = h.dims3()?;
        if let Some([gate, content, fin]) = &self.half {
            // LayerNorm still computes in f32 internally; the three small projections run on tensor cores
            let h = fused::mod_ln(h, self.ln_m.as_ref().unwrap(), 0, 1e-5)?;
            let g = candle_nn::ops::silu(&gate.forward(&h)?)?;
            let out = fin.forward(&g.mul(&content.forward(&h)?)?)?.to_dtype(DType::F32)?;
            return fused::repo_angles(&out, &self.freqs_cat, self.heads);
        }
        let h = self.norm.forward(&h.to_dtype(DType::F32)?)?;
        let g = candle_nn::ops::silu(&self.gate_map.forward(&h)?)?;
        let out = self.final_map.forward(&g.mul(&self.content_map.forward(&h)?)?)?;
        if self.fused {
            return fused::repo_angles(&out, &self.freqs_cat, self.heads);
        }
        let delta = out.reshape((b, l, self.heads, 3))?;
        // the reference's `clamp_mul(x, f)` is `x * f` in the forward pass (the tanh only shapes the gradient)
        let mut parts = Vec::with_capacity(3);
        for a in 0..3 {
            parts.push((delta.narrow(3, a, 1)?.broadcast_mul(&self.freqs[a])? * std::f64::consts::PI)?);
        }
        let ang = Tensor::cat(&parts, 3)?;
        Ok(Tensor::stack(&[&ang.cos()?, &ang.sin()?], 0)?)
    }
}

/// `UnifiedTransformerBlock`: RoPE self attention + FFN, modulated by the diffusion time through a shared adaLN plus
/// this block's own shift table (`modulation = false`: the condition refiner, which has plain affine LayerNorms).
struct Block {
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    q_norm: HeadRmsNorm,
    k_norm: HeadRmsNorm,
    out: Linear,
    mlp: FeedForward,
    shift_table: Option<Tensor>,
    heads: usize,
    fused: bool,
}

impl Block {
    fn load(vb: &VarBuilder, cfg: &DenoiserConfig, modulation: bool, dt: DType, dev: &Device) -> Result<Self> {
        let c = cfg.model_channels;
        let hd = cfg.num_head_channels;
        let a = vb.pp("attn");
        Ok(Self {
            norm1: LayerNorm::load(&vb.pp("norm1"), c, 1e-6, !modulation, dev)?,
            norm2: LayerNorm::load(&vb.pp("norm2"), c, 1e-6, !modulation, dev)?,
            qkv: Linear::load(&a.pp("qkv"), c, 3 * c, true, dt)?,
            q_norm: HeadRmsNorm::load(&a.pp("q_norm"), cfg.num_heads, hd)?,
            k_norm: HeadRmsNorm::load(&a.pp("k_norm"), cfg.num_heads, hd)?,
            out: Linear::load(&a.pp("out"), c, c, true, dt)?,
            mlp: FeedForward::load(&vb.pp("mlp"), c, cfg.mlp_ratio, dt)?,
            shift_table: if modulation { Some(vb.get((1, 6 * c), "shift_table")?.to_dtype(dt)?) } else { None },
            heads: cfg.num_heads,
            fused: fused::available(dev, dt) && hd == 64 && c <= 2048,
        })
    }

    /// Attention up to (and not including) the output projection: `[B, L, C]`.
    fn attn_core(&self, x: &Tensor, rope: &Tensor) -> Result<Tensor> {
        use ops::prof::time;
        let dev = x.device();
        let (b, l, c) = x.dims3()?;
        let d = c / self.heads;
        if self.fused {
            let qkv = time("qkv linear", dev, || self.qkv.matmul(x))?;
            let packed = time("rope+rmsnorm", dev, || fused::rope_qkv(&qkv, self.qkv.bias().unwrap(), rope, self.q_norm.gamma(), self.k_norm.gamma(), self.heads))?;
            let part = |i: usize| -> Result<Tensor> { Ok(packed.narrow(0, i, 1)?.squeeze(0)?) };
            let (q, k, v) = (part(0)?, part(1)?, part(2)?);
            let o = time("attention", dev, || ops::sdpa(&q, &k, &v))?;
            return Ok(o.reshape((b, l, c))?);
        }
        let (cos, sin) = (rope.narrow(0, 0, 1)?.squeeze(0)?, rope.narrow(0, 1, 1)?.squeeze(0)?);
        let qkv = time("qkv linear", dev, || self.qkv.forward(x))?.reshape((b, l, 3, self.heads, d))?;
        let part = |i: usize| -> Result<Tensor> { Ok(qkv.narrow(2, i, 1)?.squeeze(2)?) };
        let (q, k) = time("rope+rmsnorm", dev, || -> Result<_> {
            Ok((self.q_norm.forward(&apply_rotary(&part(0)?, &cos, &sin)?)?, self.k_norm.forward(&apply_rotary(&part(1)?, &cos, &sin)?)?))
        })?;
        let v = part(2)?.contiguous()?;
        let o = time("attention", dev, || ops::sdpa(&q, &k, &v))?;
        Ok(o.reshape((b, l, c))?)
    }

    fn attn(&self, x: &Tensor, rope: &Tensor) -> Result<Tensor> {
        self.out.forward(&self.attn_core(x, rope)?)
    }

    /// `x` `[B, L, C]`, `modv` the shared adaLN output `[1, 6C]` (None for the unmodulated refiner).
    fn forward(&self, x: &Tensor, modv: Option<&Tensor>, rope: &Tensor) -> Result<Tensor> {
        use ops::prof::time;
        let dev = x.device();
        let Some(modv) = modv else {
            let x = x.add(&self.attn(&self.norm1.forward(x)?, rope)?)?;
            return Ok(x.add(&self.mlp.forward(&self.norm2.forward(&x)?)?)?);
        };
        let m = modv.broadcast_add(self.shift_table.as_ref().unwrap())?;
        if self.fused {
            // the same maths as below, with each chain of elementwise ops in one kernel
            let h = time("norm+modulate", dev, || fused::mod_ln(x, &m, 0, 1e-6))?;
            let o = self.attn_core(&h, rope)?;
            let y = time("out linear", dev, || self.out.matmul(&o))?;
            let x = time("gate+residual", dev, || fused::bias_gate_res(x, &y, self.out.bias().unwrap(), &m, 2))?;
            let h = time("norm+modulate", dev, || fused::mod_ln(&x, &m, 3, 1e-6))?;
            let a = time("mlp", dev, || -> Result<_> { fused::bias_gelu(&self.mlp.fc1.matmul(&h)?, self.mlp.fc1.bias().unwrap()) })?;
            let y = time("mlp", dev, || self.mlp.fc2.matmul(&a))?;
            return time("gate+residual", dev, || fused::bias_gate_res(&x, &y, self.mlp.fc2.bias().unwrap(), &m, 5));
        }
        let ch = |i: usize| -> Result<Tensor> { Ok(m.chunk(6, 1)?[i].unsqueeze(1)?) };
        let (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (ch(0)?, ch(1)?, ch(2)?, ch(3)?, ch(4)?, ch(5)?);
        let h = time("norm+modulate", dev, || -> Result<_> { Ok(self.norm1.forward(x)?.broadcast_mul(&(scale_msa + 1.0)?)?.broadcast_add(&shift_msa)?) })?;
        let a = self.attn(&h, rope)?;
        let x = time("gate+residual", dev, || -> Result<_> { Ok(x.add(&a.broadcast_mul(&gate_msa)?)?) })?;
        let h = time("norm+modulate", dev, || -> Result<_> { Ok(self.norm2.forward(&x)?.broadcast_mul(&(scale_mlp + 1.0)?)?.broadcast_add(&shift_mlp)?) })?;
        let mm = time("mlp", dev, || self.mlp.forward(&h))?;
        time("gate+residual", dev, || -> Result<_> { Ok(x.add(&mm.broadcast_mul(&gate_mlp)?)?) })
    }
}

/// `PcdAbsolutePositionEmbedder` evaluated once on the stored token positions (they never change): `[1, L, C]`.
/// Computed in f32 exactly as the reference does.
fn position_embedding(pos: &Tensor, channels: usize) -> Result<Tensor> {
    let (_, l, _) = pos.dims3()?;
    let max_res = 16usize;
    let freq_dim = channels / 3 / 2;
    let res_dim = freq_dim.saturating_sub(max_res);
    let mut freqs: Vec<f32> = (0..max_res).map(|i| i as f32).collect();
    freqs.extend((0..res_dim).map(|i| i as f32 / res_dim as f32 * max_res as f32));
    let freqs: Vec<f32> = freqs.iter().take(freq_dim).map(|e| 2f32.powf(*e)).collect();
    let p: Vec<f32> = pos.flatten_all()?.to_vec1()?;
    let mut out = vec![0f32; l * channels];
    for t in 0..l {
        for a in 0..3 {
            let x = p[t * 3 + a];
            let base = t * channels + a * 2 * freq_dim;
            for (j, f) in freqs.iter().enumerate() {
                let ang = (x * f) * 2.0 * std::f32::consts::PI;
                out[base + j] = ang.sin();
                out[base + freq_dim + j] = ang.cos();
            }
        }
    }
    Ok(Tensor::from_vec(out, (1, l, channels), pos.device())?)
}

pub struct Denoiser {
    pub cfg: DenoiserConfig,
    dt: DType,
    t_embedder: ScalarEmbedder,
    ada: Linear,
    input_layer: Linear,
    cond_embedder: Linear,
    cond_embedder2: Linear,
    pos_emb: Tensor,
    noise_repo: Vec<RePo>,
    context_repo: Vec<RePo>,
    repo: Vec<RePo>,
    noise_refiner: Vec<Block>,
    context_refiner: Vec<Block>,
    cam_refiner: Mlp,
    blocks: Vec<Block>,
    shift_table: Tensor,
    out_layer: Linear,
    cam_out_layer: Linear,
    dev: Device,
}

impl Denoiser {
    pub fn load(vb: &VarBuilder, cfg: &DenoiserConfig, dt: DType, dev: &Device) -> Result<Self> {
        let c = cfg.model_channels;
        let (h, hd) = (cfg.num_heads, cfg.num_head_channels);
        let repos = |name: &str, n: usize| -> Result<Vec<RePo>> {
            (0..n).map(|i| RePo::load(&vb.pp(name).pp(i.to_string()), c, h, hd, fused::available(dev, dt), dev)).collect()
        };
        let blocks = |name: &str, n: usize, m: bool| -> Result<Vec<Block>> {
            (0..n).map(|i| Block::load(&vb.pp(name).pp(i.to_string()), cfg, m, dt, dev)).collect()
        };
        let pos_pe = vb.get((1, cfg.q_token_length, 3), "pos_pe")?;
        let pos_emb = position_embedding(&pos_pe.to_device(&Device::Cpu)?, c)?.to_device(dev)?.to_dtype(dt)?;
        Ok(Self {
            t_embedder: ScalarEmbedder::load(&vb.pp("t_embedder"), c, 10000.0, false)?,
            ada: Linear::load(&vb.pp("adaLN_modulation").pp("1"), c, 6 * c, true, DType::F32)?,
            input_layer: Linear::load(&vb.pp("input_layer"), cfg.in_channels, c, true, DType::F32)?,
            cond_embedder: Linear::load(&vb.pp("cond_embedder"), cfg.cond_channels, c, true, DType::F32)?,
            cond_embedder2: Linear::load(&vb.pp("cond_embedder2"), cfg.cond2_channels, c, true, DType::F32)?,
            pos_emb,
            noise_repo: repos("noise_repo_layers", cfg.num_refiner_blocks)?,
            context_repo: repos("context_repo_layers", cfg.num_refiner_blocks)?,
            repo: repos("repo_layers", cfg.num_blocks)?,
            noise_refiner: blocks("noise_refiner", cfg.num_refiner_blocks, true)?,
            context_refiner: blocks("context_refiner", cfg.num_refiner_blocks, false)?,
            cam_refiner: Mlp::load(&vb.pp("cam_refiner"), cfg.cam_channels, c, c, cfg.num_refiner_blocks, DType::F32)?,
            blocks: blocks("blocks", cfg.num_blocks, true)?,
            shift_table: vb.get((1, 2, c), "shift_table")?.to_dtype(DType::F32)?,
            out_layer: Linear::load(&vb.pp("out_layer"), c, cfg.in_channels, true, DType::F32)?,
            cam_out_layer: Linear::load(&vb.pp("cam_out_layer"), c, cfg.cam_channels, true, DType::F32)?,
            cfg: cfg.clone(),
            dt,
            dev: dev.clone(),
        })
    }

    /// The image-condition stream: embed and refine `feature1` `[B, Lc, 1280]` (DINOv3) and `feature2`
    /// `[B, Lc, 128]` (FLUX.2 VAE). It does not depend on the sample or the time, so it is computed once per image.
    pub fn prepare_cond(&self, f1: &Tensor, f2: &Tensor) -> Result<Tensor> {
        let mut h = self.cond_embedder.forward(&f1.to_dtype(DType::F32)?)?.add(&self.cond_embedder2.forward(&f2.to_dtype(DType::F32)?)?)?.to_dtype(self.dt)?;
        for (repo, block) in self.context_repo.iter().zip(&self.context_refiner) {
            let rope = repo.forward(&h)?;
            h = block.forward(&h, None, &rope)?;
        }
        Ok(h)
    }

    /// The velocity of `(z [B, L, 16], cam [B, 1, 5])` at diffusion time `t` in `[0, 1000]`, given the output of
    /// [`Self::prepare_cond`].
    pub fn forward(&self, z: &Tensor, cam: &Tensor, t: f32, h_cond: &Tensor) -> Result<(Tensor, Tensor)> {
        let dt = self.dt;
        let l = z.dim(1)?;
        let mut h_x = self.input_layer.forward(z)?.to_dtype(dt)?;
        let t_emb = self.t_embedder.forward(t, &self.dev)?; // [1, C]
        let t_mod = self.ada.forward(&candle_nn::ops::silu(&t_emb)?)?.to_dtype(dt)?; // [1, 6C]

        h_x = h_x.broadcast_add(&self.pos_emb)?;
        for (repo, block) in self.noise_repo.iter().zip(&self.noise_refiner) {
            let rope = repo.forward(&h_x)?;
            h_x = block.forward(&h_x, Some(&t_mod), &rope)?;
        }
        let h_cam = self.cam_refiner.forward(cam)?.to_dtype(dt)?;

        let mut h = Tensor::cat(&[&h_x, h_cond, &h_cam], 1)?;
        for (repo, block) in self.repo.iter().zip(&self.blocks) {
            let rope = ops::prof::time("repo", &self.dev, || repo.forward(&h))?;
            h = block.forward(&h, Some(&t_mod), &rope)?;
        }

        let total = h.dim(1)?;
        let hx = layer_norm_plain(&h.narrow(1, 0, l)?)?;
        let hc = layer_norm_plain(&h.narrow(1, total - 1, 1)?)?;
        let sc = self.shift_table.broadcast_add(&t_emb.unsqueeze(1)?)?; // [1, 2, C]
        let (shift, scale) = (sc.narrow(1, 0, 1)?, sc.narrow(1, 1, 1)?);
        let hx = hx.broadcast_mul(&(scale.clone() + 1.0)?)?.broadcast_add(&shift)?;
        let hc = hc.broadcast_mul(&(scale + 1.0)?)?.broadcast_add(&shift)?;
        Ok((self.out_layer.forward(&hx)?, self.cam_out_layer.forward(&hc)?))
    }
}
