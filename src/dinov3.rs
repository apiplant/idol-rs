//! DINOv3 ViT-H+/16 (`facebook/dinov3-vith16plus-pretrain-lvd1689m`): the image encoder behind `feature1`.
//! A ViT with a class token, register tokens, axial RoPE on the patch tokens, LayerScale and a gated SiLU MLP.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::ops::{self, layer_norm_plain, LayerNorm, Linear};

#[derive(Clone, Debug, Deserialize)]
pub struct Dinov3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_register_tokens: usize,
    pub patch_size: usize,
    #[serde(default = "eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "theta")]
    pub rope_theta: f64,
    #[serde(default = "yes")]
    pub use_gated_mlp: bool,
}
fn eps() -> f64 {
    1e-5
}
fn theta() -> f64 {
    100.0
}
fn yes() -> bool {
    true
}

struct Layer {
    norm1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ls1: Tensor,
    norm2: LayerNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
    ls2: Tensor,
}

pub struct Dinov3 {
    cfg: Dinov3Config,
    dt: DType,
    patch_w: Tensor, // [hidden, 3*p*p]
    patch_b: Tensor,
    cls: Tensor,
    regs: Tensor,
    layers: Vec<Layer>,
    norm: LayerNorm,
    dev: Device,
}

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

impl Dinov3 {
    pub fn load(vb: &VarBuilder, cfg: &Dinov3Config, dt: DType, dev: &Device) -> Result<Self> {
        let c = cfg.hidden_size;
        let p = cfg.patch_size;
        anyhow::ensure!(cfg.use_gated_mlp, "only the gated-MLP DINOv3 variants are supported");
        let e = vb.pp("embeddings");
        let lin = |vb: &VarBuilder, i: usize, o: usize, bias: bool| Linear::load(vb, i, o, bias, dt);
        let ln = |vb: &VarBuilder| LayerNorm::load(vb, c, cfg.layer_norm_eps as f32, true, dev);
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            let l = vb.pp("layer").pp(i.to_string());
            let a = l.pp("attention");
            let m = l.pp("mlp");
            layers.push(Layer {
                norm1: ln(&l.pp("norm1"))?,
                q: lin(&a.pp("q_proj"), c, c, true)?,
                k: lin(&a.pp("k_proj"), c, c, false)?,
                v: lin(&a.pp("v_proj"), c, c, true)?,
                o: lin(&a.pp("o_proj"), c, c, true)?,
                ls1: l.pp("layer_scale1").get(c, "lambda1")?.to_dtype(dt)?,
                norm2: ln(&l.pp("norm2"))?,
                gate: lin(&m.pp("gate_proj"), c, cfg.intermediate_size, true)?,
                up: lin(&m.pp("up_proj"), c, cfg.intermediate_size, true)?,
                down: lin(&m.pp("down_proj"), cfg.intermediate_size, c, true)?,
                ls2: l.pp("layer_scale2").get(c, "lambda1")?.to_dtype(dt)?,
            });
        }
        Ok(Self {
            patch_w: e.pp("patch_embeddings").get((c, 3, p, p), "weight")?.reshape((c, 3 * p * p))?.to_dtype(dt)?,
            patch_b: e.pp("patch_embeddings").get(c, "bias")?.to_dtype(dt)?,
            cls: e.get((1, 1, c), "cls_token")?.to_dtype(dt)?,
            regs: e.get((1, cfg.num_register_tokens, c), "register_tokens")?.to_dtype(dt)?,
            layers,
            norm: ln(&vb.pp("norm"))?,
            cfg: cfg.clone(),
            dt,
            dev: dev.clone(),
        })
    }

    /// `(cos, sin)` `[N, head_dim]` for an `ph × pw` patch grid (patch centres in [-1, 1], 2D RoPE).
    fn rope(&self, ph: usize, pw: usize) -> Result<(Tensor, Tensor)> {
        let hd = self.cfg.hidden_size / self.cfg.num_attention_heads;
        let nf = hd / 4;
        let inv: Vec<f32> = (0..nf).map(|i| 1.0 / (self.cfg.rope_theta as f32).powf(i as f32 * 4.0 / hd as f32)).collect();
        let mut cos = vec![0f32; ph * pw * hd];
        let mut sin = vec![0f32; ph * pw * hd];
        for y in 0..ph {
            for x in 0..pw {
                let n = y * pw + x;
                let co = [2.0 * ((y as f32 + 0.5) / ph as f32) - 1.0, 2.0 * ((x as f32 + 0.5) / pw as f32) - 1.0];
                for a in 0..2 {
                    for (j, f) in inv.iter().enumerate() {
                        let ang = 2.0 * std::f32::consts::PI * co[a] * f;
                        for rep in 0..2 {
                            let idx = n * hd + rep * 2 * nf + a * nf + j;
                            cos[idx] = ang.cos();
                            sin[idx] = ang.sin();
                        }
                    }
                }
            }
        }
        let mk = |v: Vec<f32>| -> Result<Tensor> { Ok(Tensor::from_vec(v, (1, ph * pw, 1, hd), &self.dev)?.to_dtype(self.dt)?) };
        Ok((mk(cos)?, mk(sin)?))
    }

    fn rotate(&self, x: &Tensor, rope: &(Tensor, Tensor), prefix: usize) -> Result<Tensor> {
        // x [B, L, H, D]; only the patch tokens after the prefix are rotated
        let (b, l, h, d) = x.dims4()?;
        let pre = x.narrow(1, 0, prefix)?;
        let pat = x.narrow(1, prefix, l - prefix)?;
        let _ = (b, h);
        let x1 = pat.narrow(3, 0, d / 2)?;
        let x2 = pat.narrow(3, d / 2, d / 2)?;
        let rot = Tensor::cat(&[&x2.neg()?, &x1], 3)?;
        let pat = pat.broadcast_mul(&rope.0)?.add(&rot.broadcast_mul(&rope.1)?)?;
        Ok(Tensor::cat(&[&pre, &pat], 1)?)
    }

    /// `images` `[B, 3, H, W]` in `[0, 1]` → `[B, 1 + registers + patches, hidden]` f32: the last hidden state,
    /// layer-normalised without affine parameters (what the denoiser is conditioned on).
    pub fn forward(&self, images: &Tensor) -> Result<Tensor> {
        let (b, _, h, w) = images.dims4()?;
        let p = self.cfg.patch_size;
        let (ph, pw) = (h / p, w / p);
        let c = self.cfg.hidden_size;
        let heads = self.cfg.num_attention_heads;
        let mean = Tensor::from_slice(&MEAN, (1, 3, 1, 1), &self.dev)?;
        let std = Tensor::from_slice(&STD, (1, 3, 1, 1), &self.dev)?;
        let x = images.to_dtype(DType::F32)?.broadcast_sub(&mean)?.broadcast_div(&std)?.to_dtype(self.dt)?;
        // patch embedding as unfold + matmul
        let x = x.reshape((b, 3, ph, p, pw, p))?.permute((0, 2, 4, 1, 3, 5))?.contiguous()?.reshape((b * ph * pw, 3 * p * p))?;
        let x = x.matmul(&self.patch_w.t()?)?.broadcast_add(&self.patch_b)?.reshape((b, ph * pw, c))?;
        let prefix = 1 + self.cfg.num_register_tokens;
        let mut x = Tensor::cat(&[&self.cls.expand((b, 1, c))?, &self.regs.expand((b, self.cfg.num_register_tokens, c))?, &x], 1)?;
        let rope = self.rope(ph, pw)?;
        let l = x.dim(1)?;
        for layer in &self.layers {
            let n = layer.norm1.forward(&x)?;
            let sh = |t: Tensor| -> Result<Tensor> { Ok(t.reshape((b, l, heads, c / heads))?) };
            let q = self.rotate(&sh(layer.q.forward(&n)?)?, &rope, prefix)?;
            let k = self.rotate(&sh(layer.k.forward(&n)?)?, &rope, prefix)?;
            let v = sh(layer.v.forward(&n)?)?;
            let a = layer.o.forward(&ops::sdpa(&q.contiguous()?, &k.contiguous()?, &v)?.reshape((b, l, c))?)?;
            x = x.add(&a.broadcast_mul(&layer.ls1)?)?;
            let n = layer.norm2.forward(&x)?;
            let m = candle_nn::ops::silu(&layer.gate.forward(&n)?)?.mul(&layer.up.forward(&n)?)?;
            x = x.add(&layer.down.forward(&m)?.broadcast_mul(&layer.ls2)?)?;
            crate::progress::check()?;
        }
        layer_norm_plain(&self.norm.forward(&x)?)
    }
}
