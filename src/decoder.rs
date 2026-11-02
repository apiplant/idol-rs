//! The two decoders that turn the denoised latent `[1, 8192, 16]` into Gaussians: the octree decoder places
//! anchor points by descending an octree, the Gaussian decoder turns each anchor into 32 Gaussians.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::gaussian::Gaussians;
use crate::ops::{self, layer_norm_plain, FeedForward, HeadRmsNorm, LayerNorm, Linear, ScalarEmbedder};
use crate::noise::Noise;

#[derive(Clone, Debug, Deserialize)]
pub struct DecoderConfig {
    #[serde(default)]
    pub in_channels: usize,
    pub model_channels: usize,
    pub cond_channels: usize,
    pub num_blocks: usize,
    pub num_heads: usize,
    #[serde(default = "four")]
    pub mlp_ratio: f64,
    #[serde(default)]
    pub representation_config: Option<RepConfig>,
}
fn four() -> f64 {
    4.0
}

#[derive(Clone, Debug, Deserialize)]
pub struct Lr {
    #[serde(rename = "_xyz")]
    pub xyz: f32,
    #[serde(rename = "_features_dc")]
    pub features_dc: f32,
    #[serde(rename = "_opacity")]
    pub opacity: f32,
    #[serde(rename = "_scaling")]
    pub scaling: f32,
    #[serde(rename = "_rotation")]
    pub rotation: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RepConfig {
    pub num_gaussians: usize,
    pub lr: Lr,
    pub perturbe_size: f32,
    #[serde(rename = "3d_filter_kernel_size")]
    pub kernel_size: f32,
    pub scaling_bias: f32,
    pub opacity_bias: f32,
}

/// `MultiHeadAttention`: RMS-normalised queries and keys; self (`to_qkv`) or cross (`to_q` + `to_kv`).
struct Attention {
    to_qkv: Option<Linear>,
    to_q: Option<Linear>,
    to_kv: Option<Linear>,
    q_norm: HeadRmsNorm,
    k_norm: HeadRmsNorm,
    to_out: Linear,
    heads: usize,
}

impl Attention {
    fn load(vb: &VarBuilder, c: usize, heads: usize, ctx: Option<usize>, dt: DType) -> Result<Self> {
        let d = c / heads;
        Ok(Self {
            to_qkv: if ctx.is_none() { Some(Linear::load(&vb.pp("to_qkv"), c, 3 * c, true, dt)?) } else { None },
            to_q: ctx.map(|_| Linear::load(&vb.pp("to_q"), c, c, true, dt)).transpose()?,
            to_kv: ctx.map(|x| Linear::load(&vb.pp("to_kv"), x, 2 * c, true, dt)).transpose()?,
            q_norm: HeadRmsNorm::load(&vb.pp("q_rms_norm"), heads, d)?,
            k_norm: HeadRmsNorm::load(&vb.pp("k_rms_norm"), heads, d)?,
            to_out: Linear::load(&vb.pp("to_out"), c, c, true, dt)?,
            heads,
        })
    }

    fn forward(&self, x: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        let h = self.heads;
        let o = if let (Some(to_q), Some(to_kv), Some(ctx)) = (&self.to_q, &self.to_kv, context) {
            let s = ctx.dim(1)?;
            let q = self.q_norm.forward(&to_q.forward(x)?.reshape((b, l, h, c / h))?)?;
            let kv = to_kv.forward(ctx)?.reshape((b, s, 2, h, c / h))?;
            let k = self.k_norm.forward(&kv.narrow(2, 0, 1)?.squeeze(2)?)?;
            let v = kv.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?;
            ops::sdpa(&q, &k, &v)?
        } else {
            let qkv = self.to_qkv.as_ref().unwrap().forward(x)?.reshape((b, l, 3, h, c / h))?;
            let p = |i: usize| -> Result<Tensor> { Ok(qkv.narrow(2, i, 1)?.squeeze(2)?) };
            let (q, k) = (self.q_norm.forward(&p(0)?)?, self.k_norm.forward(&p(1)?)?);
            ops::sdpa(&q, &k, &p(2)?.contiguous()?)?
        };
        self.to_out.forward(&o.reshape((b, l, c))?)
    }
}

/// Octree decoder block: cross attention to the latent + FFN, modulated by the octree level (shared adaLN).
struct ModBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    cross: Attention,
    mlp: FeedForward,
}

impl ModBlock {
    fn load(vb: &VarBuilder, cfg: &DecoderConfig, dt: DType, dev: &Device) -> Result<Self> {
        let c = cfg.model_channels;
        Ok(Self {
            norm1: LayerNorm::plain(c, 1e-6, dev)?,
            norm2: LayerNorm::plain(c, 1e-6, dev)?,
            cross: Attention::load(&vb.pp("cross_attn"), c, cfg.num_heads, Some(cfg.cond_channels), dt)?,
            mlp: FeedForward::load(&vb.pp("mlp"), c, cfg.mlp_ratio, dt)?,
        })
    }

    fn trace(&self, x: &Tensor, modv: &Tensor, ctx: &Tensor) -> Result<Vec<(String, Tensor)>> {
        let ch = |i: usize| -> Result<Tensor> { Ok(modv.chunk(6, 1)?[i].unsqueeze(1)?) };
        let n1 = self.norm1.forward(x)?;
        let a_in = n1.broadcast_mul(&(ch(1)? + 1.0)?)?.broadcast_add(&ch(0)?)?;
        let ca = self.cross.forward(&a_in, Some(ctx))?;
        let x1 = x.add(&ca.broadcast_mul(&ch(2)?)?)?;
        let m_in = self.norm2.forward(&x1)?.broadcast_mul(&(ch(4)? + 1.0)?)?.broadcast_add(&ch(3)?)?;
        let ff = self.mlp.forward(&m_in)?;
        let (b, l, c) = a_in.dims3()?;
        let h = self.cross.heads;
        let q = self.cross.q_norm.forward(&self.cross.to_q.as_ref().unwrap().forward(&a_in)?.reshape((b, l, h, c / h))?)?;
        let kv = self.cross.to_kv.as_ref().unwrap().forward(ctx)?.reshape((b, ctx.dim(1)?, 2, h, c / h))?;
        let k = self.cross.k_norm.forward(&kv.narrow(2, 0, 1)?.squeeze(2)?)?;
        Ok(vec![("b0_n1".into(), n1), ("b0_ain".into(), a_in), ("b0_ca".into(), ca), ("b0_x1".into(), x1), ("b0_min".into(), m_in), ("b0_ff".into(), ff), ("b0_q".into(), q), ("b0_k".into(), k)])
    }

    fn forward(&self, x: &Tensor, modv: &Tensor, ctx: &Tensor) -> Result<Tensor> {
        let ch = |i: usize| -> Result<Tensor> { Ok(modv.chunk(6, 1)?[i].unsqueeze(1)?) };
        let h = self.norm1.forward(x)?.broadcast_mul(&(ch(1)? + 1.0)?)?.broadcast_add(&ch(0)?)?;
        let x = x.add(&self.cross.forward(&h, Some(ctx))?.broadcast_mul(&ch(2)?)?)?;
        let h = self.norm2.forward(&x)?.broadcast_mul(&(ch(4)? + 1.0)?)?.broadcast_add(&ch(3)?)?;
        Ok(x.add(&self.mlp.forward(&h)?.broadcast_mul(&ch(5)?)?)?)
    }
}

/// Gaussian decoder block: self attention + cross attention to the latent + FFN.
struct CrossBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    self_attn: Attention,
    cross: Attention,
    mlp: FeedForward,
}

impl CrossBlock {
    fn load(vb: &VarBuilder, cfg: &DecoderConfig, dt: DType, dev: &Device) -> Result<Self> {
        let c = cfg.model_channels;
        Ok(Self {
            norm1: LayerNorm::plain(c, 1e-6, dev)?,
            norm2: LayerNorm::load(&vb.pp("norm2"), c, 1e-6, true, dev)?,
            norm3: LayerNorm::plain(c, 1e-6, dev)?,
            self_attn: Attention::load(&vb.pp("self_attn"), c, cfg.num_heads, None, dt)?,
            cross: Attention::load(&vb.pp("cross_attn"), c, cfg.num_heads, Some(cfg.cond_channels), dt)?,
            mlp: FeedForward::load(&vb.pp("mlp"), c, cfg.mlp_ratio, dt)?,
        })
    }

    fn forward(&self, x: &Tensor, ctx: &Tensor) -> Result<Tensor> {
        let x = x.add(&self.self_attn.forward(&self.norm1.forward(x)?, None)?)?;
        let x = x.add(&self.cross.forward(&self.norm2.forward(&x)?, Some(ctx))?)?;
        Ok(x.add(&self.mlp.forward(&self.norm3.forward(&x)?)?)?)
    }
}

/// `PcdAbsolutePositionEmbedderv2`: sinusoids of 3D points in `[0, 1]`, `[N, 3] → [N, channels]` (f32).
struct PointEmbedder {
    freqs: Tensor, // [freq_dim]
    channels: usize,
}

impl PointEmbedder {
    fn new(channels: usize, dev: &Device) -> Result<Self> {
        let freq_dim = channels / 3 / 2;
        let max_res = 10f32;
        let steps = freq_dim;
        let step = max_res / (steps - 1) as f32;
        // torch.linspace is symmetric around the middle
        let f: Vec<f32> = (0..steps)
            .map(|i| {
                let e = if i < steps / 2 { step * i as f32 } else { max_res - step * (steps - 1 - i) as f32 };
                2f32.powf(e)
            })
            .collect();
        Ok(Self { freqs: Tensor::from_vec(f, freq_dim, dev)?, channels })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        let ang = (x.unsqueeze(2)?.broadcast_mul(&self.freqs)? * std::f64::consts::PI)?; // [N, 3, F]
        let e = Tensor::cat(&[&ang.sin()?, &ang.cos()?], 2)?.reshape((n, ()))?;
        let have = e.dim(1)?;
        Ok(if have < self.channels { e.pad_with_zeros(1, 0, self.channels - have)? } else { e })
    }
}

pub struct OctreeDecoder {
    in_proj: Linear,
    pos: PointEmbedder,
    input_layer: Linear,
    l_embedder: ScalarEmbedder,
    ada: Linear,
    blocks: Vec<ModBlock>,
    out_proj: Linear,
    dt: DType,
    dev: Device,
}

impl OctreeDecoder {
    pub fn load(vb: &VarBuilder, cfg: &DecoderConfig, dt: DType, dev: &Device) -> Result<Self> {
        let c = cfg.model_channels;
        Ok(Self {
            in_proj: Linear::load(&vb.pp("in_proj"), 3, c, true, DType::F32)?,
            pos: PointEmbedder::new(c, dev)?,
            input_layer: Linear::load(&vb.pp("input_layer"), c, c, true, DType::F32)?,
            l_embedder: ScalarEmbedder::load(&vb.pp("l_embedder"), c, 1024.0, true)?,
            ada: Linear::load(&vb.pp("adaLN_modulation").pp("1"), c, 6 * c, true, DType::F32)?,
            blocks: (0..cfg.num_blocks).map(|i| ModBlock::load(&vb.pp("blocks").pp(i.to_string()), cfg, dt, dev)).collect::<Result<_>>()?,
            out_proj: Linear::load(&vb.pp("out_proj"), c, 8, true, DType::F32)?,
            dt,
            dev: dev.clone(),
        })
    }

    /// Cell centres `x [1, L, 3]` in `[0, 1]`, their octree resolution `level`, the latent `[1, Lc, C]` → the logits
    /// `[1, L, 8]` of how each cell's points split among its 8 children.
    pub fn forward(&self, x: &Tensor, level: f32, cond: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let h = self.in_proj.forward(x)?.add(&self.pos.forward(&x.reshape((b * l, 3))?)?.reshape((b, l, ()))?)?;
        let h = self.input_layer.forward(&h)?;
        let level_emb = self.ada.forward(&candle_nn::ops::silu(&self.l_embedder.forward(level, &self.dev)?)?)?.to_dtype(self.dt)?;
        let ctx = cond.to_dtype(self.dt)?;
        let mut out = h.to_dtype(self.dt)?;
        for blk in &self.blocks {
            out = blk.forward(&out, &level_emb, &ctx)?;
        }
        self.out_proj.forward(&layer_norm_plain(&out)?)
    }

    /// `forward` with its intermediates, for the parity example.
    pub fn trace(&self, x: &Tensor, level: f32, cond: &Tensor) -> Result<Vec<(String, Tensor)>> {
        let (b, l, _) = x.dims3()?;
        let pos = self.pos.forward(&x.reshape((b * l, 3))?)?.reshape((b, l, ()))?;
        let h = self.input_layer.forward(&self.in_proj.forward(x)?.add(&pos)?)?;
        let lemb = self.l_embedder.forward(level, &self.dev)?;
        let modv = self.ada.forward(&candle_nn::ops::silu(&lemb)?)?.to_dtype(self.dt)?;
        let ctx = cond.to_dtype(self.dt)?;
        let mut out = h.to_dtype(self.dt)?;
        let mut v = vec![("pos".to_string(), pos), ("h".into(), h.clone()), ("lemb".into(), lemb), ("mod".into(), modv.clone())];
        v.extend(self.blocks[0].trace(&h.to_dtype(self.dt)?, &modv, &ctx)?);
        for (i, blk) in self.blocks.iter().enumerate() {
            out = blk.forward(&out, &modv, &ctx)?;
            v.push((format!("blk{i}"), out.clone()));
        }
        Ok(v)
    }

    /// Place `num_points` anchors in `[0, 1]^3` by descending the octree to `level`. The random draws follow the
    /// reference exactly: per level one `torch.rand` per distinct (non-zero) point count, in ascending count order,
    /// then one `rand_like` for the positions inside the leaf cells.
    pub fn sample(&self, cond: &Tensor, num_points: usize, level: usize, noise: &mut Noise) -> Result<Vec<[f32; 3]>> {
        use std::collections::BTreeMap;
        let mut coords: Vec<[i32; 3]> = vec![[0, 0, 0]];
        let mut counts: Vec<u32> = vec![num_points as u32];
        for l in 1..=level {
            crate::progress::check()?;
            let res_p = (1u32 << (l - 1)) as f32;
            let np = coords.len();
            let parents: Vec<f32> = coords.iter().flat_map(|c| c.iter().map(|v| (*v as f32 + 0.5) / res_p)).collect();
            let x = Tensor::from_vec(parents, (1, np, 3), &self.dev)?;
            let probs = candle_nn::ops::softmax_last_dim(&self.forward(&x, (1u32 << l) as f32, cond)?)?;
            let probs: Vec<f32> = probs.flatten_all()?.to_vec1()?;
            let mut groups: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
            for (p, &n) in counts.iter().enumerate() {
                if n > 0 {
                    groups.entry(n).or_default().push(p);
                }
            }
            let mut split = vec![[0u32; 8]; np];
            for (n, nodes) in &groups {
                let u = noise.rand(nodes.len())?;
                for (j, &p) in nodes.iter().enumerate() {
                    split[p] = systematic_sample(&probs[p * 8..p * 8 + 8], *n, u[j]);
                }
            }
            let mut next_coords = Vec::new();
            let mut next_counts = Vec::new();
            for p in 0..np {
                for (child, &n) in split[p].iter().enumerate() {
                    if n > 0 {
                        let o = [(child & 1) as i32, ((child >> 1) & 1) as i32, ((child >> 2) & 1) as i32];
                        let c = coords[p];
                        next_coords.push([c[0] * 2 + o[0], c[1] * 2 + o[1], c[2] * 2 + o[2]]);
                        next_counts.push(n);
                    }
                }
            }
            coords = next_coords;
            counts = next_counts;
        }
        // each point lands at a uniformly random position inside its leaf cell
        let scale = (1u32 << level) as f32;
        let total: usize = counts.iter().map(|&n| n as usize).sum();
        let r = noise.rand(3 * total)?;
        let mut pts = Vec::with_capacity(total);
        let mut k = 0;
        for (c, &n) in coords.iter().zip(&counts) {
            for _ in 0..n {
                pts.push([(c[0] as f32 + r[k]) / scale, (c[1] as f32 + r[k + 1]) / scale, (c[2] as f32 + r[k + 2]) / scale]);
                k += 3;
            }
        }
        Ok(pts)
    }
}

/// The reference's `systematic_sample` for one node: split `n` points among 8 children by systematic resampling,
/// with `u` the node's uniform draw (`torch.rand`). Arithmetic in f32 in the reference's order.
pub fn systematic_sample(probs: &[f32], n: u32, u: f32) -> [u32; 8] {
    let mut out = [0u32; 8];
    if n == 0 {
        return out;
    }
    let mut p: Vec<f32> = probs.iter().map(|v| v.max(0.0)).collect();
    let sum: f32 = p.iter().sum();
    let zero = sum == 0.0;
    // probs / row_sums.clamp_min_(1): only sums above one rescale
    let d = sum.max(1.0);
    p.iter_mut().for_each(|v| *v /= d);
    if zero {
        p.iter_mut().for_each(|v| *v = 1.0 / 8.0);
    }
    let mut cdf = [0f32; 8];
    let mut acc = 0.0f32;
    for i in 0..8 {
        acc += p[i];
        // clamp(max=1 - 1e-12) is 1.0 in f32
        cdf[i] = acc.min(1.0);
    }
    let nf = n as f32;
    let u0 = u / nf;
    for i in 0..n {
        let us = (u0 + i as f32 / nf).min(1.0);
        // torch.searchsorted (left): the first bin whose cdf is >= us
        let mut bin = 0;
        while bin < 7 && cdf[bin] < us {
            bin += 1;
        }
        out[bin] += 1;
    }
    out
}

pub struct GaussianDecoder {
    in_proj: Linear,
    pos: PointEmbedder,
    input_layer: Linear,
    blocks: Vec<CrossBlock>,
    out_proj: Linear,
    perturbation: Vec<f32>, // [per_anchor, 3]
    base_offset_scale: f32,
    rep: RepConfig,
    dt: DType,
}

impl GaussianDecoder {
    pub fn load(vb: &VarBuilder, cfg: &DecoderConfig, dt: DType, dev: &Device) -> Result<Self> {
        let c = cfg.model_channels;
        let rep = cfg.representation_config.clone().ok_or_else(|| anyhow::anyhow!("decoder_gs needs a representation_config"))?;
        let per = rep.num_gaussians;
        // offset (3), colour (3), scale (3), rotation (4), opacity (1), offset scale (1) per Gaussian
        let out_dim = per * (3 + 3 + 3 + 4 + 1 + 1);
        Ok(Self {
            in_proj: Linear::load(&vb.pp("in_proj"), cfg.in_channels, c, true, DType::F32)?,
            pos: PointEmbedder::new(c, dev)?,
            input_layer: Linear::load(&vb.pp("input_layer"), c, c, true, DType::F32)?,
            blocks: (0..cfg.num_blocks).map(|i| CrossBlock::load(&vb.pp("blocks").pp(i.to_string()), cfg, dt, dev)).collect::<Result<_>>()?,
            out_proj: Linear::load(&vb.pp("out_proj"), c, out_dim, true, DType::F32)?,
            perturbation: vb.get((per, 3), "points_offset_perturbation")?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?,
            base_offset_scale: vb.get((), "base_offset_scale")?.to_dtype(DType::F32)?.to_scalar()?,
            rep,
            dt,
        })
    }

    pub fn per_anchor(&self) -> usize {
        self.rep.num_gaussians
    }

    /// anchors `[1, L, 3]`, latent `[1, Lc, C]` → per-anchor outputs `[1, L, 32 × 15]`.
    pub fn forward(&self, points: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let (b, l, c) = points.dims3()?;
        let h = self.in_proj.forward(points)?.add(&self.pos.forward(&points.reshape((b * l, c))?)?.reshape((b, l, ()))?)?;
        let h = self.input_layer.forward(&h)?;
        let ctx = cond.to_dtype(self.dt)?;
        let mut out = h.to_dtype(self.dt)?;
        for blk in &self.blocks {
            out = blk.forward(&out, &ctx)?;
        }
        self.out_proj.forward(&layer_norm_plain(&out)?)
    }

    /// Anchors → Gaussians (the reference's `to_representation` + `Gaussian`), on the CPU.
    pub fn decode(&self, anchors: &[[f32; 3]], cond: &Tensor) -> Result<Gaussians> {
        let dev = cond.device();
        let per = self.rep.num_gaussians;
        let n = anchors.len();
        let flat: Vec<f32> = anchors.iter().flatten().copied().collect();
        // long anchor lists are decoded as one sequence (they attend to each other), the result comes back in one go
        let pts = Tensor::from_vec(flat, (1, n, 3), dev)?;
        let feats: Vec<f32> = self.forward(&pts, cond)?.flatten_all()?.to_vec1()?;
        let stride = per * 15;
        let (r, lr) = (&self.rep, &self.rep.lr);
        let sb = inverse_softplus(r.scaling_bias);
        let ob = inverse_sigmoid(r.opacity_bias);
        let mut g = Gaussians::with_capacity(n * per);
        for (i, a) in anchors.iter().enumerate() {
            let f = &feats[i * stride..(i + 1) * stride];
            // layout: _xyz | _features_dc | _scaling | _rotation | _opacity | _offset_scale, each ×per
            let (xyz, dc, sc, rot, op, os) = (&f[0..per * 3], &f[per * 3..per * 6], &f[per * 6..per * 9], &f[per * 9..per * 13], &f[per * 13..per * 14], &f[per * 14..per * 15]);
            for k in 0..per {
                let offset_scale = ops::softplus(os[k] + self.base_offset_scale);
                let mut p = [0f32; 3];
                for d in 0..3 {
                    let o = xyz[k * 3 + d] * lr.xyz + self.perturbation[k * 3 + d];
                    p[d] = o.tanh() * 0.5 * r.perturbe_size * offset_scale + a[d] - 0.5;
                }
                let mut s = [0f32; 3];
                for d in 0..3 {
                    let v = ops::softplus(sc[k * 3 + d] * lr.scaling + sb);
                    s[d] = (v * v + r.kernel_size * r.kernel_size).sqrt();
                }
                let mut q = [1.0 + rot[k * 4] * lr.rotation, rot[k * 4 + 1] * lr.rotation, rot[k * 4 + 2] * lr.rotation, rot[k * 4 + 3] * lr.rotation];
                let norm = q.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                q.iter_mut().for_each(|v| *v /= norm);
                let opacity = 1.0 / (1.0 + (-(op[k] * lr.opacity + ob)).exp());
                let c = [dc[k * 3] * lr.features_dc, dc[k * 3 + 1] * lr.features_dc, dc[k * 3 + 2] * lr.features_dc];
                g.push(p, c, s, q, opacity);
            }
        }
        Ok(g)
    }
}

fn inverse_softplus(x: f32) -> f32 {
    x + (-(-x).exp_m1()).ln()
}

fn inverse_sigmoid(x: f32) -> f32 {
    (x / (1.0 - x)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn systematic_sample_conserves_and_follows_probs() {
        let probs = [0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0];
        for (i, n) in [1u32, 7, 100, 8192].into_iter().enumerate() {
            let out = systematic_sample(&probs, n, 0.137 * (i + 1) as f32);
            assert_eq!(out.iter().sum::<u32>(), n, "items are neither lost nor invented");
            assert_eq!(out[3..].iter().sum::<u32>(), 0, "zero-probability bins stay empty");
        }
        let out = systematic_sample(&probs, 1000, 0.5);
        assert!((out[0] as i32 - 500).abs() <= 1 && (out[1] as i32 - 250).abs() <= 1, "{out:?}");
    }

    #[test]
    fn systematic_sample_uniform_fallback_for_zero_rows() {
        let out = systematic_sample(&[0.0; 8], 800, 0.3);
        assert!(out.iter().all(|&c| c == 100), "{out:?}");
    }

    #[test]
    fn inverse_activations_round_trip() {
        let x = 0.004f32;
        assert!((ops::softplus(inverse_softplus(x)) - x).abs() < 1e-6);
        let p = 0.1f32;
        assert!((1.0 / (1.0 + (-inverse_sigmoid(p)).exp()) - p).abs() < 1e-6);
    }
}
