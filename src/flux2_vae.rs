//! The FLUX.2 VAE encoder (`black-forest-labs/FLUX.2-dev`, `vae/`): the image encoder behind `feature2`.
//! A KL autoencoder encoder (32 latent channels), 2×2 patchified to 128 channels and batch-norm normalised.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use crate::ops::{Conv2d, GroupNorm, Linear};

#[derive(Clone, Debug, Deserialize)]
pub struct VaeConfig {
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub latent_channels: usize,
    pub norm_num_groups: usize,
    #[serde(default = "bn_eps")]
    pub batch_norm_eps: f64,
}
fn bn_eps() -> f64 {
    1e-4
}

struct Resnet {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl Resnet {
    fn load(vb: &VarBuilder, i: usize, o: usize, groups: usize, dt: DType) -> Result<Self> {
        Ok(Self {
            norm1: GroupNorm::load(&vb.pp("norm1"), groups, i, 1e-6, dt)?,
            conv1: Conv2d::load(&vb.pp("conv1"), i, o, 3, 1, 1, true, dt)?,
            norm2: GroupNorm::load(&vb.pp("norm2"), groups, o, 1e-6, dt)?,
            conv2: Conv2d::load(&vb.pp("conv2"), o, o, 3, 1, 1, true, dt)?,
            shortcut: if i != o { Some(Conv2d::load(&vb.pp("conv_shortcut"), i, o, 1, 1, 0, true, dt)?) } else { None },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(&candle_nn::ops::silu(&self.norm1.forward(x)?)?)?;
        let h = self.conv2.forward(&candle_nn::ops::silu(&self.norm2.forward(&h)?)?)?;
        let s = match &self.shortcut {
            Some(c) => c.forward(x)?,
            None => x.clone(),
        };
        Ok(s.add(&h)?)
    }
}

struct MidAttention {
    norm: GroupNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}

impl MidAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let t = self.norm.forward(x)?.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()?; // [B, HW, C]
        let sh = |t: Tensor| -> Result<Tensor> { Ok(t.reshape((b, h * w, 1, c))?) };
        let o = crate::ops::sdpa(&sh(self.q.forward(&t)?)?, &sh(self.k.forward(&t)?)?, &sh(self.v.forward(&t)?)?)?;
        let o = self.out.forward(&o.reshape((b, h * w, c))?)?;
        Ok(x.add(&o.transpose(1, 2)?.reshape((b, c, h, w))?)?)
    }
}

pub struct Flux2Vae {
    conv_in: Conv2d,
    down: Vec<(Vec<Resnet>, Option<Conv2d>)>,
    mid: (Resnet, MidAttention, Resnet),
    norm_out: GroupNorm,
    conv_out: Conv2d,
    quant: Conv2d,
    bn_mean: Tensor,
    bn_std: Tensor,
    bn_mean_dt: Tensor,
    bn_std_dt: Tensor,
    latent: usize,
    dt: DType,
    dev: Device,
}

impl Flux2Vae {
    pub fn load(vb: &VarBuilder, cfg: &VaeConfig, dt: DType, dev: &Device) -> Result<Self> {
        let g = cfg.norm_num_groups;
        let ch = &cfg.block_out_channels;
        let e = vb.pp("encoder");
        let mut down = Vec::new();
        let mut prev = ch[0];
        for (bi, &out) in ch.iter().enumerate() {
            let db = e.pp("down_blocks").pp(bi.to_string());
            let mut resnets = Vec::new();
            for r in 0..cfg.layers_per_block {
                resnets.push(Resnet::load(&db.pp("resnets").pp(r.to_string()), if r == 0 { prev } else { out }, out, g, dt)?);
            }
            let ds = if bi + 1 < ch.len() { Some(Conv2d::load(&db.pp("downsamplers").pp("0").pp("conv"), out, out, 3, 2, 0, true, dt)?) } else { None };
            down.push((resnets, ds));
            prev = out;
        }
        let last = *ch.last().unwrap();
        let m = e.pp("mid_block");
        let a = m.pp("attentions").pp("0");
        let lin = |n: &str| Linear::load(&a.pp(n), last, last, true, dt);
        let mid = (
            Resnet::load(&m.pp("resnets").pp("0"), last, last, g, dt)?,
            MidAttention {
                norm: GroupNorm::load(&a.pp("group_norm"), g, last, 1e-6, dt)?,
                q: lin("to_q")?,
                k: lin("to_k")?,
                v: lin("to_v")?,
                out: Linear::load(&a.pp("to_out").pp("0"), last, last, true, dt)?,
            },
            Resnet::load(&m.pp("resnets").pp("1"), last, last, g, dt)?,
        );
        let n = cfg.latent_channels * 4;
        let mean = vb.pp("bn").get(n, "running_mean")?.to_dtype(DType::F32)?;
        let var = vb.pp("bn").get(n, "running_var")?.to_dtype(DType::F32)?;
        Ok(Self {
            conv_in: Conv2d::load(&e.pp("conv_in"), 3, ch[0], 3, 1, 1, true, dt)?,
            down,
            mid,
            norm_out: GroupNorm::load(&e.pp("conv_norm_out"), g, last, 1e-6, dt)?,
            conv_out: Conv2d::load(&e.pp("conv_out"), last, 2 * cfg.latent_channels, 3, 1, 1, true, dt)?,
            quant: Conv2d::load(&vb.pp("quant_conv"), 2 * cfg.latent_channels, 2 * cfg.latent_channels, 1, 1, 0, true, dt)?,
            bn_mean_dt: mean.reshape((1, n, 1, 1))?.to_dtype(dt)?,
            bn_std_dt: (var.clone() + cfg.batch_norm_eps)?.sqrt()?.reshape((1, n, 1, 1))?.to_dtype(dt)?,
            bn_mean: mean.reshape((1, n, 1, 1))?,
            bn_std: (var + cfg.batch_norm_eps)?.sqrt()?.reshape((1, n, 1, 1))?,
            latent: cfg.latent_channels,
            dt,
            dev: dev.clone(),
        })
    }

    /// `images` `[B, 3, H, W]` in `[0, 1]` → tokens `[B, 5 + (H/16)(W/16), 128]` f32: the posterior is sampled with
    /// `noise` (a flat tensor of `32 · (H/8) · (W/8)` standard normals; `None`: the posterior mean), 2×2-patchified,
    /// batch-norm normalised and prefixed by five zero tokens to line up with the DINOv3 tokens (class token and
    /// registers). The arithmetic after the encoder runs in the encoder's dtype, rounding where `diffusers` does.
    pub fn forward(&self, images: &Tensor, noise: Option<&Tensor>) -> Result<Tensor> {
        let x = ((images.to_dtype(self.dt)? * 2.0)? - 1.0)?;
        let mut h = self.conv_in.forward(&x)?;
        for (resnets, ds) in &self.down {
            for r in resnets {
                h = r.forward(&h)?;
            }
            if let Some(c) = ds {
                // diffusers pads (0, 1, 0, 1) before the stride-2 convolution
                h = c.forward(&h.pad_with_zeros(3, 0, 1)?.pad_with_zeros(2, 0, 1)?)?;
            }
            crate::progress::check()?;
        }
        h = self.mid.0.forward(&h)?;
        h = self.mid.1.forward(&h)?;
        h = self.mid.2.forward(&h)?;
        let h = self.conv_out.forward(&candle_nn::ops::silu(&self.norm_out.forward(&h)?)?)?;
        let h = self.quant.forward(&h)?;
        let mean = h.narrow(1, 0, self.latent)?;
        let latents = match noise {
            Some(noise) => {
                // DiagonalGaussianDistribution: logvar clamped, std = exp(0.5 logvar), mean + std * noise, all in the encoder dtype
                let logvar = h.narrow(1, self.latent, self.latent)?.clamp(-30.0, 20.0)?;
                let std = (logvar * 0.5)?.exp()?;
                let noise = noise.reshape(mean.dims())?.to_dtype(self.dt)?;
                mean.add(&std.mul(&noise)?)?
            }
            None => mean,
        };
        let (b, c, hh, ww) = latents.dims4()?;
        let l = latents.reshape((b, c, hh / 2, 2, ww / 2, 2))?.permute((0, 1, 3, 5, 2, 4))?.contiguous()?.reshape((b, c * 4, hh / 2, ww / 2))?;
        let l = l.broadcast_sub(&self.bn_mean_dt)?.broadcast_div(&self.bn_std_dt)?.to_dtype(DType::F32)?;
        let _ = (&self.bn_mean, &self.bn_std);
        let tokens = l.flatten_from(2)?.transpose(1, 2)?.contiguous()?; // [B, N, 128]
        let pad = Tensor::zeros((b, 5, c * 4), DType::F32, &self.dev)?;
        Ok(Tensor::cat(&[&pad, &tokens], 1)?)
    }
}
