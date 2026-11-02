//! Shared tensor ops: linear layers, norms, attention, rotary embeddings, sinusoidal embeddings.
//!
//! Every transformer here follows the reference's mixed precision: the weights of the blocks' `Linear` layers
//! live in the compute dtype (f16 on GPU), everything else (embedders, norms, RePo, output heads) stays f32,
//! and tensors are cast where the reference casts them.

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;

/// `y = x W^T + b` over the last dim of any-rank `x`. The input is cast to the weight dtype.
#[derive(Clone)]
pub struct Linear {
    w: Tensor, // [out, in]
    b: Option<Tensor>,
}

impl Linear {
    pub fn load(vb: &VarBuilder, in_dim: usize, out_dim: usize, bias: bool, dt: DType) -> Result<Self> {
        let w = vb.get((out_dim, in_dim), "weight")?.to_dtype(dt)?;
        let b = if bias { Some(vb.get(out_dim, "bias")?.to_dtype(dt)?) } else { None };
        Ok(Self { w, b })
    }

    pub fn from_tensors(w: Tensor, b: Option<Tensor>) -> Self {
        Self { w, b }
    }

    pub fn dtype(&self) -> DType {
        self.w.dtype()
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.b.as_ref()
    }

    /// `x W^T` without the bias (for the fused kernels, which add it).
    pub fn matmul(&self, x: &Tensor) -> Result<Tensor> {
        let x = if x.dtype() != self.w.dtype() { x.to_dtype(self.w.dtype())? } else { x.clone() };
        let dims = x.dims().to_vec();
        let last = *dims.last().unwrap();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let y = x.reshape((rows, last))?.contiguous()?.matmul(&self.w.t()?)?;
        let mut out = dims;
        *out.last_mut().unwrap() = self.w.dim(0)?;
        Ok(y.reshape(out)?)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.matmul(x)?;
        match &self.b {
            Some(b) => Ok(y.broadcast_add(b)?),
            None => Ok(y),
        }
    }
}

/// Plain LayerNorm over the last dim: computed in f32, returned in the input dtype (the reference's `LayerNorm32`).
#[derive(Clone)]
pub struct LayerNorm {
    w: Tensor,
    b: Tensor,
    eps: f32,
}

impl LayerNorm {
    /// `affine = false` gives the reference's `elementwise_affine=False` (ones and zeros).
    pub fn load(vb: &VarBuilder, dim: usize, eps: f32, affine: bool, dev: &Device) -> Result<Self> {
        if affine {
            Ok(Self { w: vb.get(dim, "weight")?.to_dtype(DType::F32)?, b: vb.get(dim, "bias")?.to_dtype(DType::F32)?, eps })
        } else {
            Ok(Self::plain(dim, eps, dev)?)
        }
    }

    pub fn plain(dim: usize, eps: f32, dev: &Device) -> Result<Self> {
        Ok(Self { w: Tensor::ones(dim, DType::F32, dev)?, b: Tensor::zeros(dim, DType::F32, dev)?, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dt = x.dtype();
        let x = if dt == DType::F32 { x.contiguous()? } else { x.to_dtype(DType::F32)?.contiguous()? };
        let y = candle_nn::ops::layer_norm(&x, &self.w, &self.b, self.eps)?;
        Ok(if dt == DType::F32 { y } else { y.to_dtype(dt)? })
    }
}

/// `F.layer_norm(x, x.shape[-1:])`: no affine, eps 1e-5, f32 result.
pub fn layer_norm_plain(x: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(DType::F32)?.contiguous()?;
    let d = x.dim(D::Minus1)?;
    let w = Tensor::ones(d, DType::F32, x.device())?;
    let b = Tensor::zeros(d, DType::F32, x.device())?;
    Ok(candle_nn::ops::layer_norm(&x, &w, &b, 1e-5)?)
}

/// `MultiHeadRMSNorm`: `normalize(x, dim=-1) * gamma * sqrt(head_dim)` on `[B, L, H, D]`, in f32.
#[derive(Clone)]
pub struct HeadRmsNorm {
    gamma: Tensor, // [H, D]
    ones: Tensor,
}

impl HeadRmsNorm {
    pub fn load(vb: &VarBuilder, heads: usize, dim: usize) -> Result<Self> {
        // rms_norm already is normalize(x) * sqrt(D), so gamma is used as it is
        let gamma = vb.get((heads, dim), "gamma")?.to_dtype(DType::F32)?;
        let ones = Tensor::ones(dim, DType::F32, gamma.device())?;
        Ok(Self { gamma, ones })
    }

    pub fn gamma(&self) -> &Tensor {
        &self.gamma
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dt = x.dtype();
        let x = x.to_dtype(DType::F32)?.contiguous()?;
        // normalize(x) * sqrt(D) == rms_norm(x) up to the epsilon placement
        let y = candle_nn::ops::rms_norm(&x, &self.ones, 1e-12)?;
        Ok(y.broadcast_mul(&self.gamma)?.to_dtype(dt)?)
    }
}

/// Scaled dot-product attention on `[B, L, H, D]` q and `[B, S, H, D]` k, v → `[B, L, H, D]`.
/// Flash attention on CUDA f16/bf16 when built with it, otherwise a query-chunked softmax(QK^T)V.
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let d = q.dim(D::Minus1)?;
    let scale = 1.0 / (d as f32).sqrt();
    #[cfg(feature = "flash-attn")]
    if q.device().is_cuda() && matches!(q.dtype(), DType::F16 | DType::BF16) && d <= 256 {
        return Ok(candle_flash_attn::flash_attn(&q.contiguous()?, &k.contiguous()?, &v.contiguous()?, scale, false)?);
    }
    chunked_sdpa(q, k, v, scale)
}

/// Same result as flash attention, without a kernel: queries are processed in chunks so the score matrix stays
/// under ~768 MB.
pub fn chunked_sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
    let (b, l, h, _) = q.dims4()?;
    let s = k.dim(1)?;
    let q = q.transpose(1, 2)?.contiguous()?; // [B, H, L, D]
    let kt = k.transpose(1, 2)?.transpose(2, 3)?.contiguous()?; // [B, H, D, S]
    let v = v.transpose(1, 2)?.contiguous()?; // [B, H, S, D]
    let esize = q.dtype().size_in_bytes();
    let chunk = ((768usize << 20) / (b * h * s * esize).max(1)).clamp(64, l.max(64));
    let mut outs = Vec::new();
    let mut at = 0;
    while at < l {
        let n = chunk.min(l - at);
        let qc = q.narrow(2, at, n)?;
        let scores = (qc.matmul(&kt)? * scale as f64)?;
        let p = candle_nn::ops::softmax_last_dim(&scores)?;
        outs.push(p.matmul(&v)?);
        at += n;
    }
    let o = if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 2)? };
    Ok(o.transpose(1, 2)?.contiguous()?)
}

/// tanh-approximated GELU (`nn.GELU(approximate="tanh")`).
pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu()?)
}

/// `FeedForwardNet`: Linear → GELU(tanh) → Linear.
pub struct FeedForward {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl FeedForward {
    pub fn load(vb: &VarBuilder, channels: usize, ratio: f64, dt: DType) -> Result<Self> {
        let hidden = (channels as f64 * ratio) as usize;
        let m = vb.pp("mlp");
        Ok(Self { fc1: Linear::load(&m.pp("0"), channels, hidden, true, dt)?, fc2: Linear::load(&m.pp("2"), hidden, channels, true, dt)? })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&gelu_tanh(&self.fc1.forward(x)?)?)
    }
}

/// `MLP(channels, inner, out, layers)`: `layers - 1` × (Linear → GELU(tanh)) then a Linear.
pub struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    pub fn load(vb: &VarBuilder, channels: usize, inner: usize, out: usize, layers: usize, dt: DType) -> Result<Self> {
        let m = vb.pp("mlp");
        let mut ls = Vec::new();
        for i in 0..layers - 1 {
            ls.push(Linear::load(&m.pp((2 * i).to_string()), if i == 0 { channels } else { inner }, inner, true, dt)?);
        }
        ls.push(Linear::load(&m.pp((2 * (layers - 1)).to_string()), inner, out, true, dt)?);
        Ok(Self { layers: ls })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (i, l) in self.layers.iter().enumerate() {
            x = l.forward(&x)?;
            if i + 1 < self.layers.len() {
                x = gelu_tanh(&x)?;
            }
        }
        Ok(x)
    }
}

/// Sinusoidal embedding of one scalar followed by the 2-layer SiLU MLP (`SinusoidalEmbedder`), → `[1, C]`.
pub struct ScalarEmbedder {
    fc1: Linear,
    fc2: Linear,
    freq_dim: usize,
    max_period: f64,
    two_pi: bool,
}

impl ScalarEmbedder {
    pub fn load(vb: &VarBuilder, hidden: usize, max_period: f64, two_pi: bool) -> Result<Self> {
        let freq_dim = 256;
        let m = vb.pp("mlp");
        Ok(Self {
            fc1: Linear::load(&m.pp("0"), freq_dim, hidden, true, DType::F32)?,
            fc2: Linear::load(&m.pp("2"), hidden, hidden, true, DType::F32)?,
            freq_dim,
            max_period,
            two_pi,
        })
    }

    pub fn forward(&self, t: f32, dev: &Device) -> Result<Tensor> {
        let half = self.freq_dim / 2;
        let mut e = vec![0f32; self.freq_dim];
        for i in 0..half {
            let f = (-(self.max_period.ln()) * i as f64 / half as f64).exp() as f32;
            let mut a = t * f;
            if self.two_pi {
                a = a * 2.0 * std::f32::consts::PI;
            }
            e[i] = a.cos();
            e[half + i] = a.sin();
        }
        let x = Tensor::from_vec(e, (1, self.freq_dim), dev)?;
        let h = candle_nn::ops::silu(&self.fc1.forward(&x)?)?;
        self.fc2.forward(&h)
    }
}

/// The reference's complex-product RoPE on `[B, L, H, D]`: pairs `(x[2i], x[2i+1])` are rotated by `cos`, `sin` of
/// shape `[B, L, H, D/2]` (f32). Returns the input dtype.
pub fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, l, h, d) = x.dims4()?;
    let xf = x.to_dtype(DType::F32)?.reshape((b, l, h, d / 2, 2))?;
    let x0 = xf.narrow(4, 0, 1)?.squeeze(4)?;
    let x1 = xf.narrow(4, 1, 1)?.squeeze(4)?;
    let r0 = (x0.mul(cos)? - x1.mul(sin)?)?;
    let r1 = (x0.mul(sin)? + x1.mul(cos)?)?;
    let out = Tensor::stack(&[&r0, &r1], 4)?.reshape((b, l, h, d))?;
    Ok(out.to_dtype(x.dtype())?)
}

/// `softplus` on a CPU slice (the numerically stable form).
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// 2D convolution with weights in a chosen dtype.
#[derive(Clone)]
pub struct Conv2d {
    pub w: Tensor, // [O, I, kh, kw]
    pub b: Option<Tensor>,
    pub stride: usize,
    pub pad: usize,
}

impl Conv2d {
    pub fn load(vb: &VarBuilder, i: usize, o: usize, k: usize, stride: usize, pad: usize, bias: bool, dt: DType) -> Result<Self> {
        Ok(Self {
            w: vb.get((o, i, k, k), "weight")?.to_dtype(dt)?,
            b: if bias { Some(vb.get(o, "bias")?.to_dtype(dt)?) } else { None },
            stride,
            pad,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv2d(&self.w, self.pad, self.stride, 1, 1)?;
        match &self.b {
            Some(b) => Ok(y.broadcast_add(&b.reshape((1, (), 1, 1))?)?),
            None => Ok(y),
        }
    }
}

/// `nn.GroupNorm` (statistics in f32), `x` `[B, C, H, W]`.
pub struct GroupNorm {
    inner: candle_nn::GroupNorm,
}

impl GroupNorm {
    pub fn load(vb: &VarBuilder, groups: usize, channels: usize, eps: f64, dt: DType) -> Result<Self> {
        let w = vb.get(channels, "weight")?.to_dtype(dt)?;
        let b = vb.get(channels, "bias")?.to_dtype(dt)?;
        Ok(Self { inner: candle_nn::GroupNorm::new(w, b, channels, groups, eps)? })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(candle_core::Module::forward(&self.inner, x)?)
    }
}


/// Opt-in section timing (`IDOL_PROFILE=1`): synchronises the device around each section, so only for diagnosis.
pub mod prof {
    use std::collections::BTreeMap;
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    use candle_core::Device;

    pub fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("IDOL_PROFILE").is_some())
    }

    fn table() -> &'static Mutex<BTreeMap<&'static str, (f64, u32)>> {
        static T: OnceLock<Mutex<BTreeMap<&'static str, (f64, u32)>>> = OnceLock::new();
        T.get_or_init(Default::default)
    }

    /// Time `f` under `name`.
    pub fn time<T>(name: &'static str, dev: &Device, f: impl FnOnce() -> T) -> T {
        if !on() {
            return f();
        }
        let _ = dev.synchronize();
        let t = Instant::now();
        let r = f();
        let _ = dev.synchronize();
        let mut g = table().lock().unwrap();
        let e = g.entry(name).or_insert((0.0, 0));
        e.0 += t.elapsed().as_secs_f64();
        e.1 += 1;
        r
    }

    pub fn report() {
        if !on() {
            return;
        }
        let mut g = table().lock().unwrap();
        let total: f64 = g.values().map(|v| v.0).sum();
        for (k, (t, n)) in g.iter() {
            eprintln!("[prof] {k:<18} {t:>7.3}s  {n:>6} calls  {:>5.1}%", 100.0 * t / total);
        }
        g.clear();
    }
}
