//! Fused CUDA kernels for the denoiser's memory-bound steps (see `kernels/idol.cu`): each replaces a chain of
//! candle elementwise ops, which read and write the whole `[B, L, 1024]` activation once per op, with one pass.
//! Everything falls back to the plain ops (callers check [`available`]) off CUDA and for non-f16 activations.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// Whether the fused kernels can run for activations of this dtype on this device.
pub fn available(dev: &Device, dt: DType) -> bool {
    cfg!(feature = "cuda") && dev.is_cuda() && dt == DType::F16 && std::env::var_os("IDOL_NO_FUSED").is_none()
}

#[cfg(feature = "cuda")]
mod cuda {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{CudaView, LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaDevice, CudaStorage};
    use candle_core::{CustomOp1, DType, Layout, Shape, Storage, Tensor};
    use half::f16;

    const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/idol.ptx"));

    #[derive(Clone, Copy)]
    pub enum Scalar {
        I32(i32),
        F32(f32),
        Usize(usize),
        I64(i64),
        U64(u64),
    }

    pub struct Kernel {
        pub name: &'static str,
        /// the inputs after the one the op is applied to, in kernel argument order
        pub extra: Vec<Tensor>,
        pub out_dtype: DType,
        pub out_shape: Shape,
        pub scalars: Vec<Scalar>,
        pub grid: u32,
        pub block: u32,
    }

    enum Arg<'a> {
        H(CudaView<'a, f16>),
        F(CudaView<'a, f32>),
    }

    fn cuda_err<E: std::fmt::Display>(e: E) -> candle_core::Error {
        candle_core::Error::Msg(format!("cuda: {e}"))
    }

    fn view<'a>(s: &'a CudaStorage, l: &Layout) -> candle_core::Result<Arg<'a>> {
        if !l.is_contiguous() {
            candle_core::bail!("fused kernels need contiguous inputs");
        }
        let off = l.start_offset();
        Ok(match s.dtype() {
            DType::F16 => Arg::H(s.as_cuda_slice::<f16>()?.slice(off..)),
            DType::F32 => Arg::F(s.as_cuda_slice::<f32>()?.slice(off..)),
            d => candle_core::bail!("fused kernels do not take {d:?}"),
        })
    }

    impl CustomOp1 for Kernel {
        fn name(&self) -> &'static str {
            "idol-fused"
        }

        fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &Layout) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
            candle_core::bail!("idol fused kernels run on CUDA only")
        }

        fn cuda_fwd(&self, first: &CudaStorage, first_layout: &Layout) -> candle_core::Result<(CudaStorage, Shape)> {
            let dev: &CudaDevice = first.device();
            let func = dev.get_or_load_custom_func(self.name, "idol_fused", PTX)?;
            let guards: Vec<_> = self.extra.iter().map(|t| t.storage_and_layout()).collect();
            let mut args = vec![view(first, first_layout)?];
            for (g, l) in &guards {
                match &**g {
                    Storage::Cuda(s) => args.push(view(s, l)?),
                    _ => candle_core::bail!("fused kernel input is not on the GPU"),
                }
            }
            let n = self.out_shape.elem_count();
            let cfg = LaunchConfig { grid_dim: (self.grid, 1, 1), block_dim: (self.block, 1, 1), shared_mem_bytes: 0 };
            // SAFETY: the kernel writes every element of the output before it is read
            macro_rules! run {
                ($ty:ty) => {{
                    let mut out = unsafe { dev.alloc::<$ty>(n)? };
                    let mut b = func.builder();
                    for a in &args {
                        match a {
                            Arg::H(v) => b.arg(v),
                            Arg::F(v) => b.arg(v),
                        };
                    }
                    b.arg(&mut out);
                    for s in &self.scalars {
                        match s {
                            Scalar::I32(v) => b.arg(v),
                            Scalar::F32(v) => b.arg(v),
                            Scalar::Usize(v) => b.arg(v),
                            Scalar::I64(v) => b.arg(v),
                            Scalar::U64(v) => b.arg(v),
                        };
                    }
                    unsafe { b.launch(cfg) }.map_err(cuda_err)?;
                    CudaStorage::wrap_cuda_slice(out, dev.clone())
                }};
            }
            let storage = match self.out_dtype {
                DType::F16 => run!(f16),
                DType::F32 => run!(f32),
                d => candle_core::bail!("fused kernels do not produce {d:?}"),
            };
            Ok((storage, self.out_shape.clone()))
        }
    }
}

#[cfg(feature = "cuda")]
use cuda::{Kernel, Scalar};

#[cfg(feature = "cuda")]
fn contiguous(t: &Tensor) -> Result<Tensor> {
    Ok(t.contiguous()?)
}

/// `LayerNorm(x) * (1 + scale) + shift` over the last dim of `x` `[.., C]`; `m` is the adaLN output `[1, 6C]`,
/// shift = chunk `slot`, scale = chunk `slot + 1`.
#[cfg(feature = "cuda")]
pub fn mod_ln(x: &Tensor, m: &Tensor, slot: usize, eps: f32) -> Result<Tensor> {
    let c = *x.dims().last().unwrap();
    anyhow::ensure!(c <= 2048, "mod_ln supports up to 2048 channels");
    let rows = x.elem_count() / c;
    let k = Kernel {
        name: "mod_ln_f16",
        extra: vec![contiguous(m)?],
        out_dtype: DType::F16,
        out_shape: x.shape().clone(),
        scalars: vec![Scalar::I32(c as i32), Scalar::I32(slot as i32), Scalar::F32(eps)],
        grid: rows as u32,
        block: 128,
    };
    Ok(contiguous(x)?.apply_op1_no_bwd(&k)?)
}

/// q, k, v of a fused qkv projection `qkv` `[B, L, 3C]` (no bias applied): `[3, B, L, H, 64]`, q and k rotated by
/// `cs` `[2, B, L, H, 32]` (cos, sin) and RMS-normalised with `gq`, `gk` `[H, 64]`.
#[cfg(feature = "cuda")]
pub fn rope_qkv(qkv: &Tensor, bias: &Tensor, cs: &Tensor, gq: &Tensor, gk: &Tensor, heads: usize) -> Result<Tensor> {
    let (b, l, c3) = qkv.dims3()?;
    let rows = b * l;
    let hd = c3 / 3 / heads;
    anyhow::ensure!(hd == 64, "rope_qkv needs 64-dim heads");
    let k = Kernel {
        name: "rope_qkv_f16",
        extra: vec![contiguous(bias)?, contiguous(cs)?, contiguous(gq)?, contiguous(gk)?],
        out_dtype: DType::F16,
        out_shape: (3, b, l, heads, hd).into(),
        scalars: vec![Scalar::I32(rows as i32), Scalar::I32(heads as i32), Scalar::F32(1e-12)],
        grid: (rows * heads * 32).div_ceil(256) as u32,
        block: 256,
    };
    Ok(contiguous(qkv)?.apply_op1_no_bwd(&k)?)
}

/// `x + (y + bias) * gate`, gate = chunk `slot` of the adaLN output `m` `[1, 6C]`.
#[cfg(feature = "cuda")]
pub fn bias_gate_res(x: &Tensor, y: &Tensor, bias: &Tensor, m: &Tensor, slot: usize) -> Result<Tensor> {
    let c = *x.dims().last().unwrap();
    let n2 = x.elem_count() / 2;
    let k = Kernel {
        name: "bias_gate_res_f16",
        extra: vec![contiguous(y)?, contiguous(bias)?, contiguous(m)?],
        out_dtype: DType::F16,
        out_shape: x.shape().clone(),
        scalars: vec![Scalar::I32(c as i32), Scalar::I32(slot as i32), Scalar::Usize(n2)],
        grid: n2.div_ceil(256) as u32,
        block: 256,
    };
    Ok(contiguous(x)?.apply_op1_no_bwd(&k)?)
}

/// `gelu_tanh(x + bias)`.
#[cfg(feature = "cuda")]
pub fn bias_gelu(x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let n = *x.dims().last().unwrap();
    let n2 = x.elem_count() / 2;
    let k = Kernel {
        name: "bias_gelu_f16",
        extra: vec![contiguous(bias)?],
        out_dtype: DType::F16,
        out_shape: x.shape().clone(),
        scalars: vec![Scalar::I32(n as i32), Scalar::Usize(n2)],
        grid: n2.div_ceil(256) as u32,
        block: 256,
    };
    Ok(contiguous(x)?.apply_op1_no_bwd(&k)?)
}

/// RePo rotary angles: `o` `[B, L, 3H]` f32 offsets, `freqs` `[32]` → `[2, B, L, H, 32]` (cos, sin).
#[cfg(feature = "cuda")]
pub fn repo_angles(o: &Tensor, freqs: &Tensor, heads: usize) -> Result<Tensor> {
    let (b, l, _) = o.dims3()?;
    let rows = b * l;
    let total = rows * heads * 32;
    let k = Kernel {
        name: "repo_angles_f32",
        extra: vec![contiguous(freqs)?],
        out_dtype: DType::F32,
        out_shape: (2, b, l, heads, 32).into(),
        scalars: vec![Scalar::I32(rows as i32), Scalar::I32(heads as i32)],
        grid: total.div_ceil(256) as u32,
        block: 256,
    };
    Ok(contiguous(o)?.apply_op1_no_bwd(&k)?)
}

/// What torch's `calc_execution_policy` needs of the GPU: `(multiProcessorCount, maxThreadsPerMultiProcessor)`.
#[cfg(feature = "cuda")]
pub fn sm_info(dev: &Device) -> Result<(u32, u32)> {
    use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute as A;
    let ctx = dev.as_cuda_device()?.cuda_stream().context().clone();
    Ok((ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32, ctx.attribute(A::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)? as u32))
}

/// `numel` floats of PyTorch's CUDA generator state `(seed, offset)`: `normal` is `torch.randn`, otherwise `torch.rand`.
/// Returns the values and the Philox offset increment torch would apply for this call.
#[cfg(feature = "cuda")]
pub fn philox_fill(dev: &Device, numel: usize, seed: u64, offset: u64, normal: bool, sm: (u32, u32)) -> Result<(Tensor, u64)> {
    anyhow::ensure!(numel > 0, "empty random tensor");
    // torch: block 256, grid = min(ceil(numel / 256), SMs * (maxThreadsPerSM / 256)), 4 values per thread per round
    let block = 256u64;
    let grid = (numel as u64).div_ceil(block).min(sm.0 as u64 * (sm.1 as u64 / block));
    let increment = ((numel as u64 - 1) / (block * grid * 4) + 1) * 4;
    let k = Kernel {
        name: "philox_fill_f32",
        extra: vec![],
        out_dtype: DType::F32,
        out_shape: numel.into(),
        scalars: vec![Scalar::I64(numel as i64), Scalar::U64(seed), Scalar::U64(offset), Scalar::I32(if normal { 0 } else { 1 })],
        grid: grid as u32,
        block: block as u32,
    };
    let dummy = Tensor::zeros(1, DType::F32, dev)?;
    Ok((dummy.apply_op1_no_bwd(&k)?, increment))
}

#[cfg(not(feature = "cuda"))]
mod stub {
    use super::*;
    fn no() -> anyhow::Error {
        anyhow::anyhow!("built without CUDA")
    }
    pub fn mod_ln(_: &Tensor, _: &Tensor, _: usize, _: f32) -> Result<Tensor> {
        Err(no())
    }
    pub fn rope_qkv(_: &Tensor, _: &Tensor, _: &Tensor, _: &Tensor, _: &Tensor, _: usize) -> Result<Tensor> {
        Err(no())
    }
    pub fn bias_gate_res(_: &Tensor, _: &Tensor, _: &Tensor, _: &Tensor, _: usize) -> Result<Tensor> {
        Err(no())
    }
    pub fn bias_gelu(_: &Tensor, _: &Tensor) -> Result<Tensor> {
        Err(no())
    }
    pub fn repo_angles(_: &Tensor, _: &Tensor, _: usize) -> Result<Tensor> {
        Err(no())
    }
    pub fn sm_info(_: &Device) -> Result<(u32, u32)> {
        Err(no())
    }
    pub fn philox_fill(_: &Device, _: usize, _: u64, _: u64, _: bool, _: (u32, u32)) -> Result<(Tensor, u64)> {
        Err(no())
    }
}
#[cfg(not(feature = "cuda"))]
pub use stub::*;
