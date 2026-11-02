//! The random numbers of a request. On CUDA they come from PyTorch's own generator algorithm: Philox4x32-10 through the
//! same `curand` calls, with torch's thread-to-element mapping and its Philox offset bookkeeping
//! (`torch.manual_seed(seed)`, then `randn`/`rand` draws in order), so a seed gives the same noise as the Python
//! reference on the same GPU model. (torch sizes its launch from the GPU's SM count, so streams are per GPU model, as in torch.)
//! Elsewhere the seeded CPU generator of [`crate::rng`] is used.

use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::rng::TorchGen;

pub enum Noise {
    Cpu(TorchGen, Device),
    #[cfg(feature = "cuda")]
    Cuda { dev: Device, seed: u64, offset: u64, sm: (u32, u32) },
}

impl Noise {
    pub fn new(seed: u64, dev: &Device) -> Result<Self> {
        #[cfg(feature = "cuda")]
        if dev.is_cuda() && std::env::var_os("IDOL_CPU_RNG").is_none() {
            return Ok(Noise::Cuda { dev: dev.clone(), seed, offset: 0, sm: crate::fused::sm_info(dev)? });
        }
        Ok(Noise::Cpu(TorchGen::new(seed), dev.clone()))
    }

    /// `torch.randn(numel)` as a flat f32 tensor on the device.
    pub fn randn(&mut self, numel: usize) -> Result<Tensor> {
        match self {
            Noise::Cpu(g, dev) => Ok(Tensor::from_vec(g.randn(numel), numel, dev)?),
            #[cfg(feature = "cuda")]
            Noise::Cuda { dev, seed, offset, sm } => {
                let (t, inc) = crate::fused::philox_fill(dev, numel, *seed, *offset, true, *sm)?;
                *offset += inc;
                Ok(t)
            }
        }
    }

    /// `torch.rand(numel)` (float32, `[0, 1)`) on the host.
    pub fn rand(&mut self, numel: usize) -> Result<Vec<f32>> {
        match self {
            Noise::Cpu(g, _) => Ok((0..numel).map(|_| g.uniform()).collect()),
            #[cfg(feature = "cuda")]
            Noise::Cuda { dev, seed, offset, sm } => {
                let (t, inc) = crate::fused::philox_fill(dev, numel, *seed, *offset, false, *sm)?;
                *offset += inc;
                Ok(t.to_vec1()?)
            }
        }
    }
}
