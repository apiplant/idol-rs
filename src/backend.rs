//! Device selection is a build-time choice; CUDA remains the default.
use anyhow::Result;
use candle_core::{DType, Device};

pub fn new_device(index: usize) -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        Ok(Device::new_cuda(index)?)
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        anyhow::ensure!(index == 0, "Apple Silicon has one unified-memory GPU; use --device 0");
        Ok(Device::new_metal(index)?)
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        let _ = index;
        Ok(Device::Cpu)
    }
}

/// The dtype of the transformer blocks: half precision on a GPU, f32 on the CPU.
pub fn compute_dtype(dev: &Device) -> DType {
    if dev.is_cpu() || std::env::var("IDOL_DTYPE").is_ok_and(|v| v == "f32") {
        DType::F32
    } else {
        DType::F16
    }
}

/// The dtype of the two image encoders (the reference runs DINOv3 and the VAE in bf16).
pub fn encoder_dtype(dev: &Device, fp32: bool) -> DType {
    if fp32 || dev.is_cpu() {
        DType::F32
    } else if dev.is_cuda() {
        DType::BF16
    } else {
        DType::F16
    }
}
