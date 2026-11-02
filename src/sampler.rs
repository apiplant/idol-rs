//! Euler integration of the rectified flow from noise (t = 1) to data (t = 0), with classifier-free guidance.

use anyhow::Result;
use candle_core::{DType, Tensor};

use crate::dit::Denoiser;
use crate::progress::{self, Stage};

pub struct SampleSettings {
    pub steps: usize,
    /// time shift towards high noise
    pub rescale_t: f64,
    /// guided velocity = (1 + w) v(cond) - w v(uncond)
    pub cfg_strength: f64,
    /// run the conditional and unconditional passes as one batch-2 forward (same result, faster on GPU)
    pub batch_cfg: bool,
}

/// `h_cond` is `[2, Lc, C]`: the prepared condition stream of the image, then of the all-zero condition.
pub fn sample_flow(den: &Denoiser, z: Tensor, cam: Tensor, h_cond: &Tensor, s: &SampleSettings) -> Result<(Tensor, Tensor)> {
    let steps = s.steps;
    let t_seq: Vec<f64> = (0..=steps)
        .map(|i| {
            let t = 1.0 - i as f64 / steps as f64;
            s.rescale_t * t / (1.0 + (s.rescale_t - 1.0) * t)
        })
        .collect();
    let (mut z, mut cam) = (z.to_dtype(DType::F32)?, cam.to_dtype(DType::F32)?);
    let guided = s.cfg_strength > 0.0;
    let w = s.cfg_strength;
    for i in 0..steps {
        progress::check()?;
        let (t, t_prev) = (t_seq[i], t_seq[i + 1]);
        let tv = (1000.0 * t) as f32;
        let (vz, vc) = if guided && s.batch_cfg {
            let (zz, cc) = (Tensor::cat(&[&z, &z], 0)?, Tensor::cat(&[&cam, &cam], 0)?);
            let (oz, oc) = den.forward(&zz, &cc, tv, h_cond)?;
            let g = |o: &Tensor| -> Result<Tensor> { Ok(((o.narrow(0, 0, 1)? * (1.0 + w))? - (o.narrow(0, 1, 1)? * w)?)?) };
            (g(&oz)?, g(&oc)?)
        } else if guided {
            let (cz, cc) = den.forward(&z, &cam, tv, &h_cond.narrow(0, 0, 1)?)?;
            let (uz, uc) = den.forward(&z, &cam, tv, &h_cond.narrow(0, 1, 1)?)?;
            ((((cz * (1.0 + w))? - (uz * w)?)?), (((cc * (1.0 + w))? - (uc * w)?)?))
        } else {
            den.forward(&z, &cam, tv, &h_cond.narrow(0, 0, 1)?)?
        };
        let dt = t - t_prev;
        z = (z - (vz * dt)?)?;
        cam = (cam - (vc * dt)?)?;
        progress::count(Stage::Sampling, i + 1, steps);
    }
    Ok((z, cam))
}
