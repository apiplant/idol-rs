//! Check the low-level ops against plain math: cargo run --release --example prims
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use idol_rs::ops;

fn diff(name: &str, a: &Tensor, b: &Tensor) -> Result<()> {
    let d = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?.abs()?.flatten_all()?;
    println!("{name:<28} max|diff| {:.5}  mean|diff| {:.6}", d.max(0)?.to_scalar::<f32>()?, d.mean(0)?.to_scalar::<f32>()?);
    Ok(())
}

fn main() -> Result<()> {
    let dev = idol_rs::backend::new_device(0)?;
    // rms_norm kernel vs manual normalize
    let x = Tensor::randn(0f32, 3.0, (1, 200, 16, 64), &dev)?;
    let ones = Tensor::ones(64, DType::F32, &dev)?;
    let k = candle_nn::ops::rms_norm(&x, &ones, 1e-12)?;
    let manual = x.broadcast_div(&(x.sqr()?.mean_keepdim(3)? + 1e-12)?.sqrt()?)?;
    diff("rms_norm kernel", &k, &manual)?;
    // layer_norm kernel vs manual
    let c = Tensor::randn(0f32, 3.0, (1, 200, 1024), &dev)?;
    let (w, b) = (Tensor::ones(1024, DType::F32, &dev)?, Tensor::zeros(1024, DType::F32, &dev)?);
    let ln = candle_nn::ops::layer_norm(&c, &w, &b, 1e-6)?;
    let mean = c.mean_keepdim(2)?;
    let cc = c.broadcast_sub(&mean)?;
    let ln_m = cc.broadcast_div(&(cc.sqr()?.mean_keepdim(2)? + 1e-6)?.sqrt()?)?;
    diff("layer_norm kernel", &ln, &ln_m)?;
    // flash vs chunked attention, cross-shaped
    for (lq, lk) in [(200usize, 8192usize), (1024, 1024)] {
        let q = Tensor::randn(0f32, 1.0, (1, lq, 16, 64), &dev)?.to_dtype(DType::F16)?;
        let kk = Tensor::randn(0f32, 1.0, (1, lk, 16, 64), &dev)?.to_dtype(DType::F16)?;
        let v = Tensor::randn(0f32, 1.0, (1, lk, 16, 64), &dev)?.to_dtype(DType::F16)?;
        let flash = ops::sdpa(&q, &kk, &v)?;
        let chunk = ops::chunked_sdpa(&q, &kk, &v, 0.125)?;
        // fully manual in f32
        let (qf, kf, vf) = (q.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?, kk.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?, v.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?);
        let p = candle_nn::ops::softmax_last_dim(&(qf.matmul(&kf.transpose(2, 3)?.contiguous()?)? * 0.125)?)?;
        let man = p.matmul(&vf)?.transpose(1, 2)?;
        diff(&format!("sdpa (flash) {lq}x{lk}"), &flash, &man)?;
        diff(&format!("sdpa (chunked) {lq}x{lk}"), &chunk, &man)?;
    }
    // gelu: candle's vs tanh formula
    let g = Tensor::randn(0f32, 2.0, (4096,), &dev)?;
    let tanh = ((&g * 0.044715)?.mul(&g.sqr()?)? + &g)?;
    let gt = (g.clone() * 0.5)?.mul(&((tanh * 0.7978845608028654)?.tanh()? + 1.0)?)?;
    diff("gelu vs tanh formula", &ops::gelu_tanh(&g)?, &gt)?;
    let _ = Device::Cpu;
    Ok(())
}
