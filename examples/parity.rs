//! Compare the Rust denoiser and decoders with tensors dumped by `tools/ref_dump.py`:
//!   cargo run --release --example parity -- [models] [out/ref]
use std::collections::HashMap;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use idol_rs::config::IdolConfig;
use idol_rs::decoder::{GaussianDecoder, OctreeDecoder};
use idol_rs::dit::Denoiser;

fn cmp(name: &str, mine: &Tensor, reference: &Tensor) -> Result<()> {
    let a: Vec<f32> = mine.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let b: Vec<f32> = reference.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    anyhow::ensure!(a.len() == b.len(), "{name}: {} vs {} elements", a.len(), b.len());
    let (mut dot, mut na, mut nb, mut max) = (0f64, 0f64, 0f64, 0f32);
    for (x, y) in a.iter().zip(&b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
        max = max.max((x - y).abs());
    }
    println!("{name:<14} cosine {:.6}  max|diff| {max:.4}  ref rms {:.4}  mine rms {:.4}", dot / (na.sqrt() * nb.sqrt()), (nb / b.len() as f64).sqrt(), (na / a.len() as f64).sqrt());
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let models = args.get(1).map(String::as_str).unwrap_or("models");
    let refs = args.get(2).map(String::as_str).unwrap_or("out/ref");
    let dev = idol_rs::backend::new_device(0)?;
    let dt = idol_rs::backend::compute_dtype(&dev);
    let cfg = IdolConfig::load(format!("{models}/idol/config.json").as_ref())?;
    let vb = |f: &str| -> Result<VarBuilder<'static>> { Ok(unsafe { VarBuilder::from_mmaped_safetensors(&[format!("{models}/idol/{f}.safetensors")], DType::F32, &dev)? }) };
    let load = |n: &str| -> Result<HashMap<String, Tensor>> { Ok(candle_core::safetensors::load(format!("{refs}/{n}.safetensors"), &dev)?) };

    let r = load("denoiser")?;
    let den = Denoiser::load(&vb("denoiser")?, &cfg.denoiser.args, dt, &dev)?;
    let h = den.prepare_cond(&r["f1"], &r["f2"])?;
    let (oz, oc) = den.forward(&r["z"], &r["cam"], 700.0, &h)?;
    cmp("denoiser z", &oz, &r["out_z"])?;
    cmp("denoiser cam", &oc, &r["out_cam"])?;

    let r = load("octree")?;
    let oct = OctreeDecoder::load(&vb("decoder")?, &cfg.decoder.args, dt, &dev)?;
    for (n, t) in oct.trace(&r["x"], 16.0, &r["lat"])? {
        cmp(&format!("octree {n}"), &t, &r[&n])?;
    }
    cmp("octree", &oct.forward(&r["x"], 16.0, &r["lat"])?, &r["out"])?;

    let r = load("gs")?;
    let gs = GaussianDecoder::load(&vb("decoder_gs")?, &cfg.decoder_gs.args, dt, &dev)?;
    cmp("gs raw", &gs.forward(&r["pts"], &r["lat"])?, &r["out"])?;
    let pts: Vec<f32> = r["pts"].flatten_all()?.to_vec1()?;
    let anchors: Vec<[f32; 3]> = pts.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let g = gs.decode(&anchors, &r["lat"])?;
    let t = |v: Vec<f32>, w: usize| Tensor::from_vec(v.clone(), (v.len() / w, w), &Device::Cpu);
    cmp("gs xyz", &t(g.xyz.iter().flatten().copied().collect(), 3)?, &r["xyz"].to_device(&Device::Cpu)?)?;
    cmp("gs dc", &t(g.features_dc.iter().flatten().copied().collect(), 3)?, &r["dc"].to_device(&Device::Cpu)?)?;
    cmp("gs scaling", &t(g.scaling.iter().flatten().copied().collect(), 3)?, &r["scaling"].to_device(&Device::Cpu)?)?;
    cmp("gs rotation", &t(g.rotation.iter().flatten().copied().collect(), 4)?, &r["rotation"].to_device(&Device::Cpu)?)?;
    cmp("gs opacity", &t(g.opacity.clone(), 1)?, &r["opacity"].to_device(&Device::Cpu)?)?;
    Ok(())
}
