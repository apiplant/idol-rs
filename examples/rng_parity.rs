//! Bit-exact check of the CUDA noise against `tools/ref_rng.py`:
//!   python tools/ref_rng.py 42 out/ref/rng42.safetensors && cargo run --release --example rng_parity -- 42 out/ref/rng42.safetensors
use anyhow::Result;
use candle_core::DType;
use idol_rs::noise::Noise;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let seed: u64 = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(42);
    let dev = idol_rs::backend::new_device(0)?;
    let r = candle_core::safetensors::load(a.get(2).map(String::as_str).unwrap_or("out/ref/rng42.safetensors"), &dev)?;
    let mut n = Noise::new(seed, &dev)?;
    let mut ok = true;
    let mut check = |name: &str, mine: Vec<f32>, key: &str| -> Result<()> {
        let want: Vec<f32> = r[key].to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let bad = mine.iter().zip(&want).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        println!("{name:<12} {} values, {} differ{}", want.len(), bad, if bad == 0 && mine.len() == want.len() { "  (bit-exact)" } else { "" });
        ok &= bad == 0 && mine.len() == want.len();
        Ok(())
    };
    // the VAE draws bf16 noise: the f32 normal rounded to bf16
    let vae = n.randn(32 * 128 * 128)?.to_dtype(DType::BF16)?.to_dtype(DType::F32)?.to_vec1()?;
    check("vae noise", vae, "vae_noise")?;
    check("latent", n.randn(8192 * 16)?.to_vec1()?, "latent")?;
    check("camera", n.randn(5)?.to_vec1()?, "camera")?;
    check("u1", n.rand(1)?, "u1")?;
    for k in [3usize, 40, 700, 2049] {
        check(&format!("u_{k}"), n.rand(k)?, &format!("u_{k}"))?;
    }
    check("leaf", n.rand(8192 * 3)?, "leaf")?;
    anyhow::ensure!(ok, "the noise differs from torch's");
    println!("all streams bit-exact");
    Ok(())
}
