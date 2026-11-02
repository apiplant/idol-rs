use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use idol_rs::config::{DEFAULT_GAUSSIANS, MAX_GAUSSIANS, MIN_GAUSSIANS};
use idol_rs::download::{self, log, ModelPaths};
use idol_rs::pipeline::{Engine, EngineOptions, Request};
use idol_rs::preprocess::Mode;

const IMAGE_EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "webp", "bmp"];

#[derive(Parser)]
#[command(name = "idol", version, about = "Idol: one image to a 3D Gaussian splat (candle, CUDA or Metal)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

/// `./models` when it holds the Idol weights, else `~/.cache/idol-rs/models`.
fn default_models_dir() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    let local = PathBuf::from("models");
    if local.join("idol").is_dir() {
        local
    } else {
        home.join(".cache/idol-rs/models")
    }
}

#[derive(clap::Args, Clone)]
struct Common {
    /// Models directory: `idol/`, `dinov3/` and `flux2-vae/` live in it (missing files are downloaded).
    #[arg(long, default_value_os_t = default_models_dir(), env = "IDOL_MODELS_DIR")]
    models_dir: PathBuf,
    /// GPU index.
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Run DINOv3 and the FLUX.2 VAE in f32 instead of bf16 (1.7 GB more memory).
    #[arg(long)]
    fp32_encoders: bool,
    /// Load each model only while its stage runs (fits 8-12 GB GPUs, slower for repeated requests).
    #[arg(long)]
    low_vram: bool,
}

impl Common {
    fn engine(&self) -> Result<Engine> {
        Engine::new(ModelPaths::new(&self.models_dir), EngineOptions { device: self.device, fp32_encoders: self.fp32_encoders, low_vram: self.low_vram })
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Turn an image (or a folder of images) into 3D Gaussians: writes <name>.ply, <name>_input.png, <name>_camera.json.
    Generate {
        /// An image, or a folder of images.
        #[arg(long, short)]
        image: PathBuf,
        #[arg(long, short, default_value = "outputs")]
        out: PathBuf,
        /// synthetic: renders and generated images of one object (crop 1.2); real: photos (crop 1.7).
        #[arg(long, value_enum, default_value = "synthetic")]
        mode: Mode,
        /// Override the crop: square side = object bounding box x this.
        #[arg(long)]
        crop_margin: Option<f32>,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 50)]
        steps: usize,
        /// Classifier-free guidance strength.
        #[arg(long, default_value_t = 3.0)]
        cfg_strength: f64,
        /// Time-step shift towards high noise.
        #[arg(long, default_value_t = 3.0)]
        rescale_t: f64,
        /// 32,768 to 1,048,576, rounded down to a multiple of 32.
        #[arg(long, default_value_t = DEFAULT_GAUSSIANS)]
        num_gaussians: usize,
        /// Run the guidance passes one after the other instead of as one batch (less memory, slower).
        #[arg(long)]
        no_batch_cfg: bool,
        /// Turn the result over (180 degrees about the input camera's view axis), for objects that come out upside down.
        #[arg(long)]
        flip: bool,
        #[command(flatten)]
        common: Common,
    },
    /// Start the HTTP server and web UI.
    #[cfg(feature = "serve")]
    Serve {
        #[arg(long, default_value = "0.0.0.0:8190", env = "IDOL_ADDR")]
        addr: String,
        /// Turn results over by default (requests can say `flip`: false): for models that return objects upside down.
        #[arg(long)]
        flip: bool,
        #[command(flatten)]
        common: Common,
    },
    /// Download the model files that are missing (needs a Hugging Face token for the two gated encoders).
    Download {
        #[arg(long, default_value_os_t = default_models_dir(), env = "IDOL_MODELS_DIR")]
        models_dir: PathBuf,
    },
    /// Show which model files are present.
    Models {
        #[arg(long, default_value_os_t = default_models_dir(), env = "IDOL_MODELS_DIR")]
        models_dir: PathBuf,
    },
}

fn images_in(path: &Path) -> Result<Vec<PathBuf>> {
    if !path.is_dir() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut v: Vec<PathBuf> = std::fs::read_dir(path)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()).is_some_and(|e| IMAGE_EXTENSIONS.contains(&e.to_lowercase().as_str())))
        .collect();
    v.sort();
    Ok(v)
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Generate { image, out, mode, crop_margin, seed, steps, cfg_strength, rescale_t, num_gaussians, no_batch_cfg, flip, common } => {
            if !(MIN_GAUSSIANS..=MAX_GAUSSIANS).contains(&num_gaussians) {
                bail!("--num-gaussians must be between {MIN_GAUSSIANS} and {MAX_GAUSSIANS}");
            }
            let images = images_in(&image)?;
            if images.is_empty() {
                bail!("no images found in {}", image.display());
            }
            let mut engine = common.engine()?;
            std::fs::create_dir_all(&out)?;
            for path in images {
                let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("object").to_string();
                let img = image::open(&path).with_context(|| format!("reading {}", path.display()))?;
                let mut req = Request::new(img);
                req.mode = mode;
                req.crop_margin = crop_margin;
                req.seed = seed;
                req.steps = steps;
                req.cfg_strength = cfg_strength;
                req.rescale_t = rescale_t;
                req.num_gaussians = num_gaussians / 32 * 32;
                req.batch_cfg = !no_batch_cfg && !common.low_vram;
                req.flip = flip;
                let o = engine.run(&req)?;
                o.gaussians.save_ply(&out.join(format!("{name}.ply")))?;
                o.preprocessed.save(out.join(format!("{name}_input.png")))?;
                std::fs::write(out.join(format!("{name}_camera.json")), serde_json::to_string_pretty(&o.camera)?)?;
                println!("{name}: {} ({} Gaussians, {:.1}s)", out.join(format!("{name}.ply")).display(), o.gaussians.len(), o.seconds);
            }
        }
        #[cfg(feature = "serve")]
        Cmd::Serve { addr, flip, common } => {
            let c = common.clone();
            idol_rs::serve::run(move || c.engine(), &addr, common.models_dir.clone(), common.low_vram, flip)?;
        }
        Cmd::Download { models_dir } => {
            let paths = ModelPaths::new(&models_dir);
            paths.ensure_all()?;
            log::info(&format!("all model files are in {}", models_dir.display()));
        }
        Cmd::Models { models_dir } => {
            let paths = ModelPaths::new(&models_dir);
            println!("models dir: {}", models_dir.display());
            for f in download::FILES {
                let p = paths.file(f.dir, f.name);
                let state = match std::fs::metadata(&p) {
                    Ok(m) => format!("{:>6.0} MB", m.len() as f64 / 1e6),
                    Err(_) => format!("missing (~{} MB){}", f.mb, if f.gated { ", gated" } else { "" }),
                };
                println!("  {}/{:<40} {state}", f.dir, f.name);
            }
        }
    }
    Ok(())
}
