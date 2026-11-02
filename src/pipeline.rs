//! The whole image → Gaussians pipeline: background removal, the two image encoders, the denoising loop, the octree
//! and Gaussian decoders.

use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::{DynamicImage, RgbImage};

use crate::birefnet::BiRefNet;
use crate::camera::{decode_camera, Camera};
use crate::config::{IdolConfig, DEFAULT_GAUSSIANS, MAX_GAUSSIANS, MIN_GAUSSIANS};
use crate::decoder::{GaussianDecoder, OctreeDecoder};
use crate::dinov3::{Dinov3, Dinov3Config};
use crate::dit::Denoiser;
use crate::download::{log, ModelPaths};
use crate::flux2_vae::{Flux2Vae, VaeConfig};
use crate::gaussian::Gaussians;
use crate::preprocess::{self, Mode};
use crate::progress::{self, Stage};
use crate::noise::Noise;
use crate::sampler::{sample_flow, SampleSettings};

#[derive(Clone, Debug)]
pub struct EngineOptions {
    /// GPU index
    pub device: usize,
    /// run DINOv3 and the VAE in f32 instead of bf16 (1.7 GB more memory)
    pub fp32_encoders: bool,
    /// load each model only while its stage runs, so a 8–12 GB GPU is enough (slower between requests)
    pub low_vram: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self { device: 0, fp32_encoders: false, low_vram: false }
    }
}

#[derive(Clone)]
pub struct Request {
    pub image: DynamicImage,
    pub mode: Mode,
    /// overrides the mode's crop: square side = object bounding box × this
    pub crop_margin: Option<f32>,
    pub seed: u64,
    pub steps: usize,
    pub cfg_strength: f64,
    pub rescale_t: f64,
    pub num_gaussians: usize,
    /// conditional and unconditional passes as one batch-2 forward (faster, more memory)
    pub batch_cfg: bool,
    /// turn the object over (180 degrees about the input camera's view axis): for results that come out upside down
    pub flip: bool,
}

impl Request {
    pub fn new(image: DynamicImage) -> Self {
        Self { image, mode: Mode::Synthetic, crop_margin: None, seed: 42, steps: 50, cfg_strength: 3.0, rescale_t: 3.0, num_gaussians: DEFAULT_GAUSSIANS, batch_cfg: true, flip: false }
    }
}

pub struct Output {
    pub gaussians: Gaussians,
    /// the 1024 × 1024 image the model saw (background removed, cropped)
    pub preprocessed: RgbImage,
    pub camera: Camera,
    pub seconds: f64,
}

pub struct Engine {
    pub dev: Device,
    pub cfg: IdolConfig,
    paths: ModelPaths,
    opts: EngineOptions,
    dt: DType,
    enc_dt: DType,
    birefnet: Option<BiRefNet>,
    dinov3: Option<Dinov3>,
    vae: Option<Flux2Vae>,
    denoiser: Option<Denoiser>,
    octree: Option<OctreeDecoder>,
    gs: Option<GaussianDecoder>,
}

fn vb(path: &Path, dev: &Device) -> Result<VarBuilder<'static>> {
    // the files are read-only checkpoints that are not modified while the engine runs
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, dev) }.with_context(|| format!("loading {}", path.display()))?)
}

fn step<T>(name: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let t = Instant::now();
    let r = f()?;
    log::info(&format!("{name}: {:.1}s", t.elapsed().as_secs_f64()));
    Ok(r)
}

impl Engine {
    /// Open the models directory (downloading what is missing) and read the config; weights are loaded on first
    /// use, or all at once with [`Self::preload`].
    pub fn new(paths: ModelPaths, opts: EngineOptions) -> Result<Self> {
        paths.ensure_all()?;
        let dev = crate::backend::new_device(opts.device)?;
        let cfg = IdolConfig::load(&paths.file("idol", "config.json"))?;
        Ok(Self {
            dt: crate::backend::compute_dtype(&dev),
            enc_dt: crate::backend::encoder_dtype(&dev, opts.fp32_encoders),
            dev,
            cfg,
            paths,
            opts,
            birefnet: None,
            dinov3: None,
            vae: None,
            denoiser: None,
            octree: None,
            gs: None,
        })
    }

    pub fn options(&self) -> &EngineOptions {
        &self.opts
    }

    pub fn models_dir(&self) -> &Path {
        &self.paths.root
    }

    pub fn device_name(&self) -> String {
        format!("{:?}", self.dev.location())
    }

    /// Load every model now (the server does this at start; with `low_vram` this does nothing).
    pub fn preload(&mut self) -> Result<()> {
        if self.opts.low_vram {
            return Ok(());
        }
        self.birefnet()?;
        self.dinov3()?;
        self.vae()?;
        self.denoiser()?;
        self.octree()?;
        self.gs()?;
        Ok(())
    }

    fn birefnet(&mut self) -> Result<&BiRefNet> {
        if self.birefnet.is_none() {
            let v = vb(&self.paths.file("idol", "birefnet.safetensors"), &self.dev)?;
            self.birefnet = Some(step("load birefnet", || BiRefNet::load(&v, self.dt, &self.dev))?);
        }
        Ok(self.birefnet.as_ref().unwrap())
    }

    fn dinov3(&mut self) -> Result<&Dinov3> {
        if self.dinov3.is_none() {
            let cfg: Dinov3Config = serde_json::from_str(&std::fs::read_to_string(self.paths.file("dinov3", "config.json"))?)?;
            let v = vb(&self.paths.file("dinov3", "model.safetensors"), &self.dev)?;
            self.dinov3 = Some(step("load dinov3", || Dinov3::load(&v, &cfg, self.enc_dt, &self.dev))?);
        }
        Ok(self.dinov3.as_ref().unwrap())
    }

    fn vae(&mut self) -> Result<&Flux2Vae> {
        if self.vae.is_none() {
            let cfg: VaeConfig = serde_json::from_str(&std::fs::read_to_string(self.paths.file("flux2-vae", "config.json"))?)?;
            let v = vb(&self.paths.file("flux2-vae", "diffusion_pytorch_model.safetensors"), &self.dev)?;
            self.vae = Some(step("load flux2 vae", || Flux2Vae::load(&v, &cfg, self.enc_dt, &self.dev))?);
        }
        Ok(self.vae.as_ref().unwrap())
    }

    fn denoiser(&mut self) -> Result<&Denoiser> {
        if self.denoiser.is_none() {
            let v = vb(&self.paths.file("idol", "denoiser.safetensors"), &self.dev)?;
            self.denoiser = Some(step("load denoiser", || Denoiser::load(&v, &self.cfg.denoiser.args, self.dt, &self.dev))?);
        }
        Ok(self.denoiser.as_ref().unwrap())
    }

    fn octree(&mut self) -> Result<&OctreeDecoder> {
        if self.octree.is_none() {
            let v = vb(&self.paths.file("idol", "decoder.safetensors"), &self.dev)?;
            self.octree = Some(step("load octree decoder", || OctreeDecoder::load(&v, &self.cfg.decoder.args, self.dt, &self.dev))?);
        }
        Ok(self.octree.as_ref().unwrap())
    }

    fn gs(&mut self) -> Result<&GaussianDecoder> {
        if self.gs.is_none() {
            let v = vb(&self.paths.file("idol", "decoder_gs.safetensors"), &self.dev)?;
            self.gs = Some(step("load gaussian decoder", || GaussianDecoder::load(&v, &self.cfg.decoder_gs.args, self.dt, &self.dev))?);
        }
        Ok(self.gs.as_ref().unwrap())
    }

    /// BiRefNet alpha matte of an RGB image, one byte per pixel.
    fn matte(&mut self, rgb: &image::RgbaImage) -> Result<Vec<u8>> {
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let mut chw = vec![0f32; 3 * w * h];
        for (i, p) in rgb.pixels().enumerate() {
            for c in 0..3 {
                chw[c * w * h + i] = p[c] as f32 / 255.0;
            }
        }
        let dev = self.dev.clone();
        let net = self.birefnet()?;
        let n = BiRefNet::INPUT;
        let t = Tensor::from_vec(chw, (1, 3, h, w), &dev)?.upsample_bilinear2d(n, n, true)?;
        let mean = Tensor::from_slice(&[0.485f32, 0.456, 0.406], (1, 3, 1, 1), &dev)?;
        let std = Tensor::from_slice(&[0.229f32, 0.224, 0.225], (1, 3, 1, 1), &dev)?;
        let alpha = net.forward(&t.broadcast_sub(&mean)?.broadcast_div(&std)?)?;
        let alpha = alpha.upsample_bilinear2d(h, w, true)?.clamp(0.0, 1.0)?;
        let v: Vec<f32> = alpha.flatten_all()?.to_vec1()?;
        Ok(v.into_iter().map(|a| (a * 255.0) as u8).collect())
    }

    /// Resize, remove the background when the image has no alpha, crop and composite: the image the encoders see.
    pub fn preprocess(&mut self, image: &DynamicImage, mode: Mode, crop_margin: Option<f32>) -> Result<RgbImage> {
        progress::stage(Stage::Preparing);
        let size = self.cfg.image_size as u32;
        let rgba = image.to_rgba8();
        let real_alpha = image.color().has_alpha() && preprocess::has_real_alpha(&rgba);
        let mut resized = preprocess::resize_short_side(&rgba, size);
        if !real_alpha {
            let a = step("background removal", || self.matte(&resized))?;
            for (p, a) in resized.pixels_mut().zip(a) {
                p[3] = a;
            }
            if self.opts.low_vram {
                self.birefnet = None;
            }
        }
        if let Some(d) = std::env::var_os("IDOL_DEBUG_DIR") {
            let _ = std::fs::create_dir_all(&d);
            let _ = resized.save(Path::new(&d).join("matte.png"));
        }
        preprocess::crop_and_composite(&resized, size, crop_margin.unwrap_or_else(|| mode.crop_margin()))
    }

    pub fn run(&mut self, req: &Request) -> Result<Output> {
        let t0 = Instant::now();
        if !(MIN_GAUSSIANS..=MAX_GAUSSIANS).contains(&req.num_gaussians) {
            bail!("num_gaussians must be between {MIN_GAUSSIANS} and {MAX_GAUSSIANS}");
        }
        if req.steps == 0 || req.steps > 1000 {
            bail!("steps must be between 1 and 1000");
        }
        let prepared = self.preprocess(&req.image, req.mode, req.crop_margin)?;
        let s = prepared.width() as usize;
        let pixels: Vec<f32> = {
            let mut v = vec![0f32; 3 * s * s];
            for (i, p) in prepared.pixels().enumerate() {
                for c in 0..3 {
                    v[c * s * s + i] = p[c] as f32 / 255.0;
                }
            }
            v
        };
        let pixels = Tensor::from_vec(pixels, (1, 3, s, s), &self.dev)?;
        let mut rng = Noise::new(req.seed, &self.dev)?;

        progress::stage(Stage::Encoding);
        let f1 = step("dinov3", || self.dinov3()?.forward(&pixels))?;
        if self.opts.low_vram {
            self.dinov3 = None;
        }
        // the posterior sample comes first in the reference's random stream (torch.manual_seed, then the encoder)
        let vae_noise = if std::env::var_os("IDOL_VAE_MEAN").is_some() { None } else { Some(rng.randn(32 * (s / 8) * (s / 8))?) };
        let f2 = step("flux2 vae", || self.vae()?.forward(&pixels, vae_noise.as_ref()))?;
        if self.opts.low_vram {
            self.vae = None;
        }
        if std::env::var_os("IDOL_DEBUG_DIR").is_some() {
            for (n, t) in [("dinov3", &f1), ("vae", &f2)] {
                let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
                let mean = v.iter().sum::<f32>() / v.len() as f32;
                let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
                if let Some(d) = std::env::var_os("IDOL_DEBUG_DIR") {
                    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                    let _ = std::fs::write(Path::new(&d).join(format!("{n}.f32")), bytes);
                }
                log::info(&format!("{n} tokens {:?}: mean {mean:.4} std {std:.4} finite {}", t.dims(), v.iter().all(|x| x.is_finite())));
            }
        }
        let (f1, f2) = (Tensor::cat(&[&f1, &f1.zeros_like()?], 0)?, Tensor::cat(&[&f2, &f2.zeros_like()?], 0)?);

        let (q, ic, cc) = {
            let a = &self.cfg.denoiser.args;
            (a.q_token_length, a.in_channels, a.cam_channels)
        };
        let z = rng.randn(q * ic)?.reshape((1, q, ic))?;
        let cam = rng.randn(cc)?.reshape((1, 1, cc))?;
        let settings = SampleSettings { steps: req.steps, rescale_t: req.rescale_t, cfg_strength: req.cfg_strength, batch_cfg: req.batch_cfg };
        let (latent, cam) = step("sampling", || {
            let den = self.denoiser()?;
            let h_cond = den.prepare_cond(&f1, &f2)?;
            sample_flow(den, z, cam, &h_cond, &settings)
        })?;
        crate::ops::prof::report();
        if self.opts.low_vram {
            self.denoiser = None;
        }
        if std::env::var_os("IDOL_DEBUG_DIR").is_some() {
            let v: Vec<f32> = latent.flatten_all()?.to_vec1()?;
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
            log::info(&format!("latent: mean {mean:.4} std {std:.4} finite {} camera {:?}", v.iter().all(|x| x.is_finite()), cam.flatten_all()?.to_vec1::<f32>()?));
        }
        let camera = decode_camera(&cam.flatten_all()?.to_vec1::<f32>()?);

        progress::stage(Stage::Placing);
        let per = self.gs()?.per_anchor();
        let level = self.cfg.max_voxel_level;
        let anchors = step("octree", || self.octree()?.sample(&latent, (req.num_gaussians / per).max(1), level, &mut rng))?;
        if self.opts.low_vram {
            self.octree = None;
        }
        progress::stage(Stage::Decoding);
        let mut gaussians = step("gaussians", || self.gs()?.decode(&anchors, &latent))?;
        if req.flip {
            let d = camera.dir;
            gaussians.turn_over([d[0] as f32, d[1] as f32, d[2] as f32]);
        }
        if self.opts.low_vram {
            self.gs = None;
        }
        progress::idle();
        Ok(Output { gaussians, preprocessed: prepared, camera, seconds: t0.elapsed().as_secs_f64() })
    }
}
