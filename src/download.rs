//! Fetch model files on demand: the three Idol checkpoints from `speridlabs/idol`, and the two gated image encoders
//! (DINOv3 and the FLUX.2 VAE), which need a Hugging Face token whose account accepted their terms.
//! Downloads resume from `<file>.part`. Set `IDOL_NO_DOWNLOAD=1` to turn this off, `HF_ENDPOINT` to use a mirror,
//! `HF_TOKEN` (or `huggingface-cli login`) for the gated files.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};

use crate::progress::{self, Stage};

pub const IDOL_REPO: &str = "speridlabs/idol";
pub const DINOV3_REPO: &str = "facebook/dinov3-vith16plus-pretrain-lvd1689m";
pub const FLUX2_REPO: &str = "black-forest-labs/FLUX.2-dev";

pub struct File {
    /// sub folder of the models dir
    pub dir: &'static str,
    pub name: &'static str,
    pub repo: &'static str,
    /// path inside the repo
    pub remote: &'static str,
    pub mb: u32,
    pub gated: bool,
}

pub const FILES: &[File] = &[
    File { dir: "idol", name: "config.json", repo: IDOL_REPO, remote: "config.json", mb: 1, gated: false },
    File { dir: "idol", name: "denoiser.safetensors", repo: IDOL_REPO, remote: "denoiser.safetensors", mb: 1_482, gated: false },
    File { dir: "idol", name: "decoder.safetensors", repo: IDOL_REPO, remote: "decoder.safetensors", mb: 203, gated: false },
    File { dir: "idol", name: "decoder_gs.safetensors", repo: IDOL_REPO, remote: "decoder_gs.safetensors", mb: 949, gated: false },
    File { dir: "idol", name: "birefnet.safetensors", repo: IDOL_REPO, remote: "birefnet.safetensors", mb: 444, gated: false },
    File { dir: "dinov3", name: "config.json", repo: DINOV3_REPO, remote: "config.json", mb: 1, gated: true },
    File { dir: "dinov3", name: "model.safetensors", repo: DINOV3_REPO, remote: "model.safetensors", mb: 3_363, gated: true },
    File { dir: "flux2-vae", name: "config.json", repo: FLUX2_REPO, remote: "vae/config.json", mb: 1, gated: true },
    File { dir: "flux2-vae", name: "diffusion_pytorch_model.safetensors", repo: FLUX2_REPO, remote: "vae/diffusion_pytorch_model.safetensors", mb: 336, gated: true },
];

/// Where the models live: `<models_dir>/{idol,dinov3,flux2-vae}`.
#[derive(Clone, Debug)]
pub struct ModelPaths {
    pub root: PathBuf,
}

impl ModelPaths {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn dir(&self, sub: &str) -> PathBuf {
        self.root.join(sub)
    }

    pub fn file(&self, sub: &str, name: &str) -> PathBuf {
        self.root.join(sub).join(name)
    }

    /// Every file that is not on disk yet.
    pub fn missing(&self) -> Vec<&'static File> {
        FILES.iter().filter(|f| !self.file(f.dir, f.name).is_file()).collect()
    }

    /// Download whatever is missing (checking the gated encoders' access first, before the big files).
    pub fn ensure_all(&self) -> Result<()> {
        let missing = self.missing();
        if missing.is_empty() {
            return Ok(());
        }
        if std::env::var_os("IDOL_NO_DOWNLOAD").is_some_and(|v| v != "0" && !v.is_empty()) {
            bail!(
                "model files missing in {} ({}) and downloads are disabled by IDOL_NO_DOWNLOAD",
                self.root.display(),
                missing.iter().map(|f| format!("{}/{}", f.dir, f.name)).collect::<Vec<_>>().join(", ")
            );
        }
        // the gated repos first: fail with instructions before downloading gigabytes
        for repo in [DINOV3_REPO, FLUX2_REPO] {
            if missing.iter().any(|f| f.repo == repo) {
                check_access(repo)?;
            }
        }
        for f in missing {
            let dest = self.file(f.dir, f.name);
            let url = format!("{}/{}/resolve/main/{}", endpoint(), f.repo, f.remote);
            fetch(&url, &dest, f.mb, f.gated).with_context(|| format!("downloading {}/{}", f.dir, f.name))?;
        }
        Ok(())
    }
}

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT").ok().filter(|s| !s.is_empty()).map(|s| s.trim_end_matches('/').to_string()).unwrap_or_else(|| "https://huggingface.co".into())
}

/// `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or the token file of `huggingface-cli login`.
pub fn token() -> Option<String> {
    for k in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Some(v) = std::env::var(k).ok().filter(|s| !s.trim().is_empty()) {
            return Some(v.trim().to_string());
        }
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    let hf = std::env::var_os("HF_HOME").map(PathBuf::from).unwrap_or_else(|| home.join(".cache/huggingface"));
    std::fs::read_to_string(hf.join("token")).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().redirects(0).timeout_connect(Duration::from_secs(30)).timeout_read(Duration::from_secs(60)).build()
}

/// GET `url` following redirects by hand: the token goes to the Hub only, never to the CDN it redirects to.
fn get(url: &str, token: Option<&str>, range_from: u64, head: bool) -> Result<ureq::Response, ureq::Error> {
    let a = agent();
    let mut url = url.to_string();
    let hub = endpoint();
    for _ in 0..8 {
        let mut req = if head { a.head(&url) } else { a.get(&url) };
        if let (Some(t), true) = (token, url.starts_with(&hub)) {
            req = req.set("Authorization", &format!("Bearer {t}"));
        }
        if range_from > 0 {
            req = req.set("Range", &format!("bytes={range_from}-"));
        }
        match req.call() {
            Ok(r) if (300..400).contains(&r.status()) => match r.header("location") {
                Some(l) => url = if l.starts_with("http") { l.to_string() } else { format!("{hub}{l}") },
                None => return Ok(r),
            },
            Err(ureq::Error::Status(code, r)) if (300..400).contains(&code) => match r.header("location") {
                Some(l) => url = if l.starts_with("http") { l.to_string() } else { format!("{hub}{l}") },
                None => return Err(ureq::Error::Status(code, r)),
            },
            other => return other,
        }
    }
    Err(ureq::Error::Status(310, ureq::Response::new(310, "too many redirects", "").unwrap()))
}

/// Stop with an explanation, before anything large is downloaded, when a gated repo is not readable.
pub fn check_access(repo: &str) -> Result<()> {
    let tok = token();
    let url = format!("{}/{repo}/resolve/main/config.json", endpoint());
    let probe = if repo == FLUX2_REPO { format!("{}/{repo}/resolve/main/vae/config.json", endpoint()) } else { url };
    match get(&probe, tok.as_deref(), 0, true) {
        Ok(_) => Ok(()),
        Err(ureq::Error::Status(401 | 403 | 404, _)) => {
            let why = if tok.is_none() {
                "You are not logged in to Hugging Face, and this image encoder is gated."
            } else {
                "Your Hugging Face account does not have access yet to this gated image encoder."
            };
            bail!(
                "Idol needs two image encoders that are downloaded from Hugging Face.\n{why}\n  https://huggingface.co/{repo}\n\
                 1. Accept the terms on the page above with your Hugging Face account (approval can take a while).\n\
                 2. Log in with `huggingface-cli login`, or set HF_TOKEN to a token of that account."
            )
        }
        // network trouble is not missing access: the download itself reports it
        Err(_) => Ok(()),
    }
}

fn fetch(url: &str, dest: &Path, approx_mb: u32, gated: bool) -> Result<()> {
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let part = PathBuf::from(format!("{}.part", dest.display()));
    let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let tok = token();
    let _ = gated;
    let resp = match get(url, tok.as_deref(), have, false) {
        Ok(r) => r,
        Err(ureq::Error::Status(416, _)) if have > 0 => {
            std::fs::rename(&part, dest)?;
            return Ok(());
        }
        Err(e) => return Err(anyhow!("{e}")),
    };
    let resumed = resp.status() == 206;
    let remaining: u64 = resp.header("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
    let start = if resumed { have } else { 0 };
    let total = if remaining > 0 { start + remaining } else { approx_mb as u64 * 1_000_000 };
    let name = dest.file_name().and_then(|s| s.to_str()).unwrap_or("model");
    log::info(&format!("downloading {name} ({:.2} GB){}", total as f64 / 1e9, if resumed { format!(", resuming at {:.2} GB", have as f64 / 1e9) } else { String::new() }));

    let mut out = std::fs::OpenOptions::new().create(true).write(true).append(resumed).truncate(!resumed).open(&part)?;
    let mut body = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = start;
    let mut last_log = Instant::now();
    loop {
        progress::check()?;
        let n = body.read(&mut buf).context("connection lost (run again to resume)")?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        done += n as u64;
        progress::count(Stage::Downloading, (done / 1_000_000) as usize, (total / 1_000_000) as usize);
        if last_log.elapsed() >= Duration::from_secs(5) {
            log::info(&format!("  {name}: {:.2} / {:.2} GB", done as f64 / 1e9, total as f64 / 1e9));
            last_log = Instant::now();
        }
    }
    out.flush()?;
    drop(out);
    if remaining > 0 && done != total {
        bail!("download of {name} ended early ({done} of {total} bytes); run again to resume");
    }
    std::fs::rename(&part, dest)?;
    log::info(&format!("downloaded {name}"));
    Ok(())
}

pub mod log {
    pub fn info(s: &str) {
        eprintln!("[idol] {s}");
    }
    pub fn warn(s: &str) {
        eprintln!("[idol] WARNING: {s}");
    }
}
