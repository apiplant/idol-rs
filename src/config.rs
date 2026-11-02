//! The `config.json` that ships with the Idol weights.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::decoder::DecoderConfig;
use crate::dit::DenoiserConfig;

#[derive(Clone, Debug, Deserialize)]
pub struct Part<T> {
    pub args: T,
}

#[derive(Clone, Debug, Deserialize)]
pub struct IdolConfig {
    pub denoiser: Part<DenoiserConfig>,
    pub decoder: Part<DecoderConfig>,
    pub decoder_gs: Part<DecoderConfig>,
    pub image_size: usize,
    pub max_voxel_level: usize,
}

impl IdolConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let s = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&s).with_context(|| format!("parsing {}", path.display()))
    }
}

/// Request limits shared by the CLI and the server.
pub const MIN_GAUSSIANS: usize = 32_768;
pub const MAX_GAUSSIANS: usize = 1_048_576;
pub const DEFAULT_GAUSSIANS: usize = 262_144;
