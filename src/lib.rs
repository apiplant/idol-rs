//! Idol: a single image to a 3D Gaussian splat, with CUDA or Apple Silicon (Metal) through candle.

pub mod backend;
pub mod birefnet;
pub mod camera;
pub mod config;
pub mod decoder;
pub mod dinov3;
pub mod dit;
pub mod download;
pub mod flux2_vae;
pub mod fused;
pub mod gaussian;
pub mod noise;
pub mod ops;
pub mod pipeline;
pub mod preprocess;
pub mod progress;
pub mod rng;
pub mod sampler;
#[cfg(feature = "serve")]
pub mod serve;
