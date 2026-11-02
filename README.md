# idol-rs

**One image to a 3D Gaussian splat, in Rust.**

A port of [Idol](https://github.com/speridlabs/idol-sdk) (Sperid Labs) to Rust and [candle](https://github.com/huggingface/candle):
one picture of an object in, a standard 3D Gaussian Splatting `.ply` out. It removes the background, frames the object and
generates up to 1,048,576 Gaussians, including the sides the picture never showed. One binary, no Python at run time:
a CLI, an HTTP API (ntex) and a web UI with a built-in 3D viewer (Solid.js 2).

```bash
idol generate --image photo.jpg --mode real      # writes outputs/photo.ply
idol serve                                       # web UI + API at http://127.0.0.1:8190
```

It loads the original weights as they are (`speridlabs/idol`, DINOv3, the FLUX.2 VAE): nothing is converted.

## What is in the box

| | |
|---|---|
| Output | 3D Gaussians: 32,768 to 1,048,576 (default 262,144), standard 3DGS `.ply` (y-up) |
| Denoiser | 370M parameter rectified-flow transformer, Euler sampler, classifier-free guidance |
| Image conditioning | DINOv3 ViT-H+/16 and the FLUX.2 VAE encoder |
| Decoders | octree decoder (places anchors) and Gaussian decoder (32 Gaussians per anchor) |
| Background removal | BiRefNet (Swin-L), with its deformable convolutions written out for candle |
| Backends | CUDA (flash attention), Apple Silicon through candle's Metal backend (experimental), CPU for tiny tests |

## Install

Pre-built packages are attached to every release (see [`packaging/`](packaging/README.md)):

```bash
brew install apiplant/tap/idol-rs                # macOS arm64, Linux x86_64
sudo dpkg -i idol-rs_0.1.0-1_amd64.deb           # Debian / Ubuntu
sudo pacman -U idol-rs-0.1.0-1-x86_64.pkg.tar.zst  # Arch
cargo install idol-rs                            # from source (CUDA 13 toolkit needed)
```

Linux needs an NVIDIA GPU (Ampere or newer) and the CUDA 13 runtime.

### Model files

The first run downloads what is missing into `./models` (when it holds `idol/`) or `~/.cache/idol-rs/models`
(`--models-dir` / `IDOL_MODELS_DIR` change that), about 6.6 GB:

| folder | from | size |
|---|---|---|
| `idol/` | [`speridlabs/idol`](https://huggingface.co/speridlabs/idol) | 3.1 GB |
| `dinov3/` | [`facebook/dinov3-vith16plus-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) (gated) | 3.4 GB |
| `flux2-vae/` | [`black-forest-labs/FLUX.2-dev`](https://huggingface.co/black-forest-labs/FLUX.2-dev), `vae/` only (gated) | 0.3 GB |

The two gated repositories need a Hugging Face account that accepted their terms: do that once on each page, then
`huggingface-cli login` (or export `HF_TOKEN`). `idol` checks the access before it downloads anything large, and
resumes interrupted downloads. `idol download` fetches everything up front, `idol models` shows what is present.
`HF_ENDPOINT` points the downloads at a mirror; `IDOL_NO_DOWNLOAD=1` turns them off.

## CLI

```bash
idol generate --image photo.jpg --mode real
idol generate --image renders/ --out outputs --num-gaussians 1048576
```

For each image this writes `<name>.ply` (open it in any Gaussian splat viewer), `<name>_input.png` (the image the model
saw, after background removal and cropping) and `<name>_camera.json` (the predicted camera of the input).

| flag | default | |
|---|---|---|
| `--mode` | `synthetic` | `synthetic`: renders and generated images of one object (crop 1.2); `real`: photographs (crop 1.7) |
| `--crop-margin` | from mode | square side = object bounding box × this |
| `--seed` | 42 | same seed, same object |
| `--steps` | 50 | sampling steps |
| `--cfg-strength` | 3.0 | classifier-free guidance |
| `--rescale-t` | 3.0 | time shift: higher spends more steps at high noise |
| `--num-gaussians` | 262144 | 32,768 to 1,048,576, rounded down to a multiple of 32 |
| `--fp32-encoders` | off | run DINOv3 and the VAE in f32 instead of bf16 (1.7 GB more memory) |
| `--low-vram` | off | load each model only while its stage runs, for 8–12 GB GPUs |
| `--no-batch-cfg` | off | guidance passes one after the other (less memory, slower) |
| `--flip` | off | turn the result over (see below): most pictures come out upside down |
| `--device` | 0 | GPU index |

A seed gives the same object as the Python reference's `torch.manual_seed(seed)`: on CUDA the noise (the VAE posterior
sample, the latent and camera noise, and every uniform the octree sampler draws) comes from PyTorch's own Philox algorithm,
bit for bit, in the same order (`examples/rng_parity.rs` checks that against `tools/ref_rng.py`). The arithmetic around it
differs in the last float bits, so the Gaussians are not identical but the object is: on the same seed and picture the two
point sets sit 0.0035 apart at the median (about the point spacing; another seed is six times further) and the predicted
cameras agree to three decimals. torch sizes its launches from the GPU's SM count, so streams are per GPU model, as they are in torch.
Off CUDA a seeded CPU generator is used (`IDOL_CPU_RNG=1` forces it).

## Server

```bash
idol serve --addr 0.0.0.0:8190
```

The engine is single-tenant: requests queue and run one at a time, and the models stay loaded between them.

```bash
# wait for the result
curl -s localhost:8190/v1/generate -H 'content-type: image/png' --data-binary @photo.png -o photo.ply
curl -s 'localhost:8190/v1/generate?mode=real&seed=7' --data-binary @photo.jpg -o photo.ply
# or queue and poll
id=$(curl -s localhost:8190/v1/jobs -d "{\"image\":\"$(base64 -w0 photo.png)\"}" | jq -r .id)
curl -s "localhost:8190/v1/jobs?id=$id"
curl -s "localhost:8190/v1/jobs/result?id=$id" -o photo.ply
```

| endpoint | |
|---|---|
| `POST /v1/generate` | image in (JSON with a base64 `image`, or the raw bytes), `.ply` out |
| `POST /v1/jobs`, `GET /v1/jobs?id=`, `GET /v1/jobs/result?id=&file=ply\|input\|camera`, `DELETE /v1/jobs?id=` | the same, asynchronously |
| `GET /v1/status`, `POST /v1/cancel`, `GET /v1/info`, `GET /health` | progress, cancel, server info |
| `GET /` | the web UI |
| `GET /openapi.json`, `GET /docs` | the API as OpenAPI 3.1, rendered with Scalar |

### Web UI

`idol serve` embeds a UI built with Solid.js 2 (`ui/`): drop an image, press Generate and watch the object turn in a
WebGL2 splat viewer (orbit, zoom, the model's own input view, PNG snapshots, `.ply` download), with a take history.
`pnpm dev` in `ui/` runs it against a running server; `pnpm build` produces `ui/dist`, which `build.rs` embeds in the
binary (`IDOL_REQUIRE_UI=1` makes the build fail instead of embedding none).

## Library

```rust
use idol_rs::{download::ModelPaths, pipeline::{Engine, EngineOptions, Request}};

let mut engine = Engine::new(ModelPaths::new("models"), EngineOptions::default())?;
let out = engine.run(&Request::new(image::open("photo.jpg")?))?;
out.gaussians.save_ply("photo.ply".as_ref())?;
```

`cargo add idol-rs --no-default-features` builds the library for the CPU without CUDA or the server.

## Memory

Roughly 3.7–4.4 GB of weights and a ~7 GB peak at the default 262,144 Gaussians when everything stays resident.
`--low-vram` loads one model at a time. 1,048,576 Gaussians need more headroom in the Gaussian decoder, which attends
over 32,768 anchors at once.

## Building

```bash
cd ui && pnpm install && pnpm build && cd ..     # the web UI the binary embeds
cargo build --release                            # CUDA + flash attention + server
cargo build --release --no-default-features --features metal,serve   # Apple Silicon
```

`.cargo/config.toml` pins `CUDARC_CUDA_VERSION` to the toolkit on the author's machine; override it for yours. flash
attention adds a long nvcc step to the first build; `--no-default-features --features cuda,serve` skips it and falls
back to a chunked attention that is correct but slower.

## Speed

Measured on this machine (RTX 4090), the same picture, 262,144 Gaussians, 50 steps, models already loaded:

| | one request |
|---|---|
| PyTorch reference, eager | 17.8 s |
| PyTorch reference, the SDK's default (`torch.compile`, TF32, cuDNN attention) | 13.5 s (the first call compiles: 25 s) |
| idol-rs (warm server) | 13.5 s, 12.3 s of it sampling |

So idol-rs is as fast as the compiled Python and 25 % faster than eager PyTorch, without a compile step. (The CLI
takes about 14 s because it loads the models too.) About half of the first working version's time went into
memory-bound elementwise ops, so the denoiser's chains of them are hand-written CUDA kernels (`kernels/idol.cu`, compiled
to PTX by `build.rs`): LayerNorm with the adaLN modulation, the q/k bias, rotary rotation and RMS norm, bias + GELU,
bias + gate + residual and the RePo angles. That took a request from 29.6 s to 14 s with the output unchanged (the
fused denoiser matches the reference at cosine 0.999994). What is left is matmuls and flash attention, which run near
the GPU's throughput (`IDOL_PROFILE=1` prints the per-section table; `IDOL_NO_FUSED=1` runs the plain ops).
`tools/ref_bench.py` reproduces the Python numbers.

## Checking it against the reference

`tools/ref_dump.py` dumps tensors from the PyTorch implementation and `cargo run --release --example parity` compares
them with the Rust modules: the denoiser matches at cosine 0.999994, the octree decoder at 0.999996 and every Gaussian
decoder output at 0.9998 or better. `tools/ref_encoders.py` does the same for DINOv3 (0.99993 against `transformers`) and
the FLUX.2 VAE (0.99995 for the posterior mean, against `diffusers`). `tools/ref_pipeline.py` runs the whole Python
pipeline, and `tools/render_ply.py` makes a quick numpy preview of a `.ply`. BiRefNet was checked by eye.

## Upside-down objects

Most pictures come out of the model upside down (a fox figurine, an egg chair, a tin robot, an axolotl and a backpack
in our tests); the Python reference does the same on the same inputs, so this is the model, not the port. `--flip`
(CLI), `"flip": true` (API) and the UI's "Turn the result over" switch fix it in the `.ply` itself: a proper 180° rotation
about the input camera's view axis, not a mirror, so the same face stays toward the camera and the predicted camera stays
valid. `idol serve --flip` makes it the default (a request's `flip: false` overrides). Without it the `.ply` is as the model made it.

## Differences from the reference

- Everything follows the reference code step by step; the noise streams are torch's own on CUDA (see above).
- The RePo projections of the denoiser run in half precision on the fused path (the reference keeps them in f32); `IDOL_REPO_F32=1` restores f32.
- The denoiser's image-condition stream, which does not depend on the sample or the time, is refined once per image
  instead of at every step.
- The FLUX.2 VAE posterior is sampled with torch's generator and the reference's bf16 arithmetic.

## Credits

Idol is by [Sperid Labs](https://github.com/speridlabs/idol-sdk) (Apache-2.0) and builds on
[TripoSplat](https://github.com/VAST-AI-Research/TripoSplat), [DINOv3](https://github.com/facebookresearch/dinov3),
the [FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) VAE and [BiRefNet](https://github.com/ZhengPeng7/BiRefNet).
This port is Apache-2.0 too.
