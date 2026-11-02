# Packaging

One package, `idol-rs`, carrying the `idol` binary. Linux x86_64 packages use CUDA 13 and require an NVIDIA
GPU (Ampere or newer). Apple Silicon archives use candle's Metal backend on macOS 15+ (experimental). Both download the model files on first use; the DINOv3 and FLUX.2 encoders are gated on Hugging Face, so accept their terms and run `huggingface-cli login` (or set `HF_TOKEN`) first. The `packages`,
`homebrew`, `apt` and `pacman` jobs in [`.github/workflows/release.yml`](../.github/workflows/release.yml) substitute
the `@VERSION@`, `@SHA_*@`, `@ARCH@` and `@ARCHIVE_PREFIX@` placeholders with the tag's version and the checksums of
the archives that release just built, then publish the result. None of the definitions compiles the project: they
install the binary the build jobs already produced. The `crates-io` job publishes the crate (library + binary) with
`cargo publish`.

| File | Publishes to |
| --- | --- |
| `homebrew/idol-rs.rb` | `apiplant/homebrew-tap`, as `Formula/idol-rs.rb` (Linux x86_64 and macOS arm64) |
| `pacman/PKGBUILD` | the release itself, as a `.pkg.tar.zst` asset, and `apiplant/pacman` |
| `debian/control` | the release itself, as a `.deb` asset, and `apiplant/apt` |
| `apt/apt-ftparchive.conf` | `apiplant/apt`, served at `apt.apiplant.com` |

The repositories and secrets are the shared `apiplant` ones (`HOMEBREW_TAP_TOKEN`, `CARGO_REGISTRY_TOKEN`,
`PACMAN_REPO_TOKEN` + `PACMAN_GPG_PRIVATE_KEY`/`PACMAN_GPG_PASSPHRASE`, `APT_REPO_TOKEN` +
`APT_GPG_PRIVATE_KEY`/`APT_GPG_PASSPHRASE`). A publish job whose credential is absent is skipped rather than failing
the release.

## Builds

- `ui`: `pnpm build` in `ui/`, uploaded as the `ui-dist` artifact. `binary`, `binary-arch` and `crates-io` download it
  into `ui/dist` before building, and `IDOL_REQUIRE_UI=1` makes build.rs fail instead of embedding no web UI. The crate
  ships `ui/dist`, so `cargo install` gets the UI without Node.
- `binary`: an `nvidia/cuda:13.0.2-devel-ubuntu22.04` container (nvcc and the CUDA libraries, no GPU needed),
  `cargo build --profile dist --no-default-features --features cuda,flash-attn,serve`. candle compiles its kernels for one compute
  capability, so the release uses `CUDA_COMPUTE_CAP=80` (PTX that newer GPUs JIT-compile); flash-attn is built in (the denoiser's attention is several times faster with it). The archive, the `.deb` and the Homebrew formula use it.
- `binary-arch`: the same build inside Arch against Arch's own `cuda`, used by the pacman package (a different soname).
  If rolling Arch breaks that build, the pacman package falls back to the generic archive.

`.cargo/config.toml` pins `CUDARC_CUDA_VERSION` to the toolkit on the author's machine; the CI jobs override it with
the version of the toolkit they build against.

The `.deb` depends on `libcuda1`; the CUDA 13 libraries (`libcudart`, `libcublas`, `libnvrtc`, `libcurand`)
come from NVIDIA's own repository and are not expressible as Debian dependencies. The Arch package depends on
`nvidia-utils` and `cuda`. Homebrew cannot depend on a driver or CUDA.

## Cutting a release

Bump `version` in `Cargo.toml`, commit, then:

```bash
git tag v$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2) && git push origin --tags
```

`workflow_dispatch` runs the build and packaging jobs without publishing anything (a dry run).

## Using the packages

```bash
brew tap apiplant/tap && brew install apiplant/tap/idol-rs
```

Arch Linux (one-time setup, then `pacman -Syu` picks up new releases):

```bash
curl -sSfL https://apiplant.github.io/pacman/apiplant.gpg -o /tmp/apiplant.gpg
keyid=$(gpg --show-keys --with-colons /tmp/apiplant.gpg | awk -F: '/^pub:/ { print $5; exit }') && sudo pacman-key --add /tmp/apiplant.gpg && sudo pacman-key --finger "$keyid" && sudo pacman-key --lsign-key "$keyid"
printf '\n[apiplant]\nSigLevel = Required DatabaseOptional\nServer = https://apiplant.github.io/pacman/$arch\n' | sudo tee -a /etc/pacman.conf > /dev/null
sudo pacman -Sy idol-rs
```

Debian/Ubuntu (one-time setup, then `apt upgrade` picks up new releases):

```bash
curl -sSfL https://apt.apiplant.com/apiplant-archive-keyring.gpg | sudo tee /usr/share/keyrings/apiplant.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/apiplant.gpg] https://apt.apiplant.com stable main" | sudo tee /etc/apt/sources.list.d/apiplant.list > /dev/null
sudo apt update && sudo apt install idol-rs
```

Then run `idol serve` (web UI at http://127.0.0.1:8190) or `idol generate --image photo.png`.
