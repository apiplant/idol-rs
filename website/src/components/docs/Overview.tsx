import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Section, FlagTable } from "./Prose";
import { CopyBlock } from "../Code";
import { UPSTREAM_URL } from "../../lib/links";

export function DocsOverview() {
  return (
    <DocsLayout>
      <H1>Documentation</H1>
      <Lead>
        idol-rs turns one picture of an object into a 3D Gaussian splat, on an NVIDIA GPU or Apple Silicon, in Rust (candle plus a few
        hand-written CUDA kernels). It is a port of{" "}
        <a class="text-accent hover:text-accent-dim" href={UPSTREAM_URL} target="_blank" rel="noreferrer noopener">Idol</a> by Sperid Labs and loads
        its weights as they are. It ships as a library crate and the <IC>idol</IC> command.
      </Lead>

      <Section>
        <H2>The pipeline</H2>
        <UL>
          <LI><b class="font-medium text-ink">Prepare.</b> BiRefNet removes the background (unless the picture has a real alpha channel), the alpha is eroded by a pixel, and a square crop around the object is composited on black at 1024 × 1024.</LI>
          <LI><b class="font-medium text-ink">Encode.</b> DINOv3 ViT-H+/16 and the FLUX.2 VAE encoder turn the picture into 4,101 tokens each.</LI>
          <LI><b class="font-medium text-ink">Denoise.</b> A 370M rectified-flow transformer integrates a latent of 8,192 tokens, plus one camera token, from noise with Euler steps and classifier-free guidance.</LI>
          <LI><b class="font-medium text-ink">Decode.</b> The octree decoder places up to 32,768 anchor points; the Gaussian decoder turns each into 32 Gaussians. The result is a standard 3DGS <IC>.ply</IC>.</LI>
        </UL>
      </Section>

      <Section>
        <H2>The model files</H2>
        <P>
          Everything lives under one models folder (<IC>--models-dir</IC> or <IC>IDOL_MODELS_DIR</IC>; <IC>./models</IC> when it holds
          <IC>idol/</IC>, else <IC>~/.cache/idol-rs/models</IC>). Missing files are downloaded on first use, resuming interrupted downloads.
        </P>
        <FlagTable
          rows={[
            { flag: "idol/{denoiser,decoder,decoder_gs,birefnet}.safetensors, config.json", meaning: "speridlabs/idol on Hugging Face, 3.1 GB." },
            { flag: "dinov3/model.safetensors, config.json", meaning: "facebook/dinov3-vith16plus-pretrain-lvd1689m, 3.4 GB. Gated." },
            { flag: "flux2-vae/diffusion_pytorch_model.safetensors, config.json", meaning: "The vae/ folder of black-forest-labs/FLUX.2-dev, 0.3 GB. Gated." },
          ]}
        />
        <P>
          The two gated repositories need a Hugging Face account that accepted their terms. Do that once on each page, then run{" "}
          <IC>huggingface-cli login</IC> or export <IC>HF_TOKEN</IC>. idol checks the access before it downloads anything large, and the
          token is only ever sent to the Hub, never to the CDN it redirects to. <IC>HF_ENDPOINT</IC> points the downloads at a mirror and{" "}
          <IC>IDOL_NO_DOWNLOAD=1</IC> turns them off.
        </P>
        <CopyBlock command={`idol models      # what is present\nidol download    # fetch everything up front`} />
      </Section>

      <Section>
        <H2>How it was checked</H2>
        <P>
          The denoiser, the octree decoder and every output of the Gaussian decoder were compared with the PyTorch implementation on the same
          weights and inputs: cosine similarity 0.99999 for the denoiser and at least 0.9998 for every decoder output. DINOv3 matches
          Hugging Face's implementation at 0.99993. BiRefNet and the FLUX.2 VAE were checked by eye and by statistics rather than tensor by
          tensor. <IC>examples/parity.rs</IC> and <IC>tools/ref_dump.py</IC> reproduce the comparison.
        </P>
        <P>
          Seeds carry over. On CUDA the noise (the VAE posterior sample, the latent and camera noise and the octree sampler's uniforms) is
          PyTorch's own Philox stream, bit for bit and in the same order, so the same seed gives the same object as the Python reference: the two
          point sets sit 0.0035 apart at the median, about the point spacing, against 0.021 for a different seed. The arithmetic around the noise
          still differs in the last float bits, so the Gaussians are not identical. Streams are per GPU model, as in torch.
        </P>
      </Section>

      <Section>
        <H2>Speed (RTX 4090)</H2>
        <FlagTable
          rows={[
            { flag: "idol-rs, warm server", meaning: "13.5 s: 262,144 Gaussians, 50 steps; 12.3 s of it sampling (both guidance passes batched)" },
            { flag: "PyTorch reference, torch.compile", meaning: "13.5 s (the SDK's default; the first call compiles, 25 s)" },
            { flag: "PyTorch reference, eager", meaning: "17.8 s" },
            { flag: "idol-rs before the fused kernels", meaning: "29.6 s" },
          ]}
        />
        <UL>
          <LI>The denoiser's chains of memory-bound ops (RoPE with the q/k RMS norm, LayerNorm with adaLN modulation, bias, GELU, gated residuals, the RePo angles) are single CUDA kernels in <IC>kernels/idol.cu</IC>, compiled to PTX by <IC>build.rs</IC>.</LI>
          <LI>The joint attention over about 12,000 tokens uses flash attention; a chunked softmax is the fallback without it.</LI>
          <LI>The image-condition stream does not depend on the sample or the time, so it is refined once per image, not at every step.</LI>
          <LI>Set <IC>IDOL_PROFILE=1</IC> for a per-section timing table, and <IC>IDOL_NO_FUSED=1</IC> to run the plain ops.</LI>
        </UL>
      </Section>

      <Section>
        <H2>Requirements</H2>
        <UL>
          <LI>CUDA: an NVIDIA GPU, Ampere or newer, with a driver for CUDA 13. About 7 GB at the default size; <IC>--low-vram</IC> loads one model at a time.</LI>
          <LI>Or an Apple Silicon Mac (candle's Metal backend, experimental) with macOS 15 or newer.</LI>
        </UL>
        <CopyBlock command={`idol generate --image photo.jpg --mode real`} />
      </Section>
    </DocsLayout>
  );
}
