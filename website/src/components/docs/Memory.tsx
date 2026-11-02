import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, UL, LI, IC, Section, FlagTable } from "./Prose";

export function DocsMemory() {
  return (
    <DocsLayout>
      <H1>Memory</H1>
      <Lead>Everything stays on the GPU by default. When it does not fit, load one model at a time.</Lead>

      <Section>
        <H2>What is resident</H2>
        <FlagTable
          rows={[
            { flag: "Denoiser, 370M", meaning: "f16 blocks, f32 embeddings and norms: about 0.8 GB." },
            { flag: "Octree and Gaussian decoders", meaning: "About 0.7 GB together." },
            { flag: "BiRefNet, 221M", meaning: "f16: about 0.45 GB." },
            { flag: "DINOv3 840M", meaning: "bf16: about 1.7 GB (f32 with --fp32-encoders)." },
            { flag: "FLUX.2 VAE encoder", meaning: "bf16: under 0.2 GB." },
          ]}
        />
        <P>
          Activations add to that: the denoiser attends over about 12,000 tokens with both guidance passes batched, and the VAE encoder works at
          1024 × 1024. The Python original quotes a 7.2 GB peak at the default size.
        </P>
      </Section>

      <Section>
        <H2>When it does not fit</H2>
        <UL>
          <LI><IC>--low-vram</IC>: each model is loaded when its stage starts and dropped when it ends, so only one is resident (the models are memory-mapped, so reloading is quick when they are in the page cache). It also runs the guidance passes one after the other.</LI>
          <LI><IC>--no-batch-cfg</IC>: the two guidance passes run one after the other, halving the denoiser's activations.</LI>
          <LI>Fewer Gaussians: the Gaussian decoder attends over every anchor at once, 32,768 of them at a million Gaussians, which is where the largest request needs the most headroom.</LI>
        </UL>
      </Section>
    </DocsLayout>
  );
}
