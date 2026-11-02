import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsLibrary() {
  return (
    <DocsLayout>
      <H1>As a library</H1>
      <Lead>
        The <IC>idol-rs</IC> crate (library name <IC>idol_rs</IC>) is the engine behind the <IC>idol</IC> command: model loading, BiRefNet, both
        image encoders, the denoiser and sampler, the two decoders and the PLY writer.
      </Lead>

      <Section>
        <H2>Add the dependency</H2>
        <P>
          The HTTP server (ntex, <IC>idol_rs::serve</IC>) and the <IC>idol</IC> binary are behind the default <IC>serve</IC> feature; <IC>cuda</IC>{" "}
          and <IC>flash-attn</IC> are default features too, and <IC>metal</IC> replaces them on Apple Silicon. Without any of them the crate builds for
          the CPU, which is only good for tests.
        </P>
        <CopyBlock command={`cargo add idol-rs --no-default-features`} />
      </Section>

      <Section>
        <H2>Generate a splat</H2>
        <Pre caption="src/main.rs" lang="rust">{`use idol_rs::download::ModelPaths;
use idol_rs::pipeline::{Engine, EngineOptions, Request};

// downloads what is missing, then loads each model on first use
let mut engine = Engine::new(ModelPaths::new("models"), EngineOptions::default())?;
engine.preload()?; // optional: load everything now

let mut req = Request::new(image::open("photo.jpg")?);
req.num_gaussians = 524_288;
req.seed = 7;
let out = engine.run(&req)?;

out.gaussians.save_ply("photo.ply".as_ref())?; // or out.gaussians.to_ply() for the bytes
out.preprocessed.save("photo_input.png")?;     // the image the model saw
println!("{:?}", out.camera);                   // yaw, pitch, radius, fov`}</Pre>
      </Section>

      <Section>
        <H2>The pieces</H2>
        <P>
          <IC>preprocess</IC> (crop and composite), <IC>birefnet</IC>, <IC>dinov3</IC>, <IC>flux2_vae</IC>, <IC>dit</IC> (the denoiser),{" "}
          <IC>sampler</IC>, <IC>decoder</IC> (octree and Gaussian), <IC>gaussian</IC> (the activated Gaussians and the PLY writer), <IC>fused</IC> (the CUDA
          kernels), <IC>progress</IC> (stage and step counters, and cancellation) and <IC>download</IC> are separate modules. Progress is a few
          process-wide atomics, which is what <IC>/v1/status</IC> publishes.
        </P>
        <Pre caption="Gaussians" lang="rust">{`pub struct Gaussians {
    pub xyz: Vec<[f32; 3]>,         // y-up, inside [-0.5, 0.5]^3 in the PLY
    pub features_dc: Vec<[f32; 3]>, // degree-0 spherical harmonics
    pub scaling: Vec<[f32; 3]>,
    pub rotation: Vec<[f32; 4]>,    // unit quaternions (w, x, y, z)
    pub opacity: Vec<f32>,
}`}</Pre>
      </Section>
    </DocsLayout>
  );
}
