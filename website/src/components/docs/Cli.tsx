import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Section, FlagTable } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsCli() {
  return (
    <DocsLayout>
      <H1>idol</H1>
      <Lead>Turn pictures into splats from the command line, serve the engine over HTTP, or manage the model files.</Lead>

      <Section>
        <H2>Examples</H2>
        <CopyBlock
          command={`# one picture (defaults: synthetic crop, 50 steps, 262,144 Gaussians)
idol generate --image photo.jpg

# a photograph, a looser crop
idol generate --image photo.jpg --mode real

# every picture in a folder, a million Gaussians
idol generate --image renders/ --out outputs --num-gaussians 1048576

# same seed, same object; fewer steps for a quick look
idol generate --image photo.jpg --seed 7 --steps 20`}
        />
        <P>
          For each picture it writes <IC>{"<name>.ply"}</IC> (open it in any Gaussian splat viewer), <IC>{"<name>_input.png"}</IC> (the image the model
          saw, after background removal and cropping) and <IC>{"<name>_camera.json"}</IC> (the predicted camera of the input: yaw and pitch in radians,
          radius and vertical field of view in degrees).
        </P>
      </Section>

      <Section>
        <H2>generate</H2>
        <FlagTable
          rows={[
            { flag: "-i, --image PATH", meaning: "An image or a folder of images (png, jpg, webp, bmp). A picture with a real alpha channel skips the background removal." },
            { flag: "-o, --out DIR", meaning: "Where the files go (default outputs)." },
            { flag: "--mode synthetic|real", meaning: "synthetic: renders and generated images of one object, crop 1.2. real: photographs, crop 1.7." },
            { flag: "--crop-margin X", meaning: "Overrides the crop: square side = object bounding box × X." },
            { flag: "--seed N", meaning: "Same seed, same object (default 42)." },
            { flag: "--steps N", meaning: "Euler steps (default 50)." },
            { flag: "--cfg-strength W", meaning: "Classifier-free guidance, guided = (1 + W) cond - W uncond (default 3.0; 0 turns it off)." },
            { flag: "--rescale-t T", meaning: "Time shift towards high noise (default 3.0)." },
            { flag: "--num-gaussians N", meaning: "32,768 to 1,048,576, rounded down to a multiple of 32 (default 262,144)." },
            { flag: "--fp32-encoders", meaning: "Run DINOv3 and the VAE in f32 instead of bf16 (1.7 GB more memory)." },
            { flag: "--low-vram", meaning: "Load each model only while its stage runs. Fits 8 to 12 GB cards, slower between requests." },
            { flag: "--no-batch-cfg", meaning: "Run the two guidance passes one after the other (less memory, slower)." },
            { flag: "--flip", meaning: "Turn the result over: a half turn about the input camera's view axis. Most pictures come out of the model upside down (the Python reference too)." },
            { flag: "--models-dir DIR, --device N", meaning: "Models folder (IDOL_MODELS_DIR) and GPU index." },
          ]}
        />
      </Section>

      <Section>
        <H2>serve, download, models</H2>
        <FlagTable
          rows={[
            { flag: "idol serve [--addr HOST:PORT]", meaning: "Loads everything and serves the API and the web UI (default 0.0.0.0:8190, IDOL_ADDR). Takes the same model flags, and --flip to turn results over by default." },
            { flag: "idol download", meaning: "Fetches the missing model files, checking the gated repositories' access first." },
            { flag: "idol models", meaning: "Lists the model files and whether they are present." },
          ]}
        />
      </Section>

      <Section>
        <H2>Environment</H2>
        <FlagTable
          rows={[
            { flag: "IDOL_MODELS_DIR, IDOL_ADDR", meaning: "Defaults for --models-dir and the server address." },
            { flag: "HF_TOKEN, HF_ENDPOINT, IDOL_NO_DOWNLOAD", meaning: "Hugging Face token, mirror, and a switch to forbid downloads." },
            { flag: "IDOL_PROFILE=1", meaning: "Prints a timing table of the denoiser's sections (synchronises the GPU, so it slows the run)." },
            { flag: "IDOL_NO_FUSED=1, IDOL_DTYPE=f32", meaning: "Run the plain ops instead of the fused kernels; compute in f32 (slow, for diagnosis)." },
            { flag: "IDOL_DEBUG_DIR=DIR", meaning: "Saves the matte and the encoder tokens, and logs their statistics." },
          ]}
        />
      </Section>
    </DocsLayout>
  );
}
