import { DocsLayout } from "./DocsLayout";
import { H1, H2, Lead, P, IC, Pre, Section, FlagTable } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsServer() {
  return (
    <DocsLayout>
      <H1>Web UI &amp; API</H1>
      <Lead>
        <IC>idol serve</IC> keeps every model loaded, queues requests and runs them one at a time. It serves a web UI at <IC>/</IC>, the API under{" "}
        <IC>/v1</IC>, and <IC>/docs</IC> for the API reference (rendered from <IC>/openapi.json</IC>).
      </Lead>

      <Section>
        <H2>The web UI</H2>
        <P>
          Drop, paste or pick a picture (or start from the examples), choose Render or Photo, the number of Gaussians, steps and guidance, whether to turn the result over (most come out upside down), and press
          Generate. While it works you see the stage and the denoising step; the result opens in a WebGL2 splat viewer you can orbit and zoom. The
          viewer can jump to the camera the model predicted for your picture, save a PNG of the current view, and download the <IC>.ply</IC>. Finished
          takes stay in a strip for comparing; settings persist in the browser.
        </P>
        <P>
          <IC>Ctrl+Enter</IC> generates, and <IC>?ply=URL</IC> opens a splat directly. Stop cancels the running job between steps.
        </P>
      </Section>

      <Section>
        <H2>Generate from anywhere</H2>
        <CopyBlock
          command={`# the raw picture as the body, options as query parameters; the .ply comes back
curl 'localhost:8190/v1/generate?mode=real&seed=7' --data-binary @photo.jpg -o photo.ply

# or queue it and poll
id=$(curl -s localhost:8190/v1/jobs -d "{\\"image\\":\\"$(base64 -w0 photo.png)\\"}" | jq -r .id)
curl -s "localhost:8190/v1/jobs?id=$id"
curl -s "localhost:8190/v1/jobs/result?id=$id" -o photo.ply`}
        />
        <FlagTable
          rows={[
            { flag: "POST /v1/generate", meaning: "A JSON body with a base64 image (a data: URL is fine), or the raw bytes. Returns the .ply with x-idol-job and x-idol-camera headers." },
            { flag: "POST /v1/jobs", meaning: "The same body; returns {id} at once." },
            { flag: "GET /v1/jobs?id=", meaning: "{state: queued|running|done|failed}, with meta and camera when done." },
            { flag: "GET /v1/jobs/result?id=&file=", meaning: "ply (default), input (the image the model saw, png) or camera (json)." },
            { flag: "DELETE /v1/jobs?id=", meaning: "Deletes a finished job's files." },
            { flag: "GET /v1/status, POST /v1/cancel", meaning: "What the engine is doing now (state, stage, done, total), and cancel the running job and the queue." },
            { flag: "GET /v1/info, /health, /openapi.json, /docs", meaning: "Version, device and limits; liveness; the API as OpenAPI 3.1 and its reference page (Scalar, loaded from a CDN)." },
          ]}
        />
        <P>
          Request fields: <IC>image</IC> (required), <IC>mode</IC> (<IC>synthetic</IC> or <IC>real</IC>), <IC>crop_margin</IC>, <IC>seed</IC>,{" "}
          <IC>steps</IC>, <IC>cfg_strength</IC>, <IC>rescale_t</IC>, <IC>num_gaussians</IC> and <IC>flip</IC> (turn the result over; the default is the server's <IC>--flip</IC>). With a raw body they are query parameters.
        </P>
        <Pre caption="GET /v1/status" lang="json">{`{ "state": "running", "queued": 1, "stage": "sampling", "done": 31, "total": 50 }`}</Pre>
      </Section>

      <Section>
        <H2>Developing the UI</H2>
        <CopyBlock command={`cd ui && pnpm install\nIDOL_URL=http://127.0.0.1:8190 pnpm dev   # proxies /v1 to a running idol serve\npnpm build                                # ui/dist, which build.rs embeds`} />
      </Section>
    </DocsLayout>
  );
}
