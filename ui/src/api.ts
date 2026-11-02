/* The HTTP API of `idol serve`; /openapi.json describes the same endpoints. */

export interface GenerateRequest {
  /** base64 file contents, a `data:` URL prefix is accepted */
  image: string;
  mode?: "synthetic" | "real";
  crop_margin?: number;
  seed?: number;
  steps?: number;
  cfg_strength?: number;
  rescale_t?: number;
  num_gaussians?: number;
  /** turn the result over (default: the server's setting) */
  flip?: boolean;
}

export interface Info {
  version: string;
  device: string;
  models_dir: string;
  low_vram: boolean;
  defaults: { mode: string; steps: number; cfg_strength: number; rescale_t: number; num_gaussians: number; flip: boolean };
  limits: { num_gaussians: [number, number] };
  queued: number;
}

export interface Status {
  state: "idle" | "starting" | "running";
  queued: number;
  stage: "idle" | "preparing" | "encoding" | "sampling" | "placing" | "decoding" | "downloading";
  done: number;
  total: number;
}

export interface Camera {
  yaw: number;
  pitch: number;
  radius: number;
  fov: number;
}

export interface JobState {
  state: "queued" | "running" | "done" | "failed";
  error?: string;
  meta?: { gaussians: number; seconds: number; seed: number; steps: number };
  camera?: Camera;
}

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error((await r.text()).trim() || `HTTP ${r.status}`);
  return r.json();
}

export const info = () => fetch("/v1/info").then((r) => json<Info>(r));
export const status = () => fetch("/v1/status", { cache: "no-store" }).then((r) => json<Status>(r));
export const cancel = () => fetch("/v1/cancel", { method: "POST" }).then(() => {});
export const deleteJob = (id: string) => fetch(`/v1/jobs?id=${id}`, { method: "DELETE" }).then(() => {});
export const resultUrl = (id: string, file: "ply" | "input" | "camera" = "ply") => `/v1/jobs/result?id=${id}&file=${file}`;

/** Queue a generation; resolves with the job id. */
export async function submit(req: GenerateRequest): Promise<string> {
  const r = await fetch("/v1/jobs", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(req) });
  return (await json<{ id: string }>(r)).id;
}

export const jobState = (id: string) => fetch(`/v1/jobs?id=${id}`, { cache: "no-store" }).then((r) => json<JobState>(r));

export function toDataUrl(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}
