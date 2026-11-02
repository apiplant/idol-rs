//! HTTP server: `POST /v1/generate` turns an image into a Gaussian splat (a standard 3DGS `.ply`), keeping every
//! model loaded between requests.
//!
//! The GPU engine is single-tenant, so one worker thread owns the [`Engine`] and requests queue for it; the ntex
//! handlers only parse requests and forward the result. Endpoints:
//!
//! - `POST /v1/generate`: JSON `{"image": b64, "mode": "synthetic"|"real", "crop_margin", "seed", "steps",
//!   "cfg_strength", "rescale_t", "num_gaussians"}` (only `image` is required; a `data:` URL prefix is accepted), or
//!   the raw image bytes as the body with the same fields as query parameters. Returns the `.ply`; the headers
//!   `x-idol-job` and `x-idol-camera` carry the job id and the predicted camera.
//! - `POST /v1/jobs`: the same body, returns `{"id"}` at once; `GET /v1/jobs?id=` is `{"state"}`;
//!   `GET /v1/jobs/result?id=&file=ply|input|camera|preview` fetches a finished job's files; `DELETE /v1/jobs?id=`
//!   removes them.
//! - `GET /v1/info`: models, defaults, queue length. `GET /v1/status`: what the engine is doing right now (stage and
//!   step), for progress bars. `POST /v1/cancel`. `GET /health`: `ok`.
//! - `GET /`: a web UI for all of the above (built from `ui/` and embedded). `GET /openapi.json`: the API as
//!   OpenAPI 3.1, and `GET /docs` renders it (Scalar, loaded from a CDN).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, Result};
use ntex::http::StatusCode;
use ntex::util::Bytes;
use ntex::web::{self, App, HttpRequest, HttpResponse, HttpServer};
use serde::Deserialize;
use tokio::sync::oneshot;

use crate::config::{DEFAULT_GAUSSIANS, MAX_GAUSSIANS, MIN_GAUSSIANS};
use crate::pipeline::{Engine, Request};
use crate::preprocess::Mode;

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct GenRequest {
    /// base64 file contents of the picture (png, jpeg, webp, bmp)
    #[serde(default)]
    image: String,
    mode: Option<String>,
    crop_margin: Option<f32>,
    seed: Option<u64>,
    steps: Option<usize>,
    cfg_strength: Option<f64>,
    rescale_t: Option<f64>,
    num_gaussians: Option<usize>,
    /// turn the result over (default: the server's `--flip`)
    flip: Option<bool>,
}

struct Job {
    req: GenRequest,
    /// raw image bytes when the request body was the picture itself
    raw: Option<Vec<u8>>,
    gen: u64,
    /// Folder name in the temp dir (`idol-serve-<pid>-<n>`), also the id clients poll with.
    name: String,
    /// `POST /v1/generate` waits on the finished ply here; `POST /v1/jobs` clients poll instead.
    done: Option<oneshot::Sender<Result<(Vec<u8>, String, String)>>>,
}

/// `queued` / `running` / `done` / `failed: <why>` per job id, for `GET /v1/jobs`.
fn states() -> &'static Mutex<HashMap<String, String>> {
    static S: OnceLock<Mutex<HashMap<String, String>>> = OnceLock::new();
    S.get_or_init(Default::default)
}

fn set_state(name: &str, state: impl Into<String>) {
    states().lock().unwrap().insert(name.to_string(), state.into());
}

/// Per-request directory in the temp dir. A failed request's is removed on drop; a finished one is kept (`keep`)
/// until the client deletes it (`DELETE /v1/jobs`).
struct Scratch(PathBuf, bool);
impl Scratch {
    fn keep(&mut self) {
        self.1 = true;
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        if !self.1 {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

/// A job folder name as the worker makes them: `idol-serve-<pid>-<n>`.
fn is_job_name(s: &str) -> bool {
    s.strip_prefix("idol-serve-").is_some_and(|r| r.split('-').count() == 2 && r.split('-').all(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())))
}

fn decode_b64(s: &str) -> Result<Vec<u8>> {
    let s = s.split_once("base64,").map(|(_, b)| b).unwrap_or(s);
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in s.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' | b'\n' | b'\r' | b' ' => continue,
            _ => anyhow::bail!("invalid base64"),
        } as u32;
        acc = (acc << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

fn to_request(g: &GenRequest, raw: Option<&[u8]>, default_flip: bool) -> Result<Request> {
    let bytes = match raw {
        Some(b) => b.to_vec(),
        None => decode_b64(&g.image)?,
    };
    if bytes.is_empty() {
        anyhow::bail!("no image");
    }
    let img = image::load_from_memory(&bytes).map_err(|e| anyhow!("cannot read the image: {e}"))?;
    let mut r = Request::new(img);
    r.mode = match g.mode.as_deref() {
        None | Some("synthetic") => Mode::Synthetic,
        Some("real") => Mode::Real,
        Some(m) => anyhow::bail!("mode must be synthetic or real, got {m}"),
    };
    r.crop_margin = g.crop_margin;
    r.flip = g.flip.unwrap_or(default_flip);
    r.seed = g.seed.unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|t| t.as_nanos() as u64 & 0xffff_ffff).unwrap_or(42));
    r.steps = g.steps.unwrap_or(r.steps);
    r.cfg_strength = g.cfg_strength.unwrap_or(r.cfg_strength);
    r.rescale_t = g.rescale_t.unwrap_or(r.rescale_t);
    let n = g.num_gaussians.unwrap_or(DEFAULT_GAUSSIANS);
    if !(MIN_GAUSSIANS..=MAX_GAUSSIANS).contains(&n) {
        anyhow::bail!("num_gaussians must be between {MIN_GAUSSIANS} and {MAX_GAUSSIANS}");
    }
    r.num_gaussians = n / 32 * 32;
    if r.steps == 0 || r.steps > 1000 {
        anyhow::bail!("steps must be between 1 and 1000");
    }
    Ok(r)
}

/// The engine thread: runs jobs one at a time.
fn worker(mut eng: Engine, jobs: std_mpsc::Receiver<Job>, pending: Arc<AtomicUsize>, low_vram: bool, default_flip: bool) {
    for job in jobs {
        let name = job.name.clone();
        if crate::progress::is_cancelled(job.gen) {
            pending.fetch_sub(1, Ordering::Relaxed);
            set_state(&name, "failed: cancelled");
            if let Some(done) = job.done {
                let _ = done.send(Err(anyhow!("cancelled")));
            }
            continue;
        }
        crate::progress::begin(job.gen);
        set_state(&name, "running");
        let want_bytes = job.done.is_some();
        let dir = std::env::temp_dir().join(&name);
        let mut scratch = Scratch(dir.clone(), false);
        let r = std::fs::create_dir_all(&dir).map_err(anyhow::Error::from).and_then(|_| {
            let mut req = to_request(&job.req, job.raw.as_deref(), default_flip)?;
            req.batch_cfg = !low_vram;
            let o = eng.run(&req)?;
            let ply = o.gaussians.to_ply();
            std::fs::write(dir.join("out.ply"), &ply)?;
            o.preprocessed.save(dir.join("input.png"))?;
            let camera = serde_json::to_string(&o.camera)?;
            std::fs::write(dir.join("camera.json"), &camera)?;
            std::fs::write(dir.join("meta.json"), serde_json::json!({ "gaussians": o.gaussians.len(), "seconds": o.seconds, "seed": req.seed, "steps": req.steps }).to_string())?;
            Ok((if want_bytes { ply } else { Vec::new() }, camera))
        });
        if r.is_ok() {
            scratch.keep();
        }
        match &r {
            Ok(_) => set_state(&name, "done"),
            Err(e) => {
                eprintln!("[idol] request failed: {e:#}");
                set_state(&name, format!("failed: {e:#}"));
            }
        }
        crate::progress::idle();
        pending.fetch_sub(1, Ordering::Relaxed);
        if let Some(done) = job.done {
            let _ = done.send(r.map(|(ply, cam)| (ply, name, cam)));
        }
    }
}

fn error(status: StatusCode, msg: impl std::fmt::Display) -> HttpResponse {
    HttpResponse::builder(status).content_type("text/plain; charset=utf-8").body(format!("{msg}\n"))
}

fn query(req: &HttpRequest, key: &str) -> Option<String> {
    req.query_string().split('&').find_map(|kv| kv.strip_prefix(key).and_then(|r| r.strip_prefix('=')).map(String::from))
}

fn parse_body(req: &HttpRequest, body: &Bytes) -> Result<(GenRequest, Option<Vec<u8>>), HttpResponse> {
    let ct = req.headers().get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("").to_string();
    if ct.starts_with("image/") || (!ct.contains("json") && body.first().is_some_and(|b| *b != b'{')) {
        let q = |k: &str| query(req, k);
        let parse = |k: &str| -> Result<Option<f64>, HttpResponse> { q(k).map(|v| v.parse::<f64>().map_err(|_| error(StatusCode::BAD_REQUEST, format!("bad {k}")))).transpose() };
        let g = GenRequest {
            image: String::new(),
            mode: q("mode"),
            crop_margin: parse("crop_margin")?.map(|v| v as f32),
            seed: parse("seed")?.map(|v| v as u64),
            steps: parse("steps")?.map(|v| v as usize),
            cfg_strength: parse("cfg_strength")?,
            rescale_t: parse("rescale_t")?,
            num_gaussians: parse("num_gaussians")?.map(|v| v as usize),
            flip: q("flip").map(|v| matches!(v.as_str(), "1" | "true" | "yes" | "on")),
        };
        return Ok((g, Some(body.to_vec())));
    }
    let g = serde_json::from_slice::<GenRequest>(body).map_err(|e| error(StatusCode::BAD_REQUEST, format!("bad JSON: {e}")))?;
    if g.image.trim().is_empty() {
        return Err(error(StatusCode::BAD_REQUEST, "no image"));
    }
    Ok((g, None))
}

/// Queue a request; the id names its folder in the temp dir.
fn submit(jobs: &std_mpsc::Sender<Job>, pending: &AtomicUsize, req: GenRequest, raw: Option<Vec<u8>>, done: Option<oneshot::Sender<Result<(Vec<u8>, String, String)>>>) -> Result<String, HttpResponse> {
    static N: AtomicUsize = AtomicUsize::new(0);
    let name = format!("idol-serve-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed));
    set_state(&name, "queued");
    pending.fetch_add(1, Ordering::Relaxed);
    if jobs.send(Job { req, raw, gen: crate::progress::generation(), name: name.clone(), done }).is_err() {
        pending.fetch_sub(1, Ordering::Relaxed);
        return Err(error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped"));
    }
    Ok(name)
}

async fn generate(jobs: std_mpsc::Sender<Job>, pending: Arc<AtomicUsize>, req: HttpRequest, body: Bytes) -> HttpResponse {
    let (g, raw) = match parse_body(&req, &body) {
        Ok(g) => g,
        Err(r) => return r,
    };
    let (done, rx) = oneshot::channel();
    if let Err(r) = submit(&jobs, &pending, g, raw, Some(done)) {
        return r;
    }
    match rx.await {
        Ok(Ok((ply, job, camera))) => HttpResponse::Ok().content_type("application/octet-stream").header("x-idol-job", job).header("x-idol-camera", camera).body(ply),
        Ok(Err(e)) => error(StatusCode::BAD_REQUEST, format!("{e:#}")),
        Err(_) => error(StatusCode::SERVICE_UNAVAILABLE, "engine stopped"),
    }
}

/// `POST /v1/jobs`: queue a request and return its id at once; poll `GET /v1/jobs?id=`.
async fn create_job(jobs: std_mpsc::Sender<Job>, pending: Arc<AtomicUsize>, req: HttpRequest, body: Bytes) -> HttpResponse {
    let (g, raw) = match parse_body(&req, &body) {
        Ok(g) => g,
        Err(r) => return r,
    };
    match submit(&jobs, &pending, g, raw, None) {
        Ok(id) => HttpResponse::Accepted().json(&serde_json::json!({ "id": id })),
        Err(r) => r,
    }
}

fn job_id(req: &HttpRequest) -> Option<String> {
    let id = query(req, "id")?;
    is_job_name(&id).then_some(id)
}

/// `GET /v1/jobs?id=`: `{"state": "queued" | "running" | "done" | "failed", "error"?, "meta"?, "camera"?}`. A folder
/// left by an earlier server run counts as done when its ply is there.
async fn job_state(req: HttpRequest) -> HttpResponse {
    let Some(id) = job_id(&req) else { return error(StatusCode::BAD_REQUEST, "bad job id") };
    let dir = std::env::temp_dir().join(&id);
    let st = states().lock().unwrap().get(&id).cloned();
    let st = st.or_else(|| dir.join("out.ply").is_file().then(|| "done".to_string()));
    let mut body = match st.as_deref() {
        None => return error(StatusCode::NOT_FOUND, "unknown job"),
        Some(s) => match s.strip_prefix("failed: ") {
            Some(e) => serde_json::json!({ "state": "failed", "error": e }),
            None => serde_json::json!({ "state": s }),
        },
    };
    if st.as_deref() == Some("done") {
        let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok().and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());
        body["meta"] = read("meta.json").unwrap_or_default();
        body["camera"] = read("camera.json").unwrap_or_default();
    }
    HttpResponse::Ok().header("cache-control", "no-store").json(&body)
}

/// `GET /v1/jobs/result?id=&file=`: a finished job's `ply` (default), `input` (png) or `camera` (json).
async fn job_result(req: HttpRequest) -> HttpResponse {
    let Some(id) = job_id(&req) else { return error(StatusCode::BAD_REQUEST, "bad job id") };
    let (file, ct) = match query(&req, "file").as_deref() {
        None | Some("ply") => ("out.ply", "application/octet-stream"),
        Some("input") | Some("preview") => ("input.png", "image/png"),
        Some("camera") => ("camera.json", "application/json"),
        Some(_) => return error(StatusCode::BAD_REQUEST, "file must be ply, input or camera"),
    };
    match std::fs::read(std::env::temp_dir().join(&id).join(file)) {
        Ok(b) => HttpResponse::Ok().content_type(ct).header("cache-control", "no-store").body(b),
        Err(_) => error(StatusCode::NOT_FOUND, "no result for this job"),
    }
}

/// Delete a finished job's folder (`DELETE /v1/jobs?id=idol-serve-<pid>-<n>`).
async fn delete_job(req: HttpRequest) -> HttpResponse {
    let Some(id) = job_id(&req) else { return error(StatusCode::BAD_REQUEST, "bad job id") };
    match std::fs::remove_dir_all(std::env::temp_dir().join(&id)) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
    states().lock().unwrap().remove(&id);
    HttpResponse::Ok().content_type("text/plain; charset=utf-8").body("deleted\n")
}

/// The web UI's files (`ui/dist`) as `(path, bytes)`, embedded by build.rs; empty when it was not built.
static UI: &[(&str, &[u8])] = include!(concat!(env!("OUT_DIR"), "/ui.rs"));

const DOCS_HTML: &str = r#"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>idol API</title>
  </head>
  <body>
    <script id="api-reference" data-url="/openapi.json"></script>
    <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference@1"></script>
  </body>
</html>
"#;

fn openapi() -> String {
    include_str!("openapi.json").replacen("\"VERSION\"", &format!("{:?}", env!("CARGO_PKG_VERSION")), 1)
}

fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript",
        Some("css") => "text/css",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("webp") => "image/webp",
        Some("ico") => "image/x-icon",
        Some("json") => "application/json",
        Some("woff2") => "font/woff2",
        Some("ply") => "application/octet-stream",
        _ => "application/octet-stream",
    }
}

async fn ui(req: HttpRequest) -> HttpResponse {
    let path = req.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    match UI.iter().find(|(p, _)| *p == path) {
        Some((p, bytes)) => {
            let cache = if p.starts_with("assets/") { "public, max-age=31536000, immutable" } else { "no-cache" };
            HttpResponse::Ok().content_type(content_type(p)).header("cache-control", cache).body(Bytes::from_static(bytes))
        }
        None if UI.is_empty() && path == "index.html" => error(StatusCode::NOT_FOUND, "this idol was built without its web UI: run `pnpm install && pnpm build` in ui/, then rebuild"),
        None => error(StatusCode::NOT_FOUND, "not found"),
    }
}

fn status_json(pending: &AtomicUsize) -> serde_json::Value {
    let queued = pending.load(Ordering::Relaxed);
    let (stage, done, total) = crate::progress::snapshot();
    let state = match (queued, stage) {
        (0, _) => "idle",
        (_, crate::progress::Stage::Idle) => "starting",
        _ => "running",
    };
    serde_json::json!({ "state": state, "queued": queued, "stage": stage.name(), "done": done, "total": total })
}

/// Load the engine with `load` on a dedicated thread (CUDA state stays on the thread that created it), then serve
/// it on `addr` until the process is stopped (Ctrl-C).
pub fn run<F>(load: F, addr: &str, models_dir: PathBuf, low_vram: bool, default_flip: bool) -> Result<()>
where
    F: FnOnce() -> Result<Engine> + Send + 'static,
{
    let (jobs, rx) = std_mpsc::channel::<Job>();
    let (ready_tx, ready) = std_mpsc::channel::<Result<serde_json::Value>>();
    let pending = Arc::new(AtomicUsize::new(0));
    let p2 = pending.clone();
    std::thread::Builder::new().name("idol-engine".into()).spawn(move || {
        let loaded = load().and_then(|mut e| {
            e.preload()?;
            Ok(e)
        });
        match loaded {
            Ok(eng) => {
                let info = serde_json::json!({
                    "version": env!("CARGO_PKG_VERSION"),
                    "device": eng.device_name(),
                    "models_dir": models_dir.display().to_string(),
                    "low_vram": low_vram,
                    "defaults": { "mode": "synthetic", "steps": 50, "cfg_strength": 3.0, "rescale_t": 3.0, "num_gaussians": DEFAULT_GAUSSIANS, "flip": default_flip },
                    "limits": { "num_gaussians": [MIN_GAUSSIANS, MAX_GAUSSIANS] },
                });
                let _ = ready_tx.send(Ok(info));
                worker(eng, rx, p2, low_vram, default_flip)
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        }
    })?;
    let info = ready.recv().map_err(|_| anyhow!("engine thread died while loading"))??;
    serve(jobs, pending, addr.to_string(), info)?;
    Ok(())
}

/// The startup banner: where to point a browser, and what is loaded.
fn banner(addr: &str, info: &serde_json::Value) {
    use std::io::IsTerminal;
    let tty = std::io::stderr().is_terminal();
    let c = |code: &str, t: &str| if tty { format!("\x1b[{code}m{t}\x1b[0m") } else { t.to_string() };
    let host = addr.replace("0.0.0.0", "localhost").replace("[::]", "localhost");
    let ver = info["version"].as_str().unwrap_or("");
    eprintln!();
    eprintln!("  {}  {}", c("1;38;5;141", "◆ idol"), c("2", &format!("v{ver} · one image to a 3D Gaussian splat")));
    eprintln!();
    eprintln!("  {}  {}", c("38;5;141", "web UI   "), c("4", &format!("http://{host}/")));
    eprintln!("  {}  {}", c("38;5;141", "API docs "), c("4", &format!("http://{host}/docs")));
    eprintln!();
    eprintln!("  {}  {}", c("2", "device   "), info["device"].as_str().unwrap_or("-"));
    eprintln!("  {}  {}", c("2", "models   "), info["models_dir"].as_str().unwrap_or("-"));
    eprintln!();
}

#[ntex::main(name = "idol", signals = true)]
async fn serve(jobs: std_mpsc::Sender<Job>, pending: Arc<AtomicUsize>, addr: String, info: serde_json::Value) -> std::io::Result<()> {
    banner(&addr, &info);
    HttpServer::new(async move |_| {
        let (j1, j2, p1, p2, p3, p4) = (jobs.clone(), jobs.clone(), pending.clone(), pending.clone(), pending.clone(), pending.clone());
        let info = info.clone();
        // pictures arrive base64-encoded in the JSON body, or raw
        let cfg = web::WebAppConfig::new().set_state(web::types::PayloadConfig::new(128 << 20));
        App::new()
            .with_config(cfg)
            .service(web::resource("/health").to(async || "ok\n"))
            .service(web::resource("/openapi.json").to(async || HttpResponse::Ok().content_type("application/json").body(openapi())))
            .service(web::resource("/docs").to(async || HttpResponse::Ok().content_type("text/html; charset=utf-8").body(DOCS_HTML)))
            .service(web::resource("/v1/info").route(web::get().to(move || {
                let mut i = info.clone();
                i["queued"] = serde_json::json!(p2.load(Ordering::Relaxed));
                async move { HttpResponse::Ok().json(&i) }
            })))
            .service(web::resource("/v1/status").route(web::get().to(move || {
                let s = status_json(&p3);
                async move { HttpResponse::Ok().header("cache-control", "no-store").json(&s) }
            })))
            .service(web::resource("/v1/cancel").route(web::post().to(async || {
                crate::progress::cancel_all();
                HttpResponse::Ok().content_type("text/plain; charset=utf-8").body("cancelled\n")
            })))
            .service(web::resource("/v1/generate").route(web::post().to(move |req: HttpRequest, body: Bytes| generate(j1.clone(), p1.clone(), req, body))))
            .service(
                web::resource("/v1/jobs")
                    .route(web::get().to(job_state))
                    .route(web::delete().to(delete_job))
                    .route(web::post().to(move |req: HttpRequest, body: Bytes| create_job(j2.clone(), p4.clone(), req, body))),
            )
            .service(web::resource("/v1/jobs/result").route(web::get().to(job_result)))
            .default_service(web::get().to(ui))
    })
    .bind(addr.as_str(), ntex::SharedCfg::new("idol"))?
    .workers(1)
    .run()
    .await
}

#[cfg(test)]
mod tests {
    #[test]
    fn base64_roundtrip() {
        assert_eq!(super::decode_b64("aGVsbG8gd29ybGQ=").unwrap(), b"hello world");
        assert_eq!(super::decode_b64("data:image/png;base64,aGk=").unwrap(), b"hi");
    }

    #[test]
    fn job_names() {
        assert!(super::is_job_name("idol-serve-12-3"));
        assert!(!super::is_job_name("idol-serve-12-3/../x"));
        assert!(!super::is_job_name("h3-serve-1-1"));
    }
}
