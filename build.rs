use std::path::PathBuf;

/// Writes `$OUT_DIR/ui.rs`: the built web UI (`ui/dist`, from `pnpm build` in `ui/`) as `(path, bytes)` pairs for
/// `idol serve` to embed. Without a build the list is empty and `/` says how to make one.
fn embed_ui(out: &std::path::Path) {
    fn walk(dir: &std::path::Path, root: &std::path::Path, files: &mut Vec<(String, PathBuf)>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, root, files);
            } else {
                files.push((p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/"), p.canonicalize().unwrap()));
            }
        }
    }
    let root = PathBuf::from("ui/dist");
    println!("cargo:rerun-if-changed=ui/dist");
    let mut files = Vec::new();
    walk(&root, &root, &mut files);
    files.sort();
    println!("cargo:rerun-if-env-changed=IDOL_REQUIRE_UI");
    if files.is_empty() && std::env::var_os("IDOL_REQUIRE_UI").is_some_and(|v| v != "0") {
        panic!("IDOL_REQUIRE_UI is set but ui/dist is empty: run `pnpm install && pnpm build` in ui/");
    }
    let mut src = String::from("&[\n");
    for (path, abs) in &files {
        println!("cargo:rerun-if-changed={}", abs.display());
        src += &format!("    ({path:?}, include_bytes!({:?}) as &[u8]),\n", abs.display().to_string());
    }
    src += "]\n";
    std::fs::write(out.join("ui.rs"), src).unwrap();
}

/// Compiles `kernels/idol.cu` to PTX (`$OUT_DIR/idol.ptx`) for the CUDA build; the PTX is JIT-compiled by the driver
/// for whatever GPU runs it, so one file serves every architecture from Ampere on. Other builds embed nothing.
fn build_kernels(out: &std::path::Path) {
    let ptx = out.join("idol.ptx");
    println!("cargo:rerun-if-changed=kernels/idol.cu");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        std::fs::write(&ptx, "").unwrap();
        return;
    }
    let nvcc = std::env::var_os("CUDA_PATH")
        .map(|p| PathBuf::from(p).join("bin/nvcc"))
        .filter(|p| p.is_file())
        .or_else(|| ["/opt/cuda/bin/nvcc", "/usr/local/cuda/bin/nvcc"].iter().map(PathBuf::from).find(|p| p.is_file()))
        .unwrap_or_else(|| PathBuf::from("nvcc"));
    // half-precision intrinsics need sm_53; the engine targets Ampere (80) and newer, Turing (75) works too
    let cap = std::env::var("CUDA_COMPUTE_CAP").ok().and_then(|c| c.parse::<u32>().ok()).unwrap_or(80).clamp(75, 80);
    let status = std::process::Command::new(&nvcc)
        .args(["--ptx", "-O3", &format!("-arch=compute_{cap}"), "-o"])
        .arg(&ptx)
        .arg("kernels/idol.cu")
        .status()
        .unwrap_or_else(|e| panic!("cannot run {}: {e} (the CUDA build needs nvcc)", nvcc.display()));
    assert!(status.success(), "nvcc failed on kernels/idol.cu");
}

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    embed_ui(&out);
    build_kernels(&out);
}
