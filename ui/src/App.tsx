import { For, Show, createEffect, createMemo, createSignal, onSettled, untrack } from "solid-js";
import type { JSX } from "@solidjs/web";
import { cancel as cancelRun, deleteJob, info as fetchInfo, jobState, resultUrl, status as fetchStatus, submit, toDataUrl, type Camera, type Info, type Status } from "./api";
import { SplatView, parsePly } from "./splat";
import { btn, btnGhost, btnIcon, btnLg, btnPrimary, btnSm, btnSmIcon, btnStop, chip, dot, errorBox, input, panel, panelHead, panelTitle, sectionLabel, take } from "./styles";
import { Icon, Logo, Segmented, Slider } from "./ui";

type Mode = "synthetic" | "real";

interface Take {
  id: number;
  /** the job's folder on the server (deleted with the take) */
  job: string;
  name: string;
  seed: number;
  gaussians: number;
  seconds: number;
  steps: number;
  camera?: Camera;
}

const EXAMPLES = ["fig-fox", "food-avocado", "fur-egg-chair", "obj-backpack", "toy-robot-tin", "nb-microscope", "fig-axolotl", "obj-birdhouse"];
const COUNTS = [
  { value: 32768, label: "32K" },
  { value: 65536, label: "65K" },
  { value: 131072, label: "131K" },
  { value: 262144, label: "262K" },
  { value: 524288, label: "524K" },
  { value: 1048576, label: "1M" },
];
const STAGES = [
  { id: "preparing", label: "Cutting out" },
  { id: "encoding", label: "Reading" },
  { id: "sampling", label: "Denoising" },
  { id: "placing", label: "Placing" },
  { id: "decoding", label: "Decoding" },
] as const;
const STORE = "idol.ui.v1";
let nextId = 0;

interface Saved {
  mode: Mode;
  count: number;
  steps: number;
  cfg: number;
  rescale: number;
  seed: string;
  crop: number;
  turn?: boolean;
}

function loadSaved(): Partial<Saved> {
  try {
    return JSON.parse(localStorage.getItem(STORE) ?? "{}");
  } catch {
    return {};
  }
}

const fmt = (n: number) => n.toLocaleString("en-US");
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** A click-or-drop target that hands back the image it received. */
function Drop(props: { onFile: (f: File) => void; class?: string; children: JSX.Element }) {
  let file!: HTMLInputElement;
  const [over, setOver] = createSignal(false);
  return (
    <div
      role="button"
      tabindex="0"
      class={`${props.class ?? ""} cursor-pointer transition-colors ${over() ? "!border-accent !bg-accent/10" : ""}`}
      onClick={() => file.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), file.click())}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const f = e.dataTransfer?.files?.[0];
        if (f?.type.startsWith("image/")) props.onFile(f);
      }}
    >
      <input
        ref={file}
        type="file"
        accept="image/*"
        class="hidden"
        onChange={(e) => {
          const f = e.currentTarget.files?.[0];
          if (f) props.onFile(f);
          e.currentTarget.value = "";
        }}
      />
      {props.children}
    </div>
  );
}

export function App() {
  const saved = loadSaved();
  const [info, setInfo] = createSignal<Info | null>(null);
  const [online, setOnline] = createSignal(true);
  const [status, setStatus] = createSignal<Status | null>(null);
  const [mode, setMode] = createSignal<Mode>(saved.mode ?? "synthetic");
  const [count, setCount] = createSignal(saved.count ?? 262144);
  const [steps, setSteps] = createSignal(saved.steps ?? 50);
  const [cfg, setCfg] = createSignal(saved.cfg ?? 3);
  const [rescale, setRescale] = createSignal(saved.rescale ?? 3);
  const [seed, setSeed] = createSignal(saved.seed ?? "");
  const [crop, setCrop] = createSignal(saved.crop ?? 0);
  /** turn the result over when generating; undefined: follow the server's default */
  const [turn, setTurn] = createSignal<boolean | undefined>(saved.turn);
  const [file, setFile] = createSignal<File | null>(null);
  const [previewUrl, setPreviewUrl] = createSignal("");
  const [busy, setBusy] = createSignal(false);
  const [error, setError] = createSignal("");
  const [takes, setTakes] = createSignal<Take[]>([]);
  const [current, setCurrent] = createSignal<Take | null>(null);
  const [rotate, setRotate] = createSignal(true);
  const [flip, setFlip] = createSignal(false);
  const [loadingTake, setLoadingTake] = createSignal(false);
  /** a splat opened with ?ply=<url> */
  const [external, setExternal] = createSignal(false);
  let canvas!: HTMLCanvasElement;
  let view: SplatView | undefined;
  let stopped = false;

  createEffect(
    () => JSON.stringify({ mode: mode(), count: count(), steps: steps(), cfg: cfg(), rescale: rescale(), seed: seed(), crop: crop(), turn: turn() } satisfies Saved),
    (json) => {
      try {
        localStorage.setItem(STORE, json);
      } catch {}
    },
  );

  function pick(f: File) {
    const old = untrack(previewUrl);
    if (old) URL.revokeObjectURL(old);
    setPreviewUrl(URL.createObjectURL(f));
    setFile(f);
    setError("");
  }

  async function example(name: string) {
    try {
      const r = await fetch(`/examples/${name}.webp`);
      pick(new File([await r.blob()], `${name}.webp`, { type: "image/webp" }));
      setMode("synthetic");
    } catch (e) {
      setError(String(e));
    }
  }

  async function show(t: Take) {
    setLoadingTake(true);
    setCurrent(t);
    try {
      const buf = await (await fetch(resultUrl(t.job, "ply"))).arrayBuffer();
      view?.load(parsePly(buf));
    } catch (e) {
      setError(`Could not load the splat: ${e}`);
    } finally {
      setLoadingTake(false);
    }
  }

  async function run() {
    const f = file();
    if (!f || busy()) return;
    setBusy(true);
    setError("");
    stopped = false;
    try {
      const sd = seed().trim() === "" ? Math.floor(Math.random() * 2 ** 31) : Math.floor(Number(seed()));
      if (!Number.isFinite(sd) || sd < 0) throw new Error("The seed must be a non-negative whole number.");
      const id = await submit({
        image: await toDataUrl(f),
        mode: mode(),
        seed: sd,
        steps: steps(),
        cfg_strength: cfg(),
        rescale_t: rescale(),
        num_gaussians: count(),
        flip: turn() ?? info()?.defaults.flip ?? false,
        ...(crop() > 0 ? { crop_margin: crop() } : {}),
      });
      for (;;) {
        await sleep(350);
        fetchStatus().then(setStatus).catch(() => {});
        const j = await jobState(id);
        if (j.state === "failed") {
          if (stopped || /cancelled/.test(j.error ?? "")) return;
          throw new Error(j.error ?? "The generation failed.");
        }
        if (j.state === "done") {
          const t: Take = { id: nextId++, job: id, name: f.name.replace(/\.[^.]+$/, ""), seed: j.meta?.seed ?? sd, gaussians: j.meta?.gaussians ?? count(), seconds: j.meta?.seconds ?? 0, steps: j.meta?.steps ?? steps(), camera: j.camera };
          setTakes((all) => [t, ...all]);
          await show(t);
          return;
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setStatus(null);
    }
  }

  function stop() {
    stopped = true;
    cancelRun();
  }

  async function remove(t: Take) {
    deleteJob(t.job);
    setTakes((all) => all.filter((x) => x.id !== t.id));
    if (current()?.id === t.id) {
      setCurrent(null);
      view?.clear();
    }
  }

  async function snapshot() {
    const b = await view?.snapshot();
    if (!b) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = `${current()?.name ?? "idol"}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  onSettled(() => {
    try {
      view = new SplatView(canvas);
      view.onInteract = () => setRotate(false);
    } catch (e) {
      setError(String(e));
    }
    const ply = new URLSearchParams(location.search).get("ply");
    if (ply) {
      fetch(ply)
        .then((r) => r.arrayBuffer())
        .then((b) => (view?.load(parsePly(b)), setExternal(true)))
        .catch((e) => setError(`Could not load ${ply}: ${e}`));
    }
    fetchInfo().then(setInfo).catch(() => {});
    const ping = () =>
      fetch("/health")
        .then((r) => setOnline(r.ok))
        .catch(() => setOnline(false));
    const timer = setInterval(ping, 8000);
    const key = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        run();
      }
    };
    const paste = (e: ClipboardEvent) => {
      const f = [...(e.clipboardData?.files ?? [])].find((f) => f.type.startsWith("image/"));
      if (f) pick(f);
    };
    window.addEventListener("keydown", key);
    window.addEventListener("paste", paste);
    return () => {
      clearInterval(timer);
      window.removeEventListener("keydown", key);
      window.removeEventListener("paste", paste);
      view?.destroy();
    };
  });

  createEffect(
    () => rotate(),
    (r) => {
      if (view) view.autoRotate = r;
    },
  );
  createEffect(
    () => flip(),
    (f) => {
      if (view) view.flip = f;
    },
  );

  const stageIndex = createMemo(() => STAGES.findIndex((s) => s.id === status()?.stage));
  const pct = createMemo(() => {
    const s = status();
    if (!s || !s.total) return 0;
    return Math.min(100, (s.done / s.total) * 100);
  });
  const stageLabel = createMemo(() => {
    const s = status();
    if (!s) return busy() ? "Queued" : "";
    if (s.stage === "downloading") return `Downloading models · ${s.done} / ${s.total} MB`;
    const st = STAGES.find((x) => x.id === s.stage);
    if (!st) return "Starting";
    return s.total && s.stage === "sampling" ? `${st.label} · step ${s.done} / ${s.total}` : st.label;
  });

  return (
    <div class="flex h-full flex-col">
      <header class="flex h-12 flex-none items-center gap-3 border-b border-ink-800 px-4">
        <Logo size={24} />
        <span class="text-[15px] font-semibold tracking-tight text-ink-100">idol</span>
        <span class="hidden text-ink-400 sm:inline">one image → 3D Gaussian splat</span>
        <span class="flex-1" />
        <Show when={info()}>{(i) => <span class={`${chip} hidden md:inline-flex`}>v{i().version} · {i().device}</span>}</Show>
        <span class={`${chip} gap-1.5`} title={online() ? "The server is up" : "Cannot reach the server"}>
          <span class={`${dot} ${online() ? "bg-ok" : "bg-rec"}`} />
          {online() ? "online" : "offline"}
        </span>
        <a class={`${btnGhost} !h-7 !text-xs`} href="/docs" target="_blank">
          <Icon name="book" size={13} /> API
        </a>
      </header>

      <main class="grid min-h-0 flex-1 grid-cols-1 gap-3 p-3 lg:grid-cols-[340px_minmax(0,1fr)]">
        {/* ---- controls ---- */}
        <section class={`${panel} min-h-0 overflow-y-auto`}>
          <div class={panelHead}>
            <span class={panelTitle}>Image</span>
          </div>
          <div class="flex flex-col gap-4 p-3.5">
            <Drop onFile={pick} class="group relative grid aspect-square place-items-center overflow-hidden rounded-xl border border-dashed border-ink-600 bg-ink-850 hover:border-ink-500">
              <Show
                when={previewUrl()}
                fallback={
                  <div class="flex flex-col items-center gap-2 px-6 text-center text-ink-400">
                    <Icon name="image" size={30} />
                    <span class="text-ink-200">Drop an image, click, or paste</span>
                    <span class="text-xs">A render, a generated picture or a photo of one object</span>
                  </div>
                }
              >
                {(u) => (
                  <>
                    <img src={u()} alt="input" class="size-full object-contain" />
                    <span class="absolute right-2 bottom-2 rounded-md bg-ink-950/80 px-2 py-1 text-xs text-ink-200 opacity-0 backdrop-blur transition-opacity group-hover:opacity-100">Replace</span>
                  </>
                )}
              </Show>
            </Drop>

            <div>
              <div class={sectionLabel}>Examples</div>
              <div class="grid grid-cols-4 gap-1.5">
                <For each={EXAMPLES}>
                  {(n) => (
                    <button class="aspect-square cursor-pointer overflow-hidden rounded-lg border border-ink-700 bg-ink-850 transition-colors hover:border-accent" title={n} onClick={() => example(n)}>
                      <img src={`/examples/${n}.webp`} alt={n} loading="lazy" class="size-full object-cover" />
                    </button>
                  )}
                </For>
              </div>
            </div>

            <div>
              <div class={sectionLabel}>Picture type</div>
              <Segmented
                value={mode()}
                onChange={setMode}
                options={[
                  { value: "synthetic", label: "Render", hint: "Computer generated image of one object: tight crop (1.2)" },
                  { value: "real", label: "Photo", hint: "Real-world photograph: looser crop (1.7)" },
                ]}
              />
            </div>

            <div>
              <div class={sectionLabel}>Gaussians</div>
              <Segmented value={count()} onChange={setCount} options={COUNTS} />
            </div>

            <div>
              <Slider label="Steps" value={steps()} min={10} max={100} step={1} onChange={setSteps} />
              <Slider label="Guidance" value={cfg()} min={0} max={8} step={0.5} digits={1} onChange={setCfg} />
              <details class="mt-1 text-ink-300">
                <summary class="cursor-pointer select-none py-1 text-xs text-ink-400 hover:text-ink-200">Advanced</summary>
                <Slider label="Time shift" value={rescale()} min={1} max={6} step={0.5} digits={1} onChange={setRescale} />
                <Slider label="Crop margin" value={crop()} min={0} max={3} step={0.1} digits={1} format={(v) => (v === 0 ? "auto" : v.toFixed(1))} onChange={setCrop} />
              </details>
              <label class="mt-2 flex cursor-pointer items-center justify-between gap-3 py-1.5 text-ink-300" title="Some pictures come out of the model upside down: turn the result over by half a turn about the camera's view axis">
                <span>Turn the result over</span>
                <input type="checkbox" class="size-4 cursor-pointer accent-accent" checked={turn() ?? info()?.defaults.flip ?? false} onChange={(e) => setTurn(e.currentTarget.checked)} />
              </label>
              <div class="mt-2 flex items-center gap-2">
                <input class={`${input} font-mono`} inputmode="numeric" placeholder="random seed" value={seed()} onInput={(e) => setSeed(e.currentTarget.value.replace(/[^0-9]/g, ""))} />
                <button class={`${btn} ${btnIcon} flex-none`} title="Random seed" onClick={() => setSeed(String(Math.floor(Math.random() * 2 ** 31)))}>
                  <Icon name="dice" size={16} />
                </button>
              </div>
            </div>

            <Show when={error()}>
              <div class={errorBox} role="alert">
                <Icon name="x" size={15} class="mt-0.5 flex-none text-rec" />
                <span class="min-w-0 break-words">{error()}</span>
              </div>
            </Show>

            <Show
              when={busy()}
              fallback={
                <button class={`${btnPrimary} ${btnLg}`} disabled={!file() || !online()} onClick={run} title="Ctrl+Enter">
                  <Icon name="sparkle" size={16} /> Generate
                </button>
              }
            >
              <button class={`${btnStop} ${btnLg}`} onClick={stop}>
                <Icon name="stop" size={14} /> Stop
              </button>
            </Show>
          </div>
        </section>

        {/* ---- stage ---- */}
        <section class={`${panel} relative min-h-[420px] overflow-hidden`}>
          <div class={`${panelHead} z-10`}>
            <span class={panelTitle}>Splat</span>
            <Show when={current()}>
              {(t) => (
                <span class="truncate text-xs text-ink-400">
                  {fmt(t().gaussians)} Gaussians · {t().seconds.toFixed(1)} s · seed {t().seed}
                </span>
              )}
            </Show>
            <span class="flex-1" />
            <Show when={current()}>
              {(t) => (
                <>
                  <button class={`${btn} ${btnSmIcon}`} title={rotate() ? "Stop rotating" : "Rotate"} aria-pressed={rotate() ? "true" : "false"} onClick={() => setRotate(!rotate())}>
                    <Icon name="rotate" size={14} />
                  </button>
                  <button class={`${btn} ${btnSm}`} title="Turn the object over in the viewer only (the .ply is unchanged)" aria-pressed={flip() ? "true" : "false"} onClick={() => setFlip(!flip())}>
                    <Icon name="expand" size={13} /> Flip view
                  </button>
                  <Show when={t().camera}>
                    {(c) => (
                      <button class={`${btn} ${btnSm}`} title="Look from the camera the model predicted for the picture" onClick={() => (view?.setInputView(c().yaw, c().pitch, c().radius, c().fov), setRotate(false))}>
                        <Icon name="camera" size={13} /> Input view
                      </button>
                    )}
                  </Show>
                  <button class={`${btn} ${btnSmIcon}`} title="Save a PNG of this view" onClick={snapshot}>
                    <Icon name="image" size={14} />
                  </button>
                  <a class={`${btn} ${btnSm}`} href={resultUrl(t().job, "ply")} download={`${t().name}.ply`}>
                    <Icon name="download" size={13} /> .ply
                  </a>
                </>
              )}
            </Show>
          </div>

          <div class="stage-bg relative min-h-0 flex-1">
            <canvas ref={canvas} class="absolute inset-0 size-full touch-none cursor-grab active:cursor-grabbing" />
            <Show when={!current() && !busy() && !external()}>
              <div class="pointer-events-none absolute inset-0 grid place-items-center">
                <div class="flex flex-col items-center gap-4 text-center">
                  <div class="animate-float text-accent/80">
                    <Icon name="cube" size={72} />
                  </div>
                  <div class="text-lg font-medium text-ink-100">Your object will turn here</div>
                  <div class="max-w-xs text-ink-400">Pick an image on the left and press Generate. Drag to orbit, scroll to zoom.</div>
                </div>
              </div>
            </Show>
            <Show when={current() && loadingTake()}>
              <div class="pointer-events-none absolute inset-0 grid place-items-center bg-ink-950/50 text-ink-200">Loading splat…</div>
            </Show>
            <Show when={current()}>
              {(t) => (
                <img src={resultUrl(t().job, "input")} alt="what the model saw" title="The image the model saw" class="absolute bottom-3 left-3 size-24 rounded-lg border border-ink-600 bg-black object-contain shadow-lg" />
              )}
            </Show>
            <Show when={busy()}>
              <div class="absolute inset-x-3 bottom-3 rounded-xl border border-ink-700 bg-ink-900/90 p-3 backdrop-blur">
                <div class="mb-2 flex items-center justify-between text-ink-200">
                  <span class="font-medium">{stageLabel()}</span>
                  <Show when={status()?.queued && (status()?.queued ?? 0) > 1}>
                    <span class="text-xs text-ink-400">{status()!.queued - 1} waiting</span>
                  </Show>
                </div>
                <div class="relative h-1.5 overflow-hidden rounded-full bg-ink-700">
                  <Show
                    when={pct() > 0}
                    fallback={<div class="animate-sweep absolute inset-y-0 w-1/3 rounded-full bg-gradient-to-r from-accent to-glow" />}
                  >
                    <div class="h-full rounded-full bg-gradient-to-r from-accent to-glow transition-[width] duration-300" style={{ width: `${pct()}%` }} />
                  </Show>
                </div>
                <div class="mt-2.5 flex gap-1.5">
                  <For each={STAGES}>
                    {(s, i) => (
                      <span class={`h-1 flex-1 rounded-full transition-colors ${i() < stageIndex() ? "bg-accent" : i() === stageIndex() ? "bg-accent/60" : "bg-ink-700"}`} title={s.label} />
                    )}
                  </For>
                </div>
              </div>
            </Show>
          </div>

          <Show when={takes().length}>
            <div class="flex flex-none gap-2 overflow-x-auto border-t border-ink-750 p-2">
              <For each={takes()}>
                {(t) => (
                  <div class={`${take} group relative flex-none cursor-pointer ${current()?.id === t.id ? "!border-accent" : ""}`} onClick={() => show(t)}>
                    <img src={resultUrl(t.job, "input")} alt={t.name} class="size-16 rounded-md bg-black object-contain" />
                    <button
                      class={`${btn} ${btnSmIcon} absolute -top-1.5 -right-1.5 hidden !size-5 group-hover:inline-flex`}
                      title="Delete"
                      onClick={(e) => (e.stopPropagation(), remove(t))}
                    >
                      <Icon name="x" size={11} />
                    </button>
                  </div>
                )}
              </For>
              <span class="hidden" />
            </div>
          </Show>
        </section>
      </main>
    </div>
  );
}
