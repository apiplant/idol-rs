/* A small WebGL2 Gaussian-splat viewer: parses the PLY `idol` writes, draws the splats as instanced screen-space
   ellipses (the 3DGS EWA projection) and sorts them back to front in a worker. */
import SortWorker from "./sort.worker?worker";

export interface Splats {
  count: number;
  /** x y z per splat, y-up */
  pos: Float32Array;
  /** 3 texels (RGBA32UI) per splat: centre + packed colour, then the six covariance terms */
  texels: Uint32Array;
}

const SH0 = 0.28209479177387814;

/** Parse a binary little-endian 3DGS PLY (float properties). */
export function parsePly(buf: ArrayBuffer): Splats {
  const head = new TextDecoder().decode(new Uint8Array(buf, 0, Math.min(buf.byteLength, 4096)));
  const end = head.indexOf("end_header\n");
  if (!head.startsWith("ply") || end < 0) throw new Error("not a PLY file");
  const header = head.slice(0, end);
  const count = +/element vertex (\d+)/.exec(header)![1];
  const props = [...header.matchAll(/property float (\w+)/g)].map((m) => m[1]);
  const stride = props.length;
  const col = (n: string) => {
    const i = props.indexOf(n);
    if (i < 0) throw new Error(`PLY has no ${n}`);
    return i;
  };
  const data = new Float32Array(buf.slice(end + "end_header\n".length, end + "end_header\n".length + count * stride * 4));
  const [X, Y, Z, D0, D1, D2, OP, S0, S1, S2, R0, R1, R2, R3] = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"].map(col);

  const pos = new Float32Array(count * 3);
  const texels = new Uint32Array(count * 12);
  const f = new Float32Array(texels.buffer);
  const clamp8 = (v: number) => Math.max(0, Math.min(255, Math.round(v * 255)));
  for (let i = 0; i < count; i++) {
    const o = i * stride;
    const x = data[o + X], y = data[o + Y], z = data[o + Z];
    pos[3 * i] = x; pos[3 * i + 1] = y; pos[3 * i + 2] = z;
    const t = i * 12;
    f[t] = x; f[t + 1] = y; f[t + 2] = z;
    const a = 1 / (1 + Math.exp(-data[o + OP]));
    texels[t + 3] = (clamp8(0.5 + SH0 * data[o + D0]) | (clamp8(0.5 + SH0 * data[o + D1]) << 8) | (clamp8(0.5 + SH0 * data[o + D2]) << 16) | (clamp8(a) << 24)) >>> 0;
    // Σ = M Mᵀ with M = R diag(s)
    let qw = data[o + R0], qx = data[o + R1], qy = data[o + R2], qz = data[o + R3];
    const n = Math.hypot(qw, qx, qy, qz) || 1;
    qw /= n; qx /= n; qy /= n; qz /= n;
    const r00 = 1 - 2 * (qy * qy + qz * qz), r01 = 2 * (qx * qy - qw * qz), r02 = 2 * (qx * qz + qw * qy);
    const r10 = 2 * (qx * qy + qw * qz), r11 = 1 - 2 * (qx * qx + qz * qz), r12 = 2 * (qy * qz - qw * qx);
    const r20 = 2 * (qx * qz - qw * qy), r21 = 2 * (qy * qz + qw * qx), r22 = 1 - 2 * (qx * qx + qy * qy);
    const sx = Math.exp(data[o + S0]), sy = Math.exp(data[o + S1]), sz = Math.exp(data[o + S2]);
    const m = [r00 * sx, r01 * sy, r02 * sz, r10 * sx, r11 * sy, r12 * sz, r20 * sx, r21 * sy, r22 * sz];
    const c = (i: number, j: number) => m[3 * i] * m[3 * j] + m[3 * i + 1] * m[3 * j + 1] + m[3 * i + 2] * m[3 * j + 2];
    f[t + 4] = c(0, 0); f[t + 5] = c(0, 1); f[t + 6] = c(0, 2); f[t + 7] = c(1, 1);
    f[t + 8] = c(1, 2); f[t + 9] = c(2, 2);
  }
  return { count, pos, texels };
}

const VS = `#version 300 es
precision highp float;
precision highp int;
uniform highp usampler2D u_tex;
uniform mat4 u_proj;
uniform mat4 u_view;
uniform vec2 u_focal;
uniform vec2 u_viewport;
in vec2 a_corner;
in uint a_index;
out vec4 v_color;
out vec2 v_pos;

ivec2 texel(int id) { return ivec2(id & 2047, id >> 11); }

void main() {
  int base = int(a_index) * 3;
  uvec4 a = texelFetch(u_tex, texel(base), 0);
  uvec4 b = texelFetch(u_tex, texel(base + 1), 0);
  uvec4 c = texelFetch(u_tex, texel(base + 2), 0);
  vec3 center = uintBitsToFloat(a.xyz);
  vec4 cam = u_view * vec4(center, 1.0);
  vec4 clip = u_proj * cam;
  float lim = 1.3 * clip.w;
  if (cam.z < 0.05 || clip.x < -lim || clip.x > lim || clip.y < -lim || clip.y > lim) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }
  float xx = uintBitsToFloat(b.x), xy = uintBitsToFloat(b.y), xz = uintBitsToFloat(b.z);
  float yy = uintBitsToFloat(b.w), yz = uintBitsToFloat(c.x), zz = uintBitsToFloat(c.y);
  mat3 sigma = mat3(xx, xy, xz, xy, yy, yz, xz, yz, zz);
  float iz = 1.0 / cam.z;
  mat3 J = mat3(u_focal.x * iz, 0.0, 0.0,
                0.0, u_focal.y * iz, 0.0,
                -u_focal.x * cam.x * iz * iz, -u_focal.y * cam.y * iz * iz, 0.0);
  mat3 T = J * mat3(u_view);
  mat3 cov = T * sigma * transpose(T);
  float ca = cov[0][0] + 0.3, cb = cov[0][1], cc = cov[1][1] + 0.3;
  float mid = 0.5 * (ca + cc);
  float rad = sqrt(max(0.25 * (ca - cc) * (ca - cc) + cb * cb, 0.0));
  float l1 = mid + rad, l2 = max(mid - rad, 0.05);
  vec2 dir = normalize(vec2(cb, l1 - ca) + vec2(1e-7, 0.0));
  vec2 major = min(sqrt(2.0 * l1), 1024.0) * dir;
  vec2 minor = min(sqrt(2.0 * l2), 1024.0) * vec2(dir.y, -dir.x);
  v_color = vec4(float(a.w & 255u), float((a.w >> 8) & 255u), float((a.w >> 16) & 255u), float((a.w >> 24) & 255u)) / 255.0;
  v_pos = a_corner;
  vec2 off = a_corner.x * major + a_corner.y * minor;
  vec2 ndc = clip.xy / clip.w + vec2(off.x * 2.0 / u_viewport.x, -off.y * 2.0 / u_viewport.y);
  gl_Position = vec4(ndc, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
in vec4 v_color;
in vec2 v_pos;
out vec4 o_color;
void main() {
  float a = -dot(v_pos, v_pos);
  if (a < -4.0) discard;
  float alpha = exp(a) * v_color.a;
  o_color = vec4(v_color.rgb * alpha, alpha);
}`;

type Vec3 = [number, number, number];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: Vec3): Vec3 => {
  const l = Math.hypot(...a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

export class SplatView {
  yaw = 0.6;
  pitch = 0.25;
  dist = 1.9;
  fov = 48;
  autoRotate = true;
  /** show the object upside down (some inputs come out of the model that way) */
  flip = false;
  /** called once the first frame of a new object has been drawn */
  onFrame?: () => void;

  private gl: WebGL2RenderingContext;
  private prog!: WebGLProgram;
  private tex: WebGLTexture | null = null;
  private idxBuf: WebGLBuffer;
  private vao: WebGLVertexArrayObject;
  private worker = new SortWorker();
  private count = 0;
  private sorting = false;
  private lastDir = "";
  private raf = 0;
  private dragging = false;
  private last = performance.now();
  private u: Record<string, WebGLUniformLocation | null> = {};
  private cleanup: (() => void)[] = [];
  private fresh = false;

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, premultipliedAlpha: false });
    if (!gl) throw new Error("WebGL2 is not available in this browser");
    this.gl = gl;
    this.prog = this.program(VS, FS);
    for (const n of ["u_tex", "u_proj", "u_view", "u_focal", "u_viewport"]) this.u[n] = gl.getUniformLocation(this.prog, n);
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    const quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), gl.STATIC_DRAW);
    const corner = gl.getAttribLocation(this.prog, "a_corner");
    gl.enableVertexAttribArray(corner);
    gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);
    this.idxBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.idxBuf);
    const index = gl.getAttribLocation(this.prog, "a_index");
    gl.enableVertexAttribArray(index);
    gl.vertexAttribIPointer(index, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(index, 1);
    gl.bindVertexArray(null);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    this.worker.onmessage = (e: MessageEvent) => {
      if (e.data.type !== "order") return;
      this.sorting = false;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.idxBuf);
      gl.bufferData(gl.ARRAY_BUFFER, e.data.order as Uint32Array, gl.DYNAMIC_DRAW);
      this.fresh = true;
    };
    this.bindInput();
    const resize = new ResizeObserver(() => this.resize());
    resize.observe(canvas);
    this.cleanup.push(() => resize.disconnect());
    this.raf = requestAnimationFrame(this.frame);
  }

  private program(vs: string, fs: string): WebGLProgram {
    const gl = this.gl;
    const sh = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? "shader error");
      return s;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? "link error");
    return p;
  }

  load(s: Splats) {
    const gl = this.gl;
    this.count = s.count;
    const w = 2048;
    const rows = Math.ceil((s.count * 3) / w);
    const data = new Uint32Array(w * rows * 4);
    data.set(s.texels);
    if (this.tex) gl.deleteTexture(this.tex);
    this.tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, w, rows, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, data);
    const pos = s.pos.slice();
    this.worker.postMessage({ type: "load", pos: pos.buffer }, [pos.buffer]);
    this.lastDir = "";
    this.sorting = false;
    this.fresh = false;
    this.firstFrame = true;
  }

  private firstFrame = false;

  clear() {
    this.count = 0;
  }

  /** Look at the object from the camera the model predicted for the input picture (y-up world). */
  setInputView(yaw: number, pitch: number, radius: number, fov: number) {
    // the model's frame is z-up; the PLY turns it to y-up: (x, y, z) -> (x, -z, y)
    const dx = Math.sin(yaw) * Math.cos(pitch);
    const dy = Math.cos(yaw) * Math.cos(pitch);
    const dz = Math.sin(pitch);
    const eye: Vec3 = [dx, dz, -dy];
    this.yaw = Math.atan2(eye[0], eye[2]);
    this.pitch = Math.asin(Math.max(-1, Math.min(1, eye[1])));
    this.dist = Math.min(Math.max(radius, 1.2), 4);
    this.fov = Math.min(Math.max(fov, 20), 70);
    this.autoRotate = false;
  }

  private bindInput() {
    const c = this.canvas;
    let lx = 0, ly = 0;
    const down = (e: PointerEvent) => {
      this.dragging = true;
      lx = e.clientX; ly = e.clientY;
      c.setPointerCapture(e.pointerId);
      this.autoRotate = false;
      this.onInteract?.();
    };
    const move = (e: PointerEvent) => {
      if (!this.dragging) return;
      this.yaw -= (e.clientX - lx) * 0.006;
      this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + (e.clientY - ly) * 0.006));
      lx = e.clientX; ly = e.clientY;
    };
    const up = () => (this.dragging = false);
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      this.dist = Math.max(0.6, Math.min(6, this.dist * Math.exp(e.deltaY * 0.0012)));
      this.onInteract?.();
    };
    c.addEventListener("pointerdown", down);
    c.addEventListener("pointermove", move);
    c.addEventListener("pointerup", up);
    c.addEventListener("pointercancel", up);
    c.addEventListener("wheel", wheel, { passive: false });
    this.cleanup.push(() => {
      c.removeEventListener("pointerdown", down);
      c.removeEventListener("pointermove", move);
      c.removeEventListener("pointerup", up);
      c.removeEventListener("pointercancel", up);
      c.removeEventListener("wheel", wheel);
    });
  }

  onInteract?: () => void;

  private resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  private frame = (now: number) => {
    this.raf = requestAnimationFrame(this.frame);
    const dt = Math.min((now - this.last) / 1000, 0.1);
    this.last = now;
    if (this.autoRotate && !this.dragging) this.yaw += dt * 0.35;
    this.draw();
  };

  private draw() {
    const gl = this.gl;
    this.resize();
    const { width: w, height: h } = this.canvas;
    gl.viewport(0, 0, w, h);
    gl.clearColor(0.031, 0.035, 0.051, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.count || !this.tex) return;

    const cp = Math.cos(this.pitch);
    const eye: Vec3 = [this.dist * cp * Math.sin(this.yaw), this.dist * Math.sin(this.pitch), this.dist * cp * Math.cos(this.yaw)];
    const f = norm(sub([0, 0, 0], eye));
    const r = norm(cross(f, [0, this.flip ? -1 : 1, 0]));
    const d = cross(f, r);
    // world -> camera (x right, y down, z forward), column-major
    const view = new Float32Array([
      r[0], d[0], f[0], 0,
      r[1], d[1], f[1], 0,
      r[2], d[2], f[2], 0,
      -(r[0] * eye[0] + r[1] * eye[1] + r[2] * eye[2]), -(d[0] * eye[0] + d[1] * eye[1] + d[2] * eye[2]), -(f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2]), 1,
    ]);
    const fy = h / 2 / Math.tan((this.fov * Math.PI) / 360);
    const fx = fy;
    const zn = 0.05, zf = 50;
    const proj = new Float32Array([
      (2 * fx) / w, 0, 0, 0,
      0, (-2 * fy) / h, 0, 0,
      0, 0, zf / (zf - zn), 1,
      0, 0, (-zf * zn) / (zf - zn), 0,
    ]);

    // re-sort when the view direction moved
    const key = f.map((v) => v.toFixed(3)).join();
    if (!this.sorting && key !== this.lastDir) {
      this.sorting = true;
      this.lastDir = key;
      this.worker.postMessage({ type: "sort", dir: f, id: 0 });
    }

    if (!this.fresh) return;
    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.uniform1i(this.u.u_tex, 0);
    gl.uniformMatrix4fv(this.u.u_proj, false, proj);
    gl.uniformMatrix4fv(this.u.u_view, false, view);
    gl.uniform2f(this.u.u_focal, fx, fy);
    gl.uniform2f(this.u.u_viewport, w, h);
    gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, this.count);
    gl.bindVertexArray(null);
    if (this.firstFrame) {
      this.firstFrame = false;
      this.onFrame?.();
    }
  }

  /** A PNG of the current frame. */
  snapshot(): Promise<Blob | null> {
    this.draw();
    return new Promise((res) => this.canvas.toBlob(res, "image/png"));
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    this.worker.terminate();
    this.cleanup.forEach((f) => f());
    this.gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
}
