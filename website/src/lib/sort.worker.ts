/* Depth sorting off the main thread: a 16-bit counting sort of the splats, farthest first. */
let pos: Float32Array | null = null;
let count = 0;

self.onmessage = (e: MessageEvent) => {
  const m = e.data;
  if (m.type === "load") {
    pos = new Float32Array(m.pos);
    count = pos.length / 3;
    return;
  }
  if (!pos) return;
  // m.dir: the camera's forward axis, m.eye: its position; depth = dir · (p - eye)
  const [dx, dy, dz] = m.dir as [number, number, number];
  const depth = new Float32Array(count);
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < count; i++) {
    const d = dx * pos[3 * i] + dy * pos[3 * i + 1] + dz * pos[3 * i + 2];
    depth[i] = d;
    if (d < min) min = d;
    if (d > max) max = d;
  }
  const buckets = 65536;
  const scale = (buckets - 1) / Math.max(max - min, 1e-9);
  const counts = new Uint32Array(buckets);
  const key = new Uint16Array(count);
  for (let i = 0; i < count; i++) {
    // farthest (largest depth) first
    const k = buckets - 1 - Math.floor((depth[i] - min) * scale);
    key[i] = k;
    counts[k]++;
  }
  let acc = 0;
  for (let k = 0; k < buckets; k++) {
    const c = counts[k];
    counts[k] = acc;
    acc += c;
  }
  const order = new Uint32Array(count);
  for (let i = 0; i < count; i++) order[counts[key[i]]++] = i;
  (self as unknown as Worker).postMessage({ type: "order", order, id: m.id }, [order.buffer]);
};
