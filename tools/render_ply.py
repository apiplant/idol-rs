"""Quick sanity render of an idol .ply: orthographic point splats from a few yaw angles (numpy + PIL only).
    python tools/render_ply.py out/x.ply out/x_views.png"""
import sys, numpy as np
from PIL import Image
path, out = sys.argv[1], sys.argv[2]
raw = open(path, 'rb').read()
end = raw.index(b'end_header\n') + 11
hdr = raw[:end].decode()
n = int(hdr.split('element vertex ')[1].split()[0]); props = [l.split()[-1] for l in hdr.splitlines() if l.startswith('property')]
d = np.frombuffer(raw[end:end + n * len(props) * 4], dtype='<f4').reshape(n, len(props))
c = lambda k: d[:, props.index(k)]
xyz = np.stack([c('x'), c('y'), c('z')], 1)
rgb = np.clip(0.5 + 0.28209479 * np.stack([c('f_dc_0'), c('f_dc_1'), c('f_dc_2')], 1), 0, 1)
a = 1 / (1 + np.exp(-c('opacity')))
keep = a > 0.3
xyz, rgb = xyz[keep], rgb[keep]
print(n, 'gaussians;', keep.sum(), 'opaque; bbox', xyz.min(0).round(2), xyz.max(0).round(2), 'mean rgb', rgb.mean(0).round(2))
S = 512
tiles = []
for yaw in (0, 90, 180, 270):
    t = np.radians(yaw); R = np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
    p = xyz @ R.T
    u = ((p[:, 0] + 0.6) / 1.2 * S).astype(int); v = ((0.6 - p[:, 1]) / 1.2 * S).astype(int)
    ok = (u >= 0) & (u < S) & (v >= 0) & (v < S)
    order = np.argsort(-p[:, 2])  # far first, near last overwrites
    img = np.zeros((S, S, 3), np.uint8)
    o = order[ok[order]]
    img[v[o], u[o]] = (rgb[o] * 255).astype(np.uint8)
    tiles.append(img)
Image.fromarray(np.concatenate(tiles, 1)).save(out)
