"""Dump reference tensors from the PyTorch implementation (speridlabs/idol-sdk) for the Rust parity example.

    python -I tools/ref_dump.py --sdk /path/to/idol-sdk --models models --out out/ref <denoiser|octree|gs>

`tensordict` is not needed for these components, so a tiny stand-in is installed when it is missing.
"""
import argparse, json, os, sys, types
import torch
from safetensors.torch import load_file, save_file

p = argparse.ArgumentParser()
p.add_argument('--sdk', required=True)
p.add_argument('--models', default='models')
p.add_argument('--out', default='out/ref')
p.add_argument('part')
a = p.parse_args()
try:
    import tensordict  # noqa
except ImportError:
    m = types.ModuleType('tensordict')

    class TensorDict(dict):
        def __init__(self, d=None, batch_size=None, device=None):
            super().__init__(d or {})
            self.batch_size = batch_size

        def apply(self, f):
            return TensorDict({k: f(v) for k, v in self.items()}, batch_size=self.batch_size)

    m.TensorDict = TensorDict
    sys.modules['tensordict'] = m
sys.path.insert(0, a.sdk)
os.makedirs(a.out, exist_ok=True)
dev = 'cuda'
cfg = json.load(open(f'{a.models}/idol/config.json'))
g = torch.Generator().manual_seed(1234)
rnd = lambda *s: torch.randn(*s, generator=g)


def save(name, **t):
    save_file({k: v.detach().float().cpu().contiguous() for k, v in t.items()}, f'{a.out}/{name}.safetensors')
    print('saved', name, {k: tuple(v.shape) for k, v in t.items()})


@torch.no_grad()
def main():
    if a.part == 'denoiser':
        from idol.models import LatentSeqMMFlowModel
        from tensordict import TensorDict
        m = LatentSeqMMFlowModel(**cfg['denoiser']['args'])
        m.load_state_dict(load_file(f'{a.models}/idol/denoiser.safetensors'), strict=True)
        m = m.eval().to(dev)
        z, cam, f1, f2 = rnd(1, 8192, 16), rnd(1, 1, 5), rnd(1, 4101, 1280), rnd(1, 4101, 128)
        t = torch.tensor([700.0])
        x = TensorDict({'latent': z.to(dev), 'camera': cam.to(dev)}, batch_size=1)
        c = TensorDict({'feature1': f1.to(dev), 'feature2': f2.to(dev)}, batch_size=1)
        # intermediates by hooks on the stages
        mids = {}
        def hook(name):
            def f(mod, inp, out): mids[name] = out.detach().float().cpu()
            return f
        m.noise_refiner[0].register_forward_hook(hook('noise0'))
        m.noise_refiner[1].register_forward_hook(hook('noise1'))
        m.context_refiner[1].register_forward_hook(hook('ctx1'))
        m.blocks[0].register_forward_hook(hook('block0'))
        m.blocks[23].register_forward_hook(hook('block23'))
        m.noise_repo_layers[0].register_forward_hook(hook('repo_noise0'))
        o = m(x, t.to(dev), c)
        save('denoiser', z=z, cam=cam, f1=f1, f2=f2, t=t, out_z=o['latent'], out_cam=o['camera'], **{k: v.reshape(-1) if v.is_complex() is False else torch.view_as_real(v).reshape(-1) for k, v in mids.items()})
    elif a.part == 'octree':
        from idol.models import OctreeProbabilityFixedlenDecoder
        m = OctreeProbabilityFixedlenDecoder(**cfg['decoder']['args'])
        m.load_state_dict(load_file(f'{a.models}/idol/decoder.safetensors'), strict=True)
        m = m.eval().to(dev)
        x, lat, lv = torch.rand(1, 200, 3, generator=g), rnd(1, 8192, 16), torch.tensor([16.0])
        xd, lvd, latd = x.to(dev), lv.to(dev), lat.to(dev)
        pos = m.pos_embedder(xd.view(-1, 3)).view(1, 200, -1)
        h0 = m.in_proj(xd) + pos
        h = m.input_layer(h0)
        lemb = m.l_embedder(lvd.type(h.dtype))
        mod = m.adaLN_modulation(lemb).type(m.dtype)
        out = h.type(m.dtype); ctx = latd.type(m.dtype)
        tr = {}
        for i, blk in enumerate(m.blocks):
            out = blk(out, mod, ctx); tr[f'blk{i}'] = out
        b0 = m.blocks[0]
        x0 = h.type(m.dtype)
        sm, sc, gm, sm2, sc2, gm2 = mod.chunk(6, dim=1)
        n1 = b0.norm1(x0)
        a_in = n1 * (1 + sc.unsqueeze(1)) + sm.unsqueeze(1)
        ca = b0.cross_attn(a_in, ctx)
        x1 = x0 + ca * gm.unsqueeze(1)
        n2 = b0.norm2(x1)
        m_in = n2 * (1 + sc2.unsqueeze(1)) + sm2.unsqueeze(1)
        ff = b0.mlp(m_in)
        # inside the cross attention
        ca_mod = b0.cross_attn
        qn = ca_mod.q_rms_norm(ca_mod.to_q(a_in).reshape(1, 200, 16, -1))
        kv = ca_mod.to_kv(ctx).reshape(1, 8192, 2, 16, -1)
        kn = ca_mod.k_rms_norm(kv[:, :, 0])
        save('octree', x=x, lat=lat, level=lv, out=m(xd, lvd, latd), pos=pos, h=h, lemb=lemb, mod=mod, b0_n1=n1, b0_ain=a_in, b0_ca=ca, b0_x1=x1, b0_min=m_in, b0_ff=ff, b0_q=qn, b0_k=kn, **tr)
    elif a.part == 'gs':
        from idol.models import GaussianFixedlenDecoder
        m = GaussianFixedlenDecoder(**cfg['decoder_gs']['args'])
        m.load_state_dict(load_file(f'{a.models}/idol/decoder_gs.safetensors'), strict=True)
        m = m.eval().to(dev)
        pts, lat = torch.rand(1, 300, 3, generator=g), rnd(1, 8192, 16)
        f = m(pts.to(dev), lat.to(dev))
        gs = m.to_representation(pts.to(dev), f)[0]
        save('gs', pts=pts, lat=lat, out=f, xyz=gs.xyz, dc=gs.features_dc.flatten(1), scaling=gs.scaling, rotation=gs.rotation, opacity=gs.opacity)


main()
