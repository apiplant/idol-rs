"""Time the reference pipeline (speridlabs/idol-sdk) on local weights, models already loaded, like `idol serve` would.

    python -I tools/ref_bench.py --sdk /path/to/idol-sdk --models models --image x.webp [--optimize] [--repeat 3]

--optimize is the SDK's default on GPU (torch.compile of the denoiser, batched guidance, cuDNN attention, TF32); the
first call compiles, so the later calls are the fair number. Same stand-ins as ref_pipeline.py for tensordict/encoders.
"""
import argparse, sys, time, types
import torch

p = argparse.ArgumentParser()
p.add_argument('--sdk', required=True); p.add_argument('--models', default='models'); p.add_argument('--image', required=True)
p.add_argument('--seed', type=int, default=42); p.add_argument('--steps', type=int, default=50)
p.add_argument('--num_gaussians', type=int, default=262144); p.add_argument('--repeat', type=int, default=3)
p.add_argument('--optimize', action='store_true')
a = p.parse_args()

m = types.ModuleType('tensordict')

class TensorDict(dict):
    def __init__(self, d=None, batch_size=None, device=None):
        super().__init__(d or {}); self.batch_size = batch_size
    @property
    def shape(self): return next(iter(self.values())).shape
    @property
    def device(self): return next(iter(self.values())).device
    def apply(self, f): return TensorDict({k: f(v) for k, v in self.items()}, batch_size=self.batch_size)
    def _bin(self, o, f): return TensorDict({k: f(v, o[k] if isinstance(o, dict) else o) for k, v in self.items()}, batch_size=self.batch_size)
    __add__ = lambda s, o: s._bin(o, lambda x, y: x + y)
    __sub__ = lambda s, o: s._bin(o, lambda x, y: x - y)
    __mul__ = lambda s, o: s._bin(o, lambda x, y: x * y)
    __rmul__ = lambda s, o: s._bin(o, lambda x, y: y * x)
    def __getitem__(self, k):
        if isinstance(k, str): return dict.__getitem__(self, k)
        return TensorDict({n: v[k] for n, v in self.items()}, batch_size=self.batch_size)

_cat = torch.cat
def _tdcat(ts, dim=0, **kw):
    if len(ts) and isinstance(ts[0], dict):
        return TensorDict({k: _cat([t[k] for t in ts], dim) for k in ts[0]}, batch_size=ts[0].batch_size)
    return _cat(ts, dim, **kw)
torch.cat = _tdcat
m.TensorDict = TensorDict
sys.modules['tensordict'] = m
sys.path.insert(0, a.sdk)
import idol.pipeline as P
from idol.conditioning import ImageEncoder as _Enc

P.require_encoder_access = lambda repos: None

class LocalEncoder(_Enc):
    def __init__(self, dinov3_repo, flux2_vae_repo, dinov3_dtype='bf16', device='cuda'):
        from diffusers.models import AutoencoderKLFlux2
        from transformers import AutoModel
        self.device = torch.device(device)
        self.dinov3 = AutoModel.from_pretrained(f'{a.models}/dinov3', torch_dtype=torch.bfloat16).eval().requires_grad_(False).to(self.device)
        self.flux2_vae = AutoencoderKLFlux2.from_pretrained(f'{a.models}/flux2-vae', torch_dtype=torch.bfloat16).eval().requires_grad_(False).to(self.device)

P.ImageEncoder = LocalEncoder
pipe = P.IdolPipeline(f'{a.models}/idol', device='cuda', optimize=a.optimize)
print('optimize =', pipe.optimize, '| gpu', torch.cuda.get_device_name(0), '| torch', torch.__version__, flush=True)
for i in range(a.repeat):
    torch.cuda.synchronize(); t = time.time()
    out = pipe(a.image, mode='synthetic', seed=a.seed, steps=a.steps, num_gaussians=a.num_gaussians, verbose=False)
    torch.cuda.synchronize()
    print(f'call {i + 1}: {time.time() - t:.1f} s', flush=True)
