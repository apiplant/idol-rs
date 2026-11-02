"""Run the reference pipeline (speridlabs/idol-sdk) end to end on local weights, for an orientation / quality check.

    python -I tools/ref_pipeline.py --sdk /path/to/idol-sdk --models models --image x.webp --out out/ref_x.ply [--seed 42 --steps 50]

The encoders are loaded from the local folders instead of the Hub, and `tensordict` (not installed here) is replaced
by a stand-in with the elementwise arithmetic the sampler needs.
"""
import argparse, sys, types
import torch

p = argparse.ArgumentParser()
p.add_argument('--sdk', required=True); p.add_argument('--models', default='models'); p.add_argument('--image', required=True)
p.add_argument('--out', required=True); p.add_argument('--seed', type=int, default=42); p.add_argument('--steps', type=int, default=50)
p.add_argument('--num_gaussians', type=int, default=65536); p.add_argument('--mode', default='synthetic')
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
    def _bin(self, o, f):
        return TensorDict({k: f(v, o[k] if isinstance(o, dict) else o) for k, v in self.items()}, batch_size=self.batch_size)
    __add__ = lambda s, o: s._bin(o, lambda x, y: x + y)
    __sub__ = lambda s, o: s._bin(o, lambda x, y: x - y)
    __mul__ = lambda s, o: s._bin(o, lambda x, y: x * y)
    __rmul__ = lambda s, o: s._bin(o, lambda x, y: y * x)

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
pipe = P.IdolPipeline(f'{a.models}/idol', device='cuda', optimize=False)
out = pipe(a.image, mode=a.mode, seed=a.seed, steps=a.steps, num_gaussians=a.num_gaussians, verbose=True)
out.gaussian.save_ply(a.out)
print('camera', out.camera)
