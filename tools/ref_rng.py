"""Dump PyTorch's CUDA random streams for a seed, in the order idol's pipeline draws them, to check idol-rs's generator.
    python -I tools/ref_rng.py 42 out/ref/rng.safetensors
"""
import sys, torch
from safetensors.torch import save_file
seed, out = int(sys.argv[1]), sys.argv[2]
torch.manual_seed(seed)
d = 'cuda'
t = {}
t['vae_noise'] = torch.randn(1, 32, 128, 128, device=d, dtype=torch.bfloat16).float()   # DiagonalGaussianDistribution.sample()
t['latent'] = torch.randn(1, 8192, 16, device=d)
t['camera'] = torch.randn(1, 1, 5, device=d)
t['u1'] = torch.rand(1, 1, device=d)                                                   # octree: systematic_sample, one count
for i, r in enumerate([3, 40, 700, 2049]):                                             # several group sizes
    t[f'u_{r}'] = torch.rand(r, 1, device=d)
t['leaf'] = torch.rand_like(torch.zeros(1, 8192, 3, dtype=torch.long, device=d), dtype=torch.float32)
save_file({k: v.cpu().contiguous() for k, v in t.items()}, out)
print({k: tuple(v.shape) for k, v in t.items()})
