"""Compare idol-rs's two image encoders with the reference libraries (transformers DINOv3, diffusers FLUX.2 VAE).
    python -I tools/ref_encoders.py out/obj-backpack_input.png out/debug
(run `IDOL_DEBUG_DIR=out/debug idol generate ...` first: it dumps dinov3.f32 and vae.f32)"""
import sys, numpy as np, torch
from PIL import Image
img_path, dbg = sys.argv[1], sys.argv[2]
dev = 'cuda'
img = torch.from_numpy(np.array(Image.open(img_path).convert('RGB'))).permute(2, 0, 1)[None].float() / 255
rust = lambda n, d: np.fromfile(f'{dbg}/{n}.f32', dtype='<f4').reshape(1, 4101, d)

def cmp(name, ref, mine):
    ref, mine = ref.reshape(-1).astype(np.float64), mine.reshape(-1).astype(np.float64)
    cos = ref @ mine / (np.linalg.norm(ref) * np.linalg.norm(mine))
    print(f'{name}: cosine {cos:.6f}, max abs diff {np.abs(ref - mine).max():.4f}, ref std {ref.std():.4f}')

with torch.no_grad():
    from transformers import AutoModel
    m = AutoModel.from_pretrained('models/dinov3', torch_dtype=torch.bfloat16).eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tok = m(pixel_values=((img - mean) / std).to(dev).bfloat16()).last_hidden_state.float()
    tok = torch.nn.functional.layer_norm(tok, tok.shape[-1:]).cpu().numpy()
    cmp('dinov3 all tokens', tok, rust('dinov3', 1280))
    cmp('dinov3 cls+registers', tok[:, :5], rust('dinov3', 1280)[:, :5])
    cmp('dinov3 patches', tok[:, 5:], rust('dinov3', 1280)[:, 5:])
    del m; torch.cuda.empty_cache()
    from diffusers.models import AutoencoderKLFlux2
    vae = AutoencoderKLFlux2.from_pretrained('models/flux2-vae', torch_dtype=torch.bfloat16).eval().to(dev)
    dist = vae.encode(img.to(dev).bfloat16() * 2 - 1).latent_dist
    print('posterior std (mean over elements):', float(dist.std.float().mean()))
    lat = dist.mean
    b, c, h, w = lat.shape
    lat = lat.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    mu = vae.bn.running_mean.view(1, -1, 1, 1).to(lat.dtype); sd = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(lat.dtype)
    t = ((lat - mu) / sd).float().flatten(2).transpose(1, 2).cpu().numpy()
    cmp('vae patches', t, rust('vae', 128)[:, 5:])
