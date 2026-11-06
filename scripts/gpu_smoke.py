"""Compile and exercise the exact rasterizer used by the reconstruction trainer."""
import torch
from fused_ssim import fused_ssim
from gsplat.rendering import rasterization

if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in the training environment.')
device = 'cuda'
means = torch.tensor([[0.0, 0.0, 2.0]], device=device, requires_grad=True)
quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
scales = torch.full((1, 3), 0.1, device=device)
opacities = torch.ones(1, device=device) * 0.9
colors = torch.tensor([[0.8, 0.2, 0.1]], device=device)
view = torch.eye(4, device=device)[None]
intrinsics = torch.tensor([[[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]]], device=device)
rgb, alpha, _ = rasterization(means, quats, scales, opacities, colors, view, intrinsics, 32, 32)
assert rgb.shape == (1, 32, 32, 3) and alpha.max() > 0
rgb.sum().backward()
assert means.grad is not None and torch.isfinite(means.grad).all()
sample = rgb.detach().permute(0, 3, 1, 2).contiguous()
score = fused_ssim(sample, sample)
assert torch.isfinite(score).all()
print('CUDA rasterizer forward/backward and fused SSIM passed:', torch.cuda.get_device_name(0))
