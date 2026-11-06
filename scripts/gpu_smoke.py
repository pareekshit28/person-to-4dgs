"""Exercise CUDA and the trainer's rasterizer with synchronized diagnostics."""
import os


def main():
    import torch

    print('Training PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, flush=True)
    for name in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'TORCH_EXTENSIONS_DIR'):
        print(f'{name}={os.environ.get(name, "<unset>")}', flush=True)
    print('Visible CUDA devices:', torch.cuda.device_count(), flush=True)
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable in the training environment.')

    # Ordinals are logical indices within the provider's visibility mask.
    # Preserve that mask and initialize the selected device before extensions.
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    print('Selected CUDA device:', torch.cuda.current_device(), torch.cuda.get_device_name(0), flush=True)
    print('Compute capability:', torch.cuda.get_device_capability(0), flush=True)
    basic = torch.ones((32, 32), device=device)
    assert (basic @ basic).sum().item() == 32768
    torch.cuda.synchronize(device)
    print('Basic PyTorch CUDA execution passed.', flush=True)

    from gsplat.rendering import rasterization
    from gsplat.cuda._backend import _C

    print('gsplat CUDA extension:', getattr(_C, '__file__', '<not loaded>'), flush=True)
    print('Checking packed gsplat forward/backward...', flush=True)
    means = torch.tensor([[0.0, 0.0, 2.0]], device=device, requires_grad=True)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.full((1, 3), 0.1, device=device)
    opacities = torch.ones(1, device=device) * 0.9
    colors = torch.tensor([[0.8, 0.2, 0.1]], device=device)
    view = torch.eye(4, device=device)[None]
    intrinsics = torch.tensor([[[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]]], device=device)
    rgb, alpha, _ = rasterization(means, quats, scales, opacities, colors, view, intrinsics, 32, 32, packed=True)
    torch.cuda.synchronize(device)
    assert rgb.shape == (1, 32, 32, 3) and alpha.max() > 0
    rgb.sum().backward()
    torch.cuda.synchronize(device)
    assert means.grad is not None and torch.isfinite(means.grad).all()
    print('Packed gsplat forward/backward passed.', flush=True)

    from fused_ssim import fused_ssim

    print('Checking fused SSIM...', flush=True)
    sample = rgb.detach().permute(0, 3, 1, 2).contiguous()
    score = fused_ssim(sample, sample)
    torch.cuda.synchronize(device)
    assert torch.isfinite(score).all()
    print('CUDA rasterizer forward/backward and fused SSIM passed:', torch.cuda.get_device_name(0), flush=True)


if __name__ == '__main__':
    main()
