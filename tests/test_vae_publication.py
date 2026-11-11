from contextlib import nullcontext
from pathlib import Path
from threading import Event

import numpy as np
import pytest


@pytest.fixture
def vae(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("av")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "vendor/4danyone"))
    from fdanyone.model import vae

    return torch, vae


def test_rgb_frames_preserve_pixel_values(vae):
    torch, module = vae
    video = torch.linspace(-1.5, 1.5, 3 * 2 * 5 * 7, dtype=torch.bfloat16).reshape(3, 2, 5, 7)
    scaled = video.float().add_(1.0).mul_(127.5).clamp_(0.0, 255.0)
    frames = module._rgb_frames(video)
    assert len(frames) == 2
    for index, frame in enumerate(frames):
        expected = scaled[:, index].permute(1, 2, 0).numpy().astype(np.uint8)
        np.testing.assert_array_equal(frame, expected)
        assert frame.flags.c_contiguous


def test_single_gpu_decodes_while_previous_view_is_encoding(vae, monkeypatch):
    torch, module = vae
    executor = module.VaeExecutor(None, ("cuda:0",))
    encoding_started = Event()
    decoding_next = Event()

    def run_workers(*, num_views, operation, stopped):
        operation(None, "cuda:0", tuple(range(num_views)), stopped)

    def decode_view(_model, latent, _device):
        if latent.item() == 1:
            assert encoding_started.wait(3)
            decoding_next.set()
        return torch.zeros(1, 3, 1, 1, 1)

    def encode_view(camera_id, _frames):
        if camera_id == 0:
            encoding_started.set()
            assert decoding_next.wait(3)
        return camera_id

    monkeypatch.setattr(module, "_bf16_autocast", nullcontext)
    monkeypatch.setattr(executor, "_run_workers", run_workers)
    monkeypatch.setattr(executor, "_decode_view", decode_view)
    latents = torch.arange(2).reshape(2, 1, 1, 1, 1)
    assert executor._decode_and_publish(latents, encode_view) == (0, 1)
