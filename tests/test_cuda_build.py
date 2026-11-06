import json
from pathlib import Path
import subprocess

import pytest

from scripts import install_training_cuda as native


PIN = "a" * 40
REQUIREMENTS = (
    f"pycolmap @ git+https://github.com/owner/pycolmap.git@{PIN}\n"
    f"gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@{PIN}\n"
    f"fused-ssim @ git+https://github.com/rahul-goel/fused-ssim.git@{PIN}\n"
)


def make_fingerprint(**changes):
    values = dict(torch_version="2.8.0+cu128", cuda_version="12.8", architectures=["8.0"],
                  python_tag="cpython-312", nvcc_version="release 12.8, V12.8.93", requirements=REQUIREMENTS.encode())
    values.update(changes)
    return native.fingerprint(**values)


def test_native_requirements_are_exact_and_immutable(tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text(REQUIREMENTS)
    selected = native.native_requirements(path)
    assert len(selected) == 2
    assert all("pycolmap" not in line for line in selected)
    path.write_text(REQUIREMENTS.replace(f"gsplat.git@{PIN}", "gsplat.git@main"))
    with pytest.raises(native.BuildError, match="immutable"):
        native.native_requirements(path)
    path.write_text(REQUIREMENTS + f"fused_ssim @ git+https://github.com/owner/other.git@{PIN}\n")
    with pytest.raises(native.BuildError, match="Duplicate"):
        native.native_requirements(path)


def test_native_build_fingerprint_tracks_compatibility():
    original = make_fingerprint()
    for change in ({"architectures": ["9.0"]}, {"torch_version": "2.9.0+cu128"},
                   {"python_tag": "cpython-311"}, {"nvcc_version": "release 12.8, V12.8.99"},
                   {"requirements": REQUIREMENTS.replace(PIN, "b" * 40).encode()}):
        assert make_fingerprint(**change) != original
    assert make_fingerprint(architectures=["8.0", "9.0", "8.0"]) == make_fingerprint(architectures=["9.0", "8.0"])


def test_build_command_preserves_venv_and_device_visibility():
    python = "/workspace/app/.venvs/training/bin/python"
    command = native.install_command("uv", python, ["gsplat @ pinned-source", "fused-ssim @ pinned-source"])
    assert command[command.index("--python") + 1] == python
    assert "--no-deps" in command and "--no-build-isolation" in command
    assert command.count("--reinstall-package") == 2
    env = native.build_environment(["8.0"], {"CUDA_VISIBLE_DEVICES": "GPU-example", "NVIDIA_VISIBLE_DEVICES": "void"})
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-example"
    assert env["NVIDIA_VISIBLE_DEVICES"] == "void"
    assert env["UV_NO_CACHE"] == "1" and env["BUILD_NO_CUDA"] == "0"
    assert env["TORCH_CUDA_ARCH_LIST"] == "8.0" and env["MAX_JOBS"] == "4"


def prepare_install(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "requirements-training-cuda.txt").write_text(REQUIREMENTS)
    prefix = tmp_path / "venv"
    prefix.mkdir()
    monkeypatch.setattr(native, "gpu_details", lambda: {"torch_version": "2.8.0+cu128", "cuda_version": "12.8", "architectures": ["8.0"]})
    monkeypatch.setattr(native, "nvcc_version", lambda: "release 12.8, V12.8.93")
    monkeypatch.setattr(native, "uv_executable", lambda: "uv")
    return root, prefix


def test_reuse_still_validates_and_force_rebuilds_in_fresh_process(tmp_path, monkeypatch):
    root, prefix = prepare_install(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(native.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    python = "/venv/bin/python"
    native.install_native(root=root, prefix=prefix, python_executable=python)
    assert len(calls) == 2 and calls[0][0][:3] == ["uv", "pip", "install"]
    assert calls[1][0] == [python, str(root / "scripts/gpu_smoke.py")]
    assert native.stamp_matches(prefix / native.STAMP_NAME, json.loads((prefix / native.STAMP_NAME).read_text())["fingerprint"])
    calls.clear()
    native.install_native(root=root, prefix=prefix, python_executable=python)
    assert len(calls) == 1 and calls[0][0][1].endswith("gpu_smoke.py")
    calls.clear()
    native.install_native(force=True, root=root, prefix=prefix, python_executable=python)
    assert len(calls) == 2


def test_failed_validation_invalidates_old_success_stamp(tmp_path, monkeypatch):
    root, prefix = prepare_install(tmp_path, monkeypatch)
    monkeypatch.setattr(native.subprocess, "run", lambda *args, **kwargs: None)
    native.install_native(root=root, prefix=prefix, python_executable="/venv/python")
    assert (prefix / native.STAMP_NAME).is_file()

    def fail_smoke(command, **kwargs):
        if command[-1].endswith("gpu_smoke.py"):
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(native.subprocess, "run", fail_smoke)
    with pytest.raises(native.BuildError, match="validation failed"):
        native.install_native(force=True, root=root, prefix=prefix, python_executable="/venv/python")
    assert not (prefix / native.STAMP_NAME).exists()
