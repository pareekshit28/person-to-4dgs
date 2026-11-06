"""Build and validate the two native training packages for the visible GPUs.

Run with the training environment's Python. No CUDA package is imported until
the CLI runs, and validation always happens in a fresh process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
BUILD_POLICY_VERSION = 1
NATIVE_PACKAGES = ("fused-ssim", "gsplat")
STAMP_NAME = ".person4d-cuda-build.json"


class BuildError(RuntimeError):
    pass


def native_requirements(path: Path) -> list[str]:
    """Select exactly the native packages, each pinned to an immutable Git SHA."""
    selected: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.partition("#")[0].strip()
        if not line:
            continue
        name, separator, source = line.partition("@")
        name = name.strip().lower().replace("_", "-")
        if name not in NATIVE_PACKAGES:
            continue
        if name in selected:
            raise BuildError(f"Duplicate native requirement: {name}")
        if not separator or not re.fullmatch(
            r"git\+https://github\.com/[^/\s]+/[^/@\s]+@[0-9a-f]{40}", source.strip()
        ):
            raise BuildError(f"Native requirement {name} must use an immutable 40-character Git commit.")
        selected[name] = f"{name} @ {source.strip()}"
    missing = set(NATIVE_PACKAGES) - selected.keys()
    if missing:
        raise BuildError("Missing native requirements: " + ", ".join(sorted(missing)))
    return [selected[name] for name in NATIVE_PACKAGES]


def fingerprint(*, torch_version: str, cuda_version: str, architectures: list[str],
                python_tag: str, nvcc_version: str, requirements: bytes) -> dict:
    return {
        "policy_version": BUILD_POLICY_VERSION,
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "architectures": sorted(set(architectures), key=lambda value: tuple(map(int, value.split(".")))),
        "python_tag": python_tag,
        "nvcc_version": nvcc_version,
        "requirements_sha256": hashlib.sha256(requirements).hexdigest(),
    }


def gpu_details() -> dict:
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise BuildError("CUDA is unavailable in the training environment; a GPU is required for this build.")
    if str(torch.version.cuda) != "12.8":
        raise BuildError(f"The training environment requires PyTorch CUDA 12.8, found {torch.version.cuda}.")
    torch.cuda.set_device(0)
    try:
        scan = torch.cumsum(torch.arange(8, device="cuda:0", dtype=torch.int32), 0, dtype=torch.int32)
        torch.cuda.synchronize(0)
        if scan.tolist() != [0, 1, 3, 6, 10, 15, 21, 28]:
            raise BuildError("Basic PyTorch CUDA cumulative-sum validation returned an incorrect result.")
    except RuntimeError as exc:
        raise BuildError(
            "Basic PyTorch CUDA cumulative-sum failed before native compilation. "
            f"Check the training CUDA environment before rebuilding gsplat: {exc}"
        ) from None
    capabilities = sorted({tuple(torch.cuda.get_device_capability(index)) for index in range(torch.cuda.device_count())})
    architectures = [f"{major}.{minor}" for major, minor in capabilities]
    print(f"Native build GPU: {torch.cuda.get_device_name(0)}; CUDA architectures: {';'.join(architectures)}", flush=True)
    return {"torch_version": str(torch.__version__), "cuda_version": str(torch.version.cuda), "architectures": architectures}


def nvcc_version() -> str:
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    candidate = cuda_home / "bin/nvcc"
    executable = str(candidate) if candidate.is_file() else shutil.which("nvcc")
    if not executable:
        raise BuildError("nvcc is missing. Use a CUDA 12.8 development image and run setup first.")
    result = subprocess.run([executable, "--version"], check=True, capture_output=True, text=True, timeout=30)
    version = result.stdout.strip()
    if not re.search(r"release\s+12\.8\b", version):
        raise BuildError("The native build requires the CUDA 12.8 compiler; check CUDA_HOME and PATH.")
    return version


def uv_executable() -> str:
    executable = shutil.which("uv")
    candidate = Path.home() / ".local/bin/uv"
    if executable:
        return executable
    if candidate.is_file():
        return str(candidate)
    raise BuildError("uv is missing. Run bash scripts/setup_runpod.sh first.")


def build_environment(architectures: list[str], parent: dict | None = None) -> dict:
    env = dict(os.environ if parent is None else parent)
    env.update(UV_NO_CACHE="1", BUILD_NO_CUDA="0", TORCH_CUDA_ARCH_LIST=";".join(architectures), PYTHONUNBUFFERED="1")
    env.setdefault("MAX_JOBS", "4")
    # Keep the provider's visibility mapping. Logical device zero need not be
    # physical device zero; NVIDIA_VISIBLE_DEVICES is container configuration.
    return env


def install_command(uv: str, python: str, requirements: list[str]) -> list[str]:
    return [uv, "pip", "install", "--python", python, "--no-deps", "--no-build-isolation",
            "--reinstall-package", "gsplat", "--reinstall-package", "fused-ssim", *requirements]


def stamp_matches(path: Path, expected: dict) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("fingerprint") == expected
    except (OSError, ValueError, AttributeError):
        return False


def write_stamp(path: Path, value: dict) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".person4d-cuda-build-", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump({"fingerprint": value, "validated_at": datetime.now(timezone.utc).isoformat()}, handle, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def install_native(*, force: bool = False, root: Path = ROOT, prefix: Path | None = None,
                   python_executable: str | None = None) -> None:
    # Do not resolve sys.executable: a venv executable may be a symlink to the
    # managed base Python. Resolving it would install outside this environment.
    python = sys.executable if python_executable is None else python_executable
    prefix = Path(sys.prefix) if prefix is None else prefix
    requirements_path = root / "requirements-training-cuda.txt"
    specifications = native_requirements(requirements_path)
    details = gpu_details()
    value = fingerprint(**details, python_tag=sys.implementation.cache_tag or sys.version,
                        nvcc_version=nvcc_version(), requirements=requirements_path.read_bytes())
    env = build_environment(value["architectures"])
    stamp = prefix / STAMP_NAME
    rebuild = force or not stamp_matches(stamp, value)
    if rebuild:
        # A forced rebuild can replace a previously matching binary. Invalidate
        # its stamp first so an interrupted or failed build cannot look valid.
        stamp.unlink(missing_ok=True)
        print("Building gsplat and fused SSIM from pinned source without cached wheels…", flush=True)
        try:
            subprocess.run(install_command(uv_executable(), python, specifications), cwd=root, env=env, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise BuildError(f"Native CUDA compilation failed; no build was marked valid: {exc}") from None
    else:
        print("Native build fingerprint matches this environment; validating the installed kernels…", flush=True)
    try:
        # Importing gsplat here would retain the old shared library after pip
        # replaces it. Use a new interpreter for both import and GPU validation.
        subprocess.run([python, str(root / "scripts/gpu_smoke.py")], cwd=root, env=env, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        stamp.unlink(missing_ok=True)
        raise BuildError(
            "Native CUDA validation failed; no build was marked valid. "
            f"See the GPU check output above. You can retry with bash scripts/rebuild_training_cuda.sh. ({exc})"
        ) from None
    write_stamp(stamp, value)
    print("Native CUDA build validated and fingerprint saved.", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild the native packages even when their fingerprint matches")
    args = parser.parse_args(argv)
    try:
        install_native(force=args.force)
    except (BuildError, OSError, subprocess.SubprocessError, ImportError) as exc:
        print(f"CUDA build failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
