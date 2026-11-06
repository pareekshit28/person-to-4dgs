"""Restore container-local references to existing persistent model files."""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys


def restore_model_links(models_dir: Path, inference_repo: Path, gvhmr_root: Path | None = None) -> bool:
    """Link available assets without downloading weights or loading CUDA."""
    models_dir = Path(models_dir).expanduser().resolve()
    if not models_dir.is_dir():
        return False
    inference_repo = Path(inference_repo).expanduser().resolve()
    if not (inference_repo / "fdanyone/download.py").is_file():
        raise RuntimeError(f"The inference source is missing at {inference_repo}.")
    gvhmr_root = Path(gvhmr_root) if gvhmr_root is not None else inference_repo / "third_party/GVHMR"
    sys.path.insert(0, str(inference_repo))
    try:
        download = importlib.import_module("fdanyone.download")
        download.create_classic_gvhmr_links(
            models_dir, gvhmr_root, require_models=False, require_smplx=False,
        )
    finally:
        sys.path.remove(str(inference_repo))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--inference-repo", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        restore_model_links(args.models_dir, args.inference_repo)
    except (OSError, ImportError, RuntimeError) as exc:
        print(f"Model-link initialization failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
