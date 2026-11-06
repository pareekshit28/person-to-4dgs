#!/usr/bin/env bash
# Image entrypoint: validate the baked kernels, then launch. Never build/install.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PERSON4D_WEB_PYTHON="${PERSON4D_WEB_PYTHON:-/opt/person4d/envs/web/bin/python}"
export PERSON4D_INFERENCE_PYTHON="${PERSON4D_INFERENCE_PYTHON:-/opt/person4d/envs/inference/bin/python}"
export PERSON4D_TRAINING_PYTHON="${PERSON4D_TRAINING_PYTHON:-/opt/person4d/envs/training/bin/python}"
export PERSON4D_DATA_DIR="${PERSON4D_DATA_DIR:-/workspace/person4d}"
export PERSON4D_PREBUILT_IMAGE=1
export PYTHONUNBUFFERED=1

# Importing gsplat's lazy loader with missing binaries could compile at runtime.
# Check wheel contents without importing gsplat before the short GPU check.
"$PERSON4D_TRAINING_PYTHON" - <<'PY'
from importlib.metadata import distribution
from pathlib import Path
for package, prefix in [('gsplat', 'gsplat/csrc.'), ('fused-ssim', 'fused_ssim_cuda.')]:
    installed = distribution(package)
    binaries = [p for p in installed.files or [] if str(p).replace('\\', '/').startswith(prefix) and str(p).endswith('.so')]
    if not binaries or not all(Path(installed.locate_file(p)).is_file() for p in binaries):
        raise SystemExit(f'Prebuilt CUDA binary missing for {package}. Use a complete image; startup will not compile packages.')
print('Prebuilt CUDA modules found. Checking GPU execution without compilation...', flush=True)
PY
"$PERSON4D_TRAINING_PYTHON" "$ROOT/scripts/gpu_smoke.py"
exec bash "$ROOT/scripts/start.sh"
