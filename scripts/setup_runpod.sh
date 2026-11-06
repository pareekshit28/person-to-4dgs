#!/usr/bin/env bash
# Run inside a CUDA *devel* RunPod container. All state stays on the volume.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WITH_MODELS=0
ATTENTION=auto
for arg in "$@"; do
  case "$arg" in
    --with-models) WITH_MODELS=1 ;;
    --attention=*) ATTENTION="${arg#*=}" ;;
    --help) echo 'Usage: bash scripts/setup_runpod.sh [--with-models] [--attention=auto|sage|sdpa]'; exit 0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done
case "$ATTENTION" in auto|sage|sdpa) ;; *) echo 'Invalid attention choice' >&2; exit 2;; esac
if [[ "$(uname -s)" != Linux ]]; then echo 'GPU setup requires a Linux CUDA RunPod container.' >&2; exit 1; fi
export PERSON4D_DATA_DIR="${PERSON4D_DATA_DIR:-$ROOT/data}"
python3 "$ROOT/scripts/check_storage.py" --runtime "$ROOT" --data "$PERSON4D_DATA_DIR"
if ! command -v nvidia-smi >/dev/null; then echo 'NVIDIA GPU/driver not found. Start this script inside your GPU pod.' >&2; exit 1; fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
if ! command -v nvcc >/dev/null; then echo 'CUDA compiler missing. Choose a CUDA 12.8 devel image, not a runtime image.' >&2; exit 1; fi
if ! nvcc --version | grep -q 'release 12.8'; then
  echo 'This build is pinned to CUDA 12.8. Use the documented RunPod devel image.' >&2; exit 1
fi
if [[ $EUID -eq 0 ]]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y --no-install-recommends git curl ca-certificates ffmpeg build-essential cmake ninja-build pkg-config libgl1 libglib2.0-0
else
  for tool in git curl ffmpeg ffprobe g++ cmake; do command -v "$tool" >/dev/null || { echo "Missing $tool; install system packages as root." >&2; exit 1; }; done
fi
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
mkdir -p "$PERSON4D_DATA_DIR" "$PERSON4D_DATA_DIR/cache" "$PERSON4D_DATA_DIR/tmp" "$ROOT/.venvs"
export HF_HOME="${HF_HOME:-$PERSON4D_DATA_DIR/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PERSON4D_DATA_DIR/cache/torch}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PERSON4D_DATA_DIR/cache/uv}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PERSON4D_DATA_DIR/cache/torch_extensions}"
export TMPDIR="$PERSON4D_DATA_DIR/tmp"
export MAX_JOBS="${MAX_JOBS:-4}"
free_kb="$(df -Pk "$PERSON4D_DATA_DIR" | awk 'NR==2 {print $4}')"
if (( free_kb < 80000000 )); then echo 'At least 80 GB free storage is required for environments, models and initial outputs; 200 GB is recommended.' >&2; exit 1; fi
uv python install 3.11 3.12
[[ -x .venv/bin/python ]] || uv venv --python 3.12 --seed .venv
[[ -x .venvs/inference/bin/python ]] || uv venv --python 3.11 --seed .venvs/inference
[[ -x .venvs/training/bin/python ]] || uv venv --python 3.12 --seed .venvs/training
uv pip install --python .venv/bin/python -r requirements-web.txt
for environment in inference training; do
  uv pip install --python ".venvs/$environment/bin/python" --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
done
uv pip install --python .venvs/inference/bin/python -r vendor/4danyone/requirements.txt 'torch==2.8.0' 'torchvision==0.23.0'
uv pip install --python .venvs/training/bin/python -r requirements-training.txt 'torch==2.8.0' 'torchvision==0.23.0'
uv pip install --python .venvs/training/bin/python --no-build-isolation -r requirements-training-cuda.txt

# Optional acceleration. Failure keeps the supported PyTorch SDPA path usable.
if [[ "$ATTENTION" != sdpa ]]; then
  if ! uv pip install --python .venvs/inference/bin/python --no-build-isolation \
      'git+https://github.com/thu-ml/SageAttention.git@eb615cf6cf4d221338033340ee2de1c37fbdba4a'; then
    if [[ "$ATTENTION" == sage ]]; then echo 'SageAttention installation failed.' >&2; exit 1; fi
    echo 'SageAttention unavailable; using PyTorch SDPA.' >&2
  fi
fi
PYTHONPATH="$ROOT/vendor/4danyone:$ROOT" .venvs/inference/bin/python -c 'import torch, fdanyone; assert torch.cuda.is_available(); print("Inference GPU:", torch.cuda.get_device_name(0))'
.venvs/training/bin/python scripts/gpu_smoke.py
if (( WITH_MODELS )); then
  .venvs/inference/bin/python vendor/4danyone/scripts/download_model.py \
    --model_dir "$PERSON4D_DATA_DIR/models" --gvhmr_root "$ROOT/vendor/4danyone/third_party/GVHMR"
fi
printf '\nSetup complete. Start the UI with: bash scripts/start.sh\n'
printf 'Expose HTTP port 8000 in RunPod. Upload your licensed SMPL-X model through the UI.\n'
