#!/usr/bin/env bash
# Repair the native training extensions without reinstalling models or the UI.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"
TRAINING_PYTHON="${PERSON4D_TRAINING_PYTHON:-$ROOT/.venvs/training/bin/python}"
if [[ ! -x "$TRAINING_PYTHON" ]]; then
  echo 'The training environment is missing or its base Python was removed. Run bash scripts/setup_runpod.sh first.' >&2
  exit 1
fi
exec "$TRAINING_PYTHON" "$ROOT/scripts/install_training_cuda.py" --force "$@"
