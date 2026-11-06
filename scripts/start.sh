#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then echo 'Run bash scripts/setup_runpod.sh first.' >&2; exit 1; fi
export PERSON4D_DATA_DIR="${PERSON4D_DATA_DIR:-$ROOT/data}"
mkdir -p "$PERSON4D_DATA_DIR"
mkdir -p "$PERSON4D_DATA_DIR/tmp"
export TMPDIR="$PERSON4D_DATA_DIR/tmp"
umask 077
TOKEN_FILE="$PERSON4D_DATA_DIR/access-token"
if [[ -z "${PERSON4D_TOKEN:-}" ]]; then
  if [[ ! -s "$TOKEN_FILE" ]]; then .venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))' > "$TOKEN_FILE"; fi
  export PERSON4D_TOKEN="$(cat "$TOKEN_FILE")"
fi
export HF_HOME="${HF_HOME:-$PERSON4D_DATA_DIR/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PERSON4D_DATA_DIR/cache/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PERSON4D_DATA_DIR/cache/torch_extensions}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PERSON4D_HOST="${PERSON4D_HOST:-0.0.0.0}"
export PERSON4D_PORT="${PERSON4D_PORT:-${PORT:-8000}}"
printf '\nPerson to 4DGS — port %s\n' "$PERSON4D_PORT"
printf 'Open the RunPod HTTP link for this port and sign in with this access token:\n%s\n\n' "$PERSON4D_TOKEN"
exec .venv/bin/python -m person4d
