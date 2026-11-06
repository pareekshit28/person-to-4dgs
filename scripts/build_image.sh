#!/usr/bin/env bash
# Build once on a machine/CI builder with Docker and the CUDA development image.
# The builder needs sufficient RAM/disk, but it does not need an NVIDIA GPU.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${1:-person-to-4dgs:local}"
if (( $# > 0 )); then shift; fi
export DOCKER_BUILDKIT=1
VCS_REF=unknown
if git -C "$ROOT" rev-parse HEAD >/dev/null 2>&1; then
  VCS_REF="$(git -C "$ROOT" rev-parse HEAD)"
fi
exec docker build --platform linux/amd64 \
  --build-arg "CUDA_ARCH_LIST=${PERSON4D_BUILD_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;12.0}" \
  --build-arg "MAX_JOBS=${MAX_JOBS:-2}" \
  --build-arg "VCS_REF=$VCS_REF" \
  --tag "$IMAGE" "$@" "$ROOT"
