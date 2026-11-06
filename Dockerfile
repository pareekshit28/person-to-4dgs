# syntax=docker/dockerfile:1.7
# All installation and CUDA compilation happens while building this image.
# No GPU is required by the builder: architectures are explicit, not detected.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ARG MAX_JOBS=2

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /usr/local/bin/

ENV CUDA_HOME=/usr/local/cuda \
    UV_PYTHON_INSTALL_DIR=/opt/person4d/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_LINK_MODE=hardlink \
    UV_CACHE_DIR=/tmp/person4d-uv-cache \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PERSON4D_DATA_DIR=/workspace/person4d \
    PERSON4D_WEB_PYTHON=/opt/person4d/envs/web/bin/python \
    PERSON4D_INFERENCE_PYTHON=/opt/person4d/envs/inference/bin/python \
    PERSON4D_TRAINING_PYTHON=/opt/person4d/envs/training/bin/python \
    PERSON4D_PREBUILT_IMAGE=1 \
    PERSON4D_IMAGE_CUDA_ARCH_LIST=${CUDA_ARCH_LIST} \
    PATH=/opt/person4d/envs/web/bin:${PATH}

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl ffmpeg git libgl1 \
      libglib2.0-0 libgomp1 ninja-build pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Only dependency manifests precede these layers. App-only updates reuse the
# installed environments and the compiled native modules below.
COPY requirements-web.txt requirements-training.txt requirements-training-cuda.txt /opt/person4d/requirements/
COPY vendor/4danyone/requirements.txt /opt/person4d/requirements/inference.txt

# Use one ordinary filesystem cache, not a cache mount. Hard links can then
# share identical NVIDIA libraries between the two environments. Removing the
# cache in this same RUN retains the installed files without a cache layer.
RUN uv python install 3.11 3.12 && \
    uv venv --seed --python 3.12 /opt/person4d/envs/web && \
    uv venv --seed --python 3.11 /opt/person4d/envs/inference && \
    uv venv --seed --python 3.12 /opt/person4d/envs/training && \
    uv pip install --python /opt/person4d/envs/web/bin/python \
      -r /opt/person4d/requirements/requirements-web.txt && \
    uv pip install --python /opt/person4d/envs/inference/bin/python \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.8.0 torchvision==0.23.0 && \
    uv pip install --python /opt/person4d/envs/training/bin/python \
      --index-url https://download.pytorch.org/whl/cu128 \
      torch==2.8.0 torchvision==0.23.0 && \
    uv pip install --python /opt/person4d/envs/inference/bin/python \
      -r /opt/person4d/requirements/inference.txt \
      torch==2.8.0 torchvision==0.23.0 && \
    uv pip install --python /opt/person4d/envs/training/bin/python \
      -r /opt/person4d/requirements/requirements-training.txt \
      torch==2.8.0 torchvision==0.23.0 numpy==1.26.4 && \
    uv cache clean

# Both pinned projects use PyTorch CUDAExtension. Explicit SASS targets permit
# compilation on a CPU builder; BUILD_NO_CUDA=0 prevents a JIT-only gsplat wheel.
# Built-in PyTorch SDPA is the inference backend in this image. Optional native
# attention packages are not installed at pod startup.
RUN TORCH_CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" MAX_JOBS="${MAX_JOBS}" \
    BUILD_NO_CUDA=0 FORCE_CUDA=1 UV_NO_CACHE=1 \
    uv pip install --python /opt/person4d/envs/training/bin/python \
      --no-build-isolation -r /opt/person4d/requirements/requirements-training-cuda.txt \
      torch==2.8.0 torchvision==0.23.0 numpy==1.26.4

# Import the actual installed native modules rather than gsplat's lazy loader.
# cuobjdump verifies every requested GPU architecture without launching a GPU.
# Actual forward/backward validation remains a separate, short runtime check.
RUN /opt/person4d/envs/training/bin/python - <<'PY'
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import torch

assert torch.__version__.split('+')[0] == '2.8.0', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
architectures = os.environ['PERSON4D_IMAGE_CUDA_ARCH_LIST'].split(';')
modules = {}
for name in ('gsplat.csrc', 'fused_ssim_cuda'):
    module = importlib.import_module(name)
    binary = Path(module.__file__)
    assert binary.suffix == '.so', (name, binary)
    listing = subprocess.check_output(
        ['/usr/local/cuda/bin/cuobjdump', '--list-elf', str(binary)], text=True,
    )
    for arch in architectures:
        target = 'sm_' + arch.replace('.', '')
        assert re.search(r'\b' + re.escape(target) + r'\b', listing), (name, target)
    modules[name] = str(binary)
assert hasattr(importlib.import_module('gsplat.csrc'), 'projection_ewa_3dgs_packed_fwd')
assert hasattr(importlib.import_module('fused_ssim_cuda'), 'fusedssim')
Path('/opt/person4d/image-build.json').write_text(json.dumps({
    'schema': 1,
    'torch_version': str(torch.__version__),
    'cuda_version': str(torch.version.cuda),
    'architectures': architectures,
    'native_modules': modules,
    'gpu_validated': False,
}, indent=2) + '\n')
print('Precompiled native modules and all requested architectures verified.')
PY

WORKDIR /opt/person-to-4dgs
COPY . .
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="Person to 4DGS" \
      org.opencontainers.image.source="https://github.com/pareekshit28/person-to-4dgs" \
      org.opencontainers.image.revision=${VCS_REF} \
      org.opencontainers.image.licenses="AGPL-3.0"
RUN VCS_REF="${VCS_REF}" /opt/person4d/envs/web/bin/python - <<'PY'
import json
import os
from pathlib import Path
path = Path('/opt/person4d/image-build.json')
metadata = json.loads(path.read_text())
metadata['vcs_ref'] = os.environ['VCS_REF']
path.write_text(json.dumps(metadata, indent=2) + '\n')
PY

EXPOSE 8000
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["/opt/person4d/envs/web/bin/python", "-c", "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PERSON4D_PORT','8000')+'/healthz', timeout=4).read()"]
ENTRYPOINT ["bash", "/opt/person-to-4dgs/scripts/container_start.sh"]
