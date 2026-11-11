# Person to 4DGS

Upload a static-camera video of one person performing an action. Select a
121-frame segment, reconstruct it as an animated Gaussian character, and orbit,
play, scrub, or download it in the browser. This is a self-hosted RunPod app
combining **4DAnyone → foreground/geometry preparation → FreeTimeGsVanilla**.

The repository includes the UI, server, durable GPU queue, dataset adapter,
vendored inference/reconstruction source, and a local WebGL2 player. Model
weights download onto your pod. SMPL-X is supplied separately by the user.

## Run on RunPod

### Prebuilt container (recommended)

Use the application image instead of installing the software inside each pod:

```text
ghcr.io/pareekshit28/person-to-4dgs:latest
```

Image tags and build status are available from the
[container package](https://github.com/users/pareekshit28/packages/container/package/person-to-4dgs)
and [image build workflow](https://github.com/pareekshit28/person-to-4dgs/actions/workflows/image.yml).
For reproducible deployments, select an immutable `sha-...` tag or digest.

Configure your RunPod with:

- An NVIDIA GPU supported by the image (A100, RTX 30/Ada, Hopper or RTX PRO Blackwell).
- HTTP port **8000** exposed.
- A standard Network Volume or Volume Disk for active job data.
- At least **20 GB container disk** and an initial **150 GB persistent data volume**.
- The image's default entrypoint; leave the custom start command empty.

Environments and compiled modules live at fixed paths inside the image, separate
from the mounted data. Startup checks GPU execution and starts the UI. It does
not run package installation, `apt`, `pip`, `uv`, or CUDA compilation. A new host
still needs to pull the image; cached hosts can reuse its layers.

**Reusing your current network volume:** if the source installation already put
models and jobs in `/workspace/person-to-4dgs/data`, set this pod environment
variable:

```text
PERSON4D_DATA_DIR=/workspace/person-to-4dgs/data
```

This reuses existing weights, uploads, jobs and the access token. Fresh installs
default to `/workspace/person4d`. Upload the separately licensed SMPL-X model in
the UI if it is not already in that data directory. Public models download only
when requested and missing. The container restores its local GVHMR links to
existing persistent weights each time it starts.

The published build compiles native kernels for compute capabilities
`8.0;8.6;8.9;9.0;12.0` and verifies the binaries contain those targets. Actual GPU
forward/backward validation runs on the pod, without compiling. The image uses
PyTorch SDPA for inference. A successful image build is not a claim that a full
character reconstruction has been validated on every GPU.

### Installing from source (alternative)

Recommended for the default high-quality recipe: **one H200 GPU** with a
CUDA 12.8 development container, **150 GB persistent disk for an initial job**,
16+ CPU cores and 64+ GB system RAM. An H100 80 GB or RTX PRO 6000 Blackwell
96 GB is an alternative; 200 GB provides more room for retained jobs. The upstream
generation benchmark reports its fastest tested single-GPU results on H200;
this app's complete reconstruction runtime has not yet been measured.

Use the container image:

```text
runpod/pytorch:1.5.0-cu1281-torch280-ubuntu2404
```

Expose **HTTP port 8000** in the pod configuration. Attach a standard
**Network Volume** or **Volume Disk** at `/workspace`. Use a 20 GB or larger
container disk for system packages and local Python tooling. The application,
GPU environments, models and job data live on the persistent filesystem.

**Global Volumes / GeeseFS cannot host the active application.** Object-backed
storage lacks the permissions, locks and atomic operations required by Git,
Python environments and the job/checkpoint machinery. A mount at `/workspace`
does not imply it is a standard filesystem. Check it before cloning:

```bash
findmnt -T /workspace -o TARGET,SOURCE,FSTYPE,OPTIONS
df -h / /workspace
```

If this reports `fuse.geesefs`, provision a Network Volume or Volume Disk and
deploy the pod with that storage. Increasing a Global Volume's capacity does
not fix its filesystem behavior. Keep the global volume for copied artifacts
or read-mostly files. See [RunPod's storage explanation](https://www.runpod.io/blog/global-volumes-arent-network-volumes).

Once `/workspace` uses suitable storage, run:

```bash
cd /workspace
git clone https://github.com/pareekshit28/person-to-4dgs.git
cd person-to-4dgs
bash scripts/setup_runpod.sh
bash scripts/start.sh
```

Open RunPod's **Connect → HTTP Service → port 8000**. Sign in using the access
token printed by `start.sh`. The token is retained in `data/access-token` so it
survives restart; `PERSON4D_TOKEN` can supply your own value.

The first setup installs two separate GPU environments and compiles/executes a
small rasterizer check. It first validates the active storage and stops with a
specific error on incompatible mounts. Network volumes that normalize Unix
permission bits are accepted when execution, symlinks, locks and file
replacement work; exact `chmod` mode preservation is not required.
Keep the terminal open until it completes. Re-running
setup reuses installed environments and download caches. Public weights can
also be downloaded during installation with `--with-models`.

Native reconstruction extensions are compiled for the visible GPU architectures
and installed training PyTorch/CUDA version. Setup records that build only after
the CUDA tests pass. A change of GPU architecture, PyTorch, compiler or pinned
extension source triggers a fresh build instead of reusing an incompatible wheel.

In the UI:

1. Download the public models in **Setup**.
2. Register and accept the model license at the [official SMPL-X website](https://smpl-x.is.tue.mpg.de/).
   Download its model archive and upload `models_smplx_v1_1.zip` or
   `SMPLX_NEUTRAL.npz`. The application does not need your SMPL-X credentials.
3. Upload a clear video containing one person. Select the action's start time.
4. Choose **Views** and **Training steps** independently, then start
   reconstruction. You can close the browser while the server continues processing.
5. Open the completed character in the embedded viewer and download
   `character.ftgs.ply` and `playback.json`.

Do not stop the pod until the job has finished or you have cancelled it and
retained the persistent volume. The app does not create, stop or bill RunPod pods.

## Quality and speed

| Views | Camera layout |
| --- | --- |
| Preview · 12 | One elevation |
| Balanced · 24 | One elevation |
| High quality · 48 (default) | Three elevations |

Pair any view option with **8,000 (Preview)**, **30,000 (Balanced)**,
**50,000 (High quality, default)**, or a custom budget of 1,000 to 100,000
training steps. This budget applies to Gaussian reconstruction after view generation.
For another job with the same source file, segment, views and seed, the app can
copy completed views and prepared geometry from a matching job on the same
data volume. Training uses the step count selected for the new job.

All combinations in the new-job form use a 704 × 1280 training raster and a cap
of 2,000,000 Gaussians. The foreground masks and geometry determine the seed
count. The adapter uses a 1.6 cm voxel grid and the trainer starts each Gaussian
with a standard deviation of half that spacing. More seeds can increase
training time and memory use.

Generation uses 4DAnyone-Turbo, whose upstream comparison reports quality close
to the Base model with substantially fewer denoising steps. Setup attempts
SageAttention acceleration and falls back to supported PyTorch SDPA when it
cannot be installed. Use `--attention=sdpa` to skip that optional build or
`--attention=sage` to require it. The generation backend automatically chooses
the supported low-memory path on GPUs with at most 24 GB.

Full-resolution reconstruction targets GPUs with at least 80 GB VRAM. Memory
requirements depend on the clip and the number of visible Gaussians. The UI
displays the detected GPU and any memory warning.

Use a sharp, well-lit, preferably 1080p-or-higher full-body source, with the
person visible throughout the selected action and little motion blur. Current
generation produces **121 frames at 704 × 1280**; higher source resolution does
not make the generated views native 1080p. At 24 FPS, supply at least 5.05 seconds
after the selected start; at 30 FPS, at least 4.04 seconds.

The selected performance is preserved in time. Novel views of unobserved
surfaces are generated estimates, and loose garments, hair, hands, occlusions
or incorrect pose recovery can reduce consistency. The output is animated
free-viewpoint playback of this action. Arbitrary new actions, skeletal
retargeting and relighting require additional systems.

## Included pipeline

- **Generation:** pinned 4DAnyone and GVHMR source; published weights and
  calibrated, synchronized virtual views. The source camera remains outside
  the reconstruction rig; the adapter uses the generated cameras.
- **Preparation:** decode and segment all views; save straight RGBA; serialize
  raw OpenCV/Y-up calibration as COLMAP; carve geometry at regularly spaced
  times with bounded GPU batches; initialize temporally distributed Gaussians.
- **Reconstruction:** pinned community FreeTimeGsVanilla, full-image alpha
  supervision and random backgrounds, local motion, periodic atomic checkpoints.
  All generated cameras participate in training; camera-0 diagnostics are
  explicitly in-sample, not held-out quality measurements.
- **Output:** complete animated FTGS PLY, playback metadata, training checkpoint,
  generated videos, trajectory renders and logs. The bundled renderer supports
  the temporal fields; generic static PLY viewers do not animate this file.

The original private FreeTimeGS trainer from the 4DAnyone paper is not included.
This app integrates the public reproduction. See
[4DAnyone's reconstruction note](https://github.com/ant-research/4DAnyone/blob/main/docs/nerfstudio.md),
[the paper](https://arxiv.org/html/2608.20335v1), and
[upstream speed measurements](https://github.com/ant-research/4DAnyone/blob/main/docs/inference_performance.md).

## Storage, restart and access

All uploaded videos, job records, model weights, caches and results live in
`data/` by default. Set `PERSON4D_DATA_DIR=/workspace/person4d-data` before
**both** setup and start to choose a different persistent location.

Jobs run one at a time. Cancelling terminates the active process group; saved
stages and checkpoints remain. After a crash or restart, interrupted jobs can
be resumed from the UI. The adapter reuses valid frames and hulls; training
resumes from its latest completed checkpoint. Recovered generation may rerun
its interrupted stage. The browser is not the worker and can reconnect later.

The access token protects API, inputs, outputs and the player. Use the RunPod
HTTPS link. The server is a single-owner application; do not share the token
with people who should not be able to run jobs or download footage. Models and
job data are ignored by Git.

Useful environment variables:

| Variable | Default |
| --- | --- |
| `PERSON4D_DATA_DIR` | `<repo>/data` |
| `PERSON4D_TOKEN` | Generated and retained by `scripts/start.sh` |
| `PERSON4D_PORT` / `PORT` | `8000` |
| `PERSON4D_INFERENCE_PYTHON` | `.venvs/inference/bin/python` |
| `PERSON4D_TRAINING_PYTHON` | `.venvs/training/bin/python` |
| `PERSON4D_UPLOAD_LIMIT` | `2147483648` bytes (2 GiB) |
| `MAX_JOBS` | `4` concurrent C++ compiler jobs during setup |

`Dockerfile` installs dependencies and compiles native modules at image build
time on a CPU builder with a CUDA toolkit. It needs sufficient RAM and free
disk, but no GPU. Build locally with
`bash scripts/build_image.sh person-to-4dgs:local`. The image build workflow
publishes to GHCR using the repository's GitHub Actions token. For a newly
created personal-account package, its owner must make the package public once
in GitHub Package settings to allow anonymous RunPod pulls.

## Local development and verification

If ordinary PyTorch CUDA operations succeed but the reconstruction rasterizer
fails, rebuild its native CUDA extensions against the current GPU:

```bash
cd /workspace/person-to-4dgs
git pull --ff-only
bash scripts/rebuild_training_cuda.sh
```

This rebuilds the pinned `gsplat` and `fused-ssim` extensions without cached
native wheels, then tests packed rasterization, gradients and fused SSIM in a
fresh process. It retains model weights and jobs. The command can take several
minutes to compile. A failed GPU test remains an error; the app does not bypass
the rasterizer check or claim that reconstruction is ready.

The web server and CPU tests do not need CUDA:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m person4d
```

On Windows, use `.venv\Scripts\activate` instead of `source`. GPU reconstruction
is supported on Linux. Source-only local launch has no token unless you set
`PERSON4D_TOKEN`; use `scripts/start.sh` on a public pod.

Tests cover camera transforms, temporal units, seed budgets, conversion of a
synthetic synchronized clip, durable jobs, authentication, file boundaries,
pipeline commands and cancellation. `scripts/gpu_smoke.py` runs actual CUDA
rasterization forward/backward and fused SSIM during pod setup.

**Validation boundary:** local CPU tests and browser checks can validate the
adapter and application flow. Full model installation, a real GPU reconstruction,
identity fidelity and performance must be validated on the target pod with
real footage. No successful full GPU run is claimed by this initial release.

Source revisions and changes are recorded in `vendor/SOURCES.json`. First-party
code and FreeTimeGsVanilla use AGPL-3.0; third-party components retain their own
terms, including GVHMR's non-commercial restrictions and the separate SMPL-X
license. See [THIRD_PARTY.md](THIRD_PARTY.md).
