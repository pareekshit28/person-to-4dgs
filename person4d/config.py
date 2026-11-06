from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def configured_python(variable: str, relative_default: str) -> Path:
    """Keep the venv path itself; resolving its symlink loses the environment."""
    return Path(os.environ.get(variable) or str(ROOT / relative_default)).expanduser()


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("PERSON4D_DATA_DIR", str(ROOT / "data"))).expanduser().resolve())
    token: str = field(default_factory=lambda: os.environ.get("PERSON4D_TOKEN", ""))
    inference_python: Path = field(default_factory=lambda: configured_python("PERSON4D_INFERENCE_PYTHON", ".venvs/inference/bin/python"))
    training_python: Path = field(default_factory=lambda: configured_python("PERSON4D_TRAINING_PYTHON", ".venvs/training/bin/python"))
    web_python: Path = field(default_factory=lambda: configured_python("PERSON4D_WEB_PYTHON", ".venv/bin/python"))
    upload_limit: int = field(default_factory=lambda: int(os.environ.get("PERSON4D_UPLOAD_LIMIT", str(2 * 1024**3))))
    host: str = field(default_factory=lambda: os.environ.get("PERSON4D_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.environ.get("PERSON4D_PORT", "8000")))

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def inference_repo(self) -> Path:
        return self.root / "vendor/4danyone"

    @property
    def training_repo(self) -> Path:
        return self.root / "vendor/freetimegs"


PRESETS = {
    "preview": {"label": "Preview", "views_per_layer": 12, "layer_pitches": [15], "steps": 8000, "max_points": 250000, "data_factor": 2, "description": "12 views, 352 × 640 training, 250k initial Gaussians"},
    "balanced": {"label": "Balanced", "views_per_layer": 24, "layer_pitches": [15], "steps": 30000, "max_points": 1000000, "data_factor": 1, "description": "24 views, 704 × 1280 training, 1M initial Gaussians"},
    "quality": {"label": "High detail · 80 GB+ GPU", "views_per_layer": 16, "layer_pitches": [-10, 15, 35], "steps": 50000, "max_points": 2000000, "data_factor": 1, "description": "48 views, 704 × 1280 training, 2M initial Gaussians; targets 80 GB+ GPUs"},
}

MODEL_FILES = (
    "4danyone/model.safetensors", "4danyone/smplx_to_goliath70.pt", "4danyone/Wan2.2_VAE.pth",
    "4danyone/prompt_context.safetensors", "4danyone/Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors",
    "gvhmr/gvhmr_siga24_release.ckpt", "gvhmr/epoch=10-step=25000.ckpt", "gvhmr/vitpose-h-multi-coco.pth", "gvhmr/yolov8x.pt",
    "perceptual/imagenet-vgg-verydeep-19-conv.safetensors", "birefnet/BiRefNet_config.py", "birefnet/birefnet.py", "birefnet/config.json", "birefnet/model.safetensors",
)
SMPLX_FILE = "body_models/smplx/SMPLX_NEUTRAL.npz"
