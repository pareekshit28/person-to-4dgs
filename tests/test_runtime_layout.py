import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from person4d.config import ROOT, Settings
from person4d.runtime import restore_model_links


def test_image_environment_paths_are_independent_of_source_and_persistent_data(monkeypatch, tmp_path):
    paths = {
        "PERSON4D_WEB_PYTHON": "/opt/person4d/envs/web/bin/python",
        "PERSON4D_INFERENCE_PYTHON": "/opt/person4d/envs/inference/bin/python",
        "PERSON4D_TRAINING_PYTHON": "/opt/person4d/envs/training/bin/python",
    }
    for name, value in paths.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PERSON4D_DATA_DIR", str(tmp_path / "persistent"))
    settings = Settings()
    assert settings.web_python == Path(paths["PERSON4D_WEB_PYTHON"])
    assert settings.inference_python == Path(paths["PERSON4D_INFERENCE_PYTHON"])
    assert settings.training_python == Path(paths["PERSON4D_TRAINING_PYTHON"])
    assert settings.data_dir == (tmp_path / "persistent").resolve()


def test_source_installation_keeps_checkout_environment_defaults(monkeypatch):
    for name in ("PERSON4D_WEB_PYTHON", "PERSON4D_INFERENCE_PYTHON", "PERSON4D_TRAINING_PYTHON"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings()
    assert settings.web_python == ROOT / ".venv/bin/python"
    assert settings.inference_python == ROOT / ".venvs/inference/bin/python"
    assert settings.training_python == ROOT / ".venvs/training/bin/python"


def test_model_link_initialization_does_nothing_before_models_exist(tmp_path):
    assert not restore_model_links(tmp_path / "models", tmp_path / "absent-source")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux runtime symlinks")
def test_existing_model_files_are_relinked_in_a_fresh_container_source(tmp_path):
    model = tmp_path / "persistent/models/gvhmr/gvhmr_siga24_release.ckpt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"existing-model-file")
    fresh_gvhmr = tmp_path / "new-container/GVHMR"
    (fresh_gvhmr / "hmr4d").mkdir(parents=True)
    (fresh_gvhmr / "hmr4d/__init__.py").write_text("")
    source = ROOT / "vendor/4danyone"
    assert restore_model_links(model.parent.parent, source, fresh_gvhmr)
    target = fresh_gvhmr / "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
    assert target.is_symlink() and target.read_bytes() == b"existing-model-file"
    assert restore_model_links(model.parent.parent, source, fresh_gvhmr)
    assert list((tmp_path / "persistent/models").rglob("*.ckpt")) == [model]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux startup script")
def test_start_uses_configured_image_interpreters_and_only_launches(tmp_path):
    app = tmp_path / "application"
    (app / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/start.sh", app / "scripts/start.sh")
    shutil.copy2(ROOT / "scripts/check_storage.py", app / "scripts/check_storage.py")
    package = app / "person4d"
    package.mkdir()
    (package / "__init__.py").write_text("")
    shutil.copy2(ROOT / "person4d/runtime.py", package / "runtime.py")
    (package / "__main__.py").write_text(
        "import json, os\nfrom pathlib import Path\n"
        "Path(os.environ['START_RECORD']).write_text(json.dumps({k: os.environ[k] for k in "
        "('PERSON4D_WEB_PYTHON', 'PERSON4D_INFERENCE_PYTHON', 'PERSON4D_TRAINING_PYTHON', 'PERSON4D_DATA_DIR')}))\n"
    )
    inference = app / "vendor/4danyone"
    inference.mkdir(parents=True)
    (inference / "fdanyone").symlink_to(ROOT / "vendor/4danyone/fdanyone", target_is_directory=True)
    (inference / "third_party/GVHMR/hmr4d").mkdir(parents=True)
    (inference / "third_party/GVHMR/hmr4d/__init__.py").write_text("")
    data = tmp_path / "persistent"
    model = data / "models/gvhmr/gvhmr_siga24_release.ckpt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"persistent-model")
    record = tmp_path / "started.json"
    env = dict(os.environ, PERSON4D_WEB_PYTHON=sys.executable,
               PERSON4D_INFERENCE_PYTHON=sys.executable, PERSON4D_TRAINING_PYTHON=sys.executable,
               PERSON4D_DATA_DIR=str(data), PERSON4D_TOKEN="test-token", START_RECORD=str(record), PYTHONPATH="")
    result = subprocess.run(["bash", str(app / "scripts/start.sh")], env=env,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert not (app / ".venv").exists() and not (app / ".venvs").exists()
    launched = json.loads(record.read_text())
    assert launched["PERSON4D_WEB_PYTHON"] == sys.executable
    assert launched["PERSON4D_DATA_DIR"] == str(data)
    link = inference / "third_party/GVHMR/inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
    assert link.is_symlink() and link.read_bytes() == b"persistent-model"
