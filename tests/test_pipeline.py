import json
import sys
import threading
import time
from pathlib import Path

import pytest

from person4d.config import MODEL_FILES, PRESETS, SMPLX_FILE, VIEW_LAYOUTS, Settings
from person4d.pipeline import Cancelled, Pipeline, run_command


def job_info():
    return {"id": "job", "input": {"path": "input/source.mp4"}, "options": {"preset": "quality", "steps": 1000, "target_fps": 24, "start_time": 1.5, "seed": 42}}


@pytest.mark.parametrize("voxel_size", [None, 0.016], ids=["saved-legacy-recipe", "denser-quality-recipe"])
def test_real_stage_commands_and_resume_skip_completed_work(tmp_path, monkeypatch, voxel_size):
    settings = Settings(root=tmp_path / "repo", data_dir=tmp_path / "data")
    settings.root.mkdir()
    root = settings.jobs_dir / "job"
    root.mkdir(parents=True)
    commands = []

    monkeypatch.setattr("person4d.pipeline.probe_video", lambda _: {"duration": 12, "width": 1080, "height": 1920})

    def execute(command, **kwargs):
        commands.append(command)
        if "inference.py" in command:
            generated = Path(command[command.index("--output_dir") + 1])
            (generated / "videos").mkdir(parents=True)
            (generated / "videos/00.mp4").write_bytes(b"video")
            (generated / "metadata.json").write_text(json.dumps({"output": {"frames_per_video": 121}}))
            (generated / "cameras.json").write_text(json.dumps({"cameras": [{"video": "videos/00.mp4"}]}))
        elif "person4d.adapter" in command:
            dataset = Path(command[command.index("--destination") + 1])
            dataset.mkdir()
            (dataset / "manifest.json").write_text("{}")
            (dataset / "init.npz").write_bytes(b"npz")
        elif "src/simple_trainer_freetime_4d_pure_relocation.py" in command:
            ckpts = root / "results/ckpts"
            ckpts.mkdir(parents=True)
            (ckpts / "ckpt_999.pt").write_bytes(b"checkpoint")
        elif "src/export_ftgs_ply.py" in command:
            Path(command[command.index("--output") + 1]).write_bytes(b"ply\n" + b"x" * 150)
        else:
            raise AssertionError(command)

    monkeypatch.setattr("person4d.pipeline.run_command", execute)
    updates = []
    job = job_info()
    job["recipe"] = PRESETS["quality"].copy()
    if voxel_size is None:
        job["recipe"].pop("voxel_size")
    Pipeline(settings).run(job, updates.append, threading.Event())
    assert len(commands) == 4
    inference, adapter, trainer, export = commands
    assert inference[inference.index("--views_per_layer") + 1] == "16"
    assert json.loads(inference[inference.index("--layer_pitches") + 1]) == [-10, 15, 35]
    assert inference[inference.index("--start_time") + 1] == "1.5"
    assert adapter[adapter.index("--max-points") + 1] == "2000000"
    assert adapter[adapter.index("--voxel-size") + 1] == ("0.025" if voxel_size is None else "0.016")
    if voxel_size is None:
        assert "--init-scale-world" not in trainer
    else:
        assert trainer[trainer.index("--init-scale-world") + 1] == "0.008"
    assert trainer[trainer.index("--end-frame") + 1] == "121"
    assert trainer[trainer.index("--data-factor") + 1] == "1"
    assert trainer[trainer.index("--test-every") + 1] == "0"
    assert "--render-traj-final-only" in trainer
    assert "--random-bkgd" in trainer and "transparent" in trainer
    assert trainer[trainer.index("--save-steps") + 1] == "1000"
    assert (root / "output/character.ftgs.ply").is_file()
    playback = json.loads((root / "output/playback.json").read_text())
    assert playback["fps"] == 24 and playback["frames"] == 121
    Pipeline(settings).run(job, updates.append, threading.Event())
    assert len(commands) == 4
    assert updates[-1]["stage"] == "completed"


def test_matching_jobs_reuse_preparation_and_train_independently(tmp_path, monkeypatch):
    settings = Settings(root=tmp_path / "repo", data_dir=tmp_path / "data", inference_python=Path(sys.executable))
    for path in (
        settings.root / "person4d/adapter.py",
        settings.inference_repo / "inference.py",
        settings.inference_repo / "requirements.txt",
        *(settings.models_dir / name for name in (*MODEL_FILES, SMPLX_FILE)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("asset")
    monkeypatch.setattr("person4d.pipeline.probe_video", lambda _: {"duration": 12})
    commands = []

    def execute(command, **_):
        commands.append(command)

        def value(flag):
            return command[command.index(flag) + 1]

        if "inference.py" in command:
            generated = Path(value("--output_dir"))
            (generated / "videos").mkdir(parents=True)
            (generated / "videos/00.mp4").write_bytes(b"views")
            (generated / "metadata.json").write_text(json.dumps({"output": {"frames_per_video": 121}}))
            (generated / "cameras.json").write_text(json.dumps({"cameras": [{"video": "videos/00.mp4"}]}))
        elif "person4d.adapter" in command:
            dataset = Path(value("--destination"))
            dataset.mkdir()
            (dataset / "manifest.json").write_text(json.dumps({"source": value("--source")}))
            (dataset / "init.npz").write_bytes(b"initial geometry")
        elif "--result-dir" in command:
            assert "--ckpt-path" not in command
            ckpts = Path(value("--result-dir")) / "ckpts"
            ckpts.mkdir(parents=True)
            (ckpts / f"ckpt_{int(value('--max-steps')) - 1}.pt").write_bytes(b"checkpoint")
        else:
            Path(value("--output")).write_bytes(b"ply\n" + b"x" * 150)

    monkeypatch.setattr("person4d.pipeline.run_command", execute)
    pipeline = Pipeline(settings)
    job = job_info()
    job["input"]["sha256"] = "a" * 64
    job["recipe"] = PRESETS["quality"].copy()
    pipeline.run(job, lambda _: None, threading.Event())
    assert len(commands) == 4

    job["id"] = "longer-training"
    job["options"]["steps"] = 2000
    pipeline.run(job, lambda _: None, threading.Event())
    assert len(commands) == 6
    root = settings.jobs_dir / job["id"]
    assert (root / "generated/videos/00.mp4").read_bytes() == b"views"
    manifest = json.loads((root / "dataset/manifest.json").read_text())
    assert manifest["source"] == str(root / "generated")
    assert (root / "results/ckpts/ckpt_1999.pt").is_file()
    assert "Copied dataset from job job" in (root / "job.log").read_text()

    job["id"] = "different-views"
    job["recipe"].update(VIEW_LAYOUTS[24])
    pipeline.run(job, lambda _: None, threading.Event())
    assert len(commands) == 10


def test_training_resume_passes_latest_checkpoint(tmp_path, monkeypatch):
    settings = Settings(root=tmp_path / "repo", data_dir=tmp_path / "data")
    settings.root.mkdir()
    root = settings.jobs_dir / "job"
    (root / "generated/videos").mkdir(parents=True)
    (root / "generated/videos/00.mp4").write_bytes(b"video")
    (root / "generated/metadata.json").write_text(json.dumps({"output": {"frames_per_video": 121}}))
    (root / "generated/cameras.json").write_text(json.dumps({"cameras": [{"video": "videos/00.mp4"}]}))
    (root / "dataset").mkdir()
    (root / "dataset/init.npz").write_bytes(b"npz")
    (root / "dataset/manifest.json").write_text("{}")
    ckpts = root / "results/ckpts"
    ckpts.mkdir(parents=True)
    (ckpts / "ckpt_499.pt").write_bytes(b"checkpoint")
    # A partial checkpoint never matches the published checkpoint glob.
    (ckpts / ".ckpt_999.pt.tmp").write_bytes(b"partial")
    monkeypatch.setattr("person4d.pipeline.probe_video", lambda _: {"duration": 12})
    commands = []

    def execute(command, **kwargs):
        commands.append(command)
        if "--ckpt-path" in command:
            assert command[command.index("--ckpt-path") + 1].endswith("ckpt_499.pt")
            (ckpts / "ckpt_999.pt").write_bytes(b"finished")
        elif "--output" in command:
            Path(command[command.index("--output") + 1]).write_bytes(b"ply\n" + b"x" * 150)

    monkeypatch.setattr("person4d.pipeline.run_command", execute)
    Pipeline(settings).run(job_info(), lambda _: None, threading.Event())
    assert len(commands) == 2


def test_short_segment_fails_before_gpu_commands(tmp_path, monkeypatch):
    settings = Settings(root=tmp_path, data_dir=tmp_path / "data")
    monkeypatch.setattr("person4d.pipeline.probe_video", lambda _: {"duration": 3})
    monkeypatch.setattr("person4d.pipeline.run_command", lambda *_args, **_kwargs: pytest.fail("GPU stage must not launch"))
    with pytest.raises(ValueError, match="selected segment needs"):
        Pipeline(settings).run(job_info(), lambda _: None, threading.Event())


def test_cancel_terminates_silent_subprocess(tmp_path):
    cancel = threading.Event()
    timer = threading.Timer(0.35, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(Cancelled):
            run_command([sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path, log_path=tmp_path / "job.log", cancel=cancel)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 12
