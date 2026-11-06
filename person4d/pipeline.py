"""Real subprocess pipeline; web process never imports CUDA libraries."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from .config import PRESETS, Settings


class Cancelled(RuntimeError):
    pass


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def probe_video(path: Path) -> dict:
    try:
        completed = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration", "-of", "json", str(path)], capture_output=True, text=True, timeout=30, check=True)
        data = json.loads(completed.stdout)
        stream = data.get("streams", [])[0]
        duration = float(data.get("format", {}).get("duration") or stream.get("duration") or 0)
        if duration <= 0 or not stream.get("width") or not stream.get("height"):
            raise ValueError("Video has no valid duration or raster")
        return {"duration": duration, "width": int(stream["width"]), "height": int(stream["height"]), "fps": stream.get("avg_frame_rate", "unknown"), "frames": stream.get("nb_frames")}
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is missing. Run scripts/setup_runpod.sh first.") from exc
    except (subprocess.SubprocessError, ValueError, IndexError, KeyError) as exc:
        raise ValueError("Cannot read this video. Upload a valid MP4, MOV, MKV or WebM file.") from exc


def terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        else:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, OSError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass


def run_command(command: list[str], *, cwd: Path, log_path: Path, cancel: threading.Event, update: Callable[[dict], None] | None = None, env: dict | None = None, total_steps: int | None = None) -> None:
    """Drain child output while a separate watcher can cancel a silent process."""
    command = [str(item) for item in command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        log.write("\n[command] " + json.dumps(command) + "\n")
        kwargs = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        process = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1, **kwargs)
        finished = threading.Event()

        def watcher() -> None:
            while not finished.wait(0.2):
                if cancel.is_set():
                    terminate_process(process)
                    return

        monitor = threading.Thread(target=watcher, name="process-cancel", daemon=True)
        monitor.start()
        last_update = 0.0
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                now = time.monotonic()
                if update and now - last_update > 0.8:
                    clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
                    event = None
                    try:
                        parsed = json.loads(clean)
                        if isinstance(parsed, dict) and parsed.get("event") == "progress":
                            event = {"stage_progress": parsed.get("progress"), "message": str(parsed.get("message", ""))[:240]}
                    except (ValueError, TypeError):
                        pass
                    if total_steps:
                        match = re.search(r"(\d+)\s*/\s*" + str(total_steps) + r"\b", clean)
                        if match:
                            event = {"stage_progress": min(100, int(match.group(1)) / total_steps * 100), "message": f"Training iteration {int(match.group(1)):,} / {total_steps:,}"}
                    if event:
                        update(event)
                        last_update = now
            result = process.wait()
            if cancel.is_set():
                raise Cancelled("Cancelled by user; completed stages and checkpoints were retained.")
            if result:
                raise RuntimeError(f"Process exited with code {result}. Open the job log for the upstream error.")
        finally:
            finished.set()
            if process.poll() is None:
                terminate_process(process)
            monitor.join(timeout=10)


def checkpoints(results: Path) -> list[Path]:
    def step(path: Path) -> int:
        match = re.search(r"ckpt_(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1
    return sorted((p for p in (results / "ckpts").glob("ckpt_*.pt") if p.is_file() and p.stat().st_size > 0), key=step)


class Pipeline:
    def __init__(self, settings: Settings):
        self.settings = settings

    def environment(self, inference: bool = False) -> dict:
        env = os.environ.copy()
        paths = [str(self.settings.root)]
        if inference:
            paths.append(str(self.settings.inference_repo))
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env.update(PYTHONPATH=os.pathsep.join(paths), PYTHONUNBUFFERED="1", HF_HOME=str(self.settings.data_dir / "cache/huggingface"), TORCH_HOME=str(self.settings.data_dir / "cache/torch"), CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        return env

    def run(self, job: dict, update: Callable[[dict], None], cancel: threading.Event) -> None:
        cfg = self.settings
        root = cfg.jobs_dir / job["id"]
        options = job["options"]
        # Persisted recipes make retries stable if application presets change.
        preset = job.get("recipe") or PRESETS[options["preset"]]
        steps = options.get("steps") or preset["steps"]
        generated, dataset, results = root / "generated", root / "dataset", root / "results"
        log = root / "job.log"
        markers_path = root / "stages.json"
        try:
            markers = json.loads(markers_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            markers = {}

        def stage(name: str, message: str) -> None:
            if cancel.is_set():
                raise Cancelled("Cancelled by user")
            update({"stage": name, "stage_progress": None, "message": message})

        def complete(name: str) -> None:
            markers[name] = {"completed_at": time.time()}
            atomic_json(markers_path, markers)

        def execute(command: list[str], cwd: Path, inference: bool = False, total: int | None = None) -> None:
            run_command(command, cwd=cwd, log_path=log, cancel=cancel, update=update, env=self.environment(inference), total_steps=total)

        stage("probe", "Checking the selected video segment")
        metadata = probe_video(root / job["input"]["path"])
        needed = 121 / options["target_fps"]
        if metadata["duration"] - options["start_time"] < needed - 0.001:
            raise ValueError(f"The selected segment needs at least {needed:.2f} seconds after its start time at {options['target_fps']} FPS.")
        update({"input": {**job["input"], "metadata": metadata}})
        atomic_json(root / "video.json", metadata)
        complete("probe")

        stage("generate", "Generating synchronized views with 4DAnyone")
        generated_valid = self.generated_valid(generated)
        if not generated_valid:
            if generated.exists():
                # Keep evidence of an incomplete run instead of overwriting it.
                generated.rename(root / f"generated-incomplete-{int(time.time())}")
            execute([str(cfg.inference_python), "inference.py", "--video_path", str(root / job["input"]["path"]), "--output_dir", str(generated), "--views_per_layer", str(preset["views_per_layer"]), "--layer_pitches", json.dumps(preset["layer_pitches"]), "--yaw_span", "360", "--enable_turbo", "True", "--attention_backend", "auto", "--target_fps", str(options["target_fps"]), "--start_time", str(options["start_time"]), "--model_dir", str(cfg.models_dir), "--gvhmr_root", str(cfg.inference_repo / "third_party/GVHMR"), "--seed", str(options["seed"])], cfg.inference_repo, inference=True)
            if not self.generated_valid(generated):
                raise RuntimeError("4DAnyone finished without complete target videos and camera metadata.")
        complete("generate")

        stage("adapter", "Extracting foreground frames and initial 4D geometry")
        if not (dataset / "manifest.json").is_file() or not (dataset / "init.npz").is_file():
            execute([str(cfg.inference_python), "-m", "person4d.adapter", "--source", str(generated), "--destination", str(dataset), "--model-dir", str(cfg.models_dir), "--views-repo", str(cfg.inference_repo), "--keyframe-step", "3", "--max-points", str(preset["max_points"]), "--voxel-size", "0.025", "--seed", str(options["seed"])], cfg.root, inference=True)
        if not (dataset / "manifest.json").is_file() or not (dataset / "init.npz").is_file():
            raise RuntimeError("Dataset conversion did not publish its completion manifest and initialization.")
        complete("adapter")

        stage("train", "Optimizing the animated Gaussian character")
        saved = checkpoints(results)
        latest = saved[-1] if saved else None
        latest_step = int(re.search(r"ckpt_(\d+)", latest.name).group(1)) if latest else -1
        if latest_step < steps - 1:
            save_steps = sorted(set([min(1000, steps), *range(5000, steps + 1, 5000), steps]))
            command = [str(cfg.training_python), "src/simple_trainer_freetime_4d_pure_relocation.py", "default_keyframe_small", "--data-dir", str(dataset), "--init-npz-path", str(dataset / "init.npz"), "--result-dir", str(results), "--start-frame", "0", "--end-frame", "121", "--max-steps", str(steps), "--max-samples", str(preset["max_points"]), "--data-factor", str(preset["data_factor"]), "--alpha-mode", "transparent", "--random-bkgd", "--packed", "--test-every", "0", "--lambda-foreground", "1.0", "--save-steps", *map(str, save_steps), "--eval-steps", str(steps), "--render-traj-n-frames", "121", "--render-traj-time-frames", "121", "--render-traj-fps", str(options["target_fps"]), "--render-traj-path", "fixed", "--relocation-stop-iter", str(int(steps * 0.9))]
            if latest:
                command += ["--ckpt-path", str(latest), "--resume-mode", "resume"]
            execute(command, cfg.training_repo, total=steps)
        saved = checkpoints(results)
        if not saved:
            raise RuntimeError("Training finished without a checkpoint.")
        latest = saved[-1]
        if int(re.search(r"ckpt_(\d+)", latest.name).group(1)) < steps - 1:
            raise RuntimeError("Training stopped before the requested final checkpoint.")
        complete("train")

        stage("export", "Exporting the complete animated 4DGS model")
        output = root / "output"
        output.mkdir(exist_ok=True)
        model = output / "character.ftgs.ply"
        if not model.is_file() or model.stat().st_size < 100:
            temporary_model = output / "character.partial.ftgs.ply"
            execute([str(cfg.training_python), "src/export_ftgs_ply.py", "--ckpt", str(latest), "--output", str(temporary_model)], cfg.training_repo)
            if not temporary_model.is_file() or temporary_model.stat().st_size < 100:
                raise RuntimeError("Animated model export did not produce a valid file.")
            temporary_model.replace(model)
        atomic_json(output / "playback.json", {"fps": options["target_fps"], "frames": 121, "duration_seconds": 121 / options["target_fps"], "format": "FreeTimeGsVanilla FTGS PLY v1", "model": "character.ftgs.ply", "source_start_time": options["start_time"], "preset": options["preset"], "training_steps": steps, "diagnostics": "Training-view diagnostics (no held-out cameras)", "checkpoint": str(latest.relative_to(root)).replace("\\", "/")})
        complete("export")
        update({"stage": "completed", "stage_progress": 100, "message": "Your animated character is ready to view and download."})

    @staticmethod
    def generated_valid(root: Path) -> bool:
        try:
            metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
            cameras = json.loads((root / "cameras.json").read_text(encoding="utf-8"))["cameras"]
            return metadata["output"]["frames_per_video"] == 121 and bool(cameras) and all((root / camera["video"]).is_file() and (root / camera["video"]).stat().st_size > 0 for camera in cameras)
        except (OSError, ValueError, KeyError, TypeError):
            return False
