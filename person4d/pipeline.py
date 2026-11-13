"""Real subprocess pipeline; web process never imports CUDA libraries."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable
from zipfile import ZIP_STORED, ZipFile

from .config import MODEL_FILES, PRESETS, SMPLX_FILE, Settings


class Cancelled(RuntimeError):
    pass


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def probe_video(path: Path) -> dict:
    try:
        completed = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate,nb_frames,duration,start_time:format=duration,start_time",
            "-of", "json", str(path),
        ], capture_output=True, text=True, timeout=30, check=True)
        data = json.loads(completed.stdout)
        streams = data.get("streams", [])
        stream = next(item for item in streams if item["codec_type"] == "video")
        container = data.get("format", {})
        duration = float(container.get("duration") or stream.get("duration") or 0)
        if duration <= 0 or not stream.get("width") or not stream.get("height"):
            raise ValueError("Video has no valid duration or raster")
        container_start = container.get("start_time", "N/A")
        container_start = float(container_start) if container_start != "N/A" else 0.0
        video_start = stream.get("start_time", "N/A")
        video_start = float(video_start) if video_start != "N/A" else container_start
        return {
            "duration": duration, "width": int(stream["width"]), "height": int(stream["height"]),
            "fps": stream.get("avg_frame_rate", "unknown"), "frames": stream.get("nb_frames"),
            "video_start": video_start, "container_start": container_start,
            "has_audio": any(item["codec_type"] == "audio" for item in streams),
        }
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is missing. Run scripts/setup_runpod.sh first.") from exc
    except (subprocess.SubprocessError, ValueError, StopIteration, KeyError) as exc:
        raise ValueError("Cannot read this video. Upload a valid MP4, MOV, MKV or WebM file.") from exc


def audio_export_command(source: Path, destination: Path, metadata: dict, start: float, duration: float) -> list[str]:
    # Inference selects times relative to the first video frame. Preserve source
    # timestamps to keep an audio track that starts earlier or later in sync.
    clip_start = metadata["video_start"] + start
    seek = max(0, clip_start - metadata["container_start"] - 1)
    return [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{seek:.9f}", "-copyts", "-i", str(source), "-map", "0:a:0",
        "-af", f"atrim=start={clip_start:.9f}:end={clip_start + duration:.9f},"
               f"asetpts=PTS-({clip_start:.9f})/TB,aresample=async=1:first_pts=0",
        "-t", f"{duration:.9f}", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        str(destination),
    ]


def write_scene_bundle(destination: Path, files: tuple[Path, ...], cancel: threading.Event) -> None:
    temporary = destination.with_suffix(".partial.zip")
    try:
        # Store the entries as-is so writing and opening the bundle require no
        # compression of the potentially large, full-quality model.
        with ZipFile(temporary, "w", ZIP_STORED) as archive:
            for path in files:
                with path.open("rb") as source, archive.open(path.name, "w") as entry:
                    while block := source.read(1024 * 1024):
                        if cancel.is_set():
                            raise Cancelled("Cancelled while packaging the animated scene")
                        entry.write(block)
        if cancel.is_set():
            raise Cancelled("Cancelled while packaging the animated scene")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


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

    def preparation_key(self, source: dict, generation_args: list[str], adapter_args: list[str]) -> str | None:
        """Fingerprint reusable inputs without rereading the uploaded clip or model weights."""
        sha256 = source.get("sha256", "")
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            return None
        cfg = self.settings
        inputs = [sha256, Path(source["path"]).suffix, generation_args, adapter_args]
        digest = hashlib.sha256(json.dumps(inputs).encode())
        try:
            for name in (*MODEL_FILES, SMPLX_FILE):
                stat = (cfg.models_dir / name).stat()
                digest.update(json.dumps([name, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]).encode())
            source_files = [
                cfg.root / "person4d/adapter.py",
                cfg.inference_repo / "requirements.txt",
                cfg.inference_repo / "inference.py",
            ]
            gvhmr = cfg.inference_repo / "third_party/GVHMR"
            source_suffixes = {".py", ".yaml", ".yml", ".json", ".pt", ".npy", ".npz"}
            for folder in (cfg.inference_repo / "fdanyone", gvhmr / "hmr4d", gvhmr / "tools"):
                source_files.extend(
                    path for path in folder.rglob("*")
                    if path.suffix in source_suffixes and path.is_file()
                )
            for path in sorted(source_files):
                digest.update(path.relative_to(cfg.root).as_posix().encode())
                with path.open("rb") as handle:
                    digest.update(hashlib.file_digest(handle, "sha256").digest())
            # Bind reuse to the installed inference dependencies, including optional attention backends.
            script = (
                "import importlib.metadata as metadata, sys\n"
                "versions = sorted((d.metadata['Name'] or '', d.version) for d in metadata.distributions())\n"
                "print(sys.version, versions)"
            )
            versions = subprocess.check_output(
                [str(cfg.inference_python), "-c", script],
                cwd=cfg.inference_repo,
                env=self.environment(inference=True),
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            digest.update(versions)
            return digest.hexdigest()
        except (OSError, subprocess.SubprocessError):
            return None

    def prepared_job(self, root: Path, key: str) -> Path | None:
        for markers_path in self.settings.jobs_dir.glob("*/stages.json"):
            candidate = markers_path.parent
            if candidate == root or candidate.is_symlink() or markers_path.is_symlink():
                continue
            try:
                markers = json.loads(markers_path.read_text(encoding="utf-8"))
                if markers["generate"]["preparation_key"] != key or markers["adapter"]["preparation_key"] != key:
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if self.generated_valid(candidate / "generated") and self.dataset_valid(candidate / "dataset"):
                return candidate
        return None

    @staticmethod
    def copy_prepared(source: Path, destination: Path, cancel: threading.Event, *, generated: Path | None = None) -> None:
        """Publish a private copy so jobs can be resumed and managed independently."""
        def copy_file(src: str, dst: str) -> str:
            if cancel.is_set():
                raise Cancelled("Cancelled while reusing prepared files")
            return shutil.copy2(src, dst)

        if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError("Prepared files must be local to their job")
        with tempfile.TemporaryDirectory(prefix=".reuse-", dir=destination.parent) as work:
            staged = Path(work) / destination.name
            shutil.copytree(source, staged, copy_function=copy_file)
            if generated is not None:
                manifest = json.loads((staged / "manifest.json").read_text(encoding="utf-8"))
                manifest["source"] = str(generated)
                atomic_json(staged / "manifest.json", manifest)
            if cancel.is_set():
                raise Cancelled("Cancelled while reusing prepared files")
            staged.rename(destination)

    def run(self, job: dict, update: Callable[[dict], None], cancel: threading.Event) -> None:
        cfg = self.settings
        root = cfg.jobs_dir / job["id"]
        options = job["options"]
        # Persisted recipes make retries stable if application presets change.
        recipe = job.get("recipe") or {}
        preset = recipe or PRESETS[options["preset"]]
        seed_voxel_size = recipe.get("voxel_size")
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

        def complete(name: str, preparation_key: str | None = None) -> None:
            marker = {"completed_at": time.time()}
            if preparation_key is not None:
                marker["preparation_key"] = preparation_key
            markers[name] = marker
            atomic_json(markers_path, markers)

        def execute(command: list[str], cwd: Path, inference: bool = False, total: int | None = None) -> None:
            run_command(command, cwd=cwd, log_path=log, cancel=cancel, update=update, env=self.environment(inference), total_steps=total)

        def reuse(donor: Path, destination: Path, *, source: Path | None = None) -> bool:
            try:
                self.copy_prepared(donor / destination.name, destination, cancel, generated=source)
                message = f"Copied {destination.name} from job {donor.name}"
            except (OSError, ValueError, TypeError) as exc:
                message = f"Could not copy {destination.name}: {exc}"
            with log.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[reuse] {message}\n")
            return destination.is_dir()

        stage("probe", "Checking the selected video segment")
        for leftover in root.glob(".reuse-*"):
            shutil.rmtree(leftover, ignore_errors=True)
        metadata = probe_video(root / job["input"]["path"])
        needed = 121 / options["target_fps"]
        if metadata["duration"] - options["start_time"] < needed - 0.001:
            raise ValueError(f"The selected segment needs at least {needed:.2f} seconds after its start time at {options['target_fps']} FPS.")
        update({"input": {**job["input"], "metadata": metadata}})
        atomic_json(root / "video.json", metadata)
        complete("probe")

        generation_args = [
            "--views_per_layer", str(preset["views_per_layer"]),
            "--layer_pitches", json.dumps(preset["layer_pitches"]),
            "--yaw_span", "360",
            "--enable_turbo", "True",
            "--attention_backend", "auto",
            "--target_fps", str(options["target_fps"]),
            "--start_time", str(options["start_time"]),
            "--model_dir", str(cfg.models_dir),
            "--gvhmr_root", str(cfg.inference_repo / "third_party/GVHMR"),
            "--seed", str(options["seed"]),
        ]
        adapter_args = [
            "--model-dir", str(cfg.models_dir),
            "--views-repo", str(cfg.inference_repo),
            "--keyframe-step", "3",
            "--max-points", str(preset["max_points"]),
            "--voxel-size", str(seed_voxel_size or 0.025),
            "--seed", str(options["seed"]),
        ]
        reused_job = None
        generation_key = markers.get("generate", {}).get("preparation_key")
        dataset_key = markers.get("adapter", {}).get("preparation_key")
        stage("generate", "Generating synchronized views with 4DAnyone")
        needs_generation = not self.generated_valid(generated)
        if needs_generation:
            generation_key = self.preparation_key(job["input"], generation_args, adapter_args)
            dataset_key = None
            markers.pop("generate", None)
            markers.pop("adapter", None)
            atomic_json(markers_path, markers)
            if generated.exists():
                # Keep evidence of an incomplete run instead of overwriting it.
                generated.rename(root / f"generated-incomplete-{int(time.time())}")
            if generation_key and not dataset.exists():
                reused_job = self.prepared_job(root, generation_key)
            if reused_job:
                stage("generate", "Copying synchronized views from a matching job")
                if not reuse(reused_job, generated):
                    reused_job = None
            if reused_job is None:
                execute([
                    str(cfg.inference_python), "inference.py",
                    "--video_path", str(root / job["input"]["path"]),
                    "--output_dir", str(generated), *generation_args,
                ], cfg.inference_repo, inference=True)
            if not self.generated_valid(generated):
                raise RuntimeError("4DAnyone finished without complete target videos and camera metadata.")
        complete("generate", generation_key)

        stage("adapter", "Extracting foreground frames and initial 4D geometry")
        if not self.dataset_valid(dataset):
            if reused_job:
                stage("adapter", "Copying foreground frames and geometry from a matching job")
                reuse(reused_job, dataset, source=generated)
            if not self.dataset_valid(dataset):
                execute([
                    str(cfg.inference_python), "-m", "person4d.adapter",
                    "--source", str(generated),
                    "--destination", str(dataset), *adapter_args,
                ], cfg.root, inference=True)
            dataset_key = generation_key
            if generation_key and not needs_generation and generation_key != self.preparation_key(job["input"], generation_args, adapter_args):
                dataset_key = None
        if not self.dataset_valid(dataset):
            raise RuntimeError("Dataset conversion did not publish its completion manifest and initialization.")
        complete("adapter", dataset_key)

        stage("train", "Optimizing the animated Gaussian character")
        saved = checkpoints(results)
        latest = saved[-1] if saved else None
        latest_step = int(re.search(r"ckpt_(\d+)", latest.name).group(1)) if latest else -1
        if latest_step < steps - 1:
            save_steps = sorted(set([min(1000, steps), *range(5000, steps + 1, 5000), steps]))
            command = [str(cfg.training_python), "src/simple_trainer_freetime_4d_pure_relocation.py", "default_keyframe_small", "--data-dir", str(dataset), "--init-npz-path", str(dataset / "init.npz"), "--result-dir", str(results), "--start-frame", "0", "--end-frame", "121", "--max-steps", str(steps), "--max-samples", str(preset["max_points"]), "--data-factor", str(preset["data_factor"]), "--alpha-mode", "transparent", "--random-bkgd", "--packed", "--test-every", "0", "--lambda-foreground", "1.0", "--save-steps", *map(str, save_steps), "--eval-steps", str(steps), "--render-traj-n-frames", "121", "--render-traj-time-frames", "121", "--render-traj-fps", str(options["target_fps"]), "--render-traj-path", "fixed", "--relocation-stop-iter", str(int(steps * 0.9))]
            command.append("--render-traj-final-only")
            if seed_voxel_size is not None:
                # Cover nearby hull voxels at the same timestamp when training starts.
                command += ["--init-scale-world", str(seed_voxel_size / 2)]
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
        playback = {"fps": options["target_fps"], "frames": 121, "duration_seconds": needed, "format": "FreeTimeGsVanilla FTGS PLY v1", "model": "character.ftgs.ply", "source_start_time": options["start_time"], "preset": options["preset"], "views": preset["views_per_layer"] * len(preset["layer_pitches"]), "training_steps": steps, "diagnostics": "Training-view diagnostics (no held-out cameras)", "checkpoint": str(latest.relative_to(root)).replace("\\", "/")}
        audio = output / "audio.m4a"
        if metadata.get("has_audio") and not audio.is_file():
            stage("export", "Adding audio from the selected source segment")
            temporary_audio = output / "audio.partial.m4a"
            try:
                # The continuous FTGS clock ends at frame 120.
                duration = 120 / options["target_fps"]
                execute(audio_export_command(root / job["input"]["path"], temporary_audio, metadata,
                                             options["start_time"], duration), cfg.root)
                audio_probe = subprocess.run([
                    "ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                    "stream=codec_type", "-of", "csv=p=0", str(temporary_audio),
                ], capture_output=True, text=True, check=True, timeout=15)
                if audio_probe.stdout.strip() == "audio":
                    temporary_audio.replace(audio)
            except Cancelled:
                raise
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                with log.open("a", encoding="utf-8") as handle:
                    handle.write(f"\n[audio] Source audio could not be exported: {exc}\n")
            finally:
                temporary_audio.unlink(missing_ok=True)
        if audio.is_file():
            playback["audio"] = audio.name
        playback_file = output / "playback.json"
        atomic_json(playback_file, playback)
        bundle = output / "character-with-audio.zip"
        if audio.is_file() and not bundle.is_file():
            stage("export", "Packaging the animated scene and audio")
            write_scene_bundle(bundle, (model, audio, playback_file), cancel)
        complete("export")
        update({"stage": "completed", "stage_progress": 100, "message": "Your animated character is ready to view and download."})

    @staticmethod
    def dataset_valid(root: Path) -> bool:
        return (root / "manifest.json").is_file() and (root / "init.npz").is_file()

    @staticmethod
    def generated_valid(root: Path) -> bool:
        try:
            metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
            cameras = json.loads((root / "cameras.json").read_text(encoding="utf-8"))["cameras"]
            return metadata["output"]["frames_per_video"] == 121 and bool(cameras) and all((root / camera["video"]).is_file() and (root / camera["video"]).stat().st_size > 0 for camera in cameras)
        except (OSError, ValueError, KeyError, TypeError):
            return False
