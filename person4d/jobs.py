from __future__ import annotations

import copy
import json
import mimetypes
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from filelock import FileLock, Timeout

from .config import PRESETS, Settings
from .pipeline import Cancelled, Pipeline, atomic_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


STAGE_RANGES = {"queued": (0, 0), "probe": (0, 2), "generate": (2, 40), "adapter": (40, 55), "train": (55, 97), "export": (97, 100), "completed": (100, 100)}


def artifacts_for(settings: Settings, job_id: str) -> list[dict]:
    root = settings.jobs_dir / job_id
    candidates = [(root / "output/character.ftgs.ply", "model"), (root / "output/playback.json", "metadata"), (root / "job.log", "log"), (root / "video.json", "metadata"), (root / "dataset/manifest.json", "metadata")]
    candidates += [(path, "video") for path in sorted((root / "generated/videos").glob("*.mp4"))]
    candidates += [(path, "video") for path in sorted((root / "results/videos").glob("*.mp4"))]
    candidates += [(path, "checkpoint") for path in sorted((root / "results/ckpts").glob("*.pt"))]
    result = []
    for path, kind in candidates:
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            result.append({"name": path.name, "path": relative, "url": f"/api/jobs/{job_id}/files/{quote(relative)}", "kind": kind, "type": kind, "size": path.stat().st_size})
    return result


class JobManager:
    def __init__(self, settings: Settings, pipeline: Pipeline | None = None):
        self.settings = settings
        self.pipeline = pipeline or Pipeline(settings)
        self._jobs: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self._active: str | None = None
        self.runtime_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._process_lock: FileLock | None = None

    def start(self) -> None:
        self.settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._process_lock = FileLock(str(self.settings.data_dir / "worker.lock"))
        try:
            self._process_lock.acquire(timeout=0)
        except Timeout as exc:
            raise RuntimeError("Another Person4D server owns this data directory. Run one server worker per GPU.") from exc
        with self._lock:
            for path in self.settings.jobs_dir.glob("*/job.json"):
                try:
                    job = json.loads(path.read_text(encoding="utf-8"))
                    if job["id"] != path.parent.name:
                        continue
                    if job.get("status") == "running":
                        job.update(status="interrupted", message="Server stopped during this job. Resume to reuse completed stages and checkpoints.", updated_at=utc_now())
                        atomic_json(path, job)
                    self._jobs[job["id"]] = job
                except (OSError, ValueError, KeyError):
                    continue
        self._worker = threading.Thread(target=self._run, name="person4d-worker", daemon=True)
        self._worker.start()
        self._wake.set()

    def close(self) -> None:
        self._stop.set()
        self._cancel.set()
        self._wake.set()
        if self._worker:
            self._worker.join(timeout=15)
        if self._process_lock:
            self._process_lock.release()

    @property
    def busy(self) -> bool:
        return self._active is not None or self.runtime_lock.locked()

    def _save(self, job: dict) -> None:
        atomic_json(self.settings.jobs_dir / job["id"] / "job.json", job)

    def create(self, job_id: str, *, title: str, filename: str, relative_input: str, options: dict, sha256: str) -> dict:
        now = utc_now()
        job = {"id": job_id, "title": title.strip()[:100] or Path(filename).stem[:100], "status": "queued", "stage": "queued", "progress": 0, "stage_progress": 0, "message": "Waiting for the GPU", "error": None, "created_at": now, "updated_at": now, "options": options, "recipe": copy.deepcopy(PRESETS.get(options.get("preset", "quality"), PRESETS["quality"])), "input": {"filename": filename, "path": relative_input, "url": f"/api/jobs/{job_id}/files/{quote(relative_input)}", "sha256": sha256, "metadata": None}, "artifacts": []}
        with self._lock:
            if job_id in self._jobs:
                raise ValueError("Job already exists")
            self._save(job)
            self._jobs[job_id] = job
        self._wake.set()
        return copy.deepcopy(job)

    def get(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            job = copy.deepcopy(self._jobs[job_id])
        job["artifacts"] = artifacts_for(self.settings, job_id)
        return job

    def list(self) -> list[dict]:
        with self._lock:
            ids = sorted(self._jobs, key=lambda item: self._jobs[item]["created_at"], reverse=True)
        return [self.get(job_id) for job_id in ids]

    def update(self, job_id: str, values: dict) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.update(values)
            low, high = STAGE_RANGES.get(job["stage"], (job.get("progress", 0), job.get("progress", 0)))
            part = job.get("stage_progress")
            job["progress"] = round(low + (high - low) * max(0, min(float(part or 0), 100)) / 100, 1)
            job["updated_at"] = utc_now()
            self._save(job)

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs[job_id]
            if job["status"] == "queued":
                self.update(job_id, {"status": "cancelled", "message": "Cancelled before starting"})
            elif job["status"] == "running":
                self._cancel.set()
                self.update(job_id, {"message": "Cancelling the current process; saved work will be retained"})
        return self.get(job_id)

    def retry(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs[job_id]
            if job["status"] not in ("failed", "cancelled", "interrupted"):
                raise ValueError("Only a failed, cancelled or interrupted job can be resumed")
            self.update(job_id, {"status": "queued", "stage": "queued", "stage_progress": 0, "error": None, "message": "Queued to resume completed stages and the latest checkpoint"})
        self._wake.set()
        return self.get(job_id)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=1)
            self._wake.clear()
            if self._stop.is_set():
                break
            with self._lock:
                queued = sorted((j for j in self._jobs.values() if j["status"] == "queued"), key=lambda j: j["created_at"])
                if not queued:
                    continue
                job_id = queued[0]["id"]
                self._active = job_id
                self._cancel.clear()
                self.update(job_id, {"status": "running", "error": None, "message": "Starting reconstruction"})
                job = copy.deepcopy(self._jobs[job_id])
            try:
                with self.runtime_lock:
                    if self._cancel.is_set() or self._stop.is_set():
                        raise Cancelled("Cancelled before processing")
                    self.pipeline.run(job, lambda values: self.update(job_id, values), self._cancel)
                self.update(job_id, {"status": "completed", "stage": "completed", "stage_progress": 100, "message": "Animated character ready"})
            except Cancelled as exc:
                self.update(job_id, {"status": "interrupted" if self._stop.is_set() else "cancelled", "message": str(exc)})
            except Exception as exc:
                message = str(exc)[:2000] or type(exc).__name__
                with (self.settings.jobs_dir / job_id / "job.log").open("a", encoding="utf-8") as log:
                    log.write(f"\n[error] {message}\n")
                self.update(job_id, {"status": "failed", "error": message, "message": "Reconstruction stopped. See the log, correct the problem, then resume."})
            finally:
                self._active = None
                self._wake.set()
