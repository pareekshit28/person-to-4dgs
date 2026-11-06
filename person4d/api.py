from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .config import MODEL_FILES, PRESETS, SMPLX_FILE, Settings
from .jobs import JobManager, artifacts_for, utc_now
from .pipeline import Pipeline, atomic_json, run_command
from .schemas import JobOptions, Login


def gpu_info() -> dict | None:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4, check=True)
        name, total, used = result.stdout.splitlines()[0].rsplit(",", 2)
        return {"name": name.strip(), "memory_total_mb": int(total.strip()), "memory_used_mb": int(used.strip())}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def system_state(settings: Settings, manager: JobManager) -> dict:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    gpu = gpu_info()
    missing_models = [name for name in MODEL_FILES if not (settings.models_dir / name).is_file()]
    smplx_ready = (settings.models_dir / SMPLX_FILE).is_file()
    checks = [
        {"name": "GPU", "ok": gpu is not None, "message": gpu["name"] if gpu else "No NVIDIA GPU detected"},
        {"name": "Inference environment", "ok": settings.inference_python.is_file(), "message": str(settings.inference_python)},
        {"name": "Training environment", "ok": settings.training_python.is_file(), "message": str(settings.training_python)},
        {"name": "Video tools", "ok": bool(shutil.which("ffprobe") and shutil.which("ffmpeg")), "message": "ffmpeg and ffprobe are required"},
        {"name": "4DAnyone source", "ok": (settings.inference_repo / "inference.py").is_file(), "message": "Vendored inference source"},
        {"name": "Motion recovery source", "ok": (settings.inference_repo / "third_party/GVHMR/hmr4d").is_dir(), "message": "Pinned GVHMR checkout"},
        {"name": "FreeTimeGS source", "ok": (settings.training_repo / "src/simple_trainer_freetime_4d_pure_relocation.py").is_file(), "message": "Vendored reconstruction source"},
        {"name": "Public models", "ok": not missing_models, "message": "Downloaded" if not missing_models else f"{len(missing_models)} model files remain"},
        {"name": "SMPL-X model", "ok": smplx_ready, "message": "Installed" if smplx_ready else "Upload your separately licensed SMPL-X model"},
    ]
    disk = shutil.disk_usage(settings.data_dir)
    warnings = []
    if gpu and gpu["memory_total_mb"] < 75000:
        warnings.append("The high-detail preset targets an 80 GB or larger GPU. It may exhaust memory on this GPU; use a smaller preset if necessary.")
    if disk.free < 60 * 1024**3:
        warnings.append("Less than 60 GB is free. Public model downloads and reconstruction data require substantial disk space.")
    return {"gpu": gpu, "checks": checks, "readiness": all(item["ok"] for item in checks), "model_smplx_ready": smplx_ready, "models_ready": not missing_models, "missing_models": missing_models, "disk": {"free_bytes": disk.free, "total_bytes": disk.total}, "busy": manager.busy, "presets": PRESETS, "warnings": warnings, "data_dir": str(settings.data_dir), "frames": 121}


def create_app(settings: Settings | None = None, manager: JobManager | None = None) -> FastAPI:
    settings = settings or Settings()
    manager = manager or JobManager(settings)
    setup_lock = threading.Lock()
    setup_stop = threading.Event()
    setup_thread: threading.Thread | None = None
    setup_path = settings.data_dir / "setup.json"
    setup_state = {"state": "idle", "kind": None, "message": "", "error": None, "log": ""}
    state_lock = threading.Lock()
    session_value = hmac.new(settings.token.encode(), b"person4d-session-v1", hashlib.sha256).hexdigest()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        if setup_path.is_file():
            try:
                setup_state.update(json.loads(setup_path.read_text(encoding="utf-8")))
                if setup_state["state"] == "running":
                    setup_state.update(state="failed", error="Server stopped during setup. Start setup again to continue downloading.", message="Setup interrupted")
            except (OSError, ValueError):
                pass
        manager.start()
        app.state.manager = manager
        yield
        setup_stop.set()
        manager.close()
        if setup_thread:
            setup_thread.join(timeout=12)

    app = FastAPI(title="Person4D", version="0.1.0", lifespan=lifespan, docs_url=None, redoc_url=None)

    def authenticated(request: Request) -> bool:
        if not settings.token:
            return True
        bearer = request.headers.get("authorization", "")
        return (bearer.startswith("Bearer ") and hmac.compare_digest(bearer[7:], settings.token)) or hmac.compare_digest(request.cookies.get("person4d_session", ""), session_value)

    @app.middleware("http")
    async def access_control(request: Request, call_next):
        public = request.url.path in ("/api/auth", "/api/login", "/healthz") or not (request.url.path.startswith("/api/") or request.url.path.startswith("/player/"))
        if not public and not authenticated(request):
            return JSONResponse({"detail": "Sign in with the access token from your RunPod terminal."}, status_code=401)
        # A cookie authenticates same-origin browser use. Cross-site mutation is denied.
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin:
                parsed = urlparse(origin)
                if parsed.netloc.lower() != request.headers.get("host", "").lower() or parsed.scheme not in ("http", "https"):
                    return JSONResponse({"detail": "Cross-origin requests are not allowed."}, status_code=403)
            try:
                content_length = int(request.headers.get("content-length", "0"))
            except ValueError:
                return JSONResponse({"detail": "Invalid content length"}, status_code=400)
            if content_length > settings.upload_limit + 1024 * 1024:
                return JSONResponse({"detail": f"Upload exceeds the {settings.upload_limit // 1024**2} MB limit."}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/api/auth")
    def auth(request: Request):
        return {"authenticated": authenticated(request), "token_required": bool(settings.token)}

    @app.post("/api/login")
    def login(payload: Login, request: Request):
        if settings.token and not hmac.compare_digest(payload.token, settings.token):
            raise HTTPException(401, "Incorrect access token")
        response = JSONResponse({"authenticated": True})
        forwarded = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
        response.set_cookie("person4d_session", session_value, httponly=True, secure=request.url.scheme == "https" or forwarded == "https", samesite="strict", max_age=604800, path="/")
        return response

    @app.post("/api/logout")
    def logout():
        response = JSONResponse({"authenticated": not bool(settings.token)})
        response.delete_cookie("person4d_session", path="/")
        return response

    @app.get("/api/system")
    def system():
        return system_state(settings, manager)

    @app.get("/api/setup/status")
    def setup_status():
        with state_lock:
            state = dict(setup_state)
        path = settings.data_dir / "setup.log"
        if path.is_file():
            with path.open("rb") as handle:
                handle.seek(max(0, path.stat().st_size - 48000))
                state["log"] = handle.read().decode("utf-8", errors="replace")
        return state

    def save_setup(**values):
        with state_lock:
            setup_state.update(values, updated_at=utc_now())
            atomic_json(setup_path, {k: v for k, v in setup_state.items() if k != "log"})

    def launch_setup(kind: str, source: Path | None = None):
        nonlocal setup_thread
        if not setup_lock.acquire(blocking=False):
            raise HTTPException(409, "Another setup task is running")
        if not settings.inference_python.is_file():
            setup_lock.release()
            raise HTTPException(409, "Install the environments with scripts/setup_runpod.sh first")
        save_setup(state="running", kind=kind, message="Downloading public models" if kind == "models" else "Installing your licensed SMPL-X model", error=None)

        def task():
            try:
                with manager.runtime_lock:
                    command = [str(settings.inference_python), "scripts/download_model.py" if kind == "models" else "scripts/download_smplx.py", "--model_dir", str(settings.models_dir), "--gvhmr_root", str(settings.inference_repo / "third_party/GVHMR")]
                    if source:
                        command += ["--archive_path", str(source)]
                    run_command(command, cwd=settings.inference_repo, log_path=settings.data_dir / "setup.log", cancel=setup_stop, env=Pipeline(settings).environment(True))
                save_setup(state="completed", message="Setup completed", error=None)
            except Exception as exc:
                save_setup(state="failed", message="Setup stopped", error=str(exc)[:2000])
            finally:
                if source:
                    source.unlink(missing_ok=True)
                setup_lock.release()

        setup_thread = threading.Thread(target=task, daemon=True, name="person4d-setup")
        setup_thread.start()
        return setup_status()

    @app.post("/api/setup/models")
    def setup_models():
        return launch_setup("models")

    async def write_upload(upload: UploadFile, destination: Path) -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        digest = hashlib.sha256()
        try:
            with destination.open("wb") as output:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.upload_limit:
                        raise HTTPException(413, "Uploaded file is too large")
                    digest.update(chunk)
                    output.write(chunk)
            if not size:
                raise HTTPException(400, "The uploaded file is empty")
            return digest.hexdigest()
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

    @app.post("/api/setup/smplx")
    async def setup_smplx(file: UploadFile = File(...)):
        filename = Path((file.filename or "").replace("\\", "/")).name
        if filename not in ("SMPLX_NEUTRAL.npz", "models_smplx_v1_1.zip"):
            raise HTTPException(400, "Upload SMPLX_NEUTRAL.npz or models_smplx_v1_1.zip from your licensed SMPL-X download")
        source = settings.data_dir / "setup-uploads" / uuid.uuid4().hex / filename
        await write_upload(file, source)
        try:
            return launch_setup("smplx", source)
        except Exception:
            source.unlink(missing_ok=True)
            raise

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": manager.list()}

    @app.post("/api/jobs", status_code=201)
    async def create_job(file: UploadFile = File(...), title: str = Form(""), preset: str = Form("quality"), start_time: float = Form(0), target_fps: int = Form(24), steps: int | None = Form(None), seed: int = Form(42)):
        try:
            options = JobOptions(preset=preset, start_time=start_time, target_fps=target_fps, steps=steps, seed=seed)
        except ValidationError as exc:
            raise HTTPException(422, str(exc)) from exc
        state = system_state(settings, manager)
        if not state["readiness"]:
            missing = ", ".join(item["name"] for item in state["checks"] if not item["ok"])
            raise HTTPException(409, f"Complete setup first: {missing}")
        filename = Path((file.filename or "video.mp4").replace("\\", "/")).name[:200]
        suffix = Path(filename).suffix.lower()
        if suffix not in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"):
            raise HTTPException(400, "Upload an MP4, MOV, MKV, WebM, AVI or M4V video")
        job_id = uuid.uuid4().hex
        relative = f"input/source{suffix}"
        source = settings.jobs_dir / job_id / relative
        try:
            digest = await write_upload(file, source)
            return manager.create(job_id, title=title, filename=filename, relative_input=relative, options=options.model_dump(), sha256=digest)
        except Exception:
            # This directory belongs exclusively to this unpublished upload.
            shutil.rmtree(settings.jobs_dir / job_id, ignore_errors=True)
            raise

    def get_job(job_id: str) -> dict:
        try:
            return manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(404, "Job not found") from exc

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str):
        return get_job(job_id)

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str):
        get_job(job_id)
        return manager.cancel(job_id)

    @app.post("/api/jobs/{job_id}/retry")
    @app.post("/api/jobs/{job_id}/resume")
    def job_retry(job_id: str):
        get_job(job_id)
        try:
            return manager.retry(job_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/jobs/{job_id}/logs")
    def logs(job_id: str, offset: int = 0):
        get_job(job_id)
        if offset < 0:
            raise HTTPException(422, "Offset must be nonnegative")
        path = settings.jobs_dir / job_id / "job.log"
        if not path.is_file():
            return {"text": "", "next_offset": 0}
        with path.open("rb") as handle:
            handle.seek(min(offset, path.stat().st_size))
            chunk = handle.read(256 * 1024)
            next_offset = handle.tell()
        return {"text": chunk.decode("utf-8", errors="replace"), "next_offset": next_offset}

    @app.get("/api/jobs/{job_id}/artifacts")
    def artifact_list(job_id: str):
        get_job(job_id)
        return {"artifacts": artifacts_for(settings, job_id)}

    @app.get("/api/jobs/{job_id}/files/{relative:path}")
    @app.head("/api/jobs/{job_id}/files/{relative:path}")
    def artifact_file(job_id: str, relative: str):
        job = get_job(job_id)
        root = (settings.jobs_dir / job_id).resolve()
        path = (root / relative).resolve()
        allowed = {item["path"] for item in job["artifacts"]} | {job["input"]["path"]}
        if not path.is_relative_to(root) or relative not in allowed or not path.is_file():
            raise HTTPException(404, "Artifact not found")
        return FileResponse(path, filename=path.name, content_disposition_type="inline")

    player = settings.training_repo / "player"
    if player.is_dir():
        app.mount("/player", StaticFiles(directory=player), name="player")
    static = settings.root / "person4d/static"
    if static.is_dir():
        app.mount("/", StaticFiles(directory=static, html=True), name="ui")
    return app
