import json
import threading
import time

import pytest

from person4d.config import Settings
from person4d.jobs import JobManager
from person4d.pipeline import Cancelled, atomic_json


def wait_for(predicate, timeout=5):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for worker")


class ControlledPipeline:
    def __init__(self):
        self.started = threading.Event()
        self.finish = threading.Event()
        self.ids = []

    def run(self, job, update, cancel):
        self.ids.append(job["id"])
        self.started.set()
        update({"stage": "train", "stage_progress": 50})
        while not self.finish.wait(0.01):
            if cancel.is_set():
                raise Cancelled("Cancelled")


def make_job(manager, job_id):
    return manager.create(job_id, title="Test", filename="test.mp4", relative_input="input/source.mp4", options={"preset": "preview", "start_time": 0, "target_fps": 24, "steps": 1000, "seed": 42}, sha256="abc")


def test_queue_is_serial_and_cancel_keeps_job_retryable(tmp_path):
    pipeline = ControlledPipeline()
    manager = JobManager(Settings(root=tmp_path, data_dir=tmp_path / "data"), pipeline)
    manager.start()
    try:
        make_job(manager, "first")
        assert pipeline.started.wait(2)
        make_job(manager, "second")
        assert pipeline.ids == ["first"]
        assert manager.get("first")["progress"] == 76
        manager.cancel("first")
        wait_for(lambda: manager.get("first")["status"] == "cancelled")
        wait_for(lambda: "second" in pipeline.ids)
        pipeline.finish.set()
        wait_for(lambda: manager.get("second")["status"] == "completed")
        manager.retry("first")
        wait_for(lambda: manager.get("first")["status"] == "completed")
        assert pipeline.ids == ["first", "second", "first"]
    finally:
        pipeline.finish.set()
        manager.close()


def test_restart_marks_running_interrupted_and_preserves_options(tmp_path):
    settings = Settings(root=tmp_path, data_dir=tmp_path / "data")
    manager = JobManager(settings, ControlledPipeline())
    job = make_job(manager, "recover")
    job["status"] = "running"
    atomic_json(settings.jobs_dir / "recover/job.json", job)
    restarted = JobManager(settings, ControlledPipeline())
    restarted.start()
    try:
        recovered = restarted.get("recover")
        assert recovered["status"] == "interrupted"
        assert recovered["options"] == job["options"]
        with pytest.raises(KeyError):
            restarted.retry("missing")
    finally:
        restarted.close()


def test_second_server_cannot_take_over_data_directory(tmp_path):
    settings = Settings(root=tmp_path, data_dir=tmp_path / "data")
    first = JobManager(settings, ControlledPipeline())
    second = JobManager(settings, ControlledPipeline())
    first.start()
    try:
        with pytest.raises(RuntimeError, match="Another Person4D server"):
            second.start()
    finally:
        first.close()


def test_delete_rejects_running_queued_and_open_downloads(tmp_path):
    pipeline = ControlledPipeline()
    manager = JobManager(Settings(root=tmp_path, data_dir=tmp_path / "data"), pipeline)
    manager.start()
    try:
        make_job(manager, "active")
        assert pipeline.started.wait(2)
        make_job(manager, "queued")
        for job_id in ("active", "queued"):
            with pytest.raises(ValueError, match="Stop this run"):
                manager.delete(job_id)
        manager.cancel("queued")
        with manager.read_files("queued"):
            with pytest.raises(ValueError, match="downloads"):
                manager.delete("queued")
        manager.delete("queued")
        assert [job["id"] for job in manager.list()] == ["active"]
    finally:
        pipeline.finish.set()
        manager.close()


def test_interrupted_delete_stays_out_of_history_and_reclaims_files_on_restart(tmp_path, monkeypatch):
    settings = Settings(root=tmp_path, data_dir=tmp_path / "data")
    manager = JobManager(settings, ControlledPipeline())
    make_job(manager, "recover")
    manager.cancel("recover")
    root = settings.jobs_dir / "recover"
    (root / "job.log").write_text("saved log")

    def failed_purge(path):
        (path / "job.log").unlink()
        raise OSError("Storage temporarily unavailable")

    with monkeypatch.context() as patch:
        patch.setattr("person4d.jobs.shutil.rmtree", failed_purge)
        with pytest.raises(OSError, match="Storage temporarily unavailable"):
            manager.delete("recover")
    assert not root.exists()
    assert manager.list() == []
    deleted = settings.jobs_dir / ".deleted/recover"
    assert (deleted / "job.json").is_file()

    restarted = JobManager(settings, ControlledPipeline())
    restarted.start()
    try:
        assert restarted.list() == []
        assert not deleted.exists()
    finally:
        restarted.close()
