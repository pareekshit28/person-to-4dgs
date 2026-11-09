import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from person4d.api import create_app
from person4d.config import Settings
from person4d.jobs import JobManager


class IdlePipeline:
    def run(self, job, update, cancel):
        return


@pytest.fixture
def app_setup(tmp_path):
    root = tmp_path / "app"
    (root / "person4d/static").mkdir(parents=True)
    (root / "person4d/static/index.html").write_text("Person4D")
    settings = Settings(root=root, data_dir=tmp_path / "data", token="test-secret", upload_limit=128)
    manager = JobManager(settings, IdlePipeline())
    return settings, manager, create_app(settings, manager)


def test_auth_precedes_file_access_and_cookie_login(app_setup):
    _, _, app = app_setup
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/jobs").status_code == 401
        assert client.get("/api/auth").json() == {"authenticated": False, "token_required": True}
        assert client.post("/api/login", json={"token": "wrong"}).status_code == 401
        logged_in = client.post("/api/login", json={"token": "test-secret"})
        assert logged_in.status_code == 200
        assert "httponly" in logged_in.headers["set-cookie"].lower()
        assert client.get("/api/jobs").status_code == 200
        assert client.post("/api/setup/models", headers={"Origin": "https://attacker.example"}).status_code == 403
        assert client.post("/api/logout", headers={"Origin": "http://testserver"}).status_code == 200
        assert client.get("/api/jobs").status_code == 401


def test_runpod_proxy_origin_with_rewritten_host(app_setup, monkeypatch):
    settings, manager, _ = app_setup
    monkeypatch.setenv("RUNPOD_POD_ID", "testpod")
    app = create_app(settings, manager)
    origin = f"https://testpod-{settings.port}.proxy.runpod.net"
    with TestClient(app, base_url=f"http://10.0.0.2:{settings.port}") as client:
        logged_in = client.post("/api/login", headers={"Origin": origin}, json={"token": "test-secret"})
        assert logged_in.status_code == 200
        cookie = logged_in.headers["set-cookie"].lower()
        assert "; secure" in cookie and "samesite=strict" in cookie

        # The public HTTPS connection supplies the Secure cookie to the proxy.
        session = {"Cookie": f"person4d_session={logged_in.cookies['person4d_session']}"}
        assert client.get("/api/auth", headers=session).json()["authenticated"] is True
        for untrusted_origin in (
            f"https://otherpod-{settings.port}.proxy.runpod.net",
            f"https://testpod-{settings.port + 1}.proxy.runpod.net",
            origin.replace("https://", "http://"),
            "https://attacker.example",
        ):
            headers = {**session, "Origin": untrusted_origin, "X-Forwarded-Host": untrusted_origin.split("://")[1]}
            assert client.post("/api/logout", headers=headers).status_code == 403
        assert client.post("/api/logout", headers={**session, "Origin": origin}).status_code == 200


def test_artifact_range_and_traversal_restriction(app_setup):
    settings, manager, app = app_setup
    with TestClient(app) as client:
        client.headers["Authorization"] = "Bearer test-secret"
        job = manager.create("abc", title="Test", filename="person.mp4", relative_input="input/source.mp4", options={}, sha256="digest")
        path = settings.jobs_dir / "abc/input/source.mp4"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"0123456789")
        response = client.get(job["input"]["url"], headers={"Range": "bytes=2-5"})
        assert response.status_code == 206
        assert response.content == b"2345"
        assert client.get("/api/jobs/abc/files/job.json").status_code == 404
        assert client.get("/api/jobs/abc/files/..%2F..%2Fsetup.json").status_code == 404


def test_upload_is_blocked_until_setup_is_ready(app_setup):
    _, _, app = app_setup
    with TestClient(app) as client:
        response = client.post("/api/jobs", headers={"Authorization": "Bearer test-secret"}, files={"file": ("person.mp4", b"video", "video/mp4")})
        assert response.status_code == 409
        assert "Complete setup first" in response.json()["detail"]


def test_upload_limit_and_path_safe_input(app_setup, monkeypatch):
    settings, _, app = app_setup
    monkeypatch.setattr("person4d.api.system_state", lambda *_: {"readiness": True})
    with TestClient(app) as client:
        client.headers["Authorization"] = "Bearer test-secret"
        response = client.post("/api/jobs", files={"file": ("../../person.mp4", b"video", "video/mp4")}, data={"preset": "quality", "start_time": "0", "target_fps": "24"})
        assert response.status_code == 201
        job = response.json()
        assert job["input"]["filename"] == "person.mp4"
        assert (settings.jobs_dir / job["id"] / "input/source.mp4").read_bytes() == b"video"
        large = client.post("/api/jobs", files={"file": ("person.mp4", b"x" * 129, "video/mp4")})
        assert large.status_code == 413
        assert len(list(settings.jobs_dir.glob("*/job.json"))) == 1
