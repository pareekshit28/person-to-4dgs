import errno
from pathlib import Path
import sys

import pytest

from scripts import check_storage as storage


MOUNTS = """25 1 0:1 / / rw - overlay overlay rw
36 25 0:32 / /workspace rw,nosuid,nodev - fuse.geesefs geesefs rw
37 36 0:33 / /workspace/local rw - ext4 /dev/nvme0n1 rw
38 25 0:34 / /mnt/training\\040set rw shared:9 - nfs server:/training\\040set rw
"""


def test_mountinfo_decodes_spaces_and_prefers_nested_mounts():
    mounts = storage.parse_mountinfo(MOUNTS + "not a mountinfo line\n")
    assert len(mounts) == 4
    assert storage.find_mount("/mnt/training set/jobs/one", mounts) == storage.Mount(
        "/mnt/training set", "nfs", "server:/training set"
    )
    assert storage.find_mount("/workspace/local/app", mounts).filesystem == "ext4"
    assert storage.find_mount("/workspace-other", mounts).filesystem == "overlay"
    overmounted = mounts + [storage.Mount("/workspace/local", "fuse.geesefs", "geesefs")]
    assert storage.find_mount("/workspace/local/app", overmounted).filesystem == "fuse.geesefs"


def test_geesefs_rejection_explains_usable_runpod_storage():
    mounts = storage.parse_mountinfo(MOUNTS)
    with pytest.raises(storage.StorageError) as caught:
        storage.reject_object_storage("/workspace/person4d", "Active data", mounts)
    message = str(caught.value)
    assert "fuse.geesefs" in message
    assert "Network Volume" in message and "Volume Disk" in message and "not a Global Volume" in message
    storage.reject_object_storage("/workspace/local/app", "Runtime", mounts)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux filesystem capabilities")
def test_real_posix_storage_checks_leave_no_probe_files(tmp_path):
    runtime, data = tmp_path / "runtime", tmp_path / "data"
    storage.check_storage(runtime, data)
    assert list(runtime.iterdir()) == []
    assert list(data.iterdir()) == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux filesystem capabilities")
def test_rejected_data_mount_is_detected_before_runtime_installation(tmp_path):
    runtime, data = tmp_path / "runtime", tmp_path / "data"
    fixture = f"25 1 0:1 / / rw - overlay overlay rw\n36 25 0:32 / {data} rw - fuse.geesefs geesefs rw\n"
    with pytest.raises(storage.StorageError, match="Active data"):
        storage.check_storage(runtime, data, mountinfo=fixture)
    assert not runtime.exists() and not data.exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux filesystem capabilities")
def test_permission_error_is_actionable_and_probe_is_cleaned(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(storage.os, "chmod", denied)
    with pytest.raises(storage.StorageError, match="chmod and symlinks") as caught:
        storage.probe_runtime(tmp_path)
    assert "Global Volume" not in str(caught.value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux executable and symlink semantics")
def test_normalized_network_permissions_allow_runtime_execution(tmp_path, monkeypatch):
    real_chmod = storage.os.chmod
    observed_modes = []

    def normalized(path, requested_mode, *args, **kwargs):
        real_chmod(path, 0o777, *args, **kwargs)
        observed_modes.append(Path(path).stat().st_mode & 0o777)

    monkeypatch.setattr(storage.os, "chmod", normalized)
    storage.check_storage(tmp_path / "runtime", tmp_path / "data")
    assert observed_modes and all(mode == 0o777 for mode in observed_modes)
    assert list((tmp_path / "runtime").iterdir()) == []
    assert list((tmp_path / "data").iterdir()) == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux execution errors")
def test_noexec_error_identifies_execution_without_misclassifying_volume(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(storage.subprocess, "run", denied)
    with pytest.raises(storage.StorageError, match="cannot execute a test script") as caught:
        storage.probe_runtime(tmp_path)
    assert "noexec" in str(caught.value)
    assert "Global Volume" not in str(caught.value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux filesystem capabilities")
def test_replacement_error_is_actionable_and_probe_is_cleaned(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "Operation not supported")

    monkeypatch.setattr(storage.os, "replace", denied)
    with pytest.raises(storage.StorageError, match="file locks and replacement") as caught:
        storage.probe_data(tmp_path)
    assert "Global Volume" not in str(caught.value)
    assert list(tmp_path.iterdir()) == []


def test_cli_reports_short_error_without_traceback(tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise storage.StorageError("Known storage restriction")

    monkeypatch.setattr(storage, "check_storage", fail)
    assert storage.main(["--runtime", str(tmp_path), "--data", str(tmp_path)]) == 1
    result = capsys.readouterr()
    assert result.out == ""
    assert result.err.strip() == "Storage check failed: Known storage restriction"
