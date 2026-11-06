"""Check that active installation and job storage has filesystem semantics."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile


REMEDY = (
    "Use a RunPod Volume Disk or Network Volume for the installation and active "
    "job data, not a Global Volume. Global Volumes can hold copied final artifacts."
)
OBJECT_FILESYSTEMS = {
    "geesefs", "fuse.geesefs", "goofys", "fuse.goofys", "s3fs", "fuse.s3fs",
    "mountpoint-s3", "fuse.mountpoint-s3", "gcsfuse", "fuse.gcsfuse",
}


class StorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class Mount:
    path: str
    filesystem: str
    source: str


def _unescape(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def parse_mountinfo(text: str) -> list[Mount]:
    """Parse Linux mountinfo, including escaped whitespace in mount paths."""
    mounts = []
    for line in text.splitlines():
        before, separator, after = line.partition(" - ")
        left, right = before.split(), after.split()
        if not separator or len(left) < 6 or len(right) < 2:
            continue
        mounts.append(Mount(_unescape(left[4]), right[0], _unescape(right[1])))
    return mounts


def find_mount(path: str | Path, mounts: list[Mount]) -> Mount | None:
    """Find the containing mount, preferring a nested mount over its parent."""
    target = PurePosixPath(str(path))
    # A later record at the same mount path supersedes an earlier overmounted one.
    candidates = [mount for mount in reversed(mounts) if target.is_relative_to(PurePosixPath(mount.path))]
    return max(candidates, key=lambda mount: len(PurePosixPath(mount.path).parts), default=None)


def reject_object_storage(path: str | Path, role: str, mounts: list[Mount]) -> None:
    mount = find_mount(path, mounts)
    if mount and mount.filesystem.casefold() in OBJECT_FILESYSTEMS:
        raise StorageError(
            f"{role} path {path} is on {mount.filesystem} at {mount.path}. "
            f"This object-backed mount cannot run Git/Python environments or safely "
            f"maintain the app's locks and atomic job/checkpoint updates. {REMEDY}"
        )


def probe_runtime(path: Path) -> None:
    """Exercise permissions and symlinks inside one disposable owned directory."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".person4d-storage-check-", dir=path) as temporary:
            root = Path(temporary)
            target = root / "permission-check"
            target.write_bytes(b"runtime-check")
            for mode in (0o700, 0o600):
                os.chmod(target, mode)
                if stat.S_IMODE(target.stat().st_mode) != mode:
                    raise StorageError(f"Runtime storage at {path} does not preserve permission bits. {REMEDY}")
            link = root / "symlink-check"
            link.symlink_to(target.name)
            if not link.is_symlink() or link.read_bytes() != b"runtime-check":
                raise StorageError(f"Runtime storage at {path} does not support working symlinks. {REMEDY}")
    except OSError as exc:
        raise StorageError(
            f"Runtime storage at {path} must support chmod and symlinks: {exc.strerror or exc}. {REMEDY}"
        ) from None


def probe_data(path: Path) -> None:
    """Exercise advisory locking and same-directory replacement for active jobs."""
    import fcntl

    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".person4d-storage-check-", dir=path) as temporary:
            root = Path(temporary)
            with (root / "lock-check").open("w+b") as first, (root / "lock-check").open("r+b") as second:
                fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
                try:
                    fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    fcntl.flock(second, fcntl.LOCK_UN)
                    raise StorageError(f"Active data storage at {path} does not enforce file locks. {REMEDY}")
                fcntl.flock(first, fcntl.LOCK_UN)
                fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(second, fcntl.LOCK_UN)
            destination, replacement = root / "state.json", root / "state.tmp"
            destination.write_bytes(b"old-state")
            replacement.write_bytes(b"new-state")
            os.replace(replacement, destination)
            if destination.read_bytes() != b"new-state" or replacement.exists():
                raise StorageError(f"Active data storage at {path} does not support file replacement. {REMEDY}")
    except OSError as exc:
        raise StorageError(
            f"Active data storage at {path} must support file locks and replacement: {exc.strerror or exc}. {REMEDY}"
        ) from None


def check_storage(runtime: str | Path, data: str | Path, *, mountinfo: str | None = None) -> None:
    if not sys.platform.startswith("linux"):
        raise StorageError("RunPod storage checks require Linux.")
    runtime_path, data_path = Path(runtime).resolve(), Path(data).resolve()
    if mountinfo is None:
        try:
            mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
        except OSError as exc:
            raise StorageError(f"Cannot inspect Linux storage mounts: {exc.strerror or exc}") from None
    mounts = parse_mountinfo(mountinfo)
    if not mounts:
        raise StorageError("Cannot identify the storage mounts; check /proc/self/mountinfo before setup.")
    # Reject known object storage before creating directories or writing probes.
    reject_object_storage(runtime_path, "Runtime", mounts)
    reject_object_storage(data_path, "Active data", mounts)
    probe_runtime(runtime_path)
    probe_data(data_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True, help="Application/environment directory")
    parser.add_argument("--data", type=Path, required=True, help="Models and active job data directory")
    args = parser.parse_args(argv)
    try:
        check_storage(args.runtime, args.data)
    except StorageError as exc:
        print(f"Storage check failed: {exc}", file=sys.stderr)
        return 1
    print(f"Storage checks passed: runtime={args.runtime.resolve()}, data={args.data.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
