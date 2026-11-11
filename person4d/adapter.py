"""Convert a completed 4DAnyone result to the FreeTimeGsVanilla dataset.

The control server deliberately launches this module in the 4DAnyone environment.
GPU libraries are imported only when their stage starts, so geometry contracts and
serialization can also be tested on a machine without CUDA.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from fractions import Fraction
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from typing import Callable, Iterator

import numpy as np
from PIL import Image


ADAPTER_VERSION = 2
EXPECTED_FRAMES = 121


def progress(message: str, value: float) -> None:
    print(json.dumps({"event": "progress", "stage": "adapter", "message": message,
                      "progress": round(max(0.0, min(100.0, value)), 2)}), flush=True)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def dataset_lock(destination: Path):
    """Exclude concurrent writers without treating a stale PID as a live lock."""
    lock_path = destination.parent / f".{destination.name}.adapter.lock"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another adapter is writing this dataset.") from exc
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another adapter is writing this dataset.") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def validate_cameras(payload: dict) -> list[dict]:
    """Validate the released OpenCV rig; never silently flip an axis."""
    if payload.get("camera_model") != "OPENCV":
        raise ValueError("Expected an OPENCV cameras.json from the pinned 4DAnyone release.")
    frame = payload.get("camera_frame", {})
    if frame and frame.get("name") != "opencv_camera":
        raise ValueError("Unsupported camera frame: expected OpenCV right/down/forward axes.")
    records = payload.get("cameras")
    if not isinstance(records, list) or len(records) < 3:
        raise ValueError("At least three calibrated target cameras are required.")
    if [camera.get("camera_id") for camera in records] != list(range(len(records))):
        raise ValueError("Camera IDs must be contiguous, ordered and zero-based.")
    sizes = set()
    for camera in records:
        label = f"camera {camera['camera_id']:02d}"
        intrinsic = np.asarray(camera.get("K"), dtype=np.float64)
        pose = np.asarray(camera.get("camera_to_world"), dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError(f"Invalid intrinsic matrix for {label}.")
        if (intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                or not np.allclose(intrinsic[2], [0, 0, 1])
                or not np.allclose([intrinsic[0, 1], intrinsic[1, 0]], 0)):
            raise ValueError(f"{label} needs valid unskewed PINHOLE intrinsics.")
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError(f"Invalid homogeneous camera_to_world for {label}.")
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3):
            raise ValueError(f"Camera rotation for {label} must be right-handed and orthonormal.")
        width, height = int(camera.get("image_width", 0)), int(camera.get("image_height", 0))
        if min(width, height) < 2:
            raise ValueError(f"Invalid image dimensions for {label}.")
        sizes.add((width, height))
    if len(sizes) != 1:
        raise ValueError("Target cameras must share a raster size for synchronized segmentation.")
    return records


def projection_matrices(cameras: list[dict]) -> np.ndarray:
    return np.stack([np.asarray(camera["K"], dtype=np.float64)
                     @ np.linalg.inv(np.asarray(camera["camera_to_world"], dtype=np.float64))[:3]
                     for camera in cameras])


def rotation_to_qvec(rotation: np.ndarray) -> np.ndarray:
    """Return COLMAP's wxyz quaternion for a world-to-camera rotation."""
    r = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(r)
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        q = np.array([scale / 4, (r[2, 1] - r[1, 2]) / scale,
                      (r[0, 2] - r[2, 0]) / scale, (r[1, 0] - r[0, 1]) / scale])
    else:
        axis = int(np.argmax(np.diag(r)))
        if axis == 0:
            scale = math.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q = np.array([(r[2, 1] - r[1, 2]) / scale, scale / 4,
                          (r[0, 1] + r[1, 0]) / scale, (r[0, 2] + r[2, 0]) / scale])
        elif axis == 1:
            scale = math.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q = np.array([(r[0, 2] - r[2, 0]) / scale, (r[0, 1] + r[1, 0]) / scale,
                          scale / 4, (r[1, 2] + r[2, 1]) / scale])
        else:
            scale = math.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q = np.array([(r[1, 0] - r[0, 1]) / scale, (r[0, 2] + r[2, 0]) / scale,
                          (r[1, 2] + r[2, 1]) / scale, scale / 4])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def write_colmap(directory: Path, cameras: list[dict]) -> None:
    """Write known calibration directly; no camera estimation or axis flip.

    Standard little-endian binaries are authoritative. The pinned legacy
    pycolmap fork cannot parse valid blank POINTS2D lines in a text-only model.
    Text sidecars are supplied for inspection, never relied upon by training.
    """
    directory.mkdir(parents=True, exist_ok=True)
    camera_lines = ["# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]"]
    image_lines = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME", "# POINTS2D[]"]
    for camera in cameras:
        camera_id = int(camera["camera_id"]) + 1
        k = np.asarray(camera["K"])
        parameters = " ".join(format(float(value), ".17g") for value in (k[0, 0], k[1, 1], k[0, 2], k[1, 2]))
        camera_lines.append(f"{camera_id} PINHOLE {camera['image_width']} {camera['image_height']} {parameters}")
        w2c = np.linalg.inv(np.asarray(camera["camera_to_world"], dtype=np.float64))
        pose = " ".join(format(float(value), ".17g") for value in (*rotation_to_qvec(w2c[:3, :3]), *w2c[:3, 3]))
        image_lines.extend([f"{camera_id} {pose} {camera_id} cam{camera_id - 1:02d}/000000.png", ""])
    (directory / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")
    (directory / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    # Camera normalization works with an empty reference cloud. All 4D geometry
    # comes from init.npz; skipping reference-cloud PCA avoids arbitrary axes.
    (directory / "points3D.txt").write_text("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n", encoding="utf-8")
    with (directory / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            intrinsic = np.asarray(camera["K"])
            handle.write(struct.pack("<IiQQ", int(camera["camera_id"]) + 1, 1,
                                     int(camera["image_width"]), int(camera["image_height"])))
            handle.write(struct.pack("<4d", intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]))
    with (directory / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            camera_id = int(camera["camera_id"]) + 1
            w2c = np.linalg.inv(np.asarray(camera["camera_to_world"], dtype=np.float64))
            handle.write(struct.pack("<I4d3dI", camera_id, *rotation_to_qvec(w2c[:3, :3]), *w2c[:3, 3], camera_id))
            handle.write(f"cam{camera_id - 1:02d}/000000.png".encode("utf-8") + b"\0")
            handle.write(struct.pack("<Q", 0))  # This reference image has no SfM observations.
    (directory / "points3D.bin").write_bytes(struct.pack("<Q", 0))


def keyframe_indices(num_frames: int, step: int) -> list[int]:
    if num_frames < 2 or step < 1:
        raise ValueError("At least two frames and a positive keyframe step are required.")
    return sorted(set(range(0, num_frames, step)) | {num_frames - 1})


def point_budgets(max_points: int, frames: list[int]) -> list[int]:
    if max_points < len(frames):
        raise ValueError("Point budget must include at least one point at every seed timestamp.")
    quotient, remainder = divmod(max_points, len(frames))
    return [quotient + (index < remainder) for index in range(len(frames))]


def write_initialization(path: Path, clouds: list[tuple[int, np.ndarray, np.ndarray]],
                         num_frames: int, step: int) -> int:
    denominator = max(num_frames - 1, 1)
    positions, colors, times, durations = [], [], [], []
    for frame, xyz, rgb in clouds:
        xyz, rgb = np.asarray(xyz, dtype=np.float32), np.asarray(rgb, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape or not len(xyz):
            raise ValueError(f"Invalid geometry seed at frame {frame}.")
        if not np.isfinite(xyz).all() or not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
            raise ValueError(f"Non-finite geometry or RGB outside [0,1] at frame {frame}.")
        if not 0 <= frame < num_frames:
            raise ValueError("Geometry timestamp is outside the source sequence.")
        positions.append(xyz)
        colors.append(rgb)
        times.append(np.full((len(xyz), 1), frame / denominator, dtype=np.float32))
        durations.append(np.full((len(xyz), 1), step / denominator, dtype=np.float32))
    if not positions:
        raise ValueError("No geometry seeds were generated.")
    xyz = np.concatenate(positions)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, positions=xyz, velocities=np.zeros_like(xyz), colors=np.concatenate(colors),
                 times=np.concatenate(times), durations=np.concatenate(durations),
                 has_velocity=np.zeros(len(xyz), dtype=bool), frame_start=np.int64(0),
                 frame_end=np.int64(num_frames), time_denominator=np.int64(denominator),
                 keyframe_step=np.int64(step), n_keyframes=np.int64(len(clouds)),
                 mode="person4d_visual_hull")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return len(xyz)


def _valid_rgba(path: Path, size: tuple[int, int]) -> bool:
    try:
        with Image.open(path) as image:
            image.load()
            return image.mode == "RGBA" and image.size == size and image.getchannel("A").getextrema()[1] >= 128
    except (OSError, ValueError):
        return False


def _save_rgba(path: Path, rgb: np.ndarray, alpha: np.ndarray) -> None:
    if alpha.shape != rgb.shape[:2] or alpha.dtype != np.uint8:
        raise ValueError("Segmentation returned an invalid alpha raster.")
    area = float(np.mean(alpha >= 128))
    if area < 0.0001 or area > 0.995:
        raise ValueError(f"Unusable foreground mask for {path.name}: {area:.3%} foreground. Inspect the generated view.")
    temporary = path.with_name(path.name + ".tmp")
    rgba = Image.fromarray(np.concatenate((rgb, alpha[..., None]), axis=2), "RGBA")
    rgba.save(temporary, format="PNG", compress_level=3)
    os.replace(temporary, path)


def decode_video(path: Path) -> Iterator[np.ndarray]:
    import av
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"Expected one video stream: {path.name}.")
        for frame in container.decode(container.streams.video[0]):
            yield frame.to_ndarray(format="rgb24")


class ForegroundPredictor:
    """The pinned upstream BiRefNet preprocessing, keeping one model resident."""

    def __init__(self, model_dir: Path, device: str):
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
        from fdanyone.assets import resolve_foreground_model
        from fdanyone.download import ensure_foreground_model
        ensure_foreground_model(model_dir)
        self.torch, self.device = torch, device
        self.model = AutoModelForImageSegmentation.from_pretrained(
            str(resolve_foreground_model(model_dir).resolve()), local_files_only=True, trust_remote_code=True,
        ).eval().half().to(device)
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        from torchvision.transforms.functional import to_pil_image
        images = [Image.fromarray(frame, "RGB") for frame in frames]
        inputs = self.torch.stack([self.transform(image) for image in images]).to(
            device=self.device, dtype=self.torch.float16)
        with self.torch.inference_mode():
            predictions = self.model(inputs)[-1].sigmoid().cpu()
        result = [np.asarray(to_pil_image(prediction).resize(image.size).convert("L"), dtype=np.uint8).copy()
                  for image, prediction in zip(images, predictions, strict=True)]
        del inputs, predictions
        return result

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def extract_rgba(videos: list[Path], cameras: list[dict], root: Path, num_frames: int,
                 predictor_factory: Callable, decoder: Callable = decode_video,
                 batch_size: int = 4) -> None:
    """Resume validated PNGs; source fingerprint binds cached pixels to videos."""
    predictor = None
    completed = 0
    writer = ThreadPoolExecutor(max_workers=2)
    try:
        for camera, video in zip(cameras, videos, strict=True):
            folder = root / "images" / f"cam{camera['camera_id']:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            size = (int(camera["image_width"]), int(camera["image_height"]))
            valid = [_valid_rgba(folder / f"{index:06d}.png", size) for index in range(num_frames)]
            if all(valid):
                completed += num_frames
                progress(f"Reusing camera {camera['camera_id']:02d} foreground frames", completed / (len(cameras) * num_frames) * 65)
                continue
            pending: list[tuple[int, np.ndarray]] = []
            writes: list[Future[None]] = []

            def flush() -> None:
                nonlocal predictor, writes
                if not pending:
                    return
                if predictor is None:
                    predictor = predictor_factory()
                masks = predictor([image for _, image in pending])
                if len(masks) != len(pending):
                    raise ValueError("Segmentation did not return one mask per RGB frame.")
                # Encode the prior batch while the GPU segments this one. Keep
                # at most one batch of pending writes, even on a slow volume.
                for write in writes:
                    write.result()
                writes = [writer.submit(_save_rgba, folder / f"{frame_index:06d}.png", rgb, mask)
                          for (frame_index, rgb), mask in zip(pending, masks, strict=True)]
                pending.clear()

            count = 0
            for frame_index, rgb in enumerate(decoder(video)):
                count += 1
                if frame_index >= num_frames:
                    raise ValueError(f"Generated video {video.name} has more than {num_frames} frames.")
                if rgb.dtype != np.uint8 or rgb.shape != (size[1], size[0], 3):
                    raise ValueError(f"Decoded raster does not match camera {camera['camera_id']}: {rgb.shape}.")
                if not valid[frame_index]:
                    pending.append((frame_index, rgb))
                    if len(pending) >= batch_size:
                        flush()
                completed += 1
                if count % 12 == 0:
                    progress(f"Foreground: camera {camera['camera_id']:02d}, frame {count}/{num_frames}",
                             completed / (len(cameras) * num_frames) * 65)
            flush()
            for write in writes:
                write.result()
            if count != num_frames:
                raise ValueError(f"Generated video {video.name} has {count} frames; expected {num_frames}.")
            progress(f"Foreground complete for camera {camera['camera_id']:02d}", completed / (len(cameras) * num_frames) * 65)
    finally:
        writer.shutdown()
        if predictor is not None and hasattr(predictor, "close"):
            predictor.close()


def _carving_mask(alpha: np.ndarray) -> np.ndarray:
    """Threshold alpha and dilate by two pixels using a separable 5×5 max."""
    mask = alpha >= 128
    vertical = mask.copy()
    for offset in (1, 2):
        vertical[offset:] |= mask[:-offset]
        vertical[:-offset] |= mask[offset:]
    expanded = vertical.copy()
    for offset in (1, 2):
        expanded[:, offset:] |= vertical[:, :-offset]
        expanded[:, :-offset] |= vertical[:, offset:]
    return expanded


def _rig_target(cameras: list[dict]) -> np.ndarray:
    poses = np.stack([np.asarray(camera["camera_to_world"], dtype=np.float64) for camera in cameras])
    directions = poses[:, :3, 2]
    projectors = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    system = projectors.sum(axis=0)
    if np.linalg.matrix_rank(system) < 3:
        raise ValueError("Camera rig does not constrain a bounded reconstruction volume.")
    return np.linalg.solve(system, np.einsum("bij,bj->i", projectors, poses[:, :3, 3]))


def silhouette_bounds(cameras: list[dict], masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bound each timestamp from all silhouette cones at the rig target depth.

    This conservative envelope follows a translating subject, unlike the static
    exporter's fixed cube. Carving still tests the actual calibrated silhouettes.
    """
    center = _rig_target(cameras)
    corners = []
    for camera, mask in zip(cameras, masks, strict=True):
        columns, rows = np.flatnonzero(mask.any(axis=0)), np.flatnonzero(mask.any(axis=1))
        if not len(columns) or not len(rows):
            raise ValueError(f"Empty silhouette for camera {camera['camera_id']:02d}.")
        pose, intrinsic = np.asarray(camera["camera_to_world"]), np.asarray(camera["K"])
        depth = (np.linalg.inv(pose) @ np.r_[center, 1])[2]
        if depth <= 0:
            raise ValueError("Rig target lies behind a camera.")
        for x in (columns[0], columns[-1]):
            for y in (rows[0], rows[-1]):
                point_camera = np.linalg.solve(intrinsic, np.array([x, y, 1])) * depth
                corners.append(pose[:3, :3] @ point_camera + pose[:3, 3])
    corners = np.stack(corners)
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    padding = np.maximum((upper - lower) * 0.15, 0.30)
    return lower - padding, upper + padding


def point_colors(points: np.ndarray, images: list[np.ndarray], masks: np.ndarray,
                 projections: np.ndarray, voxel_size: float) -> np.ndarray:
    sums, counts = np.zeros_like(points, dtype=np.float32), np.zeros((len(points), 1), dtype=np.float32)
    visible_sums, visible_counts = np.zeros_like(sums), np.zeros_like(counts)
    homogeneous = np.concatenate((points, np.ones((len(points), 1))), axis=1)
    height, width = masks.shape[1:]
    for image, mask, projection in zip(images, masks, projections, strict=True):
        projected = homogeneous @ projection.T
        positive = projected[:, 2] > 1e-6
        pixels = np.rint(projected[:, :2] / np.maximum(projected[:, 2:3], 1e-6)).astype(np.int64)
        valid = positive & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        ids = np.flatnonzero(valid)
        ids = ids[mask[pixels[ids, 1], pixels[ids, 0]]]
        if not len(ids):
            continue
        u, v = pixels[ids].T
        colors = image[v, u]
        sums[ids] += colors
        counts[ids] += 1

        # Pool depths over approximately one projected voxel so nearby rays
        # can reject colors from points hidden behind the hull's front surface.
        depths = projected[ids, 2]
        focal = np.linalg.norm(np.cross(projection[:2, :3], projection[2, :3]), axis=1).max()
        cell_size = max(1, math.ceil(focal * voxel_size / np.median(depths)))
        cells_x = (width + cell_size - 1) // cell_size
        cells_y = (height + cell_size - 1) // cell_size
        cells = v // cell_size * cells_x + u // cell_size
        nearest = np.full(cells_x * cells_y, np.inf, dtype=depths.dtype)
        np.minimum.at(nearest, cells, depths)
        visible = depths <= nearest[cells] + 2 * voxel_size
        visible_sums[ids[visible]] += colors[visible]
        visible_counts[ids[visible]] += 1

    if np.any(counts == 0):
        raise ValueError("Geometry contains points without foreground color observations.")
    has_visible_color = visible_counts[:, 0] > 0
    sums[has_visible_color] = visible_sums[has_visible_color]
    counts[has_visible_color] = visible_counts[has_visible_color]
    return (sums / counts / 255).astype(np.float32)


def carve_visual_hull(images: list[np.ndarray], masks: np.ndarray, cameras: list[dict],
                      budget: int, voxel_size: float, seed: int, device: str,
                      mask_tolerance: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    import torch
    lower, upper = silhouette_bounds(cameras, masks)
    axes = [np.arange(lower[axis], upper[axis], voxel_size, dtype=np.float32) for axis in range(3)]
    counts = [len(axis) for axis in axes]
    total = math.prod(counts)
    if total > 20_000_000:
        raise ValueError(f"Carving bounds require {total:,} voxels. Increase --voxel-size or inspect mask bounds.")
    projections = projection_matrices(cameras).astype(np.float32)
    torch_projections = torch.from_numpy(projections).to(device)
    torch_masks = torch.from_numpy(masks).to(device).flatten(1)
    torch_axes = [torch.from_numpy(axis).to(device) for axis in axes]
    minimum_views = len(cameras) - int(len(cameras) * mask_tolerance)
    height, width = masks.shape[1:]
    rng = np.random.default_rng(seed)
    reservoir = np.empty((0, 3), dtype=np.float32)
    scores = np.empty(0)
    found = 0
    with torch.inference_mode():
        for start in range(0, total, 65_536):
            flat = torch.arange(start, min(start + 65_536, total), device=device)
            z = flat % counts[2]
            y = flat // counts[2] % counts[1]
            x = flat // (counts[1] * counts[2])
            points = torch.stack((torch_axes[0][x], torch_axes[1][y], torch_axes[2][z]), dim=1)
            homogeneous = torch.cat((points, torch.ones((len(points), 1), device=device)), dim=1)
            projected = torch.matmul(torch_projections, homogeneous.T).transpose(1, 2)
            depth = projected[..., 2]
            pixels = torch.round(projected[..., :2] / depth[..., None].clamp_min(1e-6)).long()
            u, v = pixels[..., 0], pixels[..., 1]
            valid = (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            # Invalid projections gather at zero and are then masked out.
            offsets = torch.where(valid, v * width + u, 0)
            inside = valid & torch.gather(torch_masks, 1, offsets)
            kept = points[inside.sum(dim=0) >= minimum_views].cpu().numpy()
            found += len(kept)
            if len(kept):
                reservoir = np.concatenate((reservoir, kept))
                scores = np.concatenate((scores, rng.random(len(kept))))
                if len(scores) > budget:
                    selected = np.argpartition(scores, budget - 1)[:budget]
                    reservoir, scores = reservoir[selected], scores[selected]
    if found < 100:
        raise ValueError(f"Foreground masks and calibration produced only {found} hull voxels. Inspect generated views and silhouettes.")
    colors = point_colors(reservoir, images, masks, projections, voxel_size)
    return reservoir, colors


def _source_information(source: Path, settings: dict) -> tuple[list[dict], list[Path], dict, str]:
    metadata_path, camera_path = source / "metadata.json", source / "cameras.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cameras = validate_cameras(json.loads(camera_path.read_text(encoding="utf-8")))
    output = metadata.get("output", {})
    num_frames = int(output.get("frames_per_video", 0))
    if num_frames != EXPECTED_FRAMES:
        raise ValueError(f"Expected a completed {EXPECTED_FRAMES}-frame 4DAnyone result, found {num_frames}.")
    if int(output.get("target_views", -1)) != len(cameras):
        raise ValueError("Metadata and camera rig disagree about the number of target views.")
    try:
        fps = Fraction(str(output["fps"]))
        if fps <= 0:
            raise ValueError
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("Output metadata needs a valid positive FPS.") from exc
    videos = []
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    digest.update(metadata_path.read_bytes())
    digest.update(camera_path.read_bytes())
    for camera in cameras:
        path = (source / camera.get("video", f"videos/{camera['camera_id']:02d}.mp4")).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError("Camera video paths must name existing files inside the 4DAnyone result.")
        videos.append(path)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return cameras, videos, metadata, digest.hexdigest()


def convert(source: str | Path, destination: str | Path, model_dir: str | Path,
            views_repo: str | Path | None = None, keyframe_step: int = 3,
            max_points: int = 1_000_000, voxel_size: float = 0.025,
            seed: int = 42, device: str = "cuda:0", mask_tolerance: float = 0.05,
            *, predictor_factory: Callable | None = None, decoder: Callable = decode_video,
            carver: Callable = carve_visual_hull) -> dict:
    """Publish a complete dataset atomically, retaining private partial work."""
    source, destination, model_dir = (Path(value).expanduser().resolve() for value in (source, destination, model_dir))
    if destination == source or destination in source.parents or source in destination.parents:
        raise ValueError("Source and dataset destination must be separate sibling trees.")
    if not 0.005 <= voxel_size <= 0.1 or not 0 <= mask_tolerance <= 0.15:
        raise ValueError("Voxel size must be 0.005–0.1 m and mask tolerance 0–0.15.")
    frames = keyframe_indices(EXPECTED_FRAMES, keyframe_step)
    budgets = point_budgets(max_points, frames)
    settings = {"version": ADAPTER_VERSION, "keyframe_step": keyframe_step,
                "max_points": max_points, "voxel_size": voxel_size, "seed": seed,
                "mask_tolerance": mask_tolerance}
    cameras, videos, metadata, fingerprint = _source_information(source, settings)
    repository = Path(views_repo).resolve() if views_repo else Path(__file__).resolve().parents[1] / "vendor" / "4danyone"
    if predictor_factory is None:
        if not (repository / "fdanyone").is_dir():
            raise ValueError(f"4DAnyone source is missing: {repository}. Run scripts/setup_runpod.sh.")
        sys.path.insert(0, str(repository))
        predictor_factory = lambda: ForegroundPredictor(model_dir, device)
    with dataset_lock(destination):
        if destination.exists():
            try:
                manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError("Dataset destination exists without a completion manifest. Choose a new destination.") from exc
            if manifest.get("fingerprint") != fingerprint or not (destination / "init.npz").is_file():
                raise ValueError("Dataset destination belongs to another conversion or is incomplete.")
            if any(not _valid_rgba(destination / "images" / f"cam{camera['camera_id']:02d}" / f"{frame:06d}.png",
                                   (camera["image_width"], camera["image_height"]))
                   for camera in cameras for frame in range(EXPECTED_FRAMES)):
                raise ValueError("Published dataset has missing or damaged foreground frames.")
            progress("Reusing complete reconstruction dataset", 100)
            return manifest
        work = destination.parent / f".{destination.name}.adapter-{fingerprint[:12]}"
        work.mkdir(parents=True, exist_ok=True)
        atomic_json(work / ".adapter.json", {"fingerprint": fingerprint, "settings": settings})
        progress("Extracting synchronized target views and foreground alpha", 0)
        extract_rgba(videos, cameras, work, EXPECTED_FRAMES, predictor_factory, decoder)
        write_colmap(work / "sparse" / "0", cameras)
        seed_root = work / ".seed-cache"
        seed_root.mkdir(exist_ok=True)
        clouds = []
        for index, (frame, budget) in enumerate(zip(frames, budgets, strict=True)):
            cache = seed_root / f"{frame:06d}.npz"
            xyz = rgb = None
            if cache.is_file():
                try:
                    with np.load(cache, allow_pickle=False) as stored:
                        xyz, rgb = stored["positions"], stored["colors"]
                    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape or not 0 < len(xyz) <= budget or not np.isfinite(xyz).all() or not np.isfinite(rgb).all():
                        xyz = rgb = None
                except (OSError, ValueError, KeyError):
                    xyz = rgb = None
            if xyz is None:
                images, masks = [], []
                for camera in cameras:
                    with Image.open(work / "images" / f"cam{camera['camera_id']:02d}" / f"{frame:06d}.png") as image:
                        rgba = np.asarray(image).copy()
                    images.append(rgba[..., :3])
                    # Two pixels of carving-only tolerance retains fine features;
                    # training still receives the original soft alpha unchanged.
                    masks.append(_carving_mask(rgba[..., 3]))
                xyz, rgb = carver(images, np.stack(masks), cameras, budget, voxel_size,
                                  seed + frame, device, mask_tolerance)
                temporary = cache.with_name(cache.name + ".tmp")
                with temporary.open("wb") as handle:
                    np.savez(handle, positions=xyz, colors=rgb)
                os.replace(temporary, cache)
                del images, masks
            clouds.append((frame, xyz, rgb))
            progress(f"Geometry seeds: frame {frame}/{EXPECTED_FRAMES - 1}", 65 + (index + 1) / len(frames) * 32)
        num_points = write_initialization(work / "init.npz", clouds, EXPECTED_FRAMES, keyframe_step)
        output = metadata["output"]
        manifest = {"adapter_version": ADAPTER_VERSION, "fingerprint": fingerprint,
                    "source": str(source), "frames": EXPECTED_FRAMES, "num_frames": EXPECTED_FRAMES,
                    "cameras": len(cameras), "num_cameras": len(cameras), "fps": str(output["fps"]),
                    "width": cameras[0]["image_width"], "height": cameras[0]["image_height"],
                    "duration_seconds": float(Fraction(EXPECTED_FRAMES, 1) / Fraction(str(output["fps"]))),
                    "points": num_points, "keyframes": frames, "settings": settings,
                    "coordinate_system": "4danyone_opencv_y_up_world_meters",
                    "velocity_units": "world_units_per_source_frame", "created_at": time.time()}
        # Delete only adapter-owned seed caches after the final NPZ is durable;
        # retaining them on any failure permits an inexpensive retry.
        for cache in seed_root.glob("*.npz"):
            cache.unlink()
        seed_root.rmdir()
        atomic_json(work / "manifest.json", manifest)
        work.rename(destination)
        progress("Character reconstruction dataset ready", 100)
        return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--views-repo")
    parser.add_argument("--keyframe-step", type=int, default=3)
    parser.add_argument("--max-points", type=int, default=1_000_000)
    parser.add_argument("--voxel-size", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mask-tolerance", type=float, default=0.05)
    try:
        result = convert(**vars(parser.parse_args()))
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"event": "error", "stage": "adapter", "message": str(exc)}), file=sys.stderr, flush=True)
        raise SystemExit(1) from None
    print(json.dumps({"event": "complete", "stage": "adapter", "manifest": result}), flush=True)


if __name__ == "__main__":
    main()
