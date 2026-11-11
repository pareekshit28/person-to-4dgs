"""CPU checks for the conversion boundary, including interruption recovery."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFilter

from person4d import adapter


def cameras(count=3, width=12, height=16):
    records = []
    target = np.array([0.0, 0.9, 0.0])
    for index, angle in enumerate(np.linspace(0, 2 * np.pi, count, endpoint=False)):
        center = target + np.array([3 * np.sin(angle), 0, 3 * np.cos(angle)])
        forward = (target - center) / np.linalg.norm(target - center)
        right = np.cross(forward, [0, 1, 0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        pose = np.eye(4)
        pose[:3, :3] = np.stack([right, down, forward], axis=1)
        pose[:3, 3] = center
        records.append({"camera_id": index, "K": [[10, 0, width / 2], [0, 10, height / 2], [0, 0, 1]],
                        "camera_to_world": pose.tolist(), "image_width": width, "image_height": height,
                        "video": f"videos/{index:02d}.mp4"})
    return records


def qvec_to_rotation(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y*y + z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class GeometryContractTests(unittest.TestCase):
    def test_seed_colors_use_visible_side_and_fall_back_for_interior(self):
        rig = cameras(count=2)
        points = np.array([[0, .9, .2], [0, .9, -.2], [0, .9, 0]], dtype=np.float32)
        images = [np.full((16, 12, 3), color, dtype=np.uint8) for color in ([255, 0, 0], [0, 0, 255])]
        masks = np.ones((2, 16, 12), dtype=bool)
        colors = adapter.point_colors(points, images, masks, adapter.projection_matrices(rig), 0.016)
        np.testing.assert_array_equal(colors, [[1, 0, 0], [0, 0, 1], [.5, 0, .5]])

    def test_carving_mask_matches_max_filter_at_edges(self):
        alpha = np.zeros((15, 11), dtype=np.uint8)
        alpha[0, 10] = alpha[7, 5] = 128
        alpha[14, 0] = 255
        alpha[1, 1] = 127
        for source in (alpha, alpha[:2, :2], alpha[:1, -1:]):
            binary = Image.fromarray((source >= 128).astype(np.uint8) * 255)
            expected = np.asarray(binary.filter(ImageFilter.MaxFilter(5))) > 0
            np.testing.assert_array_equal(adapter._carving_mask(source), expected)

    def test_colmap_pose_and_projection_preserve_opencv_axes(self):
        rig = cameras()
        adapter.validate_cameras({"camera_model": "OPENCV", "cameras": rig})
        point = np.array([0.2, 1.1, 0.1, 1.0])
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            adapter.write_colmap(directory, rig)
            lines = (directory / "images.txt").read_text().splitlines()
            records = [line.split() for line in lines if line and not line.startswith("#")]
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0][-1], "cam00/000000.png")
            for record, camera in zip(records, rig):
                values = list(map(float, record[1:8]))
                exported = np.eye(4)
                exported[:3, :3] = qvec_to_rotation(values[:4])
                exported[:3, 3] = values[4:]
                expected = np.linalg.inv(camera["camera_to_world"])
                np.testing.assert_allclose(exported, expected, atol=1e-12)
                projected = np.array(camera["K"]) @ (exported @ point)[:3]
                direct = adapter.projection_matrices([camera])[0] @ point
                np.testing.assert_allclose(projected / projected[2], direct / direct[2])
            with (directory / "cameras.bin").open("rb") as handle:
                self.assertEqual(struct.unpack("<Q", handle.read(8))[0], 3)
                for index, camera in enumerate(rig):
                    header = struct.unpack("<IiQQ", handle.read(24))
                    self.assertEqual(header, (index + 1, 1, 12, 16))
                    np.testing.assert_allclose(struct.unpack("<4d", handle.read(32)), [10, 10, 6, 8])
                self.assertEqual(handle.read(), b"")
            with (directory / "images.bin").open("rb") as handle:
                self.assertEqual(struct.unpack("<Q", handle.read(8))[0], 3)
                for index, camera in enumerate(rig):
                    record = struct.unpack("<I4d3dI", handle.read(64))
                    pose = np.eye(4)
                    pose[:3, :3] = qvec_to_rotation(record[1:5])
                    pose[:3, 3] = record[5:8]
                    np.testing.assert_allclose(pose, np.linalg.inv(camera["camera_to_world"]), atol=1e-12)
                    name = bytearray()
                    while (byte := handle.read(1)) != b"\0":
                        self.assertTrue(byte)
                        name.extend(byte)
                    self.assertEqual(name.decode(), f"cam{index:02d}/000000.png")
                    self.assertEqual(struct.unpack("<Q", handle.read(8))[0], 0)
                self.assertEqual(handle.read(), b"")
            self.assertEqual((directory / "points3D.bin").read_bytes(), struct.pack("<Q", 0))

    def test_reflected_pose_and_camera_name_gaps_fail(self):
        rig = cameras()
        rig[0]["camera_to_world"][0][0] *= -1
        with self.assertRaisesRegex(ValueError, "right-handed"):
            adapter.validate_cameras({"camera_model": "OPENCV", "cameras": rig})
        rig = cameras()
        rig[1]["camera_id"] = 9
        with self.assertRaisesRegex(ValueError, "contiguous"):
            adapter.validate_cameras({"camera_model": "OPENCV", "cameras": rig})

    def test_budget_covers_all_seed_times_and_npz_uses_exclusive_end(self):
        frames = adapter.keyframe_indices(121, 7)
        self.assertEqual(frames[-1], 120)
        budgets = adapter.point_budgets(1000, frames)
        self.assertEqual(sum(budgets), 1000)
        self.assertLessEqual(max(budgets) - min(budgets), 1)
        clouds = [(frame, np.full((budget, 3), [frame / 100, 1, 2], dtype=np.float32),
                   np.full((budget, 3), 0.5, dtype=np.float32)) for frame, budget in zip(frames, budgets)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "init.npz"
            self.assertEqual(adapter.write_initialization(path, clouds, 121, 7), 1000)
            with np.load(path) as data:
                self.assertEqual(int(data["frame_end"]), 121)
                self.assertEqual(int(data["time_denominator"]), 120)
                self.assertEqual(float(data["times"].min()), 0)
                self.assertEqual(float(data["times"].max()), 1)
                np.testing.assert_allclose(data["durations"], 7 / 120)
                np.testing.assert_array_equal(data["velocities"], 0)
                np.testing.assert_allclose(data["positions"][-1], [1.2, 1, 2], atol=1e-7)
            # Exercise the actual pinned training reader rather than repeating
            # only our serialization equations in the assertions above.
            reader_path = Path(adapter.__file__).resolve().parents[1] / "vendor/freetimegs/src/init_common.py"
            spec = importlib.util.spec_from_file_location("audited_ftgs_init", reader_path)
            reader = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reader)
            loaded = reader.read_init_arrays(str(path), 0, 121)
            self.assertEqual(len(loaded["positions"]), 1000)
            np.testing.assert_allclose(loaded["positions"][-1], [1.2, 1, 2], atol=1e-7)
            np.testing.assert_allclose(loaded["times"][[0, -1]], [0, 1])
            subset = reader.read_init_arrays(str(path), 7, 50)
            np.testing.assert_allclose(subset["times"][[0, -1]], [0, 1])
            np.testing.assert_allclose(subset["durations"], 7 / 42)

    def test_silhouette_volume_follows_translating_character(self):
        rig = cameras(width=120, height=160)
        # Use a realistic focal length for this synthetic rig.
        for camera in rig:
            camera["K"] = [[100, 0, 60], [0, 100, 80], [0, 0, 1]]
        points = np.array([[x, y, z] for x in [0.5, 1.2] for y in [0.1, 1.8] for z in [-0.3, 0.3]])
        projections = adapter.projection_matrices(rig)
        masks = []
        for projection in projections:
            hp = np.c_[points, np.ones(len(points))] @ projection.T
            px = hp[:, :2] / hp[:, 2:]
            lower = np.floor(px.min(axis=0)).astype(int).clip([0, 0], [119, 159])
            upper = np.ceil(px.max(axis=0)).astype(int).clip([0, 0], [119, 159])
            mask = np.zeros((160, 120), dtype=bool)
            mask[lower[1]:upper[1]+1, lower[0]:upper[0]+1] = True
            masks.append(mask)
        lower, upper = adapter.silhouette_bounds(rig, np.stack(masks))
        self.assertTrue(np.all(points >= lower))
        self.assertTrue(np.all(points <= upper))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional for web-only CPU checks")
    def test_actual_carver_on_cpu_keeps_points_inside_calibrated_silhouettes(self):
        rig = cameras(width=120, height=160)
        for camera in rig:
            camera["K"] = [[100, 0, 60], [0, 100, 80], [0, 0, 1]]
        box = np.array([[x, y, z] for x in [-0.3, 0.3] for y in [0.1, 1.7] for z in [-0.3, 0.3]])
        projections = adapter.projection_matrices(rig)
        masks = []
        for projection in projections:
            projected = np.c_[box, np.ones(len(box))] @ projection.T
            pixels = projected[:, :2] / projected[:, 2:]
            lower = np.floor(pixels.min(axis=0)).astype(int)
            upper = np.ceil(pixels.max(axis=0)).astype(int)
            mask = np.zeros((160, 120), dtype=bool)
            mask[lower[1]:upper[1]+1, lower[0]:upper[0]+1] = True
            masks.append(mask)
        images = [np.full((160, 120, 3), [60, 20, 30], dtype=np.uint8) for _ in rig]
        xyz, rgb = adapter.carve_visual_hull(images, np.stack(masks), rig, 200, 0.1, 42, "cpu", 0)
        self.assertEqual(xyz.shape, (200, 3))
        self.assertTrue(np.all(xyz[:, 1] > -0.05))
        self.assertTrue(np.all(xyz[:, 1] < 1.9))
        for projection, mask in zip(projections, masks):
            projected = np.c_[xyz, np.ones(len(xyz))] @ projection.T
            pixels = np.rint(projected[:, :2] / projected[:, 2:]).astype(int)
            self.assertTrue(np.all(mask[pixels[:, 1], pixels[:, 0]]))
        np.testing.assert_allclose(rgb, np.broadcast_to(np.array([60, 20, 30]) / 255, rgb.shape), atol=1e-6)


class ConversionTests(unittest.TestCase):
    def source(self, root):
        source = root / "generated"
        (source / "videos").mkdir(parents=True)
        rig = cameras()
        for camera in rig:
            (source / camera["video"]).write_bytes(f"synthetic-video-{camera['camera_id']}".encode())
        (source / "cameras.json").write_text(json.dumps({"camera_model": "OPENCV", "cameras": rig}))
        (source / "metadata.json").write_text(json.dumps({"output": {"frames_per_video": 121,
            "target_views": 3, "fps": "24000/1001", "width": 12, "height": 16}}))
        return source

    @staticmethod
    def decoder(path):
        for frame in range(121):
            yield np.full((16, 12, 3), [frame, 20, 30], dtype=np.uint8)

    @staticmethod
    def predictor(frames):
        result = []
        for frame in frames:
            alpha = np.zeros(frame.shape[:2], dtype=np.uint8)
            alpha[2:-2, 3:-3] = 230
            result.append(alpha)
        return result

    @staticmethod
    def carver(images, masks, cameras, budget, voxel_size, seed, device, mask_tolerance):
        return np.full((budget, 3), [0.2, 0.9, -0.2], dtype=np.float32), np.full((budget, 3), 0.5, dtype=np.float32)

    def test_complete_conversion_and_reuse_keep_straight_rgb(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            source = self.source(root)
            destination = root / "dataset"
            result = adapter.convert(source, destination, root / "models", max_points=400,
                keyframe_step=12, predictor_factory=lambda: self.predictor,
                decoder=self.decoder, carver=self.carver)
            self.assertEqual(result["frames"], 121)
            self.assertEqual(result["fps"], "24000/1001")
            self.assertEqual(result["points"], 400)
            self.assertTrue((destination / "sparse/0/cameras.txt").is_file())
            with Image.open(destination / "images/cam01/000060.png") as image:
                rgba = np.array(image)
            np.testing.assert_array_equal(rgba[5, 5], [60, 20, 30, 230])
            # Background RGB is preserved as straight, not premultiplied alpha.
            np.testing.assert_array_equal(rgba[0, 0], [60, 20, 30, 0])
            result2 = adapter.convert(source, destination, root / "models", max_points=400,
                keyframe_step=12, predictor_factory=lambda: self.fail("Unexpected resegmentation"),
                decoder=self.decoder, carver=self.carver)
            self.assertEqual(result["fingerprint"], result2["fingerprint"])

    def test_interrupted_segmentation_reuses_valid_frames(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            source = self.source(root)
            destination = root / "dataset"
            calls = []

            def interrupted(frames):
                calls.append(len(frames))
                if len(calls) == 3:
                    raise RuntimeError("simulated interruption")
                return self.predictor(frames)

            with self.assertRaisesRegex(RuntimeError, "interruption"):
                adapter.convert(source, destination, root / "models", max_points=400,
                    keyframe_step=12, predictor_factory=lambda: interrupted,
                    decoder=self.decoder, carver=self.carver)
            self.assertFalse(destination.exists())
            cached = list(root.glob(".dataset.adapter-*/images/cam00/*.png"))
            self.assertEqual(len(cached), 8)
            processed = []

            def counting(frames):
                processed.extend(frames)
                return self.predictor(frames)

            adapter.convert(source, destination, root / "models", max_points=400,
                keyframe_step=12, predictor_factory=lambda: counting,
                decoder=self.decoder, carver=self.carver)
            self.assertEqual(len(processed), 3 * 121 - 8)

    def test_png_write_error_prevents_dataset_publication(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            source = self.source(root)
            destination = root / "dataset"
            with patch.object(adapter, "_save_rgba", side_effect=OSError("simulated write error")):
                with self.assertRaisesRegex(OSError, "write error"):
                    adapter.convert(source, destination, root / "models", max_points=400,
                        keyframe_step=12, predictor_factory=lambda: self.predictor,
                        decoder=self.decoder, carver=self.carver)
            self.assertFalse(destination.exists())

    def test_short_generated_video_never_publishes(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            source = self.source(root)

            def short(path):
                for frame in range(120):
                    yield np.zeros((16, 12, 3), dtype=np.uint8)

            with self.assertRaisesRegex(ValueError, "120 frames"):
                adapter.convert(source, root / "dataset", root / "models", max_points=400,
                    predictor_factory=lambda: self.predictor, decoder=short, carver=self.carver)
            self.assertFalse((root / "dataset").exists())

    def test_path_escape_in_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            root = Path(folder)
            source = self.source(root)
            payload = json.loads((source / "cameras.json").read_text())
            payload["cameras"][0]["video"] = "../../outside.mp4"
            (source / "cameras.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "inside"):
                adapter.convert(source, root / "dataset", root / "models", predictor_factory=lambda: self.predictor)


if __name__ == "__main__":
    unittest.main()
