"""
FreeTime Datasets Package

Provides data loading utilities for FreeTimeGS training:
- FreeTimeParser: Loads COLMAP camera poses and image paths
- FreeTimeDataset: PyTorch dataset for training/validation
- load_multiframe_colmap_points: 4D initialization from multi-frame COLMAP
- load_single_frame_with_velocity: Single-frame initialization with velocity
"""

from .FreeTime_dataset import (
    FreeTimeParser,
    FreeTimeDataset,
    load_multiframe_colmap_points,
    load_single_frame_with_velocity,
)

from .normalize import (
    similarity_from_cameras,
    transform_cameras,
    transform_points,
    align_principle_axes,
    normalize_scene,
    compute_scene_extent,
)

from .read_write_model import (
    read_model,
    write_model,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    read_cameras_text,
    read_images_text,
    read_points3D_text,
    qvec2rotmat,
    rotmat2qvec,
)

from .traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_ellipse_path_y,
    generate_spiral_path,
    generate_smooth_arc_path,
    generate_dolly_zoom_path,
    generate_fixed_camera_path,
)

__all__ = [
    # Dataset classes
    "FreeTimeParser",
    "FreeTimeDataset",
    # Initialization functions
    "load_multiframe_colmap_points",
    "load_single_frame_with_velocity",
    # Normalization functions
    "similarity_from_cameras",
    "transform_cameras",
    "transform_points",
    "align_principle_axes",
    "normalize_scene",
    "compute_scene_extent",
    # COLMAP model I/O
    "read_model",
    "write_model",
    "read_cameras_binary",
    "read_images_binary",
    "read_points3D_binary",
    "read_cameras_text",
    "read_images_text",
    "read_points3D_text",
    "qvec2rotmat",
    "rotmat2qvec",
    # Trajectory generation
    "generate_interpolated_path",
    "generate_ellipse_path_z",
    "generate_ellipse_path_y",
    "generate_spiral_path",
    "generate_smooth_arc_path",
    "generate_dolly_zoom_path",
    "generate_fixed_camera_path",
]
