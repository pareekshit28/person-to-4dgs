"""
Code borrowed from

https://github.com/google-research/multinerf/blob/5b4d4f64608ec8077222c52fdf814d40acc10bc1/internal/camera_utils.py
"""

import numpy as np
import scipy


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def average_pose(poses: np.ndarray) -> np.ndarray:
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def generate_spiral_path(
    poses,
    bounds,
    n_frames=120,
    n_rots=2,
    zrate=0.5,
    spiral_scale_f=1.0,
    spiral_scale_r=1.0,
    focus_distance=0.75,
):
    """Calculates a forward facing spiral path for rendering."""
    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of conservative near and far bounds in disparity space.
    near_bound = bounds.min()
    far_bound = bounds.max()
    # All cameras will point towards the world space point (0, 0, -focal).
    focal = 1 / (((1 - focus_distance) / near_bound + focus_distance / far_bound))
    focal = focal * spiral_scale_f

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 90, 0)
    radii = radii * spiral_scale_r
    radii = np.concatenate([radii, [1.0]])

    # Generate poses for spiral path.
    render_poses = []
    cam2world = average_pose(poses)
    up = poses[:, :3, 1].mean(0)
    for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_frames, endpoint=False):
        t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.0]
        z_axis = position - lookat
        render_poses.append(viewmatrix(z_axis, up, position))
    render_poses = np.stack(render_poses, axis=0)
    return render_poses


def generate_ellipse_path_z(
    poses: np.ndarray,
    n_frames: int = 120,
    # const_speed: bool = True,
    variation: float = 0.0,
    phase: float = 0.0,
    height: float = 0.0,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=height (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], height])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                variation
                * (
                    z_low[2]
                    + (z_high - z_low)[2]
                    * (np.cos(theta + 2 * np.pi * phase) * 0.5 + 0.5)
                )
                + height,
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # if const_speed:
    #     # Resample theta angles so that the velocity is closer to constant.
    #     lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    #     theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    #     positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(center - p, up, p) for p in positions])


def generate_ellipse_path_y(
    poses: np.ndarray,
    n_frames: int = 120,
    # const_speed: bool = True,
    variation: float = 0.0,
    phase: float = 0.0,
    height: float = 0.0,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at y=height (in middle of zero-mean capture pattern).
    offset = np.array([center[0], height, center[2]])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    y_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    y_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-z.
        # Optionally also interpolate in y to change camera height along path.
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                variation
                * (
                    y_low[1]
                    + (y_high - y_low)[1]
                    * (np.cos(theta + 2 * np.pi * phase) * 0.5 + 0.5)
                )
                + height,
                low[2] + (high - low)[2] * (np.sin(theta) * 0.5 + 0.5),
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # if const_speed:
    #     # Resample theta angles so that the velocity is closer to constant.
    #     lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    #     theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    #     positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions])


def generate_smooth_arc_path(
    poses: np.ndarray,
    n_frames: int = 120,
    arc_degrees: float = 30.0,
    height_variation: float = 0.0,
    radius_scale: float = 1.0,
) -> np.ndarray:
    """
    Generate a smooth arc camera path - gentle movement, not full 360.

    Perfect for 4D scenes where you want subtle camera motion while
    time progresses, keeping focus on the dynamic content.

    Args:
        poses: (n, 3, 4) array of input camera poses
        n_frames: number of output frames
        arc_degrees: total arc angle (e.g., 30 = ±15 degrees from center)
        height_variation: vertical movement as fraction of scene scale (0-1)
        radius_scale: scale factor for camera distance (1.0 = same as input)

    Returns:
        Array of camera poses with shape (n_frames, 3, 4)
    """
    # Calculate focal point (where cameras look at)
    center = focus_point_fn(poses)

    # Get average camera position and up vector
    avg_position = poses[:, :3, 3].mean(0)
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)

    # Calculate radius (distance from center to average camera position)
    radius = np.linalg.norm(avg_position - center) * radius_scale

    # Get the horizontal direction (perpendicular to up and look direction)
    look_dir = center - avg_position
    look_dir = look_dir / np.linalg.norm(look_dir)
    right = np.cross(look_dir, avg_up)
    right = right / np.linalg.norm(right)

    # Height range for vertical variation
    z_positions = poses[:, :3, 3][:, 2] if height_variation > 0 else None
    if z_positions is not None:
        z_range = z_positions.max() - z_positions.min()
    else:
        z_range = 0

    # Generate arc angles (smooth sine wave for natural motion)
    arc_rad = np.deg2rad(arc_degrees)
    t = np.linspace(0, 1, n_frames)
    # Smooth ease-in-out using sine
    angles = arc_rad * np.sin(t * np.pi - np.pi/2)  # Goes from -arc to +arc smoothly

    # Generate positions along the arc
    render_poses = []
    for i, angle in enumerate(angles):
        # Rotate around the center point
        # Position on arc
        offset = right * np.sin(angle) * radius + look_dir * (np.cos(angle) - 1) * radius
        position = avg_position + offset

        # Add height variation (gentle up-down)
        if height_variation > 0:
            height_offset = height_variation * z_range * np.sin(t[i] * np.pi)
            position[2] += height_offset

        # Camera looks at center
        render_poses.append(viewmatrix(center - position, avg_up, position))

    return np.stack(render_poses, axis=0)


def generate_dolly_zoom_path(
    poses: np.ndarray,
    n_frames: int = 120,
    dolly_amount: float = 0.3,
    lateral_amount: float = 0.1,
) -> np.ndarray:
    """
    Generate a smooth dolly path with optional lateral movement.

    Creates a gentle push-in or pull-out motion, good for dramatic reveals
    in 4D scenes.

    Args:
        poses: (n, 3, 4) array of input camera poses
        n_frames: number of output frames
        dolly_amount: how much to move forward/back (fraction of scene scale)
                      positive = push in, negative = pull out
        lateral_amount: how much side-to-side movement (fraction of scene scale)

    Returns:
        Array of camera poses with shape (n_frames, 3, 4)
    """
    center = focus_point_fn(poses)
    avg_position = poses[:, :3, 3].mean(0)
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)

    # Calculate scene scale
    radius = np.linalg.norm(avg_position - center)

    # Direction vectors
    look_dir = center - avg_position
    look_dir = look_dir / np.linalg.norm(look_dir)
    right = np.cross(look_dir, avg_up)
    right = right / np.linalg.norm(right)

    # Smooth parameter (0 to 1 with ease-in-out)
    t = np.linspace(0, 1, n_frames)
    smooth_t = (1 - np.cos(t * np.pi)) / 2  # Smooth ease-in-out

    render_poses = []
    for i in range(n_frames):
        # Dolly movement (forward/back)
        dolly_offset = look_dir * dolly_amount * radius * smooth_t[i]

        # Lateral movement (side to side, sine wave)
        lateral_offset = right * lateral_amount * radius * np.sin(t[i] * 2 * np.pi)

        position = avg_position + dolly_offset + lateral_offset
        render_poses.append(viewmatrix(center - position, avg_up, position))

    return np.stack(render_poses, axis=0)


def generate_fixed_camera_path(
    poses: np.ndarray,
    n_frames: int = 120,
    camera_index: int = 0,
) -> np.ndarray:
    """
    Generate a static camera path (same pose repeated).

    Useful for 4D scenes where you want to see time progression
    from a fixed viewpoint.

    Args:
        poses: (n, 3, 4) array of input camera poses
        n_frames: number of output frames
        camera_index: which input camera to use (0 = first, -1 = last)

    Returns:
        Array of camera poses with shape (n_frames, 3, 4)
    """
    selected_pose = poses[camera_index]
    return np.tile(selected_pose[np.newaxis, :, :], (n_frames, 1, 1))


def generate_interpolated_path(
    poses: np.ndarray,
    n_interp: int,
    spline_degree: int = 5,
    smoothness: float = 0.03,
    rot_weight: float = 0.1,
):
    """Creates a smooth spline path between input keyframe camera poses.

    Spline is calculated with poses in format (position, lookat-point, up-point).

    Args:
      poses: (n, 3, 4) array of input pose keyframes.
      n_interp: returned path will have n_interp * (n - 1) total poses.
      spline_degree: polynomial degree of B-spline.
      smoothness: parameter for spline smoothing, 0 forces exact interpolation.
      rot_weight: relative weighting of rotation/translation in spline solve.

    Returns:
      Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
    """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(
        points, n_interp * (points.shape[0] - 1), k=spline_degree, s=smoothness
    )
    return points_to_poses(new_points)