"""
Camera Calibration Utilities
Handles intrinsic + extrinsic calibration for driver and scene cameras.
Run scripts/calibrate_cameras.py to generate calibration files.
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Optional


def save_calibration(camera_matrix: np.ndarray,
                     dist_coeffs: np.ndarray,
                     path: str):
    data = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist()
    }
    with open(path, 'w') as f:
        yaml.dump(data, f)
    print(f"Calibration saved -> {path}")


def load_calibration(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load intrinsic calibration from YAML file."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Calibration file not found: {path}\n"
            f"Run: python scripts/calibrate_cameras.py"
        )
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    K = np.array(data['camera_matrix'], dtype=np.float64)
    D = np.array(data['dist_coeffs'], dtype=np.float64)
    return K, D


def save_extrinsics(R: np.ndarray, T: np.ndarray, path: str):
    data = {'R': R.tolist(), 'T': T.tolist()}
    with open(path, 'w') as f:
        yaml.dump(data, f)
    print(f"Extrinsics saved -> {path}")


def load_extrinsics(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load extrinsic transform (driver_cam → scene_cam) from YAML."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Extrinsics file not found: {path}\n"
            f"Run: python scripts/calibrate_cameras.py"
        )
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    R = np.array(data['R'], dtype=np.float64)
    T = np.array(data['T'], dtype=np.float64)
    return R, T


def project_gaze_to_scene(gaze_origin_driver: np.ndarray,
                           gaze_dir_driver: np.ndarray,
                           R_driver_to_scene: np.ndarray,
                           T_driver_to_scene: np.ndarray,
                           scene_K: np.ndarray,
                           scene_D: np.ndarray,
                           scene_w: int,
                           scene_h: int) -> Optional[Tuple[int, int]]:
    """
    Properly project a 3-D gaze ray from driver camera space
    into scene camera pixel coordinates.

    Args:
        gaze_origin_driver : 3-D ray origin in driver-cam space (metres)
        gaze_dir_driver    : unit gaze direction vector (driver-cam space)
        R_driver_to_scene  : 3×3 rotation  (driver → scene)
        T_driver_to_scene  : 3-D translation (driver → scene, metres)
        scene_K            : scene camera intrinsic matrix 3×3
        scene_D            : scene camera distortion coefficients
        scene_w / scene_h  : scene image dimensions

    Returns:
        (px, py) in scene image, or None if behind scene camera.
    """
    # Transform ray into scene camera coordinate system
    gaze_dir_scene = R_driver_to_scene @ gaze_dir_driver
    gaze_origin_scene = R_driver_to_scene @ gaze_origin_driver + T_driver_to_scene

    # Intersect ray with z = 5 m plane (approximate scene plane)
    if gaze_dir_scene[2] <= 0:
        return None  # Gaze points away from scene

    t = (5.0 - gaze_origin_scene[2]) / gaze_dir_scene[2]
    world_point = gaze_origin_scene + t * gaze_dir_scene

    # Project into scene image
    pt = world_point[:2] / world_point[2]
    r2 = pt[0]**2 + pt[1]**2
    k1, k2, p1, p2, k3 = (scene_D.ravel().tolist() + [0]*5)[:5]
    radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
    xd = pt[0]*radial + 2*p1*pt[0]*pt[1] + p2*(r2 + 2*pt[0]**2)
    yd = pt[1]*radial + p1*(r2 + 2*pt[1]**2) + 2*p2*pt[0]*pt[1]

    px = int(scene_K[0, 0] * xd + scene_K[0, 2])
    py = int(scene_K[1, 1] * yd + scene_K[1, 2])

    # Clamp to frame
    px = max(0, min(scene_w - 1, px))
    py = max(0, min(scene_h - 1, py))

    return px, py


def make_default_calibration(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback calibration when no calibration file exists.
    FOV ≈ 60°, no distortion.
    """
    f = width / (2 * np.tan(np.radians(30)))
    K = np.array([
        [f,   0, width  / 2],
        [0,   f, height / 2],
        [0,   0,           1]
    ], dtype=np.float64)
    D = np.zeros((5, 1), dtype=np.float64)
    return K, D
