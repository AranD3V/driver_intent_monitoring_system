"""
Personal Gaze Calibration

Fits a polynomial regression from a 4-dim feature vector
    (iris_dx, iris_dy, head_yaw_deg, head_pitch_deg)
to a 2-dim normalised scene point
    (scene_x_norm, scene_y_norm) in [0, 1]^2

This single mapping replaces the chain of (3-D iris ray → solvePnP head pose →
extrinsic transform → distortion-aware projection through a 5 m plane) with a
direct empirical fit, which:

  * removes sign-convention errors,
  * absorbs lens distortion of the driver camera,
  * absorbs any unmodelled offset between driver-cam and scene frame,
  * works for sim mode where the "scene camera" is virtual.

Fit at runtime by collecting samples while the user fixates a known on-screen
target. 9 well-spaced targets × ~30 samples each is enough for a stable fit.

Persistence:
  YAML at calibration/gaze_user.yaml — coefficients + a sample-count + a
  median residual error (px) so we can show it in the launcher.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


# Quadratic-with-cross-terms feature expansion of [dx, dy, yaw, pitch]:
#   [1, dx, dy, yaw, pitch, dx*dx, dy*dy, dx*dy, dx*yaw, dy*pitch, yaw*yaw, pitch*pitch]
# 12 features × 2 outputs = 24 fit coefficients
_FEATURE_NAMES = (
    "1", "dx", "dy", "yaw", "pitch",
    "dx*dx", "dy*dy", "dx*dy",
    "dx*yaw", "dy*pitch",
    "yaw*yaw", "pitch*pitch",
)


def _expand(features: np.ndarray) -> np.ndarray:
    """features: (N, 4) → expanded (N, 12)."""
    dx = features[:, 0]
    dy = features[:, 1]
    yaw = features[:, 2]
    pitch = features[:, 3]
    ones = np.ones_like(dx)
    return np.stack([
        ones, dx, dy, yaw, pitch,
        dx * dx, dy * dy, dx * dy,
        dx * yaw, dy * pitch,
        yaw * yaw, pitch * pitch,
    ], axis=1)


class GazeCalibrator:
    """Polynomial regression over (iris_dx, iris_dy, yaw, pitch) -> (sx, sy)."""

    # Polynomial has 12 features x 2 outputs = 24 free parameters.
    # We require comfortably more samples than that so the fit is
    # over-determined and robust to noise / blinks / outliers.
    MIN_SAMPLES = 60

    def __init__(self):
        self.coeffs: Optional[np.ndarray] = None  # (12, 2)
        self.sample_count: int = 0
        self.residual_px: float = float("nan")
        self.fit_aspect: float = float("nan")  # width / height at fit time

    @property
    def is_fit(self) -> bool:
        return self.coeffs is not None

    # ------------------------------------------------------------------ #
    #  Fit                                                                 #
    # ------------------------------------------------------------------ #

    def fit(self,
            samples: List[Dict],
            scene_w: int,
            scene_h: int) -> Tuple[float, int]:
        """
        samples: each entry is a dict with keys
            'iris_dx', 'iris_dy', 'yaw', 'pitch',     (input features)
            'target_x', 'target_y'                    (pixel targets in scene frame)

        Returns (median residual px, sample count). Raises ValueError if too few.
        """
        if len(samples) < self.MIN_SAMPLES:
            raise ValueError(
                f"Need at least {self.MIN_SAMPLES} samples, got {len(samples)}.")

        X_raw = np.array(
            [[s['iris_dx'], s['iris_dy'], s['yaw'], s['pitch']] for s in samples],
            dtype=np.float64,
        )
        Y_norm = np.array(
            [[s['target_x'] / scene_w, s['target_y'] / scene_h] for s in samples],
            dtype=np.float64,
        )

        X = _expand(X_raw)               # (N, 12)
        # Ridge-regularised least-squares for numerical stability
        lam = 1e-3
        A = X.T @ X + lam * np.eye(X.shape[1])
        B = X.T @ Y_norm
        self.coeffs = np.linalg.solve(A, B)

        # Diagnostic: median residual in pixels
        Y_pred_norm = X @ self.coeffs
        err_px = np.linalg.norm(
            (Y_pred_norm - Y_norm) * np.array([scene_w, scene_h]),
            axis=1,
        )
        self.residual_px = float(np.median(err_px))
        self.sample_count = len(samples)
        self.fit_aspect = float(scene_w) / max(1, float(scene_h))
        return self.residual_px, self.sample_count

    # ------------------------------------------------------------------ #
    #  Predict                                                             #
    # ------------------------------------------------------------------ #

    def predict(self,
                iris_dx: float, iris_dy: float,
                yaw: float, pitch: float,
                scene_w: int, scene_h: int) -> Optional[Tuple[int, int]]:
        if self.coeffs is None:
            return None
        X = _expand(np.array([[iris_dx, iris_dy, yaw, pitch]], dtype=np.float64))
        y_norm = (X @ self.coeffs)[0]                     # (2,)
        x = int(round(y_norm[0] * scene_w))
        y = int(round(y_norm[1] * scene_h))
        x = max(0, min(scene_w - 1, x))
        y = max(0, min(scene_h - 1, y))
        return x, y

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        if self.coeffs is None:
            raise RuntimeError("Cannot save an unfit calibrator.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            'coeffs': self.coeffs.tolist(),
            'feature_names': list(_FEATURE_NAMES),
            'sample_count': self.sample_count,
            'residual_px': self.residual_px,
            'fit_aspect': self.fit_aspect,
            'fit_time': time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, sort_keys=False)
        print(f"[GazeCalibrator] Saved -> {path} "
              f"(median residual {self.residual_px:.1f} px, "
              f"{self.sample_count} samples)")

    @classmethod
    def load(cls, path: str) -> Optional['GazeCalibrator']:
        p = Path(path)
        if not p.exists():
            return None
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            cal = cls()
            cal.coeffs = np.array(data['coeffs'], dtype=np.float64)
            cal.sample_count = int(data.get('sample_count', 0))
            cal.residual_px = float(data.get('residual_px', float('nan')))
            cal.fit_aspect  = float(data.get('fit_aspect',  float('nan')))
            if cal.coeffs.shape != (len(_FEATURE_NAMES), 2):
                print(f"[GazeCalibrator] Ignoring {path}: bad coeff shape "
                      f"{cal.coeffs.shape}")
                return None
            return cal
        except Exception as e:
            print(f"[GazeCalibrator] Failed to load {path}: {e}")
            return None

    def aspect_warning(self, runtime_w: int, runtime_h: int,
                       tolerance: float = 0.10) -> Optional[str]:
        """
        Returns a human-readable warning string when the calibration was
        fit at an aspect ratio that differs from the runtime scene by more
        than `tolerance`. Otherwise None.
        """
        if not np.isfinite(self.fit_aspect) or runtime_h <= 0:
            return None
        runtime_aspect = float(runtime_w) / float(runtime_h)
        if abs(runtime_aspect - self.fit_aspect) / self.fit_aspect <= tolerance:
            return None
        return (f"Calibration fit at aspect {self.fit_aspect:.2f}, "
                f"runtime scene aspect {runtime_aspect:.2f}. "
                f"Recalibrate at the actual scene resolution for best "
                f"accuracy.")
