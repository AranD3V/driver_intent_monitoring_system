"""
Driver Gaze Estimation Module (v3)

Pipeline:
  1. MediaPipe FaceLandmarker / FaceMesh → 478 landmarks (with iris).
  2. solvePnP with 6 face landmarks → head yaw / pitch / roll.
  3. Iris-relative-to-eye-center offset (sign-corrected: +x = looking right,
     +y = looking down).
  4. Eye Aspect Ratio (EAR) blink/closed-eye rejection. When eyes are
     closed we hold the last good gaze and mark `is_fixation=False`.
  5. Personal calibration (modules/gaze_calibration.GazeCalibrator) when a
     calibration/gaze_user.yaml exists — a polynomial regression from
     (iris_dx, iris_dy, yaw, pitch) directly to scene pixels. This is the
     accurate path; it works identically for both two-cam and sim modes
     because the mapping is fitted against the actual scene the driver
     looks at, regardless of how the scene image is sourced.
  6. Uncalibrated fallback: 3-D ray cast through extrinsics if available,
     else linear head-pose mapping with sane gain.
  7. Hysteresis fixation: gaze must be slow for ≥3 consecutive frames to
     count as a fixation, preventing spurious heatmap splats during
     saccades.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

import mediapipe as mp

from modules.calibration import (
    load_calibration, load_extrinsics,
    project_gaze_to_scene, make_default_calibration
)
from modules.gaze_calibration import GazeCalibrator


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str) -> str:
    """Anchor `p` at the repo root unless it's already absolute."""
    return p if Path(p).is_absolute() else str(_REPO_ROOT / p)


def _get_facemesh_class():
    try:
        fm_mod = mp.solutions.face_mesh
        return fm_mod.FaceMesh
    except AttributeError:
        return None


# 3-D canonical face model (mm) for solvePnP
FACE_3D_MODEL = np.array([
    [  0.0,    0.0,    0.0  ],   # Nose tip
    [  0.0, -330.0,  -65.0 ],   # Chin
    [-225.0,  170.0, -135.0],   # Left eye outer corner
    [ 225.0,  170.0, -135.0],   # Right eye outer corner
    [-150.0, -150.0, -125.0],   # Left mouth corner
    [ 150.0, -150.0, -125.0],   # Right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
FACE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# Left / Right iris landmark centres in MediaPipe
IRIS_LEFT_IDX  = 468
IRIS_RIGHT_IDX = 473

# Eye corner pairs (outer, inner) for normalisation
# In MediaPipe convention these are the user's left and right eyes.
EYE_CORNERS_LEFT  = (33,  133)
EYE_CORNERS_RIGHT = (263, 362)

# EAR (Eye Aspect Ratio) landmarks — one upper / one lower lid per eye.
# Lower EAR ⇒ eyelid closer to other lid ⇒ eye more closed.
EAR_LEFT_UPPER, EAR_LEFT_LOWER   = 159, 145
EAR_RIGHT_UPPER, EAR_RIGHT_LOWER = 386, 374

# Below this EAR ratio (lid-gap / eye-width), treat the eye as closed.
EAR_BLINK_RATIO = 0.16

# Hysteresis: how many consecutive slow frames count as a fixation
FIXATION_HYSTERESIS = 3
FIXATION_SPEED_PX   = 18.0   # px/frame


class GazeEstimator:
    """
    Estimates driver gaze and projects it to scene camera coordinates.
    Falls back to head-pose–based mapping when calibration is unavailable.
    """

    def __init__(self,
                 driver_calib_path:  str = 'calibration/driver_cam.yaml',
                 scene_calib_path:   str = 'calibration/scene_cam.yaml',
                 extrinsics_path:    str = 'calibration/extrinsics.yaml',
                 gaze_calib_path:    str = 'calibration/gaze_user.yaml',
                 scene_width: int = 1280,
                 scene_height: int = 720):

        # Anchor all calibration paths at the repo root so the estimator
        # works whether started from the project dir or somewhere else.
        driver_calib_path = _resolve(driver_calib_path)
        scene_calib_path  = _resolve(scene_calib_path)
        extrinsics_path   = _resolve(extrinsics_path)
        gaze_calib_path   = _resolve(gaze_calib_path)

        self.scene_w = scene_width
        self.scene_h = scene_height

        self._mode = "mesh"
        self.face_mesh = None
        self._landmarker = None

        FaceMesh = _get_facemesh_class()
        if FaceMesh is not None:
            self.face_mesh = FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            if not hasattr(mp, "tasks"):
                raise ImportError(
                    "This mediapipe version does not provide solutions or tasks APIs required for gaze estimation."
                )
            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            model_path = _resolve("models/face_landmarker.task")
            try:
                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=VisionRunningMode.IMAGE,
                    num_faces=1
                )
                self._landmarker = FaceLandmarker.create_from_options(options)
                self._mode = "tasks"
                print("[GazeEstimator] Using MediaPipe Tasks FaceLandmarker.")
            except Exception as e:
                raise ImportError(
                    "Failed to initialise MediaPipe FaceLandmarker. "
                    "Ensure a compatible face_landmarker.task model is present at models/face_landmarker.task."
                ) from e

        # Load / fall back camera calibration
        self._calibrated = False
        try:
            self.driver_K, self.driver_D = load_calibration(driver_calib_path)
            self.scene_K,  self.scene_D  = load_calibration(scene_calib_path)
            self.R_ext, self.T_ext       = load_extrinsics(extrinsics_path)
            self._calibrated = True
            print("[GazeEstimator] Camera-calibrated mode active.")
        except FileNotFoundError as e:
            print(f"[GazeEstimator] Camera calibration not available: {e}")
            self.driver_K, self.driver_D = make_default_calibration(640, 480)
            self.scene_K,  self.scene_D  = make_default_calibration(scene_width, scene_height)
            self.R_ext = np.eye(3)
            self.T_ext = np.array([0.3, 0.0, 0.5])  # rough offset (m)

        # Load personal gaze calibration if present — this is the accurate path
        self._gaze_calibrator: Optional[GazeCalibrator] = \
            GazeCalibrator.load(gaze_calib_path)
        if self._gaze_calibrator and self._gaze_calibrator.is_fit:
            print(f"[GazeEstimator] Personal gaze calibration loaded "
                  f"(median residual {self._gaze_calibrator.residual_px:.1f} px, "
                  f"{self._gaze_calibrator.sample_count} samples).")
        else:
            print("[GazeEstimator] No personal gaze calibration found at "
                  f"{gaze_calib_path}. Using uncalibrated fallback.")
            print("[GazeEstimator] For accurate tracking run:  "
                  "python scripts/calibrate_gaze.py")

        # State for velocity tracking + hysteresis fixation
        self._prev_gaze: Optional[Tuple[int, int]] = None
        self._gaze_velocity = (0.0, 0.0)
        self._smooth_gaze: Optional[Tuple[float, float]] = None
        self._slow_streak = 0
        self._last_good_gaze: Optional[Tuple[int, int]] = None
        self._eyes_closed_streak = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def process_frame(self, driver_frame: np.ndarray) -> Optional[Dict]:
        if driver_frame is None:
            return None

        rgb = cv2.cvtColor(driver_frame, cv2.COLOR_BGR2RGB)
        if self._mode == "mesh":
            results = self.face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return self._face_lost()
            landmarks = results.multi_face_landmarks[0].landmark
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)
            if not result.face_landmarks:
                return self._face_lost()
            landmarks = result.face_landmarks[0]

        h, w = driver_frame.shape[:2]

        # ── Head pose ──────────────────────────────────────────────────
        image_pts = np.array(
            [self._lm(landmarks[i], w, h) for i in FACE_LANDMARK_IDS],
            dtype=np.float64,
        )
        ok, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL, image_pts, self.driver_K, self.driver_D,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return self._face_lost()

        R, _ = cv2.Rodrigues(rvec)
        head_pose = self._rotation_to_euler(R)   # (pitch, yaw, roll) deg
        yaw_deg, pitch_deg = float(head_pose[1]), float(head_pose[0])

        # ── Iris features (sign-corrected, normalised) + EAR ───────────
        iris_dx, iris_dy, conf, ear_avg = self._iris_features(landmarks, w, h)

        eyes_closed = ear_avg < EAR_BLINK_RATIO
        if eyes_closed:
            self._eyes_closed_streak += 1
        else:
            self._eyes_closed_streak = 0

        # ── Project gaze to scene pixel ───────────────────────────────
        gaze_pt: Optional[Tuple[int, int]] = None

        if eyes_closed and self._last_good_gaze is not None:
            # Hold last good gaze through a blink, but mark it non-fixation
            gaze_pt = self._last_good_gaze
        elif self._gaze_calibrator and self._gaze_calibrator.is_fit:
            gaze_pt = self._gaze_calibrator.predict(
                iris_dx, iris_dy, yaw_deg, pitch_deg,
                self.scene_w, self.scene_h,
            )

        if gaze_pt is None:
            # Fallback path — 3-D ray through extrinsics, else linear head map
            gaze_dir_cam = self._iris_gaze_direction_3d(iris_dx, iris_dy, R)
            eye_origin = tvec.ravel() / 1000.0  # mm → m
            gaze_pt = project_gaze_to_scene(
                eye_origin, gaze_dir_cam,
                self.R_ext, self.T_ext,
                self.scene_K, self.scene_D,
                self.scene_w, self.scene_h,
            )
            if gaze_pt is None:
                gaze_pt = self._fallback_projection(head_pose)

        # ── EMA smoothing (α adapts: lower during blinks) ──────────────
        alpha = 0.40 if not eyes_closed else 0.10
        if self._smooth_gaze is None:
            self._smooth_gaze = (float(gaze_pt[0]), float(gaze_pt[1]))
        else:
            self._smooth_gaze = (
                alpha * gaze_pt[0] + (1 - alpha) * self._smooth_gaze[0],
                alpha * gaze_pt[1] + (1 - alpha) * self._smooth_gaze[1],
            )
        gaze_pt = (int(round(self._smooth_gaze[0])),
                   int(round(self._smooth_gaze[1])))

        # ── Velocity ──────────────────────────────────────────────────
        if self._prev_gaze is not None:
            self._gaze_velocity = (
                gaze_pt[0] - self._prev_gaze[0],
                gaze_pt[1] - self._prev_gaze[1],
            )
        self._prev_gaze = gaze_pt
        speed = float(np.hypot(*self._gaze_velocity))

        # ── Hysteresis fixation: slow for ≥N frames AND eyes open ─────
        if speed < FIXATION_SPEED_PX and not eyes_closed:
            self._slow_streak += 1
        else:
            self._slow_streak = 0
        is_fixation = self._slow_streak >= FIXATION_HYSTERESIS

        if not eyes_closed:
            self._last_good_gaze = gaze_pt

        # ── Confidence: combine iris confidence with eye openness ──────
        eye_open_factor = float(np.clip((ear_avg - EAR_BLINK_RATIO) /
                                        (0.10), 0.0, 1.0))
        confidence = float(conf * eye_open_factor)

        return {
            'gaze_point':       gaze_pt,
            'iris_offset':      (iris_dx, iris_dy),
            'gaze_velocity':    self._gaze_velocity,
            'gaze_speed':       speed,
            'is_fixation':      bool(is_fixation),
            'eyes_closed':      bool(eyes_closed),
            'ear':              float(ear_avg),
            'head_pose': {
                'yaw':   yaw_deg,
                'pitch': pitch_deg,
                'roll':  float(head_pose[2]),
            },
            'head_velocity': None,
            'confidence':    confidence,
            'calibrated':    bool(self._gaze_calibrator and
                                  self._gaze_calibrator.is_fit),
            'cam_calibrated': self._calibrated,
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _lm(self, lm, w, h) -> Tuple[float, float]:
        return lm.x * w, lm.y * h

    def _face_lost(self) -> Optional[Dict]:
        """Reset transient state when no face is detected."""
        self._prev_gaze = None
        self._smooth_gaze = None
        self._slow_streak = 0
        return None

    def _iris_features(self, landmarks, w, h
                       ) -> Tuple[float, float, float, float]:
        """
        Returns (iris_dx, iris_dy, confidence, ear_avg).
        - iris_dx/dy are eye-width-normalised offsets, sign-corrected so
          that iris_dx > 0 when the user looks to their right (i.e. toward
          camera-image left), and iris_dy > 0 when they look down.
        - confidence ∈ [0,1] reflects how well the iris is centred /
          how visible the eye is.
        - ear_avg is the average eye-aspect ratio across both eyes.
        """
        # Eye corners (outer = lateral, inner = nasal)
        L_out = np.array(self._lm(landmarks[EYE_CORNERS_LEFT[0]],  w, h))
        L_in  = np.array(self._lm(landmarks[EYE_CORNERS_LEFT[1]],  w, h))
        R_out = np.array(self._lm(landmarks[EYE_CORNERS_RIGHT[0]], w, h))
        R_in  = np.array(self._lm(landmarks[EYE_CORNERS_RIGHT[1]], w, h))

        left_center  = (L_out + L_in) / 2.0
        right_center = (R_out + R_in) / 2.0

        L_iris = np.array(self._lm(landmarks[IRIS_LEFT_IDX],  w, h))
        R_iris = np.array(self._lm(landmarks[IRIS_RIGHT_IDX], w, h))

        left_w  = np.linalg.norm(L_in - L_out) + 1e-6
        right_w = np.linalg.norm(R_in - R_out) + 1e-6

        # Image-space offsets (positive image x means iris toward image right)
        L_dx_img = (L_iris[0] - left_center[0])  / left_w
        R_dx_img = (R_iris[0] - right_center[0]) / right_w
        L_dy_img = (L_iris[1] - left_center[1])  / left_w
        R_dy_img = (R_iris[1] - right_center[1]) / right_w

        # Sign convention: looking RIGHT in the world means iris moves
        # toward image LEFT (selfie cam, no mirror). So world_dx = -image_dx.
        iris_dx = -0.5 * (L_dx_img + R_dx_img)
        # Looking DOWN ⇒ iris moves to image bottom ⇒ image_dy positive ⇒
        # world_dy positive. Sign already correct.
        iris_dy =  0.5 * (L_dy_img + R_dy_img)

        # Confidence: distance from iris to nearest corner / half-eye-width
        L_min = min(np.linalg.norm(L_iris - L_out),
                    np.linalg.norm(L_iris - L_in))
        R_min = min(np.linalg.norm(R_iris - R_out),
                    np.linalg.norm(R_iris - R_in))
        confidence = float(np.clip(
            (L_min / (left_w / 2) + R_min / (right_w / 2)) / 2.0,
            0.0, 1.0,
        ))

        # Eye Aspect Ratio (vertical lid gap / horizontal eye width)
        L_lid_gap = abs(landmarks[EAR_LEFT_LOWER].y -
                        landmarks[EAR_LEFT_UPPER].y) * h
        R_lid_gap = abs(landmarks[EAR_RIGHT_LOWER].y -
                        landmarks[EAR_RIGHT_UPPER].y) * h
        L_ear = L_lid_gap / left_w
        R_ear = R_lid_gap / right_w
        ear_avg = float((L_ear + R_ear) / 2.0)

        return float(iris_dx), float(iris_dy), confidence, ear_avg

    def _iris_gaze_direction_3d(self,
                                iris_dx: float,
                                iris_dy: float,
                                R: np.ndarray) -> np.ndarray:
        """
        Build a 3-D gaze direction in driver-cam space from the
        sign-corrected iris offsets. Tunable gain controls how strongly
        iris offset rotates the visual axis.
        """
        gain = 2.4   # empirical: maps eye-width-normalised offset to radians
        # Positive iris_dx (looking right in world) ⇒ +x in cam frame
        # Positive iris_dy (looking down)            ⇒ +y in cam frame
        # Forward axis pointing into the scene (out of cam)
        gaze_cam = np.array([iris_dx * gain, iris_dy * gain, 1.0])
        gaze_cam /= (np.linalg.norm(gaze_cam) + 1e-8)
        # Compose with head rotation so iris-relative gaze is expressed
        # in the world frame attached to the driver camera.
        gaze_world = R @ gaze_cam
        gaze_world /= (np.linalg.norm(gaze_world) + 1e-8)
        return gaze_world

    def _fallback_projection(self, head_pose) -> Tuple[int, int]:
        """Linear head-pose mapping when calibration and ray-cast fail."""
        pitch, yaw, _ = head_pose
        x = int(self.scene_w / 2 + (yaw   / 35.0) * (self.scene_w / 2))
        y = int(self.scene_h / 2 - (pitch / 25.0) * (self.scene_h / 3))
        x = max(0, min(self.scene_w - 1, x))
        y = max(0, min(self.scene_h - 1, y))
        return x, y

    def _rotation_to_euler(self, R: np.ndarray) -> np.ndarray:
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            pitch = np.arctan2(R[2, 1], R[2, 2])
            yaw   = np.arctan2(-R[2, 0], sy)
            roll  = np.arctan2(R[1, 0], R[0, 0])
        else:
            pitch = np.arctan2(-R[1, 2], R[1, 1])
            yaw   = np.arctan2(-R[2, 0], sy)
            roll  = 0.0
        return np.degrees([pitch, yaw, roll])

    def set_scene_resolution(self, w: int, h: int):
        """Update scene dimensions when actual resolution is known."""
        self.scene_w = w
        self.scene_h = h
        if self._gaze_calibrator and self._gaze_calibrator.is_fit:
            warn = self._gaze_calibrator.aspect_warning(w, h)
            if warn:
                print(f"[GazeEstimator] Note: {warn}")

    def cleanup(self):
        if self._mode == "mesh" and self.face_mesh is not None:
            self.face_mesh.close()
        if self._mode == "tasks" and self._landmarker is not None:
            self._landmarker.close()
