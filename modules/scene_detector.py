"""
Scene Perception Module (v2)
YOLOv8 detection + lightweight IoU-based tracker for persistent object IDs.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from ultralytics import YOLO

from modules.calibration import load_calibration, make_default_calibration


_REPO_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ #
#  Lightweight IoU tracker (no external dependency on SORT/DeepSORT)  #
# ------------------------------------------------------------------ #

class _Track:
    _next_id = 1

    def __init__(self, bbox: Dict, class_name: str):
        self.id = _Track._next_id
        _Track._next_id += 1
        self.bbox = bbox
        self.class_name = class_name
        self.age = 1
        self.hits = 1
        self.time_since_update = 0

        # Velocity estimation
        self._prev_center = np.array([bbox['center_x'], bbox['center_y']], dtype=float)
        self.velocity = np.zeros(2)

    def update(self, bbox: Dict):
        center = np.array([bbox['center_x'], bbox['center_y']], dtype=float)
        self.velocity = center - self._prev_center
        self._prev_center = center
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0

    def predict(self):
        self.time_since_update += 1
        self.age += 1


class IoUTracker:
    """Simple IoU-based multi-object tracker."""

    def __init__(self, max_age: int = 5, min_hits: int = 2, iou_thresh: float = 0.35):
        self.max_age   = max_age
        self.min_hits  = min_hits
        self.iou_thresh = iou_thresh
        self.tracks: List[_Track] = []

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Match detections to existing tracks.
        Returns enriched detections with persistent 'track_id' and 'velocity'.
        """
        for t in self.tracks:
            t.predict()

        matched, unmatched_dets = self._match(detections)

        # Update matched
        for d_idx, t_idx in matched:
            self.tracks[t_idx].update(detections[d_idx]['bbox'])
            detections[d_idx]['track_id'] = self.tracks[t_idx].id
            detections[d_idx]['velocity'] = self.tracks[t_idx].velocity.tolist()

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            t = _Track(detections[d_idx]['bbox'], detections[d_idx]['class'])
            self.tracks.append(t)
            detections[d_idx]['track_id'] = t.id
            detections[d_idx]['velocity'] = [0.0, 0.0]

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return detections

    def _match(self, detections):
        if not self.tracks or not detections:
            return [], list(range(len(detections)))

        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        for d_i, det in enumerate(detections):
            for t_i, trk in enumerate(self.tracks):
                iou_matrix[d_i, t_i] = self._iou(det['bbox'], trk.bbox)

        # Greedy matching
        matched, unmatched_dets = [], []
        used_tracks = set()
        for d_i in range(len(detections)):
            best_t = int(iou_matrix[d_i].argmax())
            if iou_matrix[d_i, best_t] >= self.iou_thresh and best_t not in used_tracks:
                matched.append((d_i, best_t))
                used_tracks.add(best_t)
            else:
                unmatched_dets.append(d_i)

        return matched, unmatched_dets

    @staticmethod
    def _iou(a: Dict, b: Dict) -> float:
        xa1, ya1, xa2, ya2 = a['x1'], a['y1'], a['x2'], a['y2']
        xb1, yb1, xb2, yb2 = b['x1'], b['y1'], b['x2'], b['y2']
        ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
        ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        return inter / (area_a + area_b - inter)


# ------------------------------------------------------------------ #
#  Scene Detector                                                      #
# ------------------------------------------------------------------ #

DRIVING_CLASSES = {
    'person', 'car', 'truck', 'bus',
    'motorcycle', 'bicycle', 'traffic light', 'stop sign'
}


class SceneDetector:

    def __init__(self,
                 model_name: str = 'yolov8n.pt',
                 confidence_threshold: float = 0.4,
                 scene_calib_path: str = 'calibration/scene_cam.yaml'):

        # Resolve relative model paths against the repo root so the tool
        # works regardless of CWD. If the user passes an absolute path or
        # a bare model name (auto-download), pass through unchanged.
        model_arg = model_name
        if not Path(model_name).is_absolute():
            local = _REPO_ROOT / model_name
            if local.exists():
                model_arg = str(local)

        try:
            self.model = YOLO(model_arg)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model '{model_arg}'. "
                f"Ensure the file exists in the project root.\n"
                f"Auto-download: python -c \"from ultralytics import YOLO; YOLO('{model_name}')\"\n"
                f"Original error: {e}"
            ) from e
        self.conf_thresh = confidence_threshold
        self.class_names = self.model.names
        self.tracker = IoUTracker()

        # Calibration for distance estimation
        if not Path(scene_calib_path).is_absolute():
            scene_calib_path = str(_REPO_ROOT / scene_calib_path)
        try:
            K, _ = load_calibration(scene_calib_path)
            self._focal_length = (K[0, 0] + K[1, 1]) / 2.0
            self._focal_length_calibrated = True
        except FileNotFoundError:
            # Provisional value until set_scene_resolution() is called
            self._focal_length = 1000.0
            self._focal_length_calibrated = False

    def process_frame(self, scene_frame: np.ndarray) -> List[Dict]:
        if scene_frame is None:
            return []

        results = self.model(scene_frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            cls_id    = int(box.cls[0])
            cls_name  = self.class_names[cls_id]
            conf      = float(box.conf[0])

            if cls_name not in DRIVING_CLASSES or conf < self.conf_thresh:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            bbox = {
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'width':    x2 - x1,
                'height':   y2 - y1,
                'center_x': (x1 + x2) // 2,
                'center_y': (y1 + y2) // 2
            }

            det: Dict = {
                'class':             cls_name,
                'bbox':              bbox,
                'confidence':        conf,
                'distance_estimate': self._estimate_distance(cls_name, bbox['height']),
                'track_id':          -1,
                'velocity':          [0.0, 0.0]
            }

            if cls_name == 'traffic light':
                det['traffic_light_state'] = self._detect_tl_state(scene_frame, bbox)

            detections.append(det)

        # Assign persistent track IDs
        detections = self.tracker.update(detections)

        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return detections

    # ------------------------------------------------------------------ #

    def set_scene_resolution(self, w: int, h: int,
                              hfov_deg: float = 70.0) -> None:
        """
        When no scene-camera calibration is loaded, derive a sensible
        focal length from the actual scene resolution and an assumed
        horizontal FOV. Without this, distance estimates are wildly off
        in simulator mode (where the virtual camera has different
        intrinsics from the 1280x720 webcam baked into the default).

        Real webcams: 60-65 deg HFOV.
        MetaDrive default RGB sensor: ~70 deg.
        CARLA default: 90 deg.
        70 deg is a safe middle ground when we don't know.
        """
        if self._focal_length_calibrated:
            return
        import math
        f = float(w) / (2.0 * math.tan(math.radians(hfov_deg / 2.0)))
        self._focal_length = f

    def _estimate_distance(self, cls_name: str, bbox_height: int) -> float:
        REAL_HEIGHTS = {
            'person': 1.7, 'car': 1.5, 'truck': 3.0, 'bus': 3.5,
            'motorcycle': 1.2, 'bicycle': 1.5,
            'traffic light': 3.0, 'stop sign': 1.2
        }
        rh = REAL_HEIGHTS.get(cls_name, 1.5)
        if bbox_height > 0:
            return round((rh * self._focal_length) / bbox_height, 2)
        return float('inf')

    def _detect_tl_state(self, frame: np.ndarray, bbox: Dict) -> str:
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        roi = frame[max(0, y1):y2, max(0, x1):x2]
        if roi.size == 0:
            return 'unknown'

        # Only look at top / middle / bottom thirds for R / Y / G
        h = roi.shape[0]
        thirds = {
            'red':    roi[:h//3],
            'yellow': roi[h//3: 2*h//3],
            'green':  roi[2*h//3:]
        }

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        counts = {}

        RED_LO1, RED_HI1 = np.array([0,  120, 120]), np.array([10,  255, 255])
        RED_LO2, RED_HI2 = np.array([160,120, 120]), np.array([180, 255, 255])
        YEL_LO,  YEL_HI  = np.array([15, 120, 120]), np.array([35,  255, 255])
        GRN_LO,  GRN_HI  = np.array([40, 100, 100]), np.array([80,  255, 255])

        top_hsv = cv2.cvtColor(roi[:h//3],    cv2.COLOR_BGR2HSV)
        mid_hsv = cv2.cvtColor(roi[h//3:2*h//3], cv2.COLOR_BGR2HSV)
        bot_hsv = cv2.cvtColor(roi[2*h//3:],  cv2.COLOR_BGR2HSV)

        counts['red']    = cv2.countNonZero(cv2.bitwise_or(
                               cv2.inRange(top_hsv, RED_LO1, RED_HI1),
                               cv2.inRange(top_hsv, RED_LO2, RED_HI2)))
        counts['yellow'] = cv2.countNonZero(cv2.inRange(mid_hsv, YEL_LO, YEL_HI))
        counts['green']  = cv2.countNonZero(cv2.inRange(bot_hsv, GRN_LO, GRN_HI))

        best = max(counts, key=counts.get)
        return best if counts[best] > 10 else 'unknown'
