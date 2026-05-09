"""
Dr(eye)ve Dataset Preprocessor
================================
Converts dr(eye)ve sequence recordings into the annotated JSON format
expected by scripts/train_intent.py.

Pipeline per subsequence
------------------------
  video_garmin.avi  ──► SceneDetector (YOLOv8) ──► AffordanceEngine
  etg_samples.txt   ──► gaze point + event type
                         └──► IntersectionEngine ──► gaze_affordance
  speed_course_coord.txt ──► vehicle telemetry (speed, heading)
  subsequences.txt  ──► intent label per frame range

Label mapping (subsequences.txt column 5)
------------------------------------------
  k  →  normal_forward        (keep-lane / straight driving)
  e  →  lane_change_prepare   (ego maneuver: turn, overtake, lane change)
  i  →  intersection_scan     (at / approaching an intersection)

Output JSON schema (one file)
------------------------------
[
  {
    "intent_label": "normal_forward",
    "seq_id": 11,
    "source": "dreyeve",
    "frames": [
      {
        "gaze_data":        { "gaze_point": [x, y],
                              "gaze_velocity": [vx, vy],
                              "is_fixation": bool,
                              "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
                              "confidence": 1.0,
                              "calibrated": true },
        "detected_objects": [ <YOLO detection dicts> ],
        "gaze_affordance":  { ... } or null,
        "vehicle":          { "speed_kmh": float, "heading_deg": float }
      },
      ...
    ]
  },
  ...
]

Usage
-----
  python scripts/prepare_dreyeve.py \\
      --dataset  "D:/TW/dr(eye)ve" \\
      --output   data/dreyeve_train.json \\
      [--frame-skip 1]          # 1 = every frame, 2 = every other, etc.
      [--min-frames 30]         # discard subsequences shorter than N processed frames
      [--seq 11 36]             # only process these sequence IDs (default: all found)
      [--no-scene]              # skip YOLO (faster, gaze-only features)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import math
import numpy as np
import cv2
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import mediapipe as mp

from modules.scene_detector      import SceneDetector
from modules.affordance_engine   import AffordanceEngine
from modules.intersection_engine import IntersectionEngine
from modules.gaze_estimator      import FACE_3D_MODEL, FACE_LANDMARK_IDS


# ------------------------------------------------------------------ #
#  JSON serialisation helper                                           #
# ------------------------------------------------------------------ #

def _to_json(obj):
    """Recursively convert numpy scalars/arrays to native Python types."""
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #

LABEL_MAP = {
    'k': 'normal_forward',
    'e': 'lane_change_prepare',
    'i': 'intersection_scan',
}

# Garmin video resolution (will be auto-detected per sequence)
_DEFAULT_W, _DEFAULT_H = 1280, 720

# TTM backfill — DR(eye)VE ETG+Garmin are synced at 25 fps (see load_head_pose).
# MAX_TTM_SEC mirrors AnticipatoryFocalLoss.max_ttm_clip in modules/losses.py
# so 'k' (background) and derived mirror_check entries collapse to the anticip
# weight floor, exactly like HDD's normal_forward sentinel.
DREYEVE_FPS = 25
MAX_TTM_SEC = 10.0


# ================================================================== #
#  Gaze loader                                                         #
# ================================================================== #

def load_gaze(etg_path: Path) -> Dict[int, Dict]:
    """
    Parse etg_samples.txt into a per-garmin-frame lookup.

    Returns
    -------
    dict: frame_gar (int) -> {
        'gaze_point':    (x, y),
        'is_fixation':   bool,
        'event_type':    str,          # dominant event for the frame
    }
    Frames where ALL samples are Blinks use center-of-frame as fallback.
    """
    frame_samples: Dict[int, List] = defaultdict(list)

    with open(etg_path) as f:
        header = f.readline()   # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                frame_gar  = int(parts[1])
                x          = float(parts[2])
                y          = float(parts[3])
                event_type = parts[4]   # Fixation / Saccade / Blink
            except (ValueError, IndexError):
                continue
            frame_samples[frame_gar].append((x, y, event_type))

    gaze_map: Dict[int, Dict] = {}
    for fgar, samples in frame_samples.items():
        fixations = [(x, y) for x, y, e in samples if e == 'Fixation'
                     and not math.isnan(x) and not math.isnan(y)]
        saccades  = [(x, y) for x, y, e in samples if e == 'Saccade'
                     and not math.isnan(x) and not math.isnan(y)]

        if fixations:
            xs = [p[0] for p in fixations]
            ys = [p[1] for p in fixations]
            gaze_map[fgar] = {
                'gaze_point':  (float(np.mean(xs)), float(np.mean(ys))),
                'is_fixation': True,
                'event_type':  'Fixation',
            }
        elif saccades:
            xs = [p[0] for p in saccades]
            ys = [p[1] for p in saccades]
            gaze_map[fgar] = {
                'gaze_point':  (float(np.mean(xs)), float(np.mean(ys))),
                'is_fixation': False,
                'event_type':  'Saccade',
            }
        else:
            # All blinks — use None; will be filled with prior or center
            gaze_map[fgar] = {
                'gaze_point':  None,
                'is_fixation': False,
                'event_type':  'Blink',
            }

    return gaze_map


# ================================================================== #
#  Vehicle telemetry loader                                            #
# ================================================================== #

def load_telemetry(speed_path: Path) -> Dict[int, Dict]:
    """
    Parse speed_course_coord.txt.

    Columns: frame_number  speed(km/h)  heading(deg)  [lat  lon]
    Returns dict: frame_number (1-indexed int) -> {speed_kmh, heading_deg}
    """
    tel: Dict[int, Dict] = {}
    with open(speed_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                frame   = int(parts[0])
                speed   = float(parts[1])
                heading = float(parts[2])
            except (ValueError, IndexError):
                continue
            tel[frame] = {'speed_kmh': speed, 'heading_deg': heading}
    return tel


# ================================================================== #
#  Head pose extractor (MediaPipe on video_etg.avi)                   #
# ================================================================== #

def _rotation_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Rotation matrix → (pitch, yaw, roll) in degrees."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2( R[2, 1],  R[2, 2]))
        yaw   = np.degrees(np.arctan2(-R[2, 0],  sy))
        roll  = np.degrees(np.arctan2( R[1, 0],  R[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-R[1, 2],  R[1, 1]))
        yaw   = np.degrees(np.arctan2(-R[2, 0],  sy))
        roll  = 0.0
    return pitch, yaw, roll


def load_head_pose(etg_video_path: Path) -> Dict[int, Dict]:
    """
    Run MediaPipe FaceMesh on video_etg.avi and extract per-frame head pose.

    The ETG and garmin cameras are synchronised frame-for-frame at 25 fps,
    so ETG frame N maps to garmin frame N (1-indexed in the returned dict).

    Returns dict: gar_frame_num (1-indexed) -> {yaw, pitch, roll}  (degrees)
    Missing / no-face frames are omitted; callers should fall back to zeros.
    """
    cap = cv2.VideoCapture(str(etg_video_path))
    if not cap.isOpened():
        print(f"  [!] Cannot open ETG video: {etg_video_path} — head pose will be zeros")
        return {}

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Approximate intrinsics (no calibration file needed for offline use)
    focal   = vid_w * 0.8
    K = np.array([[focal, 0, vid_w / 2],
                  [0, focal, vid_h / 2],
                  [0,     0,          1]], dtype=np.float64)
    D = np.zeros(4, dtype=np.float64)

    # Try legacy solutions API first, fall back to Tasks API
    use_tasks = False
    face_mesh = None
    landmarker = None

    try:
        FaceMesh = mp.solutions.face_mesh.FaceMesh
        face_mesh = FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except AttributeError:
        # mediapipe >= 0.10.21 Tasks API
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        model_path = str(Path(__file__).parent.parent / "models" / "face_landmarker.task")
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
        )
        landmarker = FaceLandmarker.create_from_options(options)
        use_tasks = True
        print("    (using MediaPipe Tasks API)")

    pose_map: Dict[int, Dict] = {}
    frame_idx = 0
    detected  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lms = None

        if use_tasks:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)
            if result.face_landmarks:
                lms = result.face_landmarks[0]
        else:
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark

        if lms is not None:
            image_pts = np.array(
                [[lms[i].x * vid_w, lms[i].y * vid_h] for i in FACE_LANDMARK_IDS],
                dtype=np.float64
            )
            ok, rvec, _ = cv2.solvePnP(
                FACE_3D_MODEL, image_pts, K, D,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ok:
                R, _ = cv2.Rodrigues(rvec)
                pitch, yaw, roll = _rotation_to_euler(R)
                gar_frame = frame_idx + 1   # 1-indexed
                pose_map[gar_frame] = {
                    'yaw':   float(yaw),
                    'pitch': float(pitch),
                    'roll':  float(roll),
                }
                detected += 1

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"    head pose: {frame_idx}/{total} frames …")

    cap.release()
    if face_mesh is not None:
        face_mesh.close()
    if landmarker is not None:
        landmarker.close()

    pct = 100.0 * detected / max(1, frame_idx)
    print(f"  Head pose extracted: {detected}/{frame_idx} frames ({pct:.1f}% face detected)")
    return pose_map


# ================================================================== #
#  Subsequence loader                                                  #
# ================================================================== #

def load_subsequences(subseq_path: Path) -> Dict[int, List[Tuple]]:
    """
    Parse subsequences.txt.

    Columns: global_idx  seq_id  start_frame  end_frame  label
    Returns dict: seq_id (int) -> [(start, end, label), ...]
    Frames are 1-indexed (matching speed_course_coord.txt).
    Only rows whose label is in LABEL_MAP are included.
    """
    seq_map: Dict[int, List] = defaultdict(list)
    with open(subseq_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                seq_id = int(parts[0])
                start  = int(parts[1])
                end    = int(parts[2])
                label  = parts[3]
            except (ValueError, IndexError):
                continue
            if label in LABEL_MAP:
                seq_map[seq_id].append((start, end, label))
    return dict(seq_map)


# ================================================================== #
#  mirror_check derivation (Option C: head-pose OR gaze deviation)    #
# ================================================================== #

def _find_mirror_windows(frames_data: List[Dict],
                         vid_w: float,
                         yaw_deg_thresh: float = 30.0,
                         gaze_frac_thresh: float = 0.35,
                         min_frames: int = 15) -> List[Tuple[int, int]]:
    """
    Scan an entry's frames_data for sustained mirror-check conditions and
    return [start_idx, end_idx) runs (indices into `frames_data` itself,
    NOT raw video frame numbers).

    A frame triggers the mirror heuristic if EITHER:
      - |head_pose.yaw| > yaw_deg_thresh                 (physical head turn)
      - is_fixation AND |gaze_x - w/2| > frac * w        (extreme lateral gaze)

    A window is emitted when the trigger holds for >= min_frames
    consecutive entries in frames_data. Operates on whatever frame_skip
    the caller used, so tune min_frames to the effective fps.

    Default 15 @ 25fps (frame_skip=1) ≈ 0.6s of sustained sideward attention,
    matching the rule-based baseline in DATASETS_AND_TRAINING.md.
    """
    center_x    = vid_w / 2.0
    gaze_margin = gaze_frac_thresh * vid_w

    flags = [False] * len(frames_data)
    for i, fd in enumerate(frames_data):
        gd = fd.get('gaze_data', {}) or {}
        hp = gd.get('head_pose', {}) or {}
        yaw = abs(float(hp.get('yaw', 0.0)))
        head_trigger = yaw > yaw_deg_thresh

        gaze_trigger = False
        if gd.get('is_fixation', False):
            gp = gd.get('gaze_point') or [center_x, 0.0]
            gaze_trigger = abs(float(gp[0]) - center_x) > gaze_margin

        flags[i] = head_trigger or gaze_trigger

    windows: List[Tuple[int, int]] = []
    i = 0
    n = len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            if j - i >= min_frames:
                windows.append((i, j))
            i = j
        else:
            i += 1
    return windows


# ================================================================== #
#  Single-sequence processor                                           #
# ================================================================== #

def process_sequence(
    seq_dir:             Path,
    seq_id:              int,
    subseqs:             List[Tuple],
    scene_det:           Optional[SceneDetector],
    aff_eng:             Optional[AffordanceEngine],
    frame_skip:          int,
    min_frames:          int,
    use_head_pose:       bool = True,
    mirror_yaw_deg:      float = 30.0,
    mirror_gaze_frac:    float = 0.35,
    mirror_min_frames:   int = 15,
) -> List[Dict]:
    """
    Process one dr(eye)ve sequence directory and return a list of
    training entries (one per labeled subsequence).
    """
    gaze_map = load_gaze(seq_dir / 'etg_samples.txt')
    tel_map  = load_telemetry(seq_dir / 'speed_course_coord.txt')

    # ── Head pose from driver-facing ETG camera ───────────────────
    pose_map: Dict[int, Dict] = {}
    if use_head_pose:
        etg_video = seq_dir / 'video_etg.avi'
        if etg_video.exists():
            print(f"  Extracting head pose from {etg_video.name} …")
            pose_map = load_head_pose(etg_video)
        else:
            print(f"  [!] video_etg.avi not found — head pose will be zeros")

    cap = cv2.VideoCapture(str(seq_dir / 'video_garmin.avi'))
    if not cap.isOpened():
        print(f"  [!] Cannot open video: {seq_dir / 'video_garmin.avi'}")
        return []

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {vid_w}x{vid_h}, {total_frames} frames")

    center_x, center_y = vid_w / 2.0, vid_h / 2.0

    # ── Pre-read gaze + fill blink frames with nearest-prior ──────
    # Fill None gaze_point entries with the last valid point
    last_valid_gaze = (center_x, center_y)
    for fgar in sorted(gaze_map.keys()):
        if gaze_map[fgar]['gaze_point'] is None:
            gaze_map[fgar]['gaze_point'] = last_valid_gaze
        else:
            last_valid_gaze = gaze_map[fgar]['gaze_point']

    entries = []

    for (start, end, raw_label) in subseqs:
        intent_label = LABEL_MAP[raw_label]

        # Clamp to actual video length (1-indexed → 0-indexed seek)
        start_0 = max(0, start - 1)
        end_0   = min(total_frames - 1, end - 1)

        if end_0 <= start_0:
            continue

        print(f"  [{seq_id}] frames {start}–{end}  label={raw_label}→{intent_label}")

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_0)

        int_engine = IntersectionEngine()   # fresh per subsequence
        frames_data = []
        prev_gaze   = None

        frame_idx = start_0  # 0-indexed
        while frame_idx <= end_0:
            ret, scene_frame = cap.read()
            if not ret:
                break

            gar_frame_num = frame_idx + 1   # 1-indexed

            # ── Gaze ────────────────────────────────────────────
            g_entry = gaze_map.get(gar_frame_num, {
                'gaze_point':  (center_x, center_y),
                'is_fixation': False,
                'event_type':  'Saccade',
            })
            gx, gy       = g_entry['gaze_point']
            is_fixation  = g_entry['is_fixation']

            # Gaze velocity (px/frame)
            if prev_gaze is not None:
                vx = gx - prev_gaze[0]
                vy = gy - prev_gaze[1]
            else:
                vx, vy = 0.0, 0.0
            prev_gaze = (gx, gy)

            # ── Scene detection ──────────────────────────────────
            if scene_det is not None and (frame_idx - start_0) % frame_skip == 0:
                detections = scene_det.process_frame(scene_frame)
                detections = aff_eng.encode_affordances(detections)
            else:
                detections = []

            # ── Gaze–object intersection ─────────────────────────
            gaze_affordance = None
            if is_fixation:
                gaze_affordance = int_engine.find_gaze_target(
                    gaze_point  = (gx, gy),
                    gaze_speed  = float(np.hypot(vx, vy)),
                    is_fixation = is_fixation,
                    detections  = detections,
                )

            # ── Head pose from ETG camera ────────────────────────
            head_pose = pose_map.get(gar_frame_num, {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0})

            # ── Vehicle telemetry ────────────────────────────────
            tel = tel_map.get(gar_frame_num, {'speed_kmh': 0.0, 'heading_deg': 0.0})

            frame_data = _to_json({
                'gaze_data': {
                    'gaze_point':   [gx, gy],
                    'gaze_velocity':[vx, vy],
                    'is_fixation':  is_fixation,
                    'head_pose':    head_pose,
                    'confidence':   1.0,
                    'calibrated':   True,
                },
                'detected_objects': detections,
                'gaze_affordance':  gaze_affordance,
                'vehicle':          tel,
            })
            frames_data.append(frame_data)

            frame_idx += frame_skip

        if len(frames_data) < min_frames:
            print(f"    skipped (only {len(frames_data)} frames < min {min_frames})")
            continue

        # TTM for the base entry.
        # 'e' (maneuver) subsequences span the maneuver itself, so window
        # start == onset → ttm=0 → anticip weight = exp(0) = 1.0 (max).
        # 'k' (background) and 'i' (intersection approach, no sharp onset)
        # use the sentinel so the anticipation term collapses to the floor.
        if raw_label == 'e':
            ttm = 0.0
        else:
            ttm = MAX_TTM_SEC

        entries.append({
            'intent_label':     intent_label,
            'seq_id':           seq_id,
            'source':           'dreyeve',
            'raw_label':        raw_label,
            'time_to_maneuver': float(ttm),
            'frames':           frames_data,
        })
        print(f"    → {len(frames_data)} frames saved as '{intent_label}' "
              f"(ttm={ttm:.1f}s)")

        # ── Derived mirror_check entries (Option C heuristic) ────────
        # Scan every subsequence (k/e/i) for sustained head-pose OR lateral-
        # gaze triggers. Emitted entries overlap frames of the parent entry;
        # the sliding-window dataset treats them as distinct training samples.
        mirror_runs = _find_mirror_windows(
            frames_data, vid_w=vid_w,
            yaw_deg_thresh=mirror_yaw_deg,
            gaze_frac_thresh=mirror_gaze_frac,
            min_frames=max(min_frames, mirror_min_frames),
        )
        for (ms, me) in mirror_runs:
            mirror_frames = frames_data[ms:me]
            if len(mirror_frames) < min_frames:
                continue
            entries.append({
                'intent_label':     'mirror_check',
                'seq_id':           seq_id,
                'source':           'dreyeve',
                'raw_label':        f'derived_mirror({raw_label})',
                'time_to_maneuver': MAX_TTM_SEC,   # inert anticipation weight
                'frames':           mirror_frames,
            })
            print(f"    + mirror_check window [{ms}:{me}] "
                  f"({len(mirror_frames)} frames)")

    cap.release()
    return entries


# ================================================================== #
#  Main                                                                #
# ================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description='Convert dr(eye)ve sequences to training JSON'
    )
    parser.add_argument('--dataset',    required=True,
                        help='Root of dr(eye)ve dataset (contains Type 1/, Type 2/, subsequences.txt)')
    parser.add_argument('--output',     default='data/dreyeve_train.json',
                        help='Output JSON path (default: data/dreyeve_train.json)')
    parser.add_argument('--frame-skip', type=int, default=1, dest='frame_skip',
                        help='Process every Nth frame (1=all, 2=every other, …)')
    parser.add_argument('--min-frames', type=int, default=30, dest='min_frames',
                        help='Minimum processed frames to keep a subsequence (default 30)')
    parser.add_argument('--seq',        type=int, nargs='+', default=None,
                        help='Only process these sequence IDs (e.g. --seq 11 36)')
    parser.add_argument('--no-scene',    action='store_true',
                        help='Skip YOLO detection (faster; gaze-only features)')
    parser.add_argument('--no-headpose', action='store_true',
                        help='Skip head pose extraction from video_etg.avi')
    # ── mirror_check derivation (Option C: head-pose OR extreme gaze) ──
    parser.add_argument('--mirror-yaw-deg',    type=float, default=30.0,
                        dest='mirror_yaw_deg',
                        help='|head yaw| above this (deg) triggers mirror_check')
    parser.add_argument('--mirror-gaze-frac',  type=float, default=0.35,
                        dest='mirror_gaze_frac',
                        help='|gaze_x - w/2| > frac*w during fixation triggers mirror_check')
    parser.add_argument('--mirror-min-frames', type=int, default=15,
                        dest='mirror_min_frames',
                        help='Min consecutive trigger-frames for a mirror_check window '
                             '(default 15 @ 25fps ≈ 0.6s sustained).')
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    out_path     = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subseq_file = dataset_root / 'subsequences.txt'
    if not subseq_file.exists():
        print(f"[ERROR] subsequences.txt not found at {subseq_file}")
        sys.exit(1)

    print("Loading subsequences …")
    all_subseqs = load_subsequences(subseq_file)
    print(f"  Found labels for {len(all_subseqs)} sequence IDs")

    # ── Discover sequence directories (Type X / <id>) ──────────────
    seq_dirs: Dict[int, Path] = {}
    for type_dir in sorted(dataset_root.iterdir()):
        if not type_dir.is_dir() or not type_dir.name.startswith('Type'):
            continue
        for seq_dir in sorted(type_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            try:
                sid = int(seq_dir.name)
            except ValueError:
                continue
            seq_dirs[sid] = seq_dir

    if not seq_dirs:
        print(f"[ERROR] No sequence folders found under {dataset_root}")
        sys.exit(1)

    if args.seq:
        seq_dirs = {k: v for k, v in seq_dirs.items() if k in args.seq}

    print(f"Sequences to process: {sorted(seq_dirs.keys())}")

    # ── Initialise modules ─────────────────────────────────────────
    if args.no_scene:
        scene_det = None
        aff_eng   = None
        print("Scene detection disabled (--no-scene)")
    else:
        print("Loading YOLOv8 …")
        scene_det = SceneDetector(
            model_name            = 'yolov8n.pt',
            confidence_threshold  = 0.35,
            scene_calib_path      = 'calibration/scene_cam.yaml',
        )
        aff_eng = AffordanceEngine('config/affordance_config.json')
        print("YOLOv8 ready.")

    # ── Process each sequence ──────────────────────────────────────
    all_entries: List[Dict] = []

    for sid, seq_dir in sorted(seq_dirs.items()):
        subseqs = all_subseqs.get(sid)
        if not subseqs:
            print(f"\nSeq {sid:02d}: no labeled subsequences — skipping")
            continue

        print(f"\nSeq {sid:02d}  ({seq_dir})  —  {len(subseqs)} subsequences")
        entries = process_sequence(
            seq_dir            = seq_dir,
            seq_id             = sid,
            subseqs            = subseqs,
            scene_det          = scene_det,
            aff_eng            = aff_eng,
            frame_skip         = args.frame_skip,
            min_frames         = args.min_frames,
            use_head_pose      = not args.no_headpose,
            mirror_yaw_deg     = args.mirror_yaw_deg,
            mirror_gaze_frac   = args.mirror_gaze_frac,
            mirror_min_frames  = args.mirror_min_frames,
        )
        all_entries.extend(entries)
        print(f"  Seq {sid:02d} → {len(entries)} entries accumulated")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Total entries : {len(all_entries)}")
    from collections import Counter
    label_counts = Counter(e['intent_label'] for e in all_entries)
    for lbl, cnt in sorted(label_counts.items()):
        total_frames = sum(len(e['frames']) for e in all_entries if e['intent_label'] == lbl)
        print(f"  {lbl:<30} {cnt:>4} seqs  {total_frames:>6} frames")

    # TTM sanity check — zero entries = degenerate distribution (all at floor).
    ttm_by_cls: Dict[str, List[float]] = defaultdict(list)
    for e in all_entries:
        ttm_by_cls[e['intent_label']].append(
            float(e.get('time_to_maneuver', float('nan')))
        )
    print("\nTTM distribution per class (seconds):")
    for lbl, ts in sorted(ttm_by_cls.items()):
        arr = np.array([t for t in ts if not np.isnan(t)], dtype=float)
        if arr.size == 0:
            print(f"  {lbl:<30}  (no TTM)")
            continue
        print(f"  {lbl:<30}  n={arr.size:4d}  min={arr.min():.2f}  "
              f"med={float(np.median(arr)):.2f}  max={arr.max():.2f}  "
              f"mean={arr.mean():.2f}")

    if not all_entries:
        print("[WARNING] No entries produced. Check sequence IDs and subsequences.txt.")
        sys.exit(0)

    # ── Write output ───────────────────────────────────────────────
    with open(out_path, 'w') as f:
        json.dump(all_entries, f, indent=2)
    print(f"\nSaved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print("\nNext step:")
    print(f"  python scripts/train_intent.py train --data {out_path} --output models/intent_dreyeve.pth")


if __name__ == '__main__':
    main()
