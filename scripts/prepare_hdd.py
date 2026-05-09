"""
Honda HDD (Driving Dataset) Preprocessor
==========================================
Converts Honda HDD pre-extracted sensor + target labels into the annotated
JSON format consumed by scripts/train_intent.py.

Expected input layout (pre-extracted, from 20200710_* tarballs):
    hdd_data/
      camera/<seq_id>/NNNNN.jpg     ← dashcam frames (30 fps, 1-indexed)
      sensor/<seq_id>.npy           ← (N, 8) CANBUS per-frame features
      target/<seq_id>.npy           ← (N,)  goal-oriented action labels

Sensor columns (8):
    0: accel_pedal        (%)
    1: steer_angle        (deg, ± ≈ 500)
    2: (unknown / rtk)
    3: velocity           (≈ km/h, 0–92)
    4: brake_pedal        (raw counts)
    5: turn_signal_left   (0/1)
    6: turn_signal_right  (0/1)
    7: yaw_rate           (deg/s)

Goal-oriented action labels (12 classes):
    0 Background
    1 intersection_passing
    2 left_turn        3 right_turn
    4 left_lane_change 5 right_lane_change
    6 left_lane_branch 7 right_lane_branch
    8 crosswalk_passing
    9 railroad_passing
    10 merge           11 U-turn

Mapping to Driver Intent Monitoring classes:
    0      -> normal_forward        (only sampled for balance)
    1,2,3,11 -> intersection_scan
    4,5,6,7,10 -> lane_change_prepare
    8      -> pedestrian_monitor
    9      -> (skipped — very few frames, not intent-relevant)

Output JSON schema matches prepare_dreyeve.py.

Usage
-----
    python scripts/prepare_hdd.py \
        --dataset "D:/TW/extracted/hdd_data" \
        --output  data/hdd_train.json \
        [--frame-skip 3]            # 30fps -> 10fps, fewer frames per segment
        [--min-frames 30]
        [--max-per-class 500]       # cap entries per class (huge dataset)
        [--background-samples 300]  # extra Background windows for normal_forward
        [--no-scene]                # skip YOLO (default: skip for speed)
        [--seq ID ID ...]           # only process these sequence IDs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import argparse
import random
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple


# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #

HDD_CLASS_NAMES = [
    'Background',                     # 0
    'intersection_passing',           # 1
    'left_turn',                      # 2
    'right_turn',                     # 3
    'left_lane_change',               # 4
    'right_lane_change',              # 5
    'left_lane_branch',               # 6
    'right_lane_branch',              # 7
    'crosswalk_passing',              # 8
    'railroad_passing',               # 9
    'merge',                          # 10
    'U-turn',                         # 11
]

LABEL_MAP = {
    1:  'intersection_scan',
    2:  'intersection_scan',
    3:  'intersection_scan',
    4:  'lane_change_prepare',
    5:  'lane_change_prepare',
    6:  'lane_change_prepare',
    7:  'lane_change_prepare',
    8:  'pedestrian_monitor',
    10: 'lane_change_prepare',
    11: 'intersection_scan',
    # 0  -> normal_forward handled separately via random background sampling
    # 9  -> railroad_passing: skipped (insufficient & not intent)
}

SCENE_W, SCENE_H = 1280, 720

# TTM backfill — HDD is 30 fps throughout (per dataset release notes).
# MAX_TTM_SEC must match AnticipatoryFocalLoss.max_ttm_clip in modules/losses.py
# so background/sentinel entries collapse the anticipation weight to its floor.
HDD_FPS      = 30
MAX_TTM_SEC  = 10.0

# Map YOLO class name -> affordance category (used when scene detection on)
CLASS_TO_AFFORDANCE = {
    'person':         'Caution',
    'bicycle':        'Caution',
    'motorcycle':     'Caution',
    'car':            'Constraint',
    'truck':          'Constraint',
    'bus':            'Constraint',
    'traffic light':  'Instructional',
    'stop sign':      'Instructional',
}


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _to_json(obj):
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def find_segments(target: np.ndarray, label: int, min_frames: int,
                  pre_context: int = 120, post_context: int = 60,
                  merge_gap: int = 60) -> List[Tuple[int, int]]:
    """Find segments containing label events (used only for Background sampling).

    HDD labels are brief (3-30 frame) event markers rather than continuous
    activity spans. We expand each event by pre_context/post_context frames
    and merge overlapping or nearby windows.
    """
    event_frames = np.where(target == label)[0]
    if len(event_frames) == 0:
        return []

    windows: List[List[int]] = []   # [start, end_exclusive]
    for f in event_frames:
        s = max(0, int(f) - pre_context)
        e = min(len(target), int(f) + post_context + 1)
        if windows and s <= windows[-1][1] + merge_gap:
            windows[-1][1] = max(windows[-1][1], e)
        else:
            windows.append([s, e])

    return [(s, e) for s, e in windows if e - s >= min_frames]


def find_event_onset_windows(target: np.ndarray, label: int,
                              window_len_frames: int,
                              pre_offsets_frames: List[int],
                              min_source_frames: int,
                              merge_gap_frames: int = 15
                              ) -> List[Tuple[int, int, int]]:
    """Multi-horizon anticipation windows per distinct event of `label`.

    For each event onset, emit one window per offset in pre_offsets_frames:
        window = [onset - offset, onset - offset + window_len_frames)
        ttm    = offset / HDD_FPS
    This produces a uniform TTM distribution across the offsets, which is
    what AnticipatoryFocalLoss needs to learn the anticipation horizon.

    Returns (start, end_exclusive, onset_abs) tuples.
    """
    event_frames = np.where(target == label)[0]
    if len(event_frames) == 0:
        return []

    # Cluster near-consecutive labels into single event onsets so we don't
    # emit duplicate anticipation windows for a multi-frame event span.
    onsets: List[int] = []
    prev = -merge_gap_frames - 1
    for f in event_frames:
        fi = int(f)
        if fi > prev + merge_gap_frames:
            onsets.append(fi)
        prev = fi

    N = len(target)
    windows: List[Tuple[int, int, int]] = []
    for onset in onsets:
        for off in pre_offsets_frames:
            start = onset - int(off)
            end   = start + window_len_frames
            if start < 0 or end > N:
                continue
            if end - start < min_source_frames:
                continue
            windows.append((start, end, onset))
    return windows


def sample_background_windows(target: np.ndarray, min_frames: int,
                              max_len: int, n_samples: int, rng) -> List[Tuple[int,int]]:
    """Sample random windows from Background (label 0) runs."""
    bg_segs = find_segments(target, 0, min_frames)
    windows = []
    for _ in range(n_samples):
        if not bg_segs:
            break
        s, e = bg_segs[rng.randrange(len(bg_segs))]
        max_possible = min(e - s, max_len)
        if max_possible < min_frames:
            continue
        win_len = rng.randint(min_frames, max_possible)
        win_start = rng.randint(s, e - win_len)
        windows.append((win_start, win_start + win_len))
    return windows


# ------------------------------------------------------------------ #
#  Frame-data builder                                                 #
# ------------------------------------------------------------------ #

def build_frame_data(sensor_row: np.ndarray,
                     prev_sensor: Optional[np.ndarray],
                     scene_detections: Optional[List[Dict]] = None) -> Dict:
    """
    Convert one (8,) sensor row into our frame_data packet.
    Gaze is placeholder (center screen, no velocity) — HDD has no gaze data.
    When scene_detections is provided, build a scene-aware gaze_affordance
    proxy (most salient object = closest/largest bbox).
    """
    accel, steer, _, vel, brake, ts_left, ts_right, yaw = sensor_row

    # NOTE: `detected_objects` intentionally dropped from output. FeatureExtractor
    # reads only `gaze_affordance`, and persisting the full detection list is a
    # memory bomb (~270KB/entry x 80k entries -> cv2 OOM on the YOLO pass).
    frame_data = {
        'gaze_data': {
            'gaze_point':    [SCENE_W / 2.0, SCENE_H / 2.0],
            'gaze_velocity': [0.0, 0.0],
            'is_fixation':   False,
            'head_pose':     {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0},
            'confidence':    0.0,
            'calibrated':    False,
        },
        'gaze_affordance':  None,
        'vehicle': {
            'speed_kmh':        float(vel),
            'steer_angle_deg':  float(steer),
            'brake':            float(brake),
            'accel_pedal':      float(accel),
            'turn_signal_left':  float(ts_left),
            'turn_signal_right': float(ts_right),
            'yaw_rate':          float(yaw),
            'heading_deg':       0.0,
        },
    }

    if scene_detections:
        frame_data['gaze_affordance'] = _build_affordance(scene_detections)
    return frame_data


def _build_affordance(detections: List[Dict]) -> Dict:
    """Pick the 'most salient' object and wrap it as a gaze_affordance.

    Saliency heuristic: pedestrians first, then closest vehicle, then
    traffic controls.  Gives the model a scene-attention proxy without
    real gaze data.
    """
    if not detections:
        return None

    def saliency(d):
        cls  = d.get('class', '')
        dist = d.get('distance_estimate') or 100.0
        # Pedestrians/cyclists dominate; then vehicles by closeness; then controls
        if cls in ('person', 'bicycle', 'motorcycle'):
            return (0, dist)
        if cls in ('car', 'truck', 'bus'):
            return (1, dist)
        if cls in ('traffic light', 'stop sign'):
            return (2, dist)
        return (3, dist)

    primary = min(detections, key=saliency)
    cls      = primary.get('class', 'none')
    dist     = primary.get('distance_estimate', 50)
    affordance = CLASS_TO_AFFORDANCE.get(cls, 'none')

    if dist < 10:
        risk = 'critical'
    elif dist < 20:
        risk = 'high'
    elif dist < 35:
        risk = 'medium'
    else:
        risk = 'low'

    return {
        'looked_object': {
            'class':    cls,
            'velocity': primary.get('velocity', [0.0, 0.0]),
            'traffic_light_state': primary.get('traffic_light_state', 'unknown'),
        },
        'affordance':    affordance,
        'gaze_duration': 30,   # treat the proxy-look as sustained
        'distance':      float(dist),
        'risk_level':    risk,
    }


# ------------------------------------------------------------------ #
#  Per-sequence processor                                              #
# ------------------------------------------------------------------ #

def process_sequence(seq_id: str,
                     sensor: np.ndarray,
                     target: np.ndarray,
                     args,
                     rng,
                     scene_detector=None,
                     camera_dir: Optional[Path] = None,
                     scene_cache: Optional[Dict] = None) -> List[Dict]:
    """Produce training entries for a single HDD sequence.

    scene_detector: optional SceneDetector instance (when --scene is on).
    camera_dir:     path to <hdd_root>/camera/<seq_id>/ for frame lookup.
    scene_cache:    per-sequence frame_idx -> detections cache so frames
                    shared across multiple label-segments aren't reprocessed.
    """
    entries: List[Dict] = []

    def _detections_for(fi: int) -> Optional[List[Dict]]:
        if scene_detector is None or camera_dir is None:
            return None
        if scene_cache is not None and fi in scene_cache:
            return scene_cache[fi]
        img_path = camera_dir / f"{fi+1:05d}.jpg"   # HDD frames 1-indexed
        if not img_path.exists():
            dets = []
        else:
            import cv2
            img = cv2.imread(str(img_path))
            dets = scene_detector.process_frame(img) if img is not None else []
        if scene_cache is not None:
            scene_cache[fi] = dets
        return dets

    # Handle mapped labels — emit multiple anticipation windows per event
    # so TTM spans the full [0, window_len/fps] range instead of saturating
    # at a single pre_context value.
    for label_int, intent in LABEL_MAP.items():
        segs = find_event_onset_windows(
            target, label_int,
            window_len_frames  = args.window_len,
            pre_offsets_frames = args.pre_offsets,
            min_source_frames  = args.min_frames * args.frame_skip,
        )
        for (s, e, onset_abs) in segs:
            ttm = min(MAX_TTM_SEC, max(0.0, (onset_abs - s) / HDD_FPS))

            frames = []
            prev = None
            for fi in range(s, e, args.frame_skip):
                dets = _detections_for(fi)
                fd = build_frame_data(sensor[fi], prev, scene_detections=dets)
                frames.append(fd)
                prev = sensor[fi]

            if len(frames) < args.min_frames:
                continue

            entries.append({
                'intent_label':     intent,
                'seq_id':           f"hdd_{seq_id}",
                'source':           'hdd',
                'raw_label':        HDD_CLASS_NAMES[label_int],
                'time_to_maneuver': float(ttm),
                'frames':           _to_json(frames),
            })

    # Background -> normal_forward
    bg_windows = sample_background_windows(
        target,
        min_frames = args.min_frames * args.frame_skip,
        max_len    = (args.max_seg_len or 300) * args.frame_skip,
        n_samples  = args.background_samples_per_seq,
        rng        = rng,
    )
    for (s, e) in bg_windows:
        frames = []
        prev = None
        for fi in range(s, e, args.frame_skip):
            dets = _detections_for(fi)
            fd = build_frame_data(sensor[fi], prev, scene_detections=dets)
            frames.append(fd)
            prev = sensor[fi]
        if len(frames) < args.min_frames:
            continue
        entries.append({
            'intent_label':     'normal_forward',
            'seq_id':           f"hdd_{seq_id}",
            'source':           'hdd',
            'raw_label':        'Background',
            'time_to_maneuver': MAX_TTM_SEC,   # sentinel → anticip-weight floor
            'frames':           _to_json(frames),
        })

    return entries


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser(description='Convert HDD to training JSON')
    ap.add_argument('--dataset', required=True,
                    help='Root containing camera/, sensor/, target/')
    ap.add_argument('--output',  default='data/hdd_train.json')
    ap.add_argument('--frame-skip', type=int, default=3, dest='frame_skip',
                    help='30fps source -> every Nth frame (3 = 10fps)')
    ap.add_argument('--min-frames', type=int, default=30, dest='min_frames',
                    help='Minimum frames per output segment')
    ap.add_argument('--max-seg-len', type=int, default=300, dest='max_seg_len',
                    help='Cap source frames per Background segment (before skip). '
                         'Mapped-label windows use --window-len instead.')
    ap.add_argument('--window-len', type=int, default=270, dest='window_len',
                    help='Source frames per event window (default 270 = 9.0s @ 30fps). '
                         'Must satisfy window_len / frame_skip >= train seq_len '
                         '(90 by default) or IntentSequenceDataset will silently '
                         'drop these entries.')
    ap.add_argument('--pre-offsets', type=int, nargs='+',
                    default=list(range(0, 121, 15)),
                    dest='pre_offsets',
                    help='Pre-onset offsets in frames; one window emitted per '
                         'offset per event. Default 0..120 step 15 = 9 '
                         'anticipation horizons (TTM 0.0s..4.0s @ 30fps).')
    ap.add_argument('--max-per-class', type=int, default=500, dest='max_per_class',
                    help='Cap total entries per intent class after all seqs')
    ap.add_argument('--background-samples-per-seq', type=int, default=4,
                    dest='background_samples_per_seq',
                    help='Random Background windows sampled per sequence')
    ap.add_argument('--seq', nargs='+', default=None,
                    help='Only process these sequence IDs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--scene', action='store_true',
                    help='Run YOLO on camera frames to populate detected_objects '
                         'and gaze_affordance (slow: ~2-3h on CPU for full HDD).')
    ap.add_argument('--yolo-weights', default='yolov8n.pt', dest='yolo_weights')
    ap.add_argument('--yolo-conf',    type=float, default=0.35, dest='yolo_conf')
    args = ap.parse_args()

    rng = random.Random(args.seed)

    ds = Path(args.dataset)
    sensor_dir = ds / 'sensor'
    target_dir = ds / 'target'
    camera_root = ds / 'camera'

    if not sensor_dir.exists() or not target_dir.exists():
        print(f"[ERROR] sensor/ or target/ not found under {ds}")
        sys.exit(1)

    # Scene detector (opt-in)
    scene_detector = None
    if args.scene:
        if not camera_root.exists():
            print(f"[ERROR] --scene requires camera/ under {ds}")
            sys.exit(1)
        print(f"Loading YOLO '{args.yolo_weights}' for scene detection...")
        from modules.scene_detector import SceneDetector
        scene_detector = SceneDetector(
            model_name=args.yolo_weights,
            confidence_threshold=args.yolo_conf,
        )

    seq_ids = sorted(p.stem for p in sensor_dir.glob('*.npy'))
    if args.seq:
        seq_ids = [s for s in seq_ids if s in args.seq]

    print(f"Sequences to process: {len(seq_ids)}"
          + ("  (scene detection ON)" if args.scene else ""))

    # Per-sequence shards: resumable after crashes. Each shard is one seq's
    # entries as standalone JSON in <output>_shards/<sid>.json.
    out = Path(args.output)
    shard_dir = out.parent / f"{out.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    import gc

    all_entries: List[Dict] = []
    t_start = time.time()
    total_frames_processed = 0
    for i, sid in enumerate(seq_ids):
        shard_path = shard_dir / f"{sid}.json"
        if shard_path.exists():
            with open(shard_path) as f:
                entries = json.load(f)
            all_entries.extend(entries)
            print(f"  [{i+1}/{len(seq_ids)}] {sid}  cached shard ({len(entries)} entries)  cum={len(all_entries)}")
            continue

        sensor = np.load(sensor_dir / f"{sid}.npy")
        target = np.load(target_dir / f"{sid}.npy")
        if len(sensor) != len(target):
            print(f"  [!] Seq {sid}: length mismatch ({len(sensor)} vs {len(target)}) - skipping")
            continue

        # Reset tracker between sequences so track IDs don't leak across drives
        if scene_detector is not None:
            scene_detector.tracker.tracks.clear()

        camera_dir  = (camera_root / sid) if args.scene else None
        scene_cache: Dict[int, List[Dict]] = {} if args.scene else None

        t0 = time.time()
        entries = process_sequence(
            sid, sensor, target, args, rng,
            scene_detector=scene_detector,
            camera_dir=camera_dir,
            scene_cache=scene_cache,
        )
        dt = time.time() - t0
        frames_this_seq = len(scene_cache) if scene_cache is not None else 0
        total_frames_processed += frames_this_seq

        with open(shard_path, 'w') as f:
            json.dump(_to_json(entries), f)
        all_entries.extend(entries)

        del scene_cache
        gc.collect()

        elapsed = time.time() - t_start
        eta = (elapsed / (i + 1)) * (len(seq_ids) - i - 1)
        print(f"  [{i+1}/{len(seq_ids)}] {sid}  "
              f"entries={len(entries)}  cum={len(all_entries)}  "
              f"{'frames=' + str(frames_this_seq) + '  ' if args.scene else ''}"
              f"seq={dt:.1f}s  eta={eta/60:.1f}min")

    # Cap per-class
    if args.max_per_class:
        by_class: Dict[str, List[Dict]] = defaultdict(list)
        for e in all_entries:
            by_class[e['intent_label']].append(e)
        capped: List[Dict] = []
        for cls, lst in by_class.items():
            if len(lst) > args.max_per_class:
                rng.shuffle(lst)
                lst = lst[:args.max_per_class]
            capped.extend(lst)
        print(f"Capped to <={args.max_per_class} per class: {len(all_entries)} -> {len(capped)}")
        all_entries = capped

    # Summary
    counts = Counter(e['intent_label'] for e in all_entries)
    total_frames = sum(len(e['frames']) for e in all_entries)
    print(f"\n{'='*55}")
    print(f"Total entries : {len(all_entries)}")
    print(f"Total frames  : {total_frames:,}")
    for lbl, cnt in sorted(counts.items()):
        f = sum(len(e['frames']) for e in all_entries if e['intent_label'] == lbl)
        print(f"  {lbl:<25} {cnt:>5} seqs  {f:>7} frames")

    # TTM sanity check — flags degenerate (all-sentinel) distributions fast.
    ttm_by_cls: Dict[str, List[float]] = defaultdict(list)
    for e in all_entries:
        ttm_by_cls[e['intent_label']].append(float(e.get('time_to_maneuver', float('nan'))))
    print("\nTTM distribution per class (seconds):")
    for lbl, ts in sorted(ttm_by_cls.items()):
        arr = np.array([t for t in ts if not np.isnan(t)], dtype=float)
        if arr.size == 0:
            print(f"  {lbl:<25}  (no TTM)")
            continue
        print(f"  {lbl:<25}  n={arr.size:4d}  min={arr.min():.2f}  "
              f"med={float(np.median(arr)):.2f}  max={arr.max():.2f}  "
              f"mean={arr.mean():.2f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_entries, f)
    print(f"\nSaved -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"Per-seq shards retained in {shard_dir} (delete to force full rerun)")


if __name__ == '__main__':
    main()
