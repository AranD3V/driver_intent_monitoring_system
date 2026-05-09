"""
Populate dead gaze dims in HDD sequences using a heuristic synthesizer.

HDD has detected_objects (from YOLO) but gaze_data is all-zero because
there is no driver-facing camera.  This script synthesizes plausible
gaze_x, gaze_y, is_fixation for each frame by treating the closest
detected object as the driver's fixation target.

gaze_affordance is also rewritten from the same closest object so that
gaze features (dims 0-4) and scene features (dims 11-33) in the feature
vector are consistent with each other.

Usage:
    python scripts/populate_gaze.py \
        --data   data/hdd_train_scene.json \
        --output data/hdd_gaze_enriched.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import numpy as np

SCENE_W, SCENE_H = 1280, 720
MAX_FIX_DIST = 40.0   # objects beyond this are treated as background (no fixation)
EMA_ALPHA    = 0.35

_AFF_MAP = {
    'person':        'Caution',
    'pedestrian':    'Caution',
    'motorcycle':    'Caution',
    'bicycle':       'Caution',
    'traffic_light': 'Instructional',
    'stop_sign':     'Instructional',
    'car':           'Constraint',
    'truck':         'Constraint',
    'bus':           'Constraint',
}


# ------------------------------------------------------------------ #
#  Per-frame heuristic                                                 #
# ------------------------------------------------------------------ #

def synthesize_frame_gaze(frame: dict):
    """
    Returns (gaze_x_norm, gaze_y_norm, fixation_prob, best_obj | None).
    Driver is assumed to fixate on the closest detected object.
    Falls back to (0.5, 0.5, 0.0, None) when no objects are detected.
    """
    objs = frame.get('detected_objects', [])
    if not objs:
        return 0.5, 0.5, 0.0, None

    best = min(objs, key=lambda o: o.get('distance_estimate', 999.0))
    dist = float(best.get('distance_estimate', 50.0))
    bbox = best.get('bbox', {})

    cx = float(bbox.get('center_x', SCENE_W / 2)) / SCENE_W
    cy = float(bbox.get('center_y', SCENE_H / 2)) / SCENE_H

    fix_prob = float(max(0.0, 1.0 - dist / MAX_FIX_DIST))

    return cx, cy, fix_prob, best


# ------------------------------------------------------------------ #
#  Temporal smoother (EMA)                                            #
# ------------------------------------------------------------------ #

def smooth_sequence(predictions: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    """Apply EMA smoothing to (T, 3) gaze predictions (x,y only)."""
    smoothed = predictions.copy()
    for t in range(1, len(smoothed)):
        smoothed[t, :2] = alpha * predictions[t, :2] + (1 - alpha) * smoothed[t - 1, :2]
    return smoothed


# ------------------------------------------------------------------ #
#  gaze_affordance builder                                             #
# ------------------------------------------------------------------ #

def _risk(dist: float) -> str:
    if dist < 5:   return 'critical'
    if dist < 15:  return 'high'
    if dist < 30:  return 'medium'
    return 'low'


def build_affordance(best_obj: dict | None, gaze_duration: float) -> dict:
    if best_obj is None:
        return {
            'looked_object': {},
            'affordance':    'none',
            'risk_level':    'low',
            'gaze_duration': 0.0,
            'distance':      50.0,
        }
    dist    = float(best_obj.get('distance_estimate', 50.0))
    cls     = best_obj.get('class', 'none')
    bbox    = best_obj.get('bbox', {})
    vel_raw = best_obj.get('velocity', [0.0, 0.0])
    vel     = vel_raw if isinstance(vel_raw, list) else [0.0, 0.0]
    return {
        'looked_object': {
            'class':               cls,
            'bbox':                bbox,
            'velocity':            vel,
            'distance_estimate':   dist,
            'confidence':          float(best_obj.get('confidence', 1.0)),
        },
        'affordance':    _AFF_MAP.get(cls, 'none'),
        'risk_level':    _risk(dist),
        'gaze_duration': gaze_duration,
        'distance':      dist,
    }


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def populate_gaze(args):
    with open(args.data) as f:
        sequences = json.load(f)
    print(f"Processing {len(sequences)} sequences from {args.data} ...")

    for seq_idx, seq in enumerate(sequences):
        frames = seq['frames']

        frame_results = [synthesize_frame_gaze(f) for f in frames]
        raw       = np.array([(r[0], r[1], r[2]) for r in frame_results], dtype=np.float32)
        best_objs = [r[3] for r in frame_results]
        preds     = smooth_sequence(raw, alpha=EMA_ALPHA)

        prev_gx, prev_gy = float(preds[0, 0]), float(preds[0, 1])
        gaze_dur_count = 0
        gaze_dur_cls   = None

        for t, frame in enumerate(frames):
            gx  = float(preds[t, 0])
            gy  = float(preds[t, 1])
            fp  = float(preds[t, 2])

            vx = float((gx - prev_gx) * SCENE_W)
            vy = float((gy - prev_gy) * SCENE_H)
            prev_gx, prev_gy = gx, gy

            frame['gaze_data'] = {
                'gaze_point':    [round(gx * SCENE_W, 2), round(gy * SCENE_H, 2)],
                'gaze_velocity': [round(vx, 4), round(vy, 4)],
                'is_fixation':   bool(fp > 0.5),
                'head_pose':     {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0},
                'confidence':    round(fp, 4),
                'calibrated':    True,
            }

            # Track how long gaze has been on the same object class.
            cur_cls = best_objs[t].get('class', None) if best_objs[t] else None
            if cur_cls is not None and cur_cls == gaze_dur_cls:
                gaze_dur_count += 1
            else:
                gaze_dur_count = 1
                gaze_dur_cls   = cur_cls

            frame['gaze_affordance'] = build_affordance(best_objs[t], float(gaze_dur_count))

        if (seq_idx + 1) % 100 == 0:
            print(f"  {seq_idx + 1}/{len(sequences)} done")

    with open(args.output, 'w') as f:
        json.dump(sequences, f)

    print(f"\nSaved {len(sequences)} sequences -> {args.output}")

    # Sanity check: mean gaze_x + std per class (variance between classes = good)
    from collections import defaultdict
    gx_per_class = defaultdict(list)
    for seq in sequences:
        lbl = seq.get('intent_label', '?')
        for frame in seq['frames']:
            gx_per_class[lbl].append(frame['gaze_data']['gaze_point'][0])
    print("\nMean gaze_x per class (640 = center, non-zero std = discriminative):")
    for lbl, vals in sorted(gx_per_class.items()):
        arr = np.array(vals)
        print(f"  {lbl:<25}  mean={arr.mean():.1f}  std={arr.std():.1f}")

    # Consistency check: gaze_point vs gaze_affordance.looked_object center
    mismatch = total = 0
    for seq in sequences:
        for frame in seq['frames']:
            gd = frame.get('gaze_data', {})
            ga = frame.get('gaze_affordance', {})
            lo = ga.get('looked_object', {})
            bbox = lo.get('bbox', {})
            if not bbox:
                continue
            total += 1
            gp = gd.get('gaze_point', [SCENE_W / 2, SCENE_H / 2])
            lx = float(bbox.get('center_x', SCENE_W / 2))
            ly = float(bbox.get('center_y', SCENE_H / 2))
            if abs(gp[0] - lx) > 50 or abs(gp[1] - ly) > 50:
                mismatch += 1
    if total:
        print(f"\nConsistency: {mismatch}/{total} frames "
              f"({100 * mismatch / total:.1f}%) have gaze_point vs "
              f"looked_object center >50px apart (target: <5%)")


def main():
    parser = argparse.ArgumentParser(description="Populate HDD gaze dims via heuristic synthesizer")
    parser.add_argument('--data',   default='data/hdd_train_scene.json')
    parser.add_argument('--output', default='data/hdd_gaze_enriched.json')
    args = parser.parse_args()
    populate_gaze(args)


if __name__ == '__main__':
    main()
