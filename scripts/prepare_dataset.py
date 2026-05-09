"""
Dataset Converter
Converts publicly available driving dataset annotations to the
training JSON format expected by train_intent.py.

Supported sources:
  - BDD-X (attention traces via gaze proxy from head pose)
  - DADA-2000 (driver attention + scene events)
  - DrFixD (driver fixation dataset)
  - Manual annotation from system logs (recommended)

Usage:
  python scripts/prepare_dataset.py \\
      --source dada \\
      --input  /data/DADA-2000 \\
      --output data/train_sequences.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
import argparse
import numpy as np


# ------------------------------------------------------------------ #
#  Intent labelling rules (heuristic, edit to match your annotations) #
# ------------------------------------------------------------------ #

INTENT_CLASSES = [
    'normal_forward',
    'mirror_check',
    'pedestrian_monitor',
    'lane_change_prepare',
    'intersection_scan'
]


def infer_intent_from_gaze(frames: list) -> str:
    """
    Heuristic intent labelling from a gaze sequence.
    Override with manual annotations when available.
    """
    if not frames:
        return 'normal_forward'

    gaze_pts = []
    obj_classes = []

    for fd in frames:
        gd = fd.get('gaze_data', {})
        gp = gd.get('gaze_point')
        if gp:
            gaze_pts.append(gp)
        ga = fd.get('gaze_affordance', {})
        lo = ga.get('looked_object', {})
        if lo:
            obj_classes.append(lo.get('class', 'none'))

    if not gaze_pts:
        return 'normal_forward'

    pts = np.array(gaze_pts)
    mean_x = pts[:, 0].mean()

    # Mirror check: gaze off to the sides consistently
    std_x = pts[:, 0].std()
    if std_x < 50 and (mean_x < 0.25 or mean_x > 0.75):
        return 'mirror_check'

    # Pedestrian monitor: looking at person for most of sequence
    from collections import Counter
    cls_counts = Counter(obj_classes)
    most_common_cls, mc_count = cls_counts.most_common(1)[0] if cls_counts else ('none', 0)
    if most_common_cls == 'person' and mc_count / max(1, len(obj_classes)) > 0.5:
        return 'pedestrian_monitor'

    # Lane change: gaze oscillating between mirrors and forward
    if std_x > 150:
        return 'lane_change_prepare'

    # Intersection scan: broad horizontal sweep
    x_range = pts[:, 0].max() - pts[:, 0].min()
    if x_range > 500:
        return 'intersection_scan'

    return 'normal_forward'


# ------------------------------------------------------------------ #
#  Loaders for specific datasets                                       #
# ------------------------------------------------------------------ #

def load_system_logs(log_dir: str, seq_len: int = 90) -> list:
    """
    Convert system log files (output of RollingLogger) to training sequences.
    Best source: logs recorded during manually-annotated driving sessions.
    """
    sequences = []
    log_files = sorted(Path(log_dir).glob('*.json'))
    print(f"Found {len(log_files)} log files in {log_dir}")

    for lf in log_files:
        with open(lf) as f:
            frames = json.load(f)

        if not isinstance(frames, list) or len(frames) < seq_len:
            continue

        # Slide window
        for start in range(0, len(frames) - seq_len + 1, seq_len // 2):
            chunk = frames[start: start + seq_len]
            intent = infer_intent_from_gaze(chunk)
            sequences.append({
                'frames':       chunk,
                'intent_label': intent,
                'source':       str(lf.name)
            })

    return sequences


def load_dada2000_stub(dada_root: str, seq_len: int = 90) -> list:
    """
    Stub loader for DADA-2000.
    DADA-2000 provides: driver gaze (2D), scene video, accident annotations.
    Download: https://github.com/JWFangit/LOTVS-DADA
    """
    sequences = []
    root = Path(dada_root)

    # DADA structure: each scenario folder has gaze_coord.csv + frames
    for scenario_dir in sorted(root.iterdir()):
        gaze_file = scenario_dir / 'gaze_coord.csv'
        if not gaze_file.exists():
            continue

        import csv
        rows = []
        with open(gaze_file) as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    rows.append((float(row[0]), float(row[1])))   # x, y (0-1)
                except (ValueError, IndexError):
                    continue

        if len(rows) < seq_len:
            continue

        # Convert gaze rows to frame_data format
        frames = []
        for gx, gy in rows:
            fd = {
                'gaze_data': {
                    'gaze_point': (int(gx * 1280), int(gy * 720)),
                    'head_pose':  {'yaw': 0, 'pitch': 0, 'roll': 0},
                    'is_fixation': True,
                    'gaze_velocity': (0, 0),
                    'confidence': 0.7
                },
                'detected_objects': [],
                'gaze_affordance': None
            }
            frames.append(fd)

        for start in range(0, len(frames) - seq_len + 1, seq_len // 2):
            chunk = frames[start: start + seq_len]
            sequences.append({
                'frames':       chunk,
                'intent_label': infer_intent_from_gaze(chunk),
                'source':       'DADA-2000'
            })

    return sequences


def load_drfixd_stub(drfixd_root: str, seq_len: int = 90) -> list:
    """
    Stub loader for DrFixD (driver fixation dataset).
    Download: https://github.com/taodeng/drfixd
    Provides per-frame fixation coordinates + saccade flags.
    """
    sequences = []
    print(f"[DrFixD] Scanning {drfixd_root} ...")
    # Implement based on actual DrFixD CSV structure
    # Each file: frame_id, fix_x, fix_y, is_fixation
    return sequences


# ------------------------------------------------------------------ #
#  Manual annotation helper                                            #
# ------------------------------------------------------------------ #

def create_annotation_template(log_file: str, output_file: str):
    """
    Create an annotation template from a log file.
    Open output_file in any JSON editor, fill in 'intent_label' for each sequence,
    then use the result for training.
    """
    with open(log_file) as f:
        frames = json.load(f)

    seq_len = 90
    template = []

    for start in range(0, len(frames) - seq_len + 1, seq_len):
        chunk = frames[start: start + seq_len]
        template.append({
            'sequence_start_frame': chunk[0].get('frame_number'),
            'sequence_end_frame':   chunk[-1].get('frame_number'),
            'intent_label':         'FILL_ME_IN',   # annotator fills this
            '_valid_labels':        INTENT_CLASSES,
            'frames':               chunk
        })

    with open(output_file, 'w') as f:
        json.dump(template, f, indent=2)

    print(f"Annotation template saved → {output_file}")
    print(f"Edit each 'intent_label' field, then use this file for training.")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',  choices=['logs', 'dada', 'drfixd', 'template'],
                        required=True)
    parser.add_argument('--input',   required=True, help='Input directory or file')
    parser.add_argument('--output',  required=True, help='Output JSON file')
    parser.add_argument('--seq-len', type=int, default=90, dest='seq_len')
    args = parser.parse_args()

    if args.source == 'logs':
        sequences = load_system_logs(args.input, args.seq_len)
    elif args.source == 'dada':
        sequences = load_dada2000_stub(args.input, args.seq_len)
    elif args.source == 'drfixd':
        sequences = load_drfixd_stub(args.input, args.seq_len)
    elif args.source == 'template':
        create_annotation_template(args.input, args.output)
        return

    if not sequences:
        print("No sequences found. Check your input path and dataset format.")
        return

    with open(args.output, 'w') as f:
        json.dump(sequences, f, indent=2)

    from collections import Counter
    intent_counts = Counter(s['intent_label'] for s in sequences)
    print(f"\nSaved {len(sequences)} sequences → {args.output}")
    print("Intent distribution:")
    for cls, cnt in intent_counts.most_common():
        print(f"  {cls:<30} {cnt}")


if __name__ == '__main__':
    main()
