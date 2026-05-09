"""
Feature-signal diagnostic: loads hdd_train.json, computes per-class mean/std for
every feature dim. Constant dims -> std ~ 0, and per-class means equal across
classes -> feature carries no discriminative signal for that dim.

Usage:
    python scripts/diagnose_features.py --data data/hdd_train.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.temporal_model import FeatureExtractor, FEATURE_DIM


FEATURE_NAMES = (
    [f"gaze_{n}" for n in ("x", "y", "vx", "vy", "fix")]
    + ["speed", "d_speed", "steer", "yaw", "brake", "turn"]
    + [f"obj_{n}" for n in ("none", "person", "car", "truck", "bus",
                             "motorcycle", "bicycle", "trafficlight", "stopsign")]
    + [f"aff_{n}" for n in ("none", "caution", "constraint", "instruct", "unknown")]
    + ["dur", "dist", "obj_vx", "obj_vy",
       "risk_lo", "risk_md", "risk_hi", "risk_cr", "tl_red"]
    + ["has_scene", "has_gaze"]
)
assert len(FEATURE_NAMES) == FEATURE_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--per-class-cap", type=int, default=30,
                    help="Sample this many entries per class for the scan")
    args = ap.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    by_class = defaultdict(list)
    for entry in data:
        by_class[entry["intent_label"]].append(entry)

    ex = FeatureExtractor()
    per_class_feats: dict = {}
    for cls, entries in by_class.items():
        sample = entries[: args.per_class_cap]
        all_vecs = []
        for e in sample:
            ex._prev_vehicle = None
            for fd in e["frames"]:
                all_vecs.append(ex.extract(fd))
        per_class_feats[cls] = np.stack(all_vecs)
        print(f"  {cls:<25} {len(entries):>5} entries -> "
              f"{per_class_feats[cls].shape[0]} frames sampled")

    print("\n" + "=" * 90)
    print(f"{'idx':>3} {'feature':<15} {'global_std':>11}  per-class mean (constant flag)")
    print("=" * 90)

    global_cat = np.concatenate(list(per_class_feats.values()), axis=0)
    global_std = global_cat.std(axis=0)

    dead = []
    for i in range(FEATURE_DIM):
        gstd = float(global_std[i])
        means = [float(per_class_feats[c][:, i].mean()) for c in per_class_feats]
        mean_spread = float(np.ptp(means))
        flag = " [DEAD]" if gstd < 1e-6 else (" [flat]" if mean_spread < 1e-3 else "")
        if gstd < 1e-6:
            dead.append(FEATURE_NAMES[i])
        means_str = "  ".join(f"{m:+.3f}" for m in means)
        print(f"{i:>3} {FEATURE_NAMES[i]:<15} {gstd:>11.4f}   {means_str}{flag}")

    print("=" * 90)
    print(f"\nDead dims ({len(dead)}/{FEATURE_DIM}): {dead}")
    if len(dead) >= 20:
        print("\n[CRITICAL] >=20 feature dims are constant across every sample.")
        print("           Re-run prepare_hdd.py with --scene to populate YOLO-based")
        print("           object/affordance/context features.")


if __name__ == "__main__":
    main()
