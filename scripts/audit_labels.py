"""
Label audit — sanity-check existing dataset labels against vehicle telemetry.

For every sequence we compute a small set of telemetry signals (turn signal
usage, |yaw_rate|, |steer_deg|, |delta_speed|, gaze availability) and check
whether the labeled intent is *plausible* given those signals. Sequences that
fail the plausibility check are written to a CSV for review.

Usage:
  python scripts/audit_labels.py --data data/hdd_train.json
  python scripts/audit_labels.py --data data/*.json --out audit_report.csv

Run after `prepare_dataset.py`, before training. A clean audit pass is the
single cheapest way to gain accuracy at this dataset size.
"""

import sys, json, csv, glob, argparse
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Plausibility rules ─────────────────────────────────────────────────────
#
# Each rule returns a list of warnings (empty if the label is plausible).
# Rules are intentionally LENIENT — we only flag clear mismatches, not
# borderline cases, because false flags waste annotator time more than
# missed flags.

# Telemetry thresholds — tuned to HDD signal ranges (see audit run output).
TS_MIN_FRAMES_FOR_LANE_CHANGE = 5     # at 30Hz ~ 0.17s
YAW_RATE_TURN_THRESHOLD       = 5.0   # deg/s sustained for >0.5s
STEER_TURN_THRESHOLD          = 30.0  # deg
NORMAL_FORWARD_MAX_YAW_FRAC   = 0.20  # fraction of frames with high yaw


def _seq_telemetry(seq: dict) -> dict:
    """Extract aggregate telemetry stats for one sequence."""
    frames = seq.get('frames', [])
    n = max(1, len(frames))

    ts_left  = sum(1 for f in frames if (f.get('vehicle') or {}).get('turn_signal_left',  0) > 0.5)
    ts_right = sum(1 for f in frames if (f.get('vehicle') or {}).get('turn_signal_right', 0) > 0.5)

    yaws   = [(f.get('vehicle') or {}).get('yaw_rate', 0.0) or 0.0 for f in frames]
    steers = [(f.get('vehicle') or {}).get('steer_angle_deg', 0.0) or 0.0 for f in frames]
    speeds = [(f.get('vehicle') or {}).get('speed_kmh', 0.0) or 0.0 for f in frames]

    high_yaw_frac = sum(1 for y in yaws if abs(y) > YAW_RATE_TURN_THRESHOLD) / n
    max_abs_yaw   = max((abs(y) for y in yaws),   default=0.0)
    max_abs_steer = max((abs(s) for s in steers), default=0.0)
    speed_delta   = (max(speeds) - min(speeds)) if speeds else 0.0
    mean_speed    = sum(speeds) / n if speeds else 0.0

    # Net yaw direction (sum of yaw rates) — left-turn is positive in HDD
    net_yaw = sum(yaws)

    has_real_gaze = any(
        (f.get('gaze_data') or {}).get('gaze_point', [640, 360]) != [640, 360]
        for f in frames
    )

    return {
        'n_frames':       n,
        'ts_left':        ts_left,
        'ts_right':       ts_right,
        'high_yaw_frac':  high_yaw_frac,
        'max_abs_yaw':    max_abs_yaw,
        'max_abs_steer':  max_abs_steer,
        'speed_delta':    speed_delta,
        'mean_speed':     mean_speed,
        'net_yaw':        net_yaw,
        'has_real_gaze':  has_real_gaze,
    }


def _check_plausibility(label: str, raw_label: str, t: dict) -> list:
    """Return list of warning strings; empty list = plausible."""
    warns = []

    if label == 'normal_forward':
        # Should not have sustained turn or aggressive turn signals.
        if t['high_yaw_frac'] > NORMAL_FORWARD_MAX_YAW_FRAC:
            warns.append(f"normal_forward but high_yaw_frac={t['high_yaw_frac']:.2f}")
        if t['ts_left'] > 20 or t['ts_right'] > 20:
            warns.append(f"normal_forward but turn_signal_frames L={t['ts_left']} R={t['ts_right']}")
        if t['max_abs_steer'] > 100:
            warns.append(f"normal_forward but max_steer={t['max_abs_steer']:.0f}deg")

    elif label == 'lane_change_prepare':
        # Expect at least *some* signal: turn signal OR lateral yaw.
        no_signal = (t['ts_left'] < TS_MIN_FRAMES_FOR_LANE_CHANGE
                     and t['ts_right'] < TS_MIN_FRAMES_FOR_LANE_CHANGE
                     and t['max_abs_yaw'] < YAW_RATE_TURN_THRESHOLD)
        if no_signal:
            warns.append("lane_change_prepare but no turn-signal and low yaw")

        # Direction mismatch — strong signal of mislabel
        if 'left' in raw_label and t['ts_right'] > 4 * max(1, t['ts_left']):
            warns.append(f"raw=left but ts_right={t['ts_right']} >> ts_left={t['ts_left']}")
        if 'right' in raw_label and t['ts_left'] > 4 * max(1, t['ts_right']):
            warns.append(f"raw=right but ts_left={t['ts_left']} >> ts_right={t['ts_right']}")

    elif label == 'intersection_scan':
        # Sub-cases: turn vs straight intersection passing.
        is_turn   = raw_label in ('left_turn', 'right_turn', 'U-turn')
        is_branch = raw_label in ('left_lane_branch', 'right_lane_branch')

        if is_turn:
            if t['high_yaw_frac'] < 0.05 and t['max_abs_yaw'] < YAW_RATE_TURN_THRESHOLD:
                warns.append(f"raw={raw_label} but no turn dynamics (high_yaw_frac={t['high_yaw_frac']:.2f})")
            if raw_label == 'left_turn' and t['net_yaw'] < 0:
                warns.append(f"left_turn but net_yaw={t['net_yaw']:.1f} (right-leaning)")
            if raw_label == 'right_turn' and t['net_yaw'] > 0:
                warns.append(f"right_turn but net_yaw={t['net_yaw']:.1f} (left-leaning)")

        # branches/passings: keep lenient

    elif label == 'pedestrian_monitor':
        # Telemetry alone can't confirm — only flag if vehicle is clearly turning
        if t['high_yaw_frac'] > 0.40:
            warns.append(f"pedestrian_monitor but vehicle is turning hard ({t['high_yaw_frac']:.2f})")

    return warns


# ── Main audit driver ─────────────────────────────────────────────────────

def audit(paths: list, out_csv: Path) -> None:
    flagged = []
    total = 0
    by_label_total   = Counter()
    by_label_flagged = Counter()

    for p in paths:
        try:
            data = json.load(open(p))
        except Exception as e:
            print(f"[!] {p}: {e}")
            continue

        for idx, seq in enumerate(data):
            total += 1
            label     = seq.get('intent_label', '')
            raw_label = seq.get('raw_label', '')
            seq_id    = seq.get('seq_id', f"{Path(p).stem}_{idx}")

            t      = _seq_telemetry(seq)
            warns  = _check_plausibility(label, raw_label, t)
            by_label_total[label] += 1

            if warns:
                by_label_flagged[label] += 1
                flagged.append({
                    'source':         Path(p).name,
                    'seq_id':         seq_id,
                    'idx':            idx,
                    'intent_label':   label,
                    'raw_label':      raw_label,
                    'n_frames':       t['n_frames'],
                    'ts_left':        t['ts_left'],
                    'ts_right':       t['ts_right'],
                    'high_yaw_frac':  f"{t['high_yaw_frac']:.3f}",
                    'max_abs_yaw':    f"{t['max_abs_yaw']:.2f}",
                    'max_abs_steer':  f"{t['max_abs_steer']:.1f}",
                    'speed_delta':    f"{t['speed_delta']:.1f}",
                    'mean_speed':     f"{t['mean_speed']:.1f}",
                    'net_yaw':        f"{t['net_yaw']:.1f}",
                    'has_real_gaze':  int(t['has_real_gaze']),
                    'warnings':       ' | '.join(warns),
                })

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\nAudited {total} sequences across {len(paths)} file(s)")
    print(f"Flagged {len(flagged)} ({100*len(flagged)/max(1,total):.1f}%)")
    print("\nPer-class flag rate:")
    for lbl in sorted(by_label_total):
        n = by_label_total[lbl]
        f = by_label_flagged[lbl]
        print(f"  {lbl:25s}  {f:4d} / {n:4d}  ({100*f/max(1,n):.1f}%)")

    # ── Write CSV ─────────────────────────────────────────────────────────
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if flagged:
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(flagged[0].keys()))
            w.writeheader()
            w.writerows(flagged)
        print(f"\nFlagged sequences -> {out_csv}")
    else:
        print("\nNo issues found, no CSV written.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True,
                    help='Dataset JSON file(s) or glob(s)')
    ap.add_argument('--out',  default='reports/audit_flagged.csv',
                    help='CSV path for flagged sequences')
    args = ap.parse_args()

    paths = []
    for pat in args.data:
        matched = sorted(glob.glob(pat))
        paths.extend(matched if matched else [pat])
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        print("No matching files."); sys.exit(1)

    audit(paths, Path(args.out))


if __name__ == '__main__':
    main()
