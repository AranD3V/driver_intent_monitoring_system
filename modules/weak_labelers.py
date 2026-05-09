"""
Weak labelers — high-precision rule-based intent classifiers.

Each labeler returns a (label, confidence) tuple or (None, 0.0) if it abstains.
Confidence is the labeler's prior precision on labeled data, learnable via
`calibrate_labelers`. Combine multiple labelers with consensus voting in
`scripts/consensus_label.py`.

Design principle: high precision, low recall. We'd rather skip 60% of the
unlabeled corpus than emit one bad label that pollutes training. Use the
confidence to weight examples in the loss, not to filter borderline cases.
"""

from typing import Optional, Tuple, Dict, List
from collections import Counter

INTENT_4CLASS = (
    'normal_forward',
    'lane_change_prepare',
    'intersection_scan',
    'pedestrian_monitor',
)

# ── Sequence-level telemetry helpers ─────────────────────────────────────

def _telemetry(frames: List[Dict]) -> Dict:
    n = max(1, len(frames))
    ts_left  = sum(1 for f in frames if (f.get('vehicle') or {}).get('turn_signal_left',  0) > 0.5)
    ts_right = sum(1 for f in frames if (f.get('vehicle') or {}).get('turn_signal_right', 0) > 0.5)
    yaws   = [(f.get('vehicle') or {}).get('yaw_rate', 0.0) or 0.0 for f in frames]
    steers = [(f.get('vehicle') or {}).get('steer_angle_deg', 0.0) or 0.0 for f in frames]
    speeds = [(f.get('vehicle') or {}).get('speed_kmh', 0.0) or 0.0 for f in frames]

    return {
        'n':              n,
        'ts_left':        ts_left,
        'ts_right':       ts_right,
        'ts_left_frac':   ts_left  / n,
        'ts_right_frac':  ts_right / n,
        'high_yaw_frac':  sum(1 for y in yaws if abs(y) > 5.0) / n,
        'max_abs_yaw':    max((abs(y) for y in yaws),   default=0.0),
        'max_abs_steer':  max((abs(s) for s in steers), default=0.0),
        'mean_abs_steer': sum(abs(s) for s in steers) / n if steers else 0.0,
        'net_yaw':        sum(yaws),
        'mean_speed':     sum(speeds) / n if speeds else 0.0,
        'speed_delta':    (max(speeds) - min(speeds)) if speeds else 0.0,
    }


def _scene_object_freq(frames: List[Dict]) -> Counter:
    """Histogram of looked-at object classes from gaze_affordance."""
    cls = Counter()
    for f in frames:
        ga = f.get('gaze_affordance')
        if ga and isinstance(ga, dict):
            obj = (ga.get('looked_object') or {}).get('class')
            if obj:
                cls[obj] += 1
    return cls


# ── Individual labelers ──────────────────────────────────────────────────

def label_telemetry(seq: Dict) -> Tuple[Optional[str], float]:
    """
    Vehicle-dynamics labeler. Reasons over turn signals + yaw + steering.
    Highest precision on lane_change_prepare and turn-style intersection_scan.

    Confidence is the per-bucket precision measured on hdd_train.json:
    - turn_signal_dominant + low_yaw  -> lane_change_prepare  (~90%)
    - sustained_high_yaw_or_steer     -> intersection_scan    (~85%)
    - flat_telemetry_long_window      -> normal_forward       (~80%)
    """
    t = _telemetry(seq.get('frames', []))

    # Lane change: dominant turn signal + low/moderate yaw, no sustained turn
    ts_dominant = (t['ts_left_frac'] > 0.10 or t['ts_right_frac'] > 0.10)
    ts_assymmetric = abs(t['ts_left'] - t['ts_right']) >= 5
    moderate_yaw = t['high_yaw_frac'] < 0.30 and t['max_abs_yaw'] < 15.0
    if ts_dominant and ts_assymmetric and moderate_yaw:
        return 'lane_change_prepare', 0.90

    # Turn-style intersection_scan: sustained yaw OR very high steer
    if t['high_yaw_frac'] > 0.20 or t['max_abs_steer'] > 200:
        return 'intersection_scan', 0.85

    # Normal forward: minimal vehicle dynamics across a meaningful window
    if (t['n'] >= 60                       # ~2s at 30Hz
            and t['high_yaw_frac'] < 0.05
            and t['ts_left'] < 5 and t['ts_right'] < 5
            and t['max_abs_steer'] < 40
            and t['mean_speed'] > 5):       # actually moving, not stopped
        return 'normal_forward', 0.80

    return None, 0.0


def label_raw_signal(seq: Dict) -> Tuple[Optional[str], float]:
    """
    Trust the raw_label when telemetry CONFIRMS direction — this is essentially
    a guarded passthrough that filters mislabeled raws via direction check.
    """
    raw = seq.get('raw_label', '')
    t = _telemetry(seq.get('frames', []))

    if raw in ('left_lane_change', 'left_lane_branch', 'merge'):
        if t['ts_left'] > t['ts_right'] or t['net_yaw'] >= 0:
            return 'lane_change_prepare', 0.92
        return None, 0.0

    if raw in ('right_lane_change', 'right_lane_branch'):
        if t['ts_right'] > t['ts_left'] or t['net_yaw'] <= 0:
            return 'lane_change_prepare', 0.92
        return None, 0.0

    if raw in ('left_turn', 'U-turn'):
        if t['net_yaw'] >= 0 and (t['high_yaw_frac'] > 0.10 or t['max_abs_steer'] > 100):
            return 'intersection_scan', 0.90
        return None, 0.0

    if raw == 'right_turn':
        if t['net_yaw'] <= 0 and (t['high_yaw_frac'] > 0.10 or t['max_abs_steer'] > 100):
            return 'intersection_scan', 0.90
        return None, 0.0

    if raw == 'intersection_passing':
        return 'intersection_scan', 0.80

    if raw == 'crosswalk_passing':
        return 'pedestrian_monitor', 0.80

    if raw == 'Background':
        # Background overlaps with maneuvers — only trust if telemetry agrees
        if (t['high_yaw_frac'] < 0.05
                and t['ts_left'] < 5 and t['ts_right'] < 5
                and t['max_abs_steer'] < 40):
            return 'normal_forward', 0.85
        return None, 0.0

    return None, 0.0


def label_gaze_object(seq: Dict) -> Tuple[Optional[str], float]:
    """
    Scene-object labeler. Only useful on sequences that have gaze_affordance
    populated (real driver-cam sessions, not bare HDD).
    """
    cls_hist = _scene_object_freq(seq.get('frames', []))
    if not cls_hist:
        return None, 0.0

    n = sum(cls_hist.values())
    if n < 10:
        return None, 0.0

    person_frac = cls_hist.get('person', 0) / n
    if person_frac > 0.40:
        return 'pedestrian_monitor', 0.85

    light_frac = (cls_hist.get('traffic_light', 0) + cls_hist.get('stop_sign', 0)) / n
    if light_frac > 0.30:
        return 'intersection_scan', 0.75

    return None, 0.0


# ── Public API ───────────────────────────────────────────────────────────

LABELERS = {
    'telemetry':   label_telemetry,
    'raw_signal':  label_raw_signal,
    'gaze_object': label_gaze_object,
}


def run_all(seq: Dict) -> Dict[str, Tuple[Optional[str], float]]:
    """Run every labeler on one sequence; returns {name: (label, conf)}."""
    return {name: fn(seq) for name, fn in LABELERS.items()}


def consensus(votes: Dict[str, Tuple[Optional[str], float]],
              min_voters: int = 2) -> Tuple[Optional[str], float, List[str]]:
    """
    Two-of-N consensus. Returns (label, weighted_confidence, agreeing_voters).
    Confidence = mean of agreeing labelers' confidences. None if no consensus.
    """
    by_label: Dict[str, list] = {}
    voter_by_label: Dict[str, list] = {}
    for name, (lbl, conf) in votes.items():
        if lbl is None:
            continue
        by_label.setdefault(lbl, []).append(conf)
        voter_by_label.setdefault(lbl, []).append(name)

    if not by_label:
        return None, 0.0, []

    best_label = max(by_label, key=lambda k: (len(by_label[k]), sum(by_label[k])))
    if len(by_label[best_label]) < min_voters:
        return None, 0.0, []

    confs = by_label[best_label]
    return best_label, sum(confs) / len(confs), voter_by_label[best_label]
