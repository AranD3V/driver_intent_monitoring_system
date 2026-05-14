"""
Simulation Labeler — extracts ground-truth labels per frame from MetaDrive's
world state. Produces three things every tick:

  1. lane / road context   (broken vs solid, white vs yellow, wrong side, ...)
  2. current vehicle action (accelerate / brake / coast / steer_L|R / lane_change)
  3. violations            (overspeed, harsh_brake, crossed_solid, off_road, ...)

These labels are oracle-grade (read straight from the sim) — far cleaner than
anything YOLO could infer from the rendered RGB. They're persisted alongside
the per-frame session log and rolled up into a per-run violations.csv.

The labeler does NOT depend on MetaDrive being importable — when the
underlying capture isn't a MetaDriveCapture (or sim_info isn't available)
every method returns empty dicts so the rest of the pipeline doesn't care.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple


# ── Action thresholds (tuned against MetaDrive default action space) ──────
ACCEL_THRESHOLD = 0.15      # info['acceleration'] above this -> "accelerate"
BRAKE_THRESHOLD = -0.15     # info['acceleration'] below this -> "brake"
STEER_THRESHOLD = 0.10      # |info['steering']| above this -> "steer_L/R"
LANE_CHANGE_HOLD_S = 0.6    # sustained steer + still on road -> lane change

# ── Violation thresholds ──────────────────────────────────────────────────
OVERSPEED_HIGH_KMH = 10.0   # over the lane limit
OVERSPEED_CRIT_KMH = 25.0
HARSH_ACCEL_MPS2   = 3.5
HARSH_BRAKE_MPS2   = 4.0
HARSH_STEER        = 0.7    # |steering| close to lock = harsh swerve
WRONG_SIDE_COS     = -0.3   # cos(velocity, lane_direction) below this
COOLDOWN_S         = 2.0    # de-dupe identical violations within this window


# ── Lane marking taxonomy ─────────────────────────────────────────────────
# Maps MetaDrive's PGLineType string codes to short labels used in logs.
_LINE_TYPE_MAP = {
    'ROAD_LINE_BROKEN_SINGLE_WHITE': 'broken',
    'ROAD_LINE_SOLID_SINGLE_WHITE':  'solid',
    'ROAD_EDGE_BOUNDARY':            'edge',
    'GUARDRAIL':                     'guardrail',
    'UNKNOWN_LINE':                  'none',
}


def _classify_line(line_type, line_color) -> str:
    """Returns a label like 'broken_white' / 'solid_yellow' / 'edge'."""
    # line_type may already be a string (PGLineType members are str constants)
    raw = str(line_type)
    base = _LINE_TYPE_MAP.get(raw, 'none')
    if base in ('edge', 'guardrail', 'none'):
        return base
    # PGLineColor.YELLOW ~= (1.0, 0.78, 0.0, 1); GREY ~= (1,1,1,1) (white)
    is_yellow = False
    try:
        r, g, b = float(line_color[0]), float(line_color[1]), float(line_color[2])
        is_yellow = (r > 0.8 and g > 0.5 and b < 0.3)
    except Exception:
        pass
    return f"{base}_{'yellow' if is_yellow else 'white'}"


class SimLabeler:
    """
    Stateless-per-frame labeller for the MetaDrive ego vehicle.

    The capture object (MetaDriveCapture) is expected to expose:
      - get_sim_state() -> dict   with 'info' from env.step() and 'lane' fields
      - get_telemetry() -> dict   (already exists in modules/metadrive_capture.py)

    If either is missing, label() returns a minimal dict and no violations.
    """

    def __init__(self):
        self._action_window: Deque[Tuple[float, str]] = deque(maxlen=60)
        # Per-violation last-fired timestamp for de-duplication
        self._cooldown: Dict[str, float] = {}
        # Speed history for harsh-accel detection (timestamp, kmh)
        self._speed_hist: Deque[Tuple[float, float]] = deque(maxlen=30)
        # Frames so far where ego was on a wrong-direction lane
        self._wrong_side_frames = 0
        # Most recently observed nearest-light readings for ran_red_light:
        # we remember the previous frame's light id + state + distance so we
        # can detect "ego just passed under a red light".
        self._prev_light_id:   Optional[str]   = None
        self._prev_light_state: Optional[str]  = None
        self._prev_light_dist: float           = 1e6

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def label(self,
              capture,
              telemetry: Optional[Dict] = None) -> Dict:
        """
        Build a per-frame label dict. Returns:
          {
            'lane': {
                'left':  'broken_white'|'solid_white'|'solid_yellow'|...,
                'right': '...',
                'speed_limit_kmh': float,
                'lateral_offset_m': float,
                'lane_direction_deg': float,
            },
            'action': 'accelerate'|'brake'|'coast'|'steer_left'|'steer_right'
                      |'lane_change_left'|'lane_change_right',
            'wrong_side': bool,
            'on_lane': bool,
            'navigation': 'forward'|'left'|'right',
            'crash': {'vehicle': bool, 'object': bool, 'human': bool,
                      'building': bool, 'sidewalk': bool},
            'violations': [
                {'kind': 'overspeed_high', 'severity': 'high',
                 'message': 'OVERSPEED 95/80 km/h', 'time': 12.3},
                ...
            ],
          }
        """
        if capture is None or not hasattr(capture, 'get_sim_state'):
            return {}
        try:
            sim = capture.get_sim_state() or {}
        except Exception:
            return {}
        if not sim:
            return {}

        info: Dict = sim.get('info') or {}
        lane: Dict = sim.get('lane') or {}
        agent: Dict = sim.get('agent') or {}
        tele: Dict = telemetry or {}
        light_ahead: Optional[Dict] = sim.get('traffic_light_ahead')

        now = time.time()

        # ── Lane marking labels ──────────────────────────────────────
        left_label  = _classify_line(*lane.get('left_marking',  ('UNKNOWN_LINE', (1, 1, 1, 1))))
        right_label = _classify_line(*lane.get('right_marking', ('UNKNOWN_LINE', (1, 1, 1, 1))))

        # ── Current action ───────────────────────────────────────────
        action = self._classify_action(info, lane, now)

        # ── Wrong-side detection ─────────────────────────────────────
        wrong_side = self._is_wrong_side(agent, lane)
        if wrong_side:
            self._wrong_side_frames += 1
        else:
            self._wrong_side_frames = 0

        # ── Crash flags from info ────────────────────────────────────
        crash = {
            'vehicle':  bool(info.get('crash_vehicle')),
            'object':   bool(info.get('crash_object')),
            'human':    bool(info.get('crash_human')),
            'building': bool(info.get('crash_building')),
            'sidewalk': bool(info.get('crash_sidewalk')),
        }

        on_lane = bool(agent.get('on_lane', True)) and not bool(info.get('out_of_road'))

        # ── Navigation hint (used as oracle for "is turning?") ───────
        nav = 'forward'
        if info.get('navigation_left'):
            nav = 'left'
        elif info.get('navigation_right'):
            nav = 'right'

        # ── Violations ───────────────────────────────────────────────
        violations = self._violations(
            info=info, lane=lane, tele=tele,
            wrong_side=wrong_side, crash=crash,
            left_label=left_label, right_label=right_label,
            light_ahead=light_ahead,
            now=now,
        )

        # Normalise traffic-light field for the output dict (drop the
        # bookkeeping fields that don't belong in the public label).
        tl_out = None
        if light_ahead:
            tl_out = {
                'state':      light_ahead.get('state', 'unknown'),
                'distance_m': float(light_ahead.get('distance_m', 0.0)),
            }

        return {
            'lane': {
                'left':              left_label,
                'right':             right_label,
                'speed_limit_kmh':   float(lane.get('speed_limit_kmh', 0.0)),
                'lateral_offset_m':  float(lane.get('lateral_offset_m', 0.0)),
                'lane_direction_deg': float(lane.get('lane_direction_deg', 0.0)),
            },
            'action':              action,
            'wrong_side':          wrong_side,
            'on_lane':             on_lane,
            'navigation':          nav,
            'crash':               crash,
            'traffic_light_ahead': tl_out,
            'violations':          violations,
        }

    # ------------------------------------------------------------------ #
    #  Action classification                                              #
    # ------------------------------------------------------------------ #

    def _classify_action(self, info: Dict, lane: Dict, now: float) -> str:
        steer = float(info.get('steering', 0.0))
        accel = float(info.get('acceleration', 0.0))

        # Lane-change detection: sustained steer while staying on road.
        # Look back over the last LANE_CHANGE_HOLD_S of inputs.
        if abs(steer) > STEER_THRESHOLD:
            cur = 'steer_left' if steer > 0 else 'steer_right'
            self._action_window.append((now, cur))
            # If we've been steering in the same direction for the hold
            # window AND ego is still on the road, upgrade to lane change.
            cutoff = now - LANE_CHANGE_HOLD_S
            same_dir = [a for t, a in self._action_window
                        if t >= cutoff and a == cur]
            if len(same_dir) >= max(5, int(LANE_CHANGE_HOLD_S * 10)):
                return 'lane_change_left' if cur == 'steer_left' else 'lane_change_right'
            return cur

        # No significant steer -> longitudinal action
        self._action_window.append((now, 'straight'))
        if accel > ACCEL_THRESHOLD:
            return 'accelerate'
        if accel < BRAKE_THRESHOLD:
            return 'brake'
        return 'coast'

    # ------------------------------------------------------------------ #
    #  Wrong-side detection                                               #
    # ------------------------------------------------------------------ #

    def _is_wrong_side(self, agent: Dict, lane: Dict) -> bool:
        """
        cos(velocity_vec, lane_direction_vec) deeply negative => wrong way.
        Falls back to False when either vector is missing or near zero.
        """
        try:
            vx, vy = float(agent['velocity'][0]), float(agent['velocity'][1])
            dx, dy = float(lane['lane_direction'][0]), float(lane['lane_direction'][1])
        except (KeyError, TypeError, IndexError):
            return False
        v_mag = math.hypot(vx, vy)
        d_mag = math.hypot(dx, dy)
        if v_mag < 0.5 or d_mag < 1e-3:
            return False
        cos = (vx * dx + vy * dy) / (v_mag * d_mag)
        return cos < WRONG_SIDE_COS

    # ------------------------------------------------------------------ #
    #  Violation rollup                                                   #
    # ------------------------------------------------------------------ #

    def _violations(self, *,
                    info: Dict, lane: Dict, tele: Dict,
                    wrong_side: bool, crash: Dict,
                    left_label: str, right_label: str,
                    light_ahead: Optional[Dict],
                    now: float) -> List[Dict]:
        out: List[Dict] = []

        # ── Ran-red-light detection ──────────────────────────────────
        # Logic: we were approaching a red light at distance d1 last frame,
        # and on the current frame either the light is no longer ahead
        # (we passed it) OR it's behind us / its distance jumped. Combined
        # with ego speed > a few km/h, that's a ran-red.
        cur_id    = light_ahead.get('light_id') if light_ahead else None
        cur_state = light_ahead.get('state')    if light_ahead else None
        cur_dist  = float(light_ahead.get('distance_m', 1e6)) if light_ahead else 1e6
        speed     = float(tele.get('speed_kmh', 0.0) or 0.0)
        passed_red = (
            self._prev_light_id is not None
            and self._prev_light_state == 'red'
            and self._prev_light_dist < 8.0
            and (cur_id != self._prev_light_id or cur_dist > self._prev_light_dist + 2.0)
            and speed > 5.0
        )
        if passed_red:
            self._emit(out, 'ran_red_light', 'critical',
                       'RAN A RED LIGHT', now, cooldown_s=3.0)
        # Yellow-light-blown advisory: passed under yellow at >40 km/h.
        passed_yellow = (
            self._prev_light_id is not None
            and self._prev_light_state == 'yellow'
            and self._prev_light_dist < 8.0
            and (cur_id != self._prev_light_id or cur_dist > self._prev_light_dist + 2.0)
            and speed > 40.0
        )
        if passed_yellow:
            self._emit(out, 'ran_yellow_light', 'advisory',
                       'RAN A YELLOW LIGHT', now, cooldown_s=3.0)

        # Update tracker
        self._prev_light_id    = cur_id
        self._prev_light_state = cur_state
        self._prev_light_dist  = cur_dist

        speed = float(tele.get('speed_kmh', 0.0) or 0.0)
        limit = float(lane.get('speed_limit_kmh', 0.0)
                      or tele.get('speed_limit_kmh', 0.0) or 0.0)

        # Overspeed
        if limit > 0 and speed - limit >= OVERSPEED_CRIT_KMH:
            self._emit(out, 'overspeed_critical', 'critical',
                       f'OVERSPEED {speed:.0f}/{limit:.0f} km/h', now)
        elif limit > 0 and speed - limit >= OVERSPEED_HIGH_KMH:
            self._emit(out, 'overspeed', 'high',
                       f'OVERSPEEDING {speed:.0f}/{limit:.0f} km/h', now)

        # Harsh accel / brake (m/s^2 derived from speed history)
        self._speed_hist.append((now, speed))
        if len(self._speed_hist) >= 5:
            t0, s0 = self._speed_hist[0]
            dt = now - t0
            if dt >= 0.3:
                accel_mps2 = ((speed - s0) / 3.6) / dt
                if accel_mps2 <= -HARSH_BRAKE_MPS2:
                    self._emit(out, 'harsh_brake', 'high',
                               f'HARSH BRAKE {abs(accel_mps2):.1f} m/s^2', now)
                elif accel_mps2 >= HARSH_ACCEL_MPS2:
                    self._emit(out, 'harsh_accel', 'high',
                               f'HARSH ACCEL {accel_mps2:.1f} m/s^2', now)

        # Harsh steer (close to lock)
        steer = float(info.get('steering', 0.0))
        if abs(steer) >= HARSH_STEER:
            self._emit(out, 'harsh_steer', 'advisory',
                       f'SHARP SWERVE steer={steer:+.2f}', now)

        # Wrong-side / wrong-way
        if wrong_side and self._wrong_side_frames >= 5:
            self._emit(out, 'wrong_side', 'critical',
                       'WRONG WAY - turn around', now)

        # Crossed a solid line on either side while drifting wide
        # The proxy: ego flagged out_of_road AND nearer side is a solid line.
        if bool(info.get('out_of_road')):
            offset = float(lane.get('lateral_offset_m', 0.0))
            crossed = None
            if offset > 0.5 and 'solid' in right_label:
                crossed = right_label
            elif offset < -0.5 and 'solid' in left_label:
                crossed = left_label
            if crossed:
                self._emit(out, 'crossed_solid_line', 'high',
                           f'CROSSED SOLID LINE ({crossed})', now)
            else:
                self._emit(out, 'off_road', 'high',
                           'OFF ROAD', now)

        # Crashes
        if crash['vehicle']:
            self._emit(out, 'collision_vehicle', 'critical',
                       'COLLISION with vehicle', now, cooldown_s=0.5)
        if crash['object']:
            self._emit(out, 'cone_hit', 'high',
                       'HIT TRAFFIC OBJECT (cone/barrier)', now, cooldown_s=0.5)
        if crash['human']:
            self._emit(out, 'collision_pedestrian', 'critical',
                       'COLLISION with pedestrian', now, cooldown_s=0.5)
        if crash['sidewalk']:
            self._emit(out, 'on_sidewalk', 'high',
                       'DRIVING ON SIDEWALK', now)

        return out

    def _emit(self, out: List[Dict], kind: str, severity: str,
              message: str, now: float,
              cooldown_s: float = COOLDOWN_S) -> None:
        last = self._cooldown.get(kind, 0.0)
        if now - last < cooldown_s:
            return
        self._cooldown[kind] = now
        out.append({
            'kind':     kind,
            'severity': severity,
            'message':  message,
            'time':     round(now, 3),
        })

    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._action_window.clear()
        self._cooldown.clear()
        self._speed_hist.clear()
        self._wrong_side_frames = 0
        self._prev_light_id = None
        self._prev_light_state = None
        self._prev_light_dist = 1e6
