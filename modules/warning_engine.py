"""
Warning Engine

Evaluates trigger conditions from the affordance specification table
and emits driver-facing warning prompts. Warnings fire for any object
in the scene that meets its trigger condition, whether or not the
driver is gazing at it -- the safety-critical case is the one the
driver MISSES.

Trigger table (graduated -- earlier warning, later critical):
  Person         Caution        High       Distance < 15 m
  Person         Caution        Critical   Distance <  8 m
  Bicycle        Caution        High       Distance < 20 m
  Bicycle        Caution        Critical   Distance <  8 m
  Car/Truck/Bus  Constraint     High       Distance < 25 m   <-- early SLOW DOWN
  Car/Truck/Bus  Constraint     Critical   Distance <  8 m   <-- BRAKE
  Traffic Light  Instructional  High       Red state
  Stop Sign      Instructional  High       Distance < 20 m
  Merging Gap    Opportunity    Advisory   Gap > 8 m

Plus: any vehicle/person bbox covering more than the BBOX_AREA_*
fraction of the scene image triggers an unconditional critical warning
even if the distance estimate is unreliable (e.g. uncalibrated cam,
simulator scene). This guarantees we ALWAYS warn when something is
huge on screen.

Driver-state triggers (independent of the scene):
  Drowsiness     Eyes closed >= 1.0 s -> high, >= 2.0 s -> critical
  Overspeeding   Speed > limit + 10 km/h -> high,  + 25 km/h -> critical
  Rash driving   |dV/dt| > 4 m/s^2 over ~0.5 s -> high (>3 m/s^2 -> advisory)
"""

import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple


# Bbox-area fallback: if the object covers more than this fraction of
# the scene, fire a critical warning regardless of distance estimate.
BBOX_AREA_CRITICAL = 0.18    # ~18 % of the scene
BBOX_AREA_HIGH     = 0.07    # ~7  % of the scene


def _dist(det): return det.get('distance_estimate', float('inf'))


def _bbox_area_frac(det) -> float:
    """Bbox area as a fraction of the scene area, when available."""
    bb = det.get('bbox') or {}
    w = bb.get('width')
    h = bb.get('height')
    sw = det.get('_scene_w')
    sh = det.get('_scene_h')
    if not (w and h and sw and sh):
        return 0.0
    return float(w) * float(h) / (float(sw) * float(sh) + 1e-9)


# Each entry: (classes, condition_fn(det) -> bool, severity, message)
_TRIGGERS = [
    # --- Person: tiered warning ---
    (
        ('person', 'pedestrian'),
        lambda d: _dist(d) < 8 or _bbox_area_frac(d) > BBOX_AREA_CRITICAL,
        'critical',
        'PERSON AHEAD - BRAKE',
    ),
    (
        ('person', 'pedestrian'),
        lambda d: _dist(d) < 15 or _bbox_area_frac(d) > BBOX_AREA_HIGH,
        'high',
        'PEDESTRIAN AHEAD - {dist:.0f} m',
    ),

    # --- Cyclist ---
    (
        ('bicycle',),
        lambda d: _dist(d) < 8 or _bbox_area_frac(d) > BBOX_AREA_CRITICAL,
        'critical',
        'CYCLIST CLOSE - BRAKE',
    ),
    (
        ('bicycle',),
        lambda d: _dist(d) < 20 or _bbox_area_frac(d) > BBOX_AREA_HIGH,
        'high',
        'CYCLIST AHEAD - {dist:.0f} m',
    ),

    # --- Vehicle: tiered warning is the big fix here ---
    (
        ('car', 'truck', 'bus'),
        lambda d: _dist(d) < 8 or _bbox_area_frac(d) > BBOX_AREA_CRITICAL,
        'critical',
        'VEHICLE TOO CLOSE - BRAKE',
    ),
    (
        ('car', 'truck', 'bus'),
        lambda d: _dist(d) < 25 or _bbox_area_frac(d) > BBOX_AREA_HIGH,
        'high',
        'VEHICLE AHEAD - SLOW DOWN ({dist:.0f} m)',
    ),

    # --- Traffic control ---
    (
        ('traffic light',),
        lambda d: d.get('traffic_light_state') == 'red',
        'high',
        'RED LIGHT - STOP',
    ),
    (
        ('stop sign',),
        lambda d: _dist(d) < 20,
        'high',
        'STOP SIGN AHEAD - {dist:.0f} m',
    ),

    # --- Advisory ---
    (
        ('merging_gap',),
        lambda d: d.get('gap_size', 0) > 8,
        'advisory',
        'MERGING GAP - {gap:.0f} m',
    ),
]

# Frames suppressed after a same-track warning fires; lowered so the
# graduated high->critical can re-fire when the situation worsens.
_COOLDOWN_FRAMES = 18

# If the driver was looking at an object within this many evaluate() calls,
# downgrade its warning severity by one step — driver is already aware.
# critical is NEVER fully suppressed; advisories are dropped entirely.
_GAZE_RECENT_FRAMES = 30
_DOWNGRADE = {'critical': 'high', 'high': 'advisory', 'advisory': None}

SEVERITY_ORDER = {'critical': 0, 'high': 1, 'advisory': 2}


# ── Driver-state thresholds ──────────────────────────────────────────── #
# Drowsiness (sustained eye closure)
DROWSY_HIGH_S       = 1.0     # eyes closed at least this long -> high
DROWSY_CRIT_S       = 2.0     # ... -> critical (likely micro-sleep)

# Overspeeding (delta over the posted limit)
OVERSPEED_HIGH_KMH  = 10.0
OVERSPEED_CRIT_KMH  = 25.0

# Rash driving (longitudinal acceleration magnitude)
RASH_ADVISORY_MPS2  = 3.0
RASH_HIGH_MPS2      = 4.5

# Min ego speed to even consider "rash" (don't warn at standstill)
RASH_MIN_SPEED_KMH  = 8.0

# Cooldown for driver-state warnings (seconds, real time)
DRIVER_STATE_COOLDOWN_S = 4.0


class WarningEngine:

    def __init__(self):
        # track_id → frames since last warning fired
        self._cooldown: Dict[int, int] = defaultdict(lambda: _COOLDOWN_FRAMES)

        # track_id → frames since driver last gazed at it. 0 = looking now.
        # Inflates each tick; reset to 0 when gaze lands on the track.
        self._gaze_recency: Dict[int, int] = defaultdict(
            lambda: _GAZE_RECENT_FRAMES + 1)

        # Driver-state internal state
        self._eyes_closed_started_at: Optional[float] = None
        # Rolling window of (timestamp, speed_kmh) used to compute |dV/dt|
        self._speed_history: Deque[Tuple[float, float]] = deque(maxlen=30)
        # Per-warning-key cooldown timestamp
        self._driver_warn_cooldown: Dict[str, float] = {}

    def evaluate(self,
                 detections: List[Dict],
                 gaze_affordance: Optional[Dict] = None,
                 scene_size: Optional[tuple] = None,
                 gaze_data: Optional[Dict] = None,
                 telemetry: Optional[Dict] = None) -> List[Dict]:
        """
        Returns a list of active warning dicts, sorted most-severe first:
          {
            'severity':   'critical' | 'high' | 'advisory',
            'message':    str,
            'affordance': str,
            'class':      str,
            'track_id':   int | None,
            'gaze_miss':  bool,
          }

        scene_size: optional (width, height) of the scene image.  When
        provided, the bbox-area fallback trigger fires for objects that
        cover a large fraction of the scene even if their distance
        estimate is unreliable -- the failsafe for sim scenes and
        uncalibrated cameras.

        gaze_data / telemetry: when provided, evaluate_driver_state() also
        emits drowsiness, overspeed, and rash-driving warnings into the
        same returned list -- so they flow through the voice assistant,
        warnings.csv, and run summary identically to scene warnings.
        """
        # Age all cooldowns + gaze-recency counters
        for tid in list(self._cooldown):
            self._cooldown[tid] += 1
        for tid in list(self._gaze_recency):
            self._gaze_recency[tid] += 1

        # Inject scene size into each det so the trigger lambdas can read it
        if scene_size is not None:
            sw, sh = scene_size
            for det in detections:
                det['_scene_w'] = sw
                det['_scene_h'] = sh

        gazed_id = None
        if gaze_affordance and gaze_affordance.get('looked_object'):
            gazed_id = gaze_affordance['looked_object'].get('track_id')
            if gazed_id is not None:
                self._gaze_recency[gazed_id] = 0

        active: List[Dict] = []

        for det in detections:
            cls = det['class']
            tid = det.get('track_id', -1)

            for classes, condition, severity, template in _TRIGGERS:
                if cls not in classes:
                    continue
                if not condition(det):
                    continue
                if self._cooldown[tid] < _COOLDOWN_FRAMES:
                    continue    # still in cooldown for this track

                dist = _dist(det)
                gap  = det.get('gap_size', 0.0)
                state = det.get('traffic_light_state', '')

                msg = template.format(
                    dist=dist  if dist != float('inf') else 0,
                    gap=gap,
                    state=state,
                )

                # Gaze-aware severity downgrade: if the driver was looking at
                # this track within the recent window, they're already aware,
                # so step the severity down. critical -> high, high -> advisory,
                # advisory -> dropped entirely. Never suppresses a critical
                # outright (driver awareness ≠ avoidance).
                effective_severity = severity
                if self._gaze_recency[tid] <= _GAZE_RECENT_FRAMES:
                    downgraded = _DOWNGRADE.get(severity, severity)
                    if downgraded is None:
                        self._cooldown[tid] = 0
                        break
                    effective_severity = downgraded

                active.append({
                    'severity':   effective_severity,
                    'message':    msg,
                    'affordance': det.get('affordance', 'Unknown'),
                    'class':      cls,
                    'track_id':   tid,
                    'gaze_miss':  (tid != gazed_id),
                })
                self._cooldown[tid] = 0     # reset cooldown
                break                        # one warning per object per tick

        # Driver-state warnings (drowsiness / overspeed / rash) flow into
        # the same list so voice + CSV + summary handle them uniformly.
        active.extend(self.evaluate_driver_state(gaze_data, telemetry))

        # Sort: critical first, then high, then advisory
        active.sort(key=lambda w: SEVERITY_ORDER.get(w['severity'], 9))
        return active

    # ------------------------------------------------------------------ #
    #  Driver-state warnings                                               #
    # ------------------------------------------------------------------ #

    def evaluate_driver_state(self,
                              gaze_data: Optional[Dict],
                              telemetry: Optional[Dict]) -> List[Dict]:
        """
        Returns drowsiness / overspeeding / rash-driving warnings.
        Either argument may be None (e.g. two-cam mode has no telemetry,
        or gaze can be lost on a frame); the relevant triggers just skip.
        """
        out: List[Dict] = []
        now = time.time()

        # ── Drowsiness ─────────────────────────────────────────────────
        if gaze_data is not None:
            eyes_closed = bool(gaze_data.get('eyes_closed', False))
            if eyes_closed:
                if self._eyes_closed_started_at is None:
                    self._eyes_closed_started_at = now
                elapsed = now - self._eyes_closed_started_at
                if elapsed >= DROWSY_CRIT_S:
                    self._emit_driver_state(out, 'drowsiness_critical',
                                            'critical',
                                            f'EYES CLOSED {elapsed:.1f}s - WAKE UP',
                                            now)
                elif elapsed >= DROWSY_HIGH_S:
                    self._emit_driver_state(out, 'drowsiness_high',
                                            'high',
                                            f'EYES CLOSED - PULL OVER',
                                            now)
            else:
                self._eyes_closed_started_at = None

        # ── Speed-derived warnings ─────────────────────────────────────
        if telemetry is not None:
            speed = float(telemetry.get('speed_kmh', 0.0) or 0.0)
            limit = float(telemetry.get('speed_limit_kmh', 0.0) or 0.0)

            # Overspeed
            if limit > 0:
                over = speed - limit
                if over >= OVERSPEED_CRIT_KMH:
                    self._emit_driver_state(out, 'overspeed_critical',
                                            'critical',
                                            f'OVERSPEED CRITICAL - {speed:.0f}/{limit:.0f} km/h',
                                            now)
                elif over >= OVERSPEED_HIGH_KMH:
                    self._emit_driver_state(out, 'overspeed_high',
                                            'high',
                                            f'OVERSPEEDING - {speed:.0f}/{limit:.0f} km/h',
                                            now)

            # Rash driving via longitudinal accel
            self._speed_history.append((now, speed))
            if (speed >= RASH_MIN_SPEED_KMH
                    and len(self._speed_history) >= 5):
                t0, s0 = self._speed_history[0]
                dt = now - t0
                if dt >= 0.3:
                    # km/h -> m/s, then accel m/s^2
                    dv = (speed - s0) / 3.6
                    accel = dv / dt
                    a_mag = abs(accel)
                    if a_mag >= RASH_HIGH_MPS2:
                        kind = 'BRAKING' if accel < 0 else 'ACCEL'
                        self._emit_driver_state(out, 'rash_high',
                                                'high',
                                                f'HARD {kind} - {a_mag:.1f} m/s2',
                                                now)
                    elif a_mag >= RASH_ADVISORY_MPS2:
                        kind = 'BRAKING' if accel < 0 else 'ACCEL'
                        self._emit_driver_state(out, 'rash_advisory',
                                                'advisory',
                                                f'AGGRESSIVE {kind} - smooth out',
                                                now)
        return out

    def _emit_driver_state(self, out: List[Dict],
                           key: str, severity: str,
                           message: str, now: float) -> None:
        """De-duplicate driver-state warnings using a wall-clock cooldown."""
        last = self._driver_warn_cooldown.get(key, 0.0)
        if now - last < DRIVER_STATE_COOLDOWN_S:
            return
        self._driver_warn_cooldown[key] = now
        out.append({
            'severity':   severity,
            'message':    message,
            'affordance': 'DriverState',
            'class':      key.split('_')[0],   # drowsiness / overspeed / rash
            'track_id':   None,
            'gaze_miss':  False,
        })

    def reset(self):
        self._cooldown.clear()
        self._gaze_recency.clear()
        self._eyes_closed_started_at = None
        self._speed_history.clear()
        self._driver_warn_cooldown.clear()
