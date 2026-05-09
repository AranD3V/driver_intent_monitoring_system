"""
Unit tests for the safety-critical pieces:
  - WarningEngine: tiered triggers + bbox-area fallback + cooldown
  - VoiceAssistant: humanizer covers every warning template
  - RunSession: aggregates correctly and writes the artefact bundle
  - SceneDetector.set_scene_resolution: focal length matches FOV
  - AffordanceEngine: ASCII-only strings (cv2.putText safe)

Run:
    python tests/test_safety_chain.py
Exits 0 on pass, 1 on any failure.
"""
from __future__ import annotations

import math
import sys
import time
import json
import shutil
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from modules.warning_engine import WarningEngine, BBOX_AREA_CRITICAL, BBOX_AREA_HIGH
from modules.voice_assistant import VoiceAssistant
from modules.run_session    import RunSession
from modules.affordance_engine import AffordanceEngine


# ─────────────────────────────────────────────────────────────────────── #
def _make_det(cls: str, dist: float,
              bbox_frac: float = 0.02,
              tl_state: str = None,
              track_id: int = None,
              scene_size=(1280, 720)) -> dict:
    area = scene_size[0] * scene_size[1]
    side = int(math.sqrt(bbox_frac * area))
    if track_id is None:
        track_id = abs(hash((cls, dist, bbox_frac))) % 100000
    d = {
        'class': cls,
        'distance_estimate': dist,
        'track_id': track_id,
        'confidence': 0.9,
        'bbox': {'x1': 0, 'y1': 0, 'x2': side, 'y2': side,
                 'width': side, 'height': side,
                 'center_x': side // 2, 'center_y': side // 2},
        'velocity': [0.0, 0.0],
    }
    if tl_state:
        d['traffic_light_state'] = tl_state
    return d


# ─────────────────────────────────────────────────────────────────────── #
class WarningEngineTests(unittest.TestCase):
    """Tier thresholds, bbox-area fallback, cooldown semantics."""

    SCENE = (1280, 720)

    def setUp(self):
        self.eng = WarningEngine()

    def _eval(self, det):
        self.eng.reset()
        return self.eng.evaluate([det], scene_size=self.SCENE)

    # --- Vehicles ------------------------------------------------------
    def test_car_30m_no_warning(self):
        self.assertEqual(self._eval(_make_det('car', 30.0)), [])

    def test_car_20m_high(self):
        w = self._eval(_make_det('car', 20.0))[0]
        self.assertEqual(w['severity'], 'high')
        self.assertIn('VEHICLE AHEAD', w['message'])

    def test_car_5m_critical(self):
        w = self._eval(_make_det('car', 5.0))[0]
        self.assertEqual(w['severity'], 'critical')
        self.assertIn('VEHICLE TOO CLOSE', w['message'])

    def test_truck_uses_vehicle_rules(self):
        self.assertEqual(self._eval(_make_det('truck', 5.0))[0]['severity'], 'critical')
        self.assertEqual(self._eval(_make_det('bus',   5.0))[0]['severity'], 'critical')

    def test_bbox_area_fallback_critical(self):
        # 25% of scene area, distance unreliably large
        w = self._eval(_make_det('car', 100.0, bbox_frac=0.25))[0]
        self.assertEqual(w['severity'], 'critical')

    def test_bbox_area_fallback_high(self):
        w = self._eval(_make_det('car', 100.0, bbox_frac=0.10))[0]
        self.assertEqual(w['severity'], 'high')

    # --- Pedestrians ----------------------------------------------------
    def test_person_20m_no_warning(self):
        self.assertEqual(self._eval(_make_det('person', 20.0)), [])

    def test_person_12m_high(self):
        self.assertEqual(self._eval(_make_det('person', 12.0))[0]['severity'], 'high')

    def test_person_6m_critical(self):
        self.assertEqual(self._eval(_make_det('person', 6.0))[0]['severity'], 'critical')

    # --- Cyclists -------------------------------------------------------
    def test_bicycle_15m_high(self):
        self.assertEqual(self._eval(_make_det('bicycle', 15.0))[0]['severity'], 'high')

    def test_bicycle_5m_critical(self):
        self.assertEqual(self._eval(_make_det('bicycle', 5.0))[0]['severity'], 'critical')

    # --- Traffic control -----------------------------------------------
    def test_red_light(self):
        self.assertEqual(self._eval(_make_det('traffic light', 30.0,
                                              tl_state='red'))[0]['severity'], 'high')

    def test_green_light_no_warning(self):
        self.assertEqual(self._eval(_make_det('traffic light', 30.0,
                                              tl_state='green')), [])

    def test_stop_sign_close(self):
        self.assertEqual(self._eval(_make_det('stop sign', 18.0))[0]['severity'], 'high')

    # --- Cooldown -------------------------------------------------------
    def test_cooldown_suppresses_repeat(self):
        det = _make_det('car', 5.0, track_id=42)
        first  = self.eng.evaluate([det], scene_size=self.SCENE)
        second = self.eng.evaluate([det], scene_size=self.SCENE)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 0,
                         "Same track in cooldown should not re-fire")

    def test_gaze_miss_flag(self):
        det = _make_det('car', 5.0, track_id=7)
        gaze_aff = {'looked_object': {'track_id': 7}}
        w = self.eng.evaluate([det], gaze_aff, scene_size=self.SCENE)[0]
        self.assertFalse(w['gaze_miss'])

        self.eng.reset()
        gaze_aff = {'looked_object': {'track_id': 99}}
        w = self.eng.evaluate([det], gaze_aff, scene_size=self.SCENE)[0]
        self.assertTrue(w['gaze_miss'])


# ─────────────────────────────────────────────────────────────────────── #
class DriverStateTests(unittest.TestCase):
    """Drowsiness, overspeeding, rash-driving warnings."""

    def setUp(self):
        self.eng = WarningEngine()

    # --- Drowsiness ---------------------------------------------------
    def test_eyes_closed_brief_no_warn(self):
        # First tick -> starts the timer; <1s elapsed -> no warning
        out = self.eng.evaluate_driver_state(
            {'eyes_closed': True}, telemetry=None)
        self.assertEqual(out, [])

    def test_eyes_closed_high_after_one_second(self):
        # Manually backdate the timer so we don't have to actually sleep
        self.eng._eyes_closed_started_at = time.time() - 1.2
        out = self.eng.evaluate_driver_state(
            {'eyes_closed': True}, telemetry=None)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]['severity'], 'high')
        self.assertIn('EYES CLOSED', out[0]['message'])

    def test_eyes_closed_critical_after_two_seconds(self):
        self.eng._eyes_closed_started_at = time.time() - 2.5
        out = self.eng.evaluate_driver_state(
            {'eyes_closed': True}, telemetry=None)
        self.assertEqual(out[0]['severity'], 'critical')

    def test_eyes_open_resets_timer(self):
        self.eng._eyes_closed_started_at = time.time() - 5.0
        self.eng.evaluate_driver_state({'eyes_closed': False}, telemetry=None)
        self.assertIsNone(self.eng._eyes_closed_started_at)

    # --- Overspeeding -------------------------------------------------
    def test_overspeed_below_buffer_no_warn(self):
        out = self.eng.evaluate_driver_state(
            None, telemetry={'speed_kmh': 65, 'speed_limit_kmh': 60})
        self.assertEqual(out, [])

    def test_overspeed_high(self):
        out = self.eng.evaluate_driver_state(
            None, telemetry={'speed_kmh': 75, 'speed_limit_kmh': 60})
        sevs = [w['severity'] for w in out]
        self.assertIn('high', sevs)

    def test_overspeed_critical(self):
        out = self.eng.evaluate_driver_state(
            None, telemetry={'speed_kmh': 100, 'speed_limit_kmh': 60})
        self.assertTrue(any(w['severity'] == 'critical' for w in out))

    # --- Rash driving (longitudinal accel) ----------------------------
    def test_hard_braking_detected(self):
        # Fake five timestamps with rapid speed drop: 60 -> 30 km/h in 0.5s
        # That's 8.33 m/s drop in 0.5s -> 16.7 m/s^2 deceleration.
        now = time.time()
        for i, (t_off, v) in enumerate([(-0.5, 60), (-0.4, 55),
                                         (-0.3, 50), (-0.2, 45),
                                         (-0.1, 35)]):
            self.eng._speed_history.append((now + t_off, v))
        out = self.eng.evaluate_driver_state(
            None, telemetry={'speed_kmh': 30, 'speed_limit_kmh': 60})
        # speed_history append + rash check happens inside eval; needs >=8 km/h
        msgs = [w['message'] for w in out]
        self.assertTrue(any('HARD BRAKING' in m for m in msgs),
                        f"Expected HARD BRAKING, got {msgs}")

    def test_no_rash_below_min_speed(self):
        now = time.time()
        for t_off, v in [(-0.4, 5), (-0.3, 4), (-0.2, 3), (-0.1, 2)]:
            self.eng._speed_history.append((now + t_off, v))
        out = self.eng.evaluate_driver_state(
            None, telemetry={'speed_kmh': 1, 'speed_limit_kmh': 60})
        # No rash warning allowed at low speed (we don't warn at standstill)
        self.assertFalse(any('HARD BRAKING' in w['message'] or
                             'AGGRESSIVE' in w['message'] for w in out))

    # --- Cooldown semantics -------------------------------------------
    def test_driver_state_cooldown(self):
        self.eng._eyes_closed_started_at = time.time() - 2.5
        a = self.eng.evaluate_driver_state(
            {'eyes_closed': True}, telemetry=None)
        b = self.eng.evaluate_driver_state(
            {'eyes_closed': True}, telemetry=None)
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 0,
                         "Same driver-state warning must be suppressed by cooldown")

    # --- Plumbed through main evaluate() ------------------------------
    def test_driver_state_flows_through_evaluate(self):
        self.eng._eyes_closed_started_at = time.time() - 2.5
        warns = self.eng.evaluate(
            detections=[],
            gaze_data={'eyes_closed': True},
            telemetry={'speed_kmh': 100, 'speed_limit_kmh': 60},
            scene_size=(1280, 720),
        )
        sevs = [w['severity'] for w in warns]
        self.assertIn('critical', sevs)
        # Critical ones must be sorted before any other entries
        self.assertEqual(warns[0]['severity'], 'critical')


# ─────────────────────────────────────────────────────────────────────── #
class VoiceHumanizerTests(unittest.TestCase):
    """Every warning template must have an associated humanized utterance."""

    def setUp(self):
        # Don't actually start the worker thread
        self.va = VoiceAssistant(enabled=False)

    def test_critical_vehicle(self):
        out = self.va._humanize({
            'message': 'VEHICLE TOO CLOSE - BRAKE',
            'severity': 'critical', 'gaze_miss': True
        })
        self.assertIn('Brake', out)
        self.assertIn('Vehicle', out)

    def test_high_vehicle_with_gaze(self):
        # When driver IS looking at the object, prefix is dropped
        out = self.va._humanize({
            'message': 'VEHICLE AHEAD - SLOW DOWN (20 m)',
            'severity': 'high', 'gaze_miss': False
        })
        self.assertNotIn('Caution', out)
        self.assertIn('Vehicle', out)

    def test_high_vehicle_without_gaze(self):
        out = self.va._humanize({
            'message': 'VEHICLE AHEAD - SLOW DOWN (20 m)',
            'severity': 'high', 'gaze_miss': True
        })
        self.assertIn('Caution', out)
        self.assertIn('Vehicle', out)

    def test_pedestrian_critical(self):
        out = self.va._humanize({
            'message': 'PERSON AHEAD - BRAKE',
            'severity': 'critical', 'gaze_miss': True
        })
        self.assertIn('Pedestrian', out)
        self.assertIn('Brake', out)

    def test_pedestrian_high(self):
        out = self.va._humanize({
            'message': 'PEDESTRIAN AHEAD - 12 m',
            'severity': 'high', 'gaze_miss': True
        })
        self.assertIn('Pedestrian', out)
        self.assertIn('slow down', out.lower())

    def test_cyclist_critical(self):
        out = self.va._humanize({
            'message': 'CYCLIST CLOSE - BRAKE',
            'severity': 'critical', 'gaze_miss': True
        })
        self.assertIn('Cyclist', out)
        self.assertIn('Brake', out)

    def test_red_light(self):
        out = self.va._humanize({
            'message': 'RED LIGHT - STOP',
            'severity': 'high', 'gaze_miss': True
        })
        self.assertIn('Red light', out)

    def test_stop_sign(self):
        out = self.va._humanize({
            'message': 'STOP SIGN AHEAD - 18 m',
            'severity': 'high', 'gaze_miss': True
        })
        self.assertIn('Stop sign', out)

    # --- Driver-state phrases ------------------------------------------
    def test_drowsy_critical(self):
        out = self.va._humanize({
            'message': 'EYES CLOSED 2.4s - WAKE UP',
            'severity': 'critical', 'gaze_miss': False, 'class': 'drowsiness',
        })
        self.assertIn('Wake up', out)
        self.assertIn('Pull over', out)

    def test_drowsy_high(self):
        out = self.va._humanize({
            'message': 'EYES CLOSED - PULL OVER',
            'severity': 'high', 'gaze_miss': False, 'class': 'drowsiness',
        })
        self.assertIn('drowsy', out.lower())
        self.assertNotIn('Caution', out)   # no scene prefix

    def test_overspeed_critical(self):
        out = self.va._humanize({
            'message': 'OVERSPEED CRITICAL - 95/60 km/h',
            'severity': 'critical', 'gaze_miss': False, 'class': 'overspeed',
        })
        self.assertIn('overspeed', out.lower())
        self.assertIn('immediately', out.lower())

    def test_overspeed_high(self):
        out = self.va._humanize({
            'message': 'OVERSPEEDING - 75/60 km/h',
            'severity': 'high', 'gaze_miss': False, 'class': 'overspeed',
        })
        self.assertIn('speed limit', out.lower())

    def test_hard_braking(self):
        out = self.va._humanize({
            'message': 'HARD BRAKING - 5.2 m/s2',
            'severity': 'high', 'gaze_miss': False, 'class': 'rash',
        })
        self.assertIn('Hard braking', out)

    def test_aggressive_accel_advisory(self):
        out = self.va._humanize({
            'message': 'AGGRESSIVE ACCEL - smooth out',
            'severity': 'advisory', 'gaze_miss': False, 'class': 'rash',
        })
        self.assertIn('gradually', out.lower())


# ─────────────────────────────────────────────────────────────────────── #
class RunSessionTests(unittest.TestCase):
    """End-to-end: session aggregates frames and writes a complete bundle."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='runsess_'))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_lifecycle(self):
        rs = RunSession(mode='unit_test', root=str(self.tmp),
                        model_path='rule-based',
                        driver_source='cam0', scene_source='cam1')

        # Feed 5 synthetic frames
        for i in range(5):
            fd = {
                'detected_objects': [
                    {'class': 'car', 'risk_level': 'high',
                     'affordance': 'Constraint'}
                ],
                'gaze_data': {'is_fixation': True},
                'gaze_affordance': {
                    'looked_object': {'class': 'car'},
                    'affordance': 'Constraint', 'risk_level': 'high'
                },
                'intent_prediction': {'intent': 'normal_forward',
                                       'confidence': 0.9},
                'warnings': [{
                    'severity': 'high', 'class': 'car', 'track_id': 1,
                    'affordance': 'Constraint',
                    'message': 'VEHICLE AHEAD - 20 m',
                    'gaze_miss': False,
                }] if i % 2 == 0 else [],
            }
            rs.update(fd, fps=30.0)

        summary = rs.finalize()

        # --- Summary contents -------------------------------------------------
        self.assertEqual(summary['frame_count'], 5)
        self.assertEqual(summary['warnings_total'], 3)   # i=0,2,4
        self.assertEqual(summary['warnings_by_severity']['high'], 3)
        self.assertEqual(summary['intent_distribution']['normal_forward'], 5)
        self.assertEqual(summary['gaze_object_dwell_frames']['car'], 5)
        self.assertEqual(summary['fixation_pct'], 100.0)

        # --- Disk artefacts ---------------------------------------------------
        self.assertTrue(Path(rs.summary_json_path).exists())
        self.assertTrue(Path(rs.summary_md_path).exists())
        self.assertTrue(Path(rs.warnings_csv_path).exists())
        self.assertTrue(Path(rs.voice_log_path).exists())

        # warnings.csv header row + 3 data rows
        with open(rs.warnings_csv_path, encoding='utf-8') as f:
            lines = f.read().strip().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn('severity', lines[0])

        # summary.json must be parseable and well-formed
        with open(rs.summary_json_path, encoding='utf-8') as f:
            j = json.load(f)
        self.assertIn('artefacts', j)
        self.assertEqual(j['frame_count'], 5)


# ─────────────────────────────────────────────────────────────────────── #
class FocalLengthTests(unittest.TestCase):
    """SceneDetector.set_scene_resolution must yield realistic distances."""

    def _build(self):
        from modules.scene_detector import SceneDetector
        sd = SceneDetector.__new__(SceneDetector)
        sd._focal_length = 1000.0
        sd._focal_length_calibrated = False
        return sd

    def test_metadrive_640x480_70deg(self):
        sd = self._build()
        sd.set_scene_resolution(640, 480, hfov_deg=70.0)
        self.assertAlmostEqual(sd._focal_length, 457.0, delta=2.0)

    def test_carla_1280x720_90deg(self):
        sd = self._build()
        sd.set_scene_resolution(1280, 720, hfov_deg=90.0)
        self.assertAlmostEqual(sd._focal_length, 640.0, delta=2.0)

    def test_distance_roundtrip(self):
        """A 1.5m car at 20m -> bbox px -> distance back ~= 20m."""
        sd = self._build()
        sd.set_scene_resolution(640, 480, hfov_deg=70.0)
        car_px = (1.5 * sd._focal_length) / 20.0
        dist_back = sd._estimate_distance('car', int(round(car_px)))
        self.assertAlmostEqual(dist_back, 20.0, delta=1.0)

    def test_calibrated_skips_override(self):
        """If a calibration file was loaded, FOV-based overrides do nothing."""
        sd = self._build()
        sd._focal_length = 750.0
        sd._focal_length_calibrated = True
        sd.set_scene_resolution(640, 480, hfov_deg=70.0)
        self.assertEqual(sd._focal_length, 750.0)


# ─────────────────────────────────────────────────────────────────────── #
class AffordanceEngineTests(unittest.TestCase):
    """All HUD-bound strings must be ASCII (cv2.putText doesn't do UTF-8)."""

    def test_no_unicode_in_details(self):
        ae = AffordanceEngine(str(_REPO / 'config' / 'affordance_config.json'))
        for cls in ('person', 'pedestrian', 'bicycle', 'car', 'truck', 'bus',
                    'motorcycle', 'stop sign', 'merging_gap'):
            for d in (3, 10, 20, 50):
                det = {'class': cls, 'distance_estimate': d,
                       'bbox': {'x1':0,'y1':0,'x2':10,'y2':10,
                                'width':10,'height':10,
                                'center_x':5,'center_y':5}}
                out = ae._detail(det, ae.affordance_map.get(cls, 'Unknown'))
                self.assertEqual(out, out.encode('ascii', 'ignore').decode('ascii'),
                                 f"Non-ASCII in detail string for {cls} @ {d}m: {out!r}")


# ─────────────────────────────────────────────────────────────────────── #
if __name__ == '__main__':
    unittest.main(verbosity=2)
