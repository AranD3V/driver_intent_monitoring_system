"""
Affordance Encoding Engine (v3)
Context-aware affordance mapping with risk assessment.
Affordances and trigger conditions match the project specification table:

  Object         Affordance      Risk       Trigger Condition
  ─────────────────────────────────────────────────────────
  Person         Caution         Critical   Distance < 5 m
  Bicycle        Caution         High       Distance < 15 m
  Car / Truck    Constraint      Critical   Distance < 5 m
  Traffic Light  Instructional   High       Red state
  Stop Sign      Instructional   High       Distance < 15 m
  Merging Gap    Opportunity     Low        Gap > 8 m
"""

import json
from typing import Dict, List


class AffordanceEngine:

    def __init__(self, config_path: str):
        with open(config_path) as f:
            cfg = json.load(f)
        self.affordance_map = cfg['affordance_mapping']
        dc = cfg.get('distance_thresholds', {})
        self.D_CRIT   = dc.get('critical', 5)
        self.D_HIGH   = dc.get('high',    15)
        self.D_MEDIUM = dc.get('medium',  30)

    def encode_affordances(self, detections: List[Dict]) -> List[Dict]:
        for det in detections:
            base = self.affordance_map.get(det['class'], 'Unknown')
            det['affordance']        = base
            det['affordance_detail'] = self._detail(det, base)
            det['risk_level']        = self._risk(det)
        return detections

    # ------------------------------------------------------------------ #

    def _detail(self, det: Dict, base: str) -> str:
        cls  = det['class']
        dist = det.get('distance_estimate', float('inf'))

        if cls == 'traffic light':
            return {
                'red':    'Stop Required',
                'yellow': 'Prepare to Stop',
                'green':  'Proceed',
                'unknown':'Observe Signal',
            }.get(det.get('traffic_light_state', 'unknown'), 'Observe Signal')

        if cls == 'stop sign':
            return 'Stop Immediately' if dist < self.D_HIGH else 'Prepare to Stop'

        if cls in ('person', 'pedestrian'):
            if dist < self.D_CRIT:  return 'Critical - Pedestrian < 5 m'
            if dist < self.D_HIGH:  return 'High Caution - Pedestrian < 15 m'
            return 'Monitor Pedestrian'

        if cls == 'bicycle':
            if dist < self.D_HIGH:  return 'High Caution - Cyclist < 15 m'
            return 'Monitor Cyclist'

        if cls in ('car', 'truck', 'bus'):
            if dist < self.D_CRIT:  return 'Critical - Vehicle < 5 m'
            if dist < self.D_HIGH:  return 'Maintain Safe Distance'
            return 'Monitor Traffic'

        if cls == 'motorcycle':
            if dist < self.D_HIGH:  return 'Caution - Motorcycle Nearby'
            return 'Monitor Motorcycle'

        if cls == 'merging_gap':
            gap = det.get('gap_size', 0)
            return 'Merging Opportunity' if gap > 8 else 'Gap Too Small'

        return base

    def _risk(self, det: Dict) -> str:
        cls  = det['class']
        dist = det.get('distance_estimate', float('inf'))

        # Person — Critical < 5 m, else High < 15 m
        if cls in ('person', 'pedestrian'):
            if dist < self.D_CRIT:  return 'critical'
            if dist < self.D_HIGH:  return 'high'
            return 'low'

        # Bicycle — High < 15 m
        if cls == 'bicycle':
            return 'high' if dist < self.D_HIGH else 'low'

        # Car / Truck / Bus — Critical < 5 m, Medium < 15 m
        if cls in ('car', 'truck', 'bus'):
            if dist < self.D_CRIT:  return 'critical'
            if dist < self.D_HIGH:  return 'medium'
            return 'low'

        # Traffic light — High when red and within 20 m
        if cls == 'traffic light':
            if det.get('traffic_light_state') == 'red' and dist < 20:
                return 'high'
            return 'low'

        # Stop sign — High < 15 m
        if cls == 'stop sign':
            return 'high' if dist < self.D_HIGH else 'low'

        # Motorcycle — medium < 15 m
        if cls == 'motorcycle':
            return 'medium' if dist < self.D_HIGH else 'low'

        # Merging gap — always low (it's an opportunity, not a threat)
        if cls == 'merging_gap':
            return 'low'

        return 'low'

    # ── Colour helpers (used by visualize and gaze_affordance_map) ───────── #

    @staticmethod
    def affordance_color(affordance: str):
        return {
            'Caution':       (  0, 165, 255),   # orange
            'Constraint':    (  0,  40, 210),   # red
            'Instructional': ( 30, 215, 215),   # yellow-cyan
            'Opportunity':   (200, 230,  30),   # lime-green
            'Unknown':       (128, 128, 128),   # grey
        }.get(affordance, (255, 255, 255))

    @staticmethod
    def risk_color(risk_level: str):
        return {
            'low':      (  0, 210,   0),
            'medium':   (  0, 220, 200),
            'high':     (  0, 140, 255),
            'critical': (  0,   0, 255),
        }.get(risk_level, (255, 255, 255))
