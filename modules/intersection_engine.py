"""
Gaze–Object Intersection Engine (v2)
Uses persistent track IDs for stable gaze-duration counting.
Adds fixation detection to filter out saccades.
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class IntersectionEngine:

    MIN_FIXATION_FRAMES = 5   # ignore gaze < N frames (saccade filter)

    def __init__(self):
        # track_id → frame counter of how long gazed at
        self._gaze_counters: Dict[int, int] = defaultdict(int)
        self._current_track_id: Optional[int] = None
        self._history: List[Dict] = []

    # ------------------------------------------------------------------ #
    def find_gaze_target(self,
                         gaze_point: Tuple[int, int],
                         gaze_speed: float,
                         is_fixation: bool,
                         detections: List[Dict]) -> Optional[Dict]:
        """
        Args:
            gaze_point  : (x, y) in scene frame pixels
            gaze_speed  : gaze velocity magnitude (px/frame)
            is_fixation : True when gaze speed below threshold
            detections  : list of objects with bboxes and track_ids
        Returns:
            Dict with looked_object, affordance, gaze_duration, risk_level
            or None if no valid intersection.
        """
        # Saccade: don't assign gaze target
        if not is_fixation:
            return None

        gx, gy = gaze_point
        candidates = [d for d in detections if self._inside(gx, gy, d['bbox'])]

        if not candidates:
            self._decay_counters()
            return None

        # Prefer highest-confidence candidate
        target = max(candidates, key=lambda d: d['confidence'])
        tid = target.get('track_id', -1)

        # Reset counters for tracks not currently gazed at
        self._decay_counters(active_id=tid)

        # Increment counter for this track
        self._gaze_counters[tid] += 1
        duration = self._gaze_counters[tid]

        # Require minimum fixation duration before reporting
        if duration < self.MIN_FIXATION_FRAMES:
            return None

        self._current_track_id = tid

        result = {
            'looked_object':    target,
            'affordance':       target.get('affordance', 'Unknown'),
            'affordance_detail':target.get('affordance_detail', ''),
            'gaze_duration':    duration,
            'risk_level':       target.get('risk_level', 'low'),
            'distance':         target.get('distance_estimate', None)
        }

        self._history.append({
            'class':      target['class'],
            'affordance': target.get('affordance'),
            'duration':   duration,
            'risk_level': target.get('risk_level')
        })
        if len(self._history) > 300:
            self._history = self._history[-300:]

        return result

    # ------------------------------------------------------------------ #
    def get_statistics(self) -> Dict:
        if not self._history:
            return {}
        classes = [h['class'] for h in self._history]
        from collections import Counter
        class_counts = Counter(classes)
        return {
            'total_gazes':       len(self._history),
            'class_distribution': dict(class_counts),
            'most_viewed':        class_counts.most_common(1)[0][0]
        }

    def reset(self):
        self._gaze_counters.clear()
        self._current_track_id = None
        self._history.clear()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _inside(x, y, bbox) -> bool:
        return bbox['x1'] <= x <= bbox['x2'] and bbox['y1'] <= y <= bbox['y2']

    def _decay_counters(self, active_id: Optional[int] = None):
        """Gradually reduce counters for non-active tracks."""
        for tid in list(self._gaze_counters):
            if tid != active_id:
                self._gaze_counters[tid] = max(0, self._gaze_counters[tid] - 1)
                if self._gaze_counters[tid] == 0:
                    del self._gaze_counters[tid]
