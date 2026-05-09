"""
Temporal Smoothing + Confidence Gate
Prevents jittery intent predictions by majority-vote smoothing
and blocking low-confidence outputs.
"""

from collections import deque, Counter
from typing import Dict, Optional


class IntentSmoother:
    """
    Majority-vote smoothing window over intent predictions.
    Holds the previous confirmed intent when confidence is low.
    """

    def __init__(self,
                 window_size: int = 7,
                 min_ratio: float = 0.60,
                 confidence_threshold: float = 0.60):
        """
        Args:
            window_size          : number of recent predictions to consider
            min_ratio            : fraction of window that must agree
            confidence_threshold : per-frame model confidence required
        """
        self.window_size    = window_size
        self.min_ratio      = min_ratio
        self.conf_threshold = confidence_threshold

        self._history: deque = deque(maxlen=window_size)
        self._confirmed: Optional[str] = None

    def update(self, raw: Dict) -> Dict:
        """
        Args:
            raw : prediction dict from TemporalIntentPredictor.predict()
        Returns:
            smoothed prediction dict (same schema, with extra keys)
        """
        if not raw.get('ready', False):
            return raw

        # Reject low-confidence frames
        if raw['confidence'] < self.conf_threshold:
            return {
                **raw,
                'intent': self._confirmed or raw['intent'],
                'smoothed': True,
                'uncertain': True
            }

        self._history.append(raw['intent'])

        counts = Counter(self._history)
        top_intent, top_count = counts.most_common(1)[0]
        ratio = top_count / len(self._history)

        if ratio >= self.min_ratio:
            self._confirmed = top_intent

        return {
            **raw,
            'intent': self._confirmed or raw['intent'],
            'smoothed': True,
            'uncertain': self._confirmed is None
        }

    def reset(self):
        self._history.clear()
        self._confirmed = None
