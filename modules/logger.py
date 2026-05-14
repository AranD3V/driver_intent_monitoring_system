"""
Rolling Frame Logger
Buffers recent frames in memory and flushes batches to disk.
Prevents memory growth during long sessions.
"""

import json
import time
import logging
from pathlib import Path
from collections import deque
from typing import Dict, Optional


class RollingLogger:

    def __init__(self,
                 log_dir: str = 'logs',
                 buffer_size: int = 500,
                 flush_interval: int = 500,
                 session_id: Optional[str] = None):
        """
        Args:
            log_dir        : directory to write JSON batches
            buffer_size    : max frames kept in RAM
            flush_interval : write to disk every N frames
            session_id     : optional id to embed in batch filenames so they
                             align with the parent run directory; defaults to
                             the wall-clock time when the logger was created.
        """
        self.log_dir       = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.buffer        = deque(maxlen=buffer_size)
        self.flush_interval = flush_interval
        self._frame_count  = 0
        self._session_id   = (session_id
                              if session_id is not None
                              else str(int(time.time())))

        logging.basicConfig(level=logging.INFO)
        self._log = logging.getLogger('RollingLogger')

    def log(self, frame_data: Dict):
        serialized = self._serialize(frame_data)
        self.buffer.append(serialized)
        self._frame_count += 1

        if self._frame_count % self.flush_interval == 0:
            self._flush()

    def flush_final(self):
        """Call at end of session to flush remaining buffer."""
        self._flush()
        self._log.info(f"Session complete. Total frames: {self._frame_count}")

    # ------------------------------------------------------------------ #

    def _flush(self):
        if not self.buffer:
            return
        batch = list(self.buffer)
        path  = self.log_dir / f"session_{self._session_id}_batch_{self._frame_count:07d}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(batch, f, default=str)
        self._log.info(f"Flushed {len(batch)} frames -> {path}")
        self.buffer.clear()

    @staticmethod
    def _serialize(fd: Dict) -> Dict:
        out: Dict = {
            'timestamp':    fd.get('timestamp'),
            'frame_number': fd.get('frame_number')
        }

        gd = fd.get('gaze_data')
        if gd:
            out['gaze_data'] = {
                'gaze_point':  gd.get('gaze_point'),
                'head_pose':   gd.get('head_pose'),
                'gaze_speed':  gd.get('gaze_speed'),
                'is_fixation': gd.get('is_fixation'),
                'confidence':  gd.get('confidence'),
                'calibrated':  gd.get('calibrated')
            }

        out['detected_objects'] = [
            {
                'class':      o['class'],
                'track_id':   o.get('track_id'),
                'bbox':       o['bbox'],
                'confidence': o['confidence'],
                'affordance': o.get('affordance'),
                'risk_level': o.get('risk_level'),
                'distance':   o.get('distance_estimate'),
                'velocity':   o.get('velocity')
            }
            for o in fd.get('detected_objects', [])
        ]

        ga = fd.get('gaze_affordance')
        if ga:
            lo = ga.get('looked_object', {})
            out['gaze_affordance'] = {
                'class':         lo.get('class'),
                'track_id':      lo.get('track_id'),
                'affordance':    ga.get('affordance'),
                'gaze_duration': ga.get('gaze_duration'),
                'risk_level':    ga.get('risk_level'),
                'distance':      ga.get('distance')
            }

        ip = fd.get('intent_prediction')
        if ip:
            out['intent_prediction'] = {
                'intent':        ip.get('intent'),
                'confidence':    ip.get('confidence'),
                'uncertain':     ip.get('uncertain', False),
                'probabilities': ip.get('probabilities')
            }

        # Sim oracle labels (MetaDrive only). Persisted in full so the
        # session-log batches double as a labelled training set for the
        # progressive-training loop in scripts/train_intent.py.
        sim = fd.get('sim_label')
        if sim:
            out['sim_label'] = sim

        return out
