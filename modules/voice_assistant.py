"""
Voice Assistant — driver-facing safe-drive prompt engine.

Turns WarningEngine output and intent predictions into spoken / beeped
guidance so the driver can keep their eyes on the road.

Design:
- Background worker thread consumes a priority queue of utterances.
- pyttsx3 is used when available (offline, Windows SAPI / macOS NSSpeech /
  espeak on Linux). When pyttsx3 is missing, falls back to a short
  severity-coded beep on Windows (winsound) and a console print elsewhere.
- Per-message cooldown prevents spam: the same phrase will not be repeated
  more than once every `cooldown_s` seconds.
- Severity ordering: critical > high > advisory > coaching.
  A higher-priority utterance preempts the queue.
- Coaching prompts are derived from the current intent prediction
  (e.g. "Lane change — check your mirrors") and only fire on intent
  transitions, never on every frame.
"""

from __future__ import annotations

import time
import threading
import platform
from collections import deque
from queue import PriorityQueue, Empty
from typing import Dict, List, Optional

# ── Optional dependencies — degrade gracefully ──────────────────────────────
try:
    import pyttsx3
    _TTS_AVAILABLE = True
except Exception:
    _TTS_AVAILABLE = False

try:
    import winsound
    _WINSOUND_AVAILABLE = platform.system() == 'Windows'
except Exception:
    _WINSOUND_AVAILABLE = False


# ── Severity → (priority, beep frequency, beep ms) ──────────────────────────
_SEV_PRIORITY: Dict[str, int] = {
    'critical': 0,
    'high':     1,
    'advisory': 2,
    'coaching': 3,
}

_SEV_BEEP: Dict[str, tuple] = {
    'critical': (1200, 250),
    'high':     ( 900, 180),
    'advisory': ( 700, 120),
    'coaching': ( 600,  90),
}

# ── Coaching prompts keyed by predicted intent ──────────────────────────────
_INTENT_COACHING: Dict[str, str] = {
    'lane_change_prepare': "Lane change detected. Check your mirrors and blind spot.",
    'intersection_scan':   "Approaching intersection. Scan left and right.",
    'mirror_check':        "Mirror check noted.",
    'pedestrian_monitor':  "Pedestrian in view. Maintain attention.",
}

# ── Critical-condition coaching when no warning fires but driver is distracted
_DISTRACTION_PROMPT = "Eyes on the road."


class _Utterance:
    """Hashable comparable wrapper for the priority queue."""
    __slots__ = ('priority', 'tstamp', 'text', 'severity', 'key')

    def __init__(self, priority: int, text: str,
                 severity: str, key: str):
        self.priority = priority
        self.tstamp   = time.time()
        self.text     = text
        self.severity = severity
        self.key      = key   # used for cooldown tracking

    # PriorityQueue ordering: lower priority value first, then FIFO
    def __lt__(self, other: '_Utterance') -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.tstamp < other.tstamp


class VoiceAssistant:
    """
    Background driver-assist prompt engine.

        va = VoiceAssistant()
        va.start()
        ...
        va.dispatch_warnings(warnings_list)
        va.dispatch_intent(intent_pred_dict)
        va.dispatch_distraction(is_distracted=True)
        ...
        va.shutdown()
    """

    def __init__(self,
                 enabled: bool   = True,
                 rate_wpm: int   = 175,
                 cooldown_s: float = 4.0,
                 max_queue: int  = 8):
        self.enabled    = enabled
        self.cooldown_s = cooldown_s
        self._queue: PriorityQueue = PriorityQueue(maxsize=max_queue)
        self._last_spoken: Dict[str, float] = {}
        self._spoken_log: deque = deque(maxlen=512)
        self._last_intent: Optional[str] = None
        self._engine = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._mode: str = 'silent'   # 'tts' | 'beep' | 'silent'

        if not self.enabled:
            return

        if _TTS_AVAILABLE:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty('rate', rate_wpm)
                self._mode = 'tts'
            except Exception as e:
                print(f"[VoiceAssistant] pyttsx3 init failed ({e}); "
                      f"falling back to beep.")
                self._engine = None
                self._mode = 'beep' if _WINSOUND_AVAILABLE else 'silent'
        else:
            self._mode = 'beep' if _WINSOUND_AVAILABLE else 'silent'

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if not self.enabled or self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name='voice-assistant')
        self._thread.start()
        print(f"[VoiceAssistant] Started in '{self._mode}' mode.")

    def shutdown(self) -> None:
        self._running = False
        # Push a sentinel so the worker wakes up
        try:
            self._queue.put_nowait(_Utterance(99, '', 'coaching', '__exit__'))
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=2)
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  Public dispatch API                                                 #
    # ------------------------------------------------------------------ #

    def dispatch_warnings(self, warnings: Optional[List[Dict]]) -> None:
        """Convert WarningEngine output into spoken prompts."""
        if not warnings:
            return
        # Speak only the most-severe new warning per tick
        top = warnings[0]
        sev = top.get('severity', 'advisory')
        if sev not in _SEV_PRIORITY:
            sev = 'advisory'
        text = self._humanize(top)
        key  = f"warn:{top.get('class','?')}:{top.get('track_id','?')}:{sev}"
        self._enqueue(text, sev, key)

    def dispatch_intent(self, intent_pred: Optional[Dict]) -> None:
        """
        Speak a coaching tip whenever the smoothed intent transitions to
        a new actionable class. We don't speak on every frame.
        """
        if not intent_pred:
            return
        intent = intent_pred.get('intent')
        conf   = intent_pred.get('confidence', 0.0)
        if not intent or conf < 0.55:
            return
        if intent == self._last_intent:
            return
        self._last_intent = intent
        prompt = _INTENT_COACHING.get(intent)
        if prompt:
            self._enqueue(prompt, 'coaching', f"intent:{intent}")

    def dispatch_distraction(self, is_distracted: bool) -> None:
        """Trigger the 'eyes on the road' nudge when gaze is off-screen."""
        if is_distracted:
            self._enqueue(_DISTRACTION_PROMPT, 'high', 'distraction')

    @property
    def history(self) -> List[Dict]:
        """Return a copy of recent utterances — used by the run summary."""
        return list(self._spoken_log)

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _humanize(warn: Dict) -> str:
        """Turn the terse banner message into a friendlier spoken line."""
        msg = warn.get('message', '')
        cls = (warn.get('class') or '').lower()
        sev = warn.get('severity', 'advisory')
        gaze_miss = warn.get('gaze_miss', False)

        # Severity prefix gives the driver an instant reaction cue
        prefix = {
            'critical': "Brake! ",
            'high':     "Caution. ",
            'advisory': "Heads up. ",
        }.get(sev, "")

        # If the driver is already gazing at the object, downgrade the prefix
        if not gaze_miss and sev != 'critical':
            prefix = ""

        # Shorten well-known patterns into natural speech
        cleaned = msg.replace('-', ' ').replace('  ', ' ').strip()
        upper = msg.upper()
        if 'VEHICLE TOO CLOSE' in upper:
            cleaned = "Vehicle too close. Brake now."
        elif 'VEHICLE AHEAD' in upper:
            cleaned = "Vehicle ahead. Slow down."
        elif 'PERSON AHEAD' in upper or 'PEDESTRIAN' in upper:
            cleaned = ("Pedestrian ahead. Brake now."
                       if sev == 'critical'
                       else "Pedestrian ahead, slow down.")
        elif 'CYCLIST CLOSE' in upper:
            cleaned = "Cyclist close. Brake now."
        elif 'CYCLIST' in upper:
            cleaned = "Cyclist ahead, give space."
        elif 'RED LIGHT' in upper:
            cleaned = "Red light, prepare to stop."
        elif 'STOP SIGN' in upper:
            cleaned = "Stop sign ahead."
        elif 'MERGING GAP' in upper:
            cleaned = "Merging gap available on your side."
        # Driver-state warnings -- talk about the driver, not the scene.
        # Return directly so the scene-warning prefix isn't applied.
        elif 'EYES CLOSED' in upper and sev == 'critical':
            return "Wake up! Eyes closed too long. Pull over now."
        elif 'EYES CLOSED' in upper:
            return "You look drowsy. Take a break."
        elif 'OVERSPEED CRITICAL' in upper:
            return "Critical overspeed. Slow down immediately."
        elif 'OVERSPEED' in upper:
            return "You are over the speed limit. Slow down."
        elif 'HARD BRAKING' in upper:
            return "Hard braking detected. Maintain a safer gap."
        elif 'HARD ACCEL' in upper:
            return "Hard acceleration. Drive more smoothly."
        elif 'AGGRESSIVE BRAKING' in upper:
            return "Brake more gradually."
        elif 'AGGRESSIVE ACCEL' in upper:
            return "Accelerate more gradually."
        elif cls and cls not in cleaned.lower():
            cleaned = f"{cls.title()} alert."

        return (prefix + cleaned).strip()

    def _enqueue(self, text: str, severity: str, key: str) -> None:
        if not self.enabled or not text:
            return
        with self._lock:
            now = time.time()
            last = self._last_spoken.get(key, 0.0)
            if now - last < self.cooldown_s:
                return
            self._last_spoken[key] = now

        priority = _SEV_PRIORITY.get(severity, 2)
        utt = _Utterance(priority, text, severity, key)

        # If the queue is full, drop the oldest lowest-priority item
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except Empty:
                pass
        try:
            self._queue.put_nowait(utt)
        except Exception:
            pass

    def _worker(self) -> None:
        while self._running:
            try:
                utt = self._queue.get(timeout=0.2)
            except Empty:
                continue
            if utt.key == '__exit__':
                break

            self._spoken_log.append({
                'time':     utt.tstamp,
                'severity': utt.severity,
                'text':     utt.text,
            })

            if self._mode == 'tts' and self._engine is not None:
                try:
                    # Pre-pend a short beep on critical so the driver
                    # snaps to attention before the speech starts.
                    if utt.severity == 'critical' and _WINSOUND_AVAILABLE:
                        f, ms = _SEV_BEEP['critical']
                        winsound.Beep(f, ms)
                    self._engine.say(utt.text)
                    self._engine.runAndWait()
                except Exception as e:
                    print(f"[VoiceAssistant] TTS error: {e}")
            elif self._mode == 'beep' and _WINSOUND_AVAILABLE:
                f, ms = _SEV_BEEP.get(utt.severity, _SEV_BEEP['advisory'])
                try:
                    winsound.Beep(f, ms)
                    print(f"[VoiceAssistant] {utt.severity.upper()}: "
                          f"{utt.text}")
                except Exception:
                    print(f"[VoiceAssistant] {utt.severity.upper()}: "
                          f"{utt.text}")
            else:
                # Silent mode: just log to stdout
                print(f"[VoiceAssistant] {utt.severity.upper()}: {utt.text}")
