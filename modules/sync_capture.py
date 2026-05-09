"""
Synchronized Camera Capture
Software frame-level synchronization between driver and scene cameras.
"""

import cv2
import time
import threading
import numpy as np
from queue import Queue, Empty
from typing import Optional, Tuple


class _CameraThread(threading.Thread):
    """Background thread that continuously reads from one camera."""

    def __init__(self, source, name: str, buffer_size: int = 2):
        super().__init__(daemon=True, name=name)

        self._is_file = isinstance(source, str)

        # Try different backends for Windows compatibility
        import sys
        if sys.platform == 'win32' and not self._is_file:
            self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(source)
        else:
            self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {source}")

        # Try to set consistent resolution / fps
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Frame interval for throttling file playback to ~30 fps
        self._frame_interval = 1.0 / 30.0 if self._is_file else 0.0

        # Warmup - discard first few frames (live cameras only)
        if not self._is_file:
            for _ in range(5):
                self.cap.read()

        self._queue: Queue = Queue(maxsize=buffer_size)
        self._running = False
        print(f"[{name}] Initialized and ready")

    def run(self):
        self._running = True
        consecutive_failures = 0
        max_failures = 10
        _last_frame_time = 0.0

        try:
            while self._running:
                # Throttle file sources to ~30 fps so they don't race ahead
                # of the live driver webcam and break timestamp sync.
                if self._is_file:
                    now = time.perf_counter()
                    elapsed = now - _last_frame_time
                    if elapsed < self._frame_interval:
                        time.sleep(self._frame_interval - elapsed)
                    _last_frame_time = time.perf_counter()

                ret, frame = self.cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print(f"[{self.name}] Too many read failures, stopping")
                        self._queue.put(None)
                        break
                    time.sleep(0.01)  # Brief pause before retry
                    continue

                consecutive_failures = 0  # Reset on success
                ts = time.perf_counter()

                # Discard old frame if queue is full (stay real-time)
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except Empty:
                        pass
                self._queue.put((frame, ts))
        finally:
            self.cap.release()

    def get_frame(self, timeout: float = 0.1) -> Tuple[Optional[np.ndarray], float]:
        try:
            item = self._queue.get(timeout=timeout)
            if item is None:
                return None, 0.0
            return item
        except Empty:
            return None, 0.0

    def stop(self):
        self._running = False   # cap released by run() finally


class SynchronizedCapture:
    """
    Reads driver + scene cameras in separate threads and matches
    frame pairs by closest timestamp within `max_sync_ms` ms.
    """

    def __init__(self,
                 driver_source,
                 scene_source,
                 max_sync_ms: float = 33.0):
        """
        Args:
            driver_source : camera index or video path for driver cam
            scene_source  : camera index or video path for scene cam
            max_sync_ms   : max allowed timestamp gap (ms) between frame pair
        """
        self._driver_thread = _CameraThread(driver_source, 'driver-cam')
        self._scene_thread  = _CameraThread(scene_source,  'scene-cam')
        self.max_sync_sec   = max_sync_ms / 1000.0

        self._driver_thread.start()
        self._scene_thread.start()

        print(f"[SyncCapture] Cameras started. Max sync gap: {max_sync_ms:.0f} ms")

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Returns (driver_frame, scene_frame, avg_timestamp)
        or (None, None, 0) at end of stream.
        """
        driver_frame, ts_d = self._driver_thread.get_frame()
        scene_frame,  ts_s = self._scene_thread.get_frame()

        if driver_frame is None or scene_frame is None:
            return None, None, 0.0

        gap = abs(ts_d - ts_s)
        if gap > self.max_sync_sec:
            # Drop the later frame and try once more from that camera
            if ts_d > ts_s:
                scene_frame, ts_s = self._scene_thread.get_frame()
            else:
                driver_frame, ts_d = self._driver_thread.get_frame()

        return driver_frame, scene_frame, (ts_d + ts_s) / 2.0

    def release(self):
        self._driver_thread.stop()
        self._scene_thread.stop()
        self._driver_thread.join(timeout=2)
        self._scene_thread.join(timeout=2)
        print("[SyncCapture] Released.")
