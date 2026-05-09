"""
MetaDrive Scene Source
Replaces the physical scene camera with a MetaDrive RGB camera feed
for live demo mode. The driver camera is still a real webcam.

Architecture note
-----------------
MetaDrive is built on Panda3D, which is NOT thread-safe.  env.step() and
env.reset() must be called from the SAME thread that created the environment
(the main thread).  Therefore this class does NOT use a background thread for
MetaDrive.  Instead, read() steps the simulation inline and is called from the
main pipeline loop.  Only the driver webcam runs in a background thread.

Usage:
    capture = MetaDriveCapture(driver_source=0)
    driver_frame, scene_frame, ts = capture.read()   # call from main thread
    capture.release()
"""

import sys
import time
import threading
import numpy as np
import cv2
from queue import Queue, Empty
from typing import Optional, Tuple


# ------------------------------------------------------------------ #
#  Driver webcam — background thread                                   #
# ------------------------------------------------------------------ #

class _DriverThread(threading.Thread):
    """Continuously reads the driver-facing webcam into a 1-frame buffer."""

    def __init__(self, source: int):
        super().__init__(daemon=True, name='driver-cam')
        if sys.platform == 'win32':
            self._cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(source)
        else:
            self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open driver camera: {source}")

        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Warmup
        for _ in range(5):
            self._cap.read()

        self._queue: Queue = Queue(maxsize=1)
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except Empty:
                    pass
            self._queue.put(frame)

    def get_frame(self) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=0.05)
        except Empty:
            return None

    def stop(self):
        self._running = False
        self._cap.release()


# ------------------------------------------------------------------ #
#  MetaDrive capture — main-thread sim stepping                        #
# ------------------------------------------------------------------ #

class MetaDriveCapture:
    """
    Drop-in replacement for SynchronizedCapture using MetaDrive as the
    scene source.  Must be used from the main thread only.

    Args:
        driver_source : webcam index for the driver-facing camera
        width         : scene frame width  in pixels
        height        : scene frame height in pixels
        num_scenarios : number of random road layouts to cycle through
        manual        : if True, keyboard control via the Panda3D window
                        (W/S = throttle/brake, A/D = steer left/right);
                        if False, IDM autopilot drives the vehicle.
    """

    def __init__(self,
                 driver_source: int = 0,
                 width:  int = 800,
                 height: int = 450,
                 num_scenarios: int = 10,
                 manual: bool = False):

        self.width   = width
        self.height  = height
        self._manual = manual

        # ── Driver webcam ─────────────────────────────────────────────
        self._driver = _DriverThread(driver_source)
        self._driver.start()

        # ── MetaDrive ─────────────────────────────────────────────────
        try:
            from metadrive import MetaDriveEnv
        except ImportError:
            raise ImportError(
                "MetaDrive is not installed.\n"
                "Run: pip install metadrive-simulator"
            )

        RGBCamera = self._get_rgb_camera_class()

        env_config = {
            "num_scenarios":     num_scenarios,
            "start_seed":        42,
            "use_render":        True,
            "image_observation": True,
            "image_on_cuda":     False,
            "norm_pixel":        True,
            "stack_size":        1,        # avoid (H,W,3,N) stacked tensor
            "sensors": {
                "rgb_camera": (RGBCamera, width, height)
            },
            # Without image_source, the agent observation 'image' is empty
            # and YOLO sees a black scene. This is the MetaDrive convention
            # for binding a named sensor to the per-agent obs dict.
            "vehicle_config": {"image_source": "rgb_camera"},
            "interface_panel":   [],
            "show_logo":         False,
            "traffic_density":   0.1,
        }

        if manual:
            # MetaDrive reads W/A/S/D (or arrow keys) from the Panda3D window.
            env_config["manual_control"] = True
        else:
            from metadrive.policy.idm_policy import IDMPolicy
            env_config["agent_policy"] = IDMPolicy

        self._env = MetaDriveEnv(env_config)

        self._obs, _ = self._env.reset()
        self._terminated = False
        self._running = True

        if manual:
            print("[MetaDriveCapture] Ready: MetaDrive (manual control) + driver webcam.")
            print("[MetaDriveCapture] Controls: W/Up=accelerate  S/Down=brake  A/Left=steer left  D/Right=steer right")
            print("[MetaDriveCapture] Focus the MetaDrive window to receive key input.")
        else:
            print("[MetaDriveCapture] Ready: MetaDrive (autopilot) + driver webcam.")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Step the MetaDrive simulation one tick and return:
            (driver_frame, scene_frame, timestamp)

        MUST be called from the main thread (Panda3D requirement).
        Returns (None, None, 0.0) if the driver camera fails.
        """
        if not self._running:
            return None, None, 0.0

        # ── Step sim ──────────────────────────────────────────────────
        # In manual mode MetaDrive reads the Panda3D keyboard state itself;
        # the action value here is unused but the call is still required.
        try:
            step_result = self._env.step([0.0, 0.0])
        except Exception as e:
            print(f"[MetaDriveCapture] Step error: {e}")
            return None, None, 0.0

        # Gymnasium (5-tuple) vs old API (4-tuple)
        if len(step_result) == 5:
            obs, _reward, terminated, truncated, _info = step_result
            done = terminated or truncated
        else:
            obs, _reward, done, _info = step_result

        if done:
            try:
                obs, _ = self._env.reset()
            except Exception:
                pass

        # ── Scene frame ───────────────────────────────────────────────
        scene_frame = self._extract_frame(obs)
        ts = time.perf_counter()

        # ── Driver frame ──────────────────────────────────────────────
        driver_frame = self._driver.get_frame()
        if driver_frame is None:
            return None, None, 0.0

        return driver_frame, scene_frame, ts

    def get_telemetry(self) -> dict:
        """
        Latest MetaDrive ego-vehicle telemetry snapshot.
        Used by the warning engine for overspeeding / rash-driving detection.

        Returns:
            {'speed_kmh', 'speed_limit_kmh', 'steering', 'throttle', 'brake'}
        """
        try:
            agent = self._env.agent
            speed_kmh = float(getattr(agent, 'speed_km_h', 0.0) or 0.0)
            # MetaDrive's max_speed_km_h is the upper allowed; treat as limit.
            speed_limit = float(getattr(agent, 'max_speed_km_h', 80.0) or 80.0)
            ctrl = getattr(agent, 'last_current_action', None)
            steer = throttle = brake = 0.0
            if ctrl is not None and hasattr(ctrl, '__len__') and len(ctrl) >= 2:
                steer = float(ctrl[0])
                # MetaDrive's second action is throttle/brake combined: >0
                # accelerates, <0 brakes.
                t_b = float(ctrl[1])
                throttle = max(0.0, t_b)
                brake    = max(0.0, -t_b)
            return {
                'speed_kmh':       round(speed_kmh, 1),
                'speed_limit_kmh': round(speed_limit, 1),
                'steering':        round(steer, 3),
                'throttle':        round(throttle, 3),
                'brake':           round(brake, 3),
            }
        except Exception:
            return {'speed_kmh': 0.0, 'speed_limit_kmh': 0.0,
                    'steering': 0.0, 'throttle': 0.0, 'brake': 0.0}

    def release(self):
        self._running = False
        self._driver.stop()
        self._driver.join(timeout=2)
        try:
            self._env.close()
        except Exception:
            pass
        print("[MetaDriveCapture] Released.")

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _extract_frame(self, obs) -> Optional[np.ndarray]:
        """
        Convert a MetaDrive observation to a BGR uint8 numpy array.

        MetaDrive's image observation can come back in any of:
            (H, W, 3)            single frame
            (H, W, 3, N)         stacked across N timesteps  <-- common!
            (H, W, 4)            with alpha
        We collapse the time-stack to the most recent frame, drop alpha,
        and only invoke env.render(mode='rgb_array') as a last resort
        because in manual mode it returns a blank frame.
        """
        try:
            if isinstance(obs, dict) and "image" in obs:
                frame = np.asarray(obs["image"])
                # Stacked obs is (H, W, C, T) — pick the latest frame
                if frame.ndim == 4:
                    frame = frame[..., -1]
                # norm_pixel=True -> float [0,1]; convert to uint8
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
                # Drop alpha
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[..., :3]
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        except Exception as e:
            print(f"[MetaDriveCapture] image obs extract failed: {e}")

        # Fallback: pull from the named RGB sensor directly. This works
        # in both manual and autopilot modes and avoids env.render() which
        # is unreliable when use_render=True opens its own Panda3D window.
        try:
            cam = self._env.engine.get_sensor("rgb_camera")
            arr = cam.perceive(to_float=False, new_parent_node=self._env.agent.origin)
            arr = np.asarray(arr)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[..., :3]
            if arr.shape[1] != self.width or arr.shape[0] != self.height:
                arr = cv2.resize(arr, (self.width, self.height))
            return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[MetaDriveCapture] sensor.perceive fallback failed: {e}")

        return None

    @staticmethod
    def _get_rgb_camera_class():
        """Import RGBCamera — import path moved between MetaDrive versions."""
        for module_path in (
            "metadrive.component.sensors.rgb_camera",
            "metadrive.obs.image_obs",
        ):
            try:
                import importlib
                mod = importlib.import_module(module_path)
                return mod.RGBCamera
            except (ImportError, AttributeError):
                continue
        raise ImportError(
            "Cannot locate RGBCamera in your MetaDrive installation. "
            "Try: pip install --upgrade metadrive-simulator"
        )
