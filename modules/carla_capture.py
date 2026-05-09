"""
CARLA Simulator Capture

Supplements the scene camera with a CARLA RGB sensor mounted on the ego
vehicle.  The driver camera still uses a real webcam so that MediaPipe
gaze estimation runs exactly as in live mode.

Interface mirrors SynchronizedCapture:
    read()    → (driver_frame, scene_frame, timestamp)
    release() → cleanup

Bonus attribute:
    get_telemetry() → dict with speed_kmh, location, rotation, traffic_light
"""

import time
import threading
import numpy as np
import cv2
import sys
from queue import Queue, Empty
from typing import Optional, Tuple, Dict

try:
    import carla
    _CARLA_AVAILABLE = True
except ImportError:
    _CARLA_AVAILABLE = False


# Camera mount relative to the ego vehicle root (windshield centre)
_CAM_TRANSFORM_ARGS = dict(x=1.5, z=2.4)


class CarlaCapture:
    """
    Drop-in replacement for SynchronizedCapture that sources the scene
    feed from a CARLA RGB sensor while keeping a real webcam for the
    driver (gaze) camera.

    Args:
        driver_source   : webcam index for the real driver-face camera
        host            : CARLA server hostname (default 'localhost')
        port            : CARLA server port     (default 2000)
        timeout         : connection timeout in seconds
        scene_width     : RGB sensor output width  (px)
        scene_height    : RGB sensor output height (px)
        scene_fov       : RGB sensor horizontal FOV (degrees)
        ego_role        : role_name attribute to look for on the ego vehicle;
                          if not found a Tesla Model 3 is spawned on autopilot
    """

    def __init__(self,
                 driver_source: int = 0,
                 host: str = 'localhost',
                 port: int = 2000,
                 timeout: float = 10.0,
                 scene_width: int = 1280,
                 scene_height: int = 720,
                 scene_fov: float = 90.0,
                 ego_role: str = 'hero'):

        if not _CARLA_AVAILABLE:
            raise ImportError(
                "The 'carla' Python package is not installed.\n"
                "Install it with:  pip install carla\n"
                "or copy the .egg from your CARLA release's PythonAPI/."
            )

        # ── Driver camera (real webcam) ──────────────────────────────
        self._driver_cap = self._open_webcam(driver_source)

        # ── CARLA connection ─────────────────────────────────────────
        print(f"[CarlaCapture] Connecting to CARLA at {host}:{port}...")
        self._client = carla.Client(host, port)
        self._client.set_timeout(timeout)
        self._world  = self._client.get_world()
        print(f"[CarlaCapture] Connected. Map: {self._world.get_map().name}")

        # ── Ego vehicle ───────────────────────────────────────────────
        self._spawned_ego = False
        self._ego = self._find_ego(ego_role)
        if self._ego is None:
            self._ego = self._spawn_ego()
            self._spawned_ego = True

        # ── RGB camera sensor ─────────────────────────────────────────
        self._scene_queue: Queue = Queue(maxsize=2)
        self._sensor = self._attach_rgb_camera(scene_width, scene_height, scene_fov)

        # ── Telemetry state ───────────────────────────────────────────
        self._telemetry: Dict = {}
        self._tele_lock = threading.Lock()

        print("[CarlaCapture] Ready: driver cam = webcam, scene cam = CARLA RGB sensor.")

    # ================================================================ #
    #  Public API                                                        #
    # ================================================================ #

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Returns (driver_frame, scene_frame, timestamp).
        Returns (None, None, 0.0) when either source fails.
        """
        ret, driver_frame = self._driver_cap.read()
        if not ret:
            return None, None, 0.0

        try:
            scene_frame, ts = self._scene_queue.get(timeout=0.15)
        except Empty:
            return None, None, 0.0

        return driver_frame, scene_frame, ts

    def get_telemetry(self) -> Dict:
        """Latest CARLA vehicle + environment telemetry snapshot."""
        with self._tele_lock:
            return dict(self._telemetry)

    def release(self):
        if self._sensor and self._sensor.is_alive:
            self._sensor.stop()
            self._sensor.destroy()
        # Only destroy ego if we spawned it; leave pre-existing actors alone.
        if self._spawned_ego and self._ego and self._ego.is_alive:
            self._ego.destroy()
        self._driver_cap.release()
        print("[CarlaCapture] Released.")

    # ================================================================ #
    #  Private helpers                                                   #
    # ================================================================ #

    @staticmethod
    def _open_webcam(source: int) -> cv2.VideoCapture:
        if sys.platform == 'win32':
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open driver webcam at index {source}.\n"
                "Run: python scripts/identify_cameras.py"
            )
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(5):          # discard warmup frames
            cap.read()
        return cap

    def _find_ego(self, role: str) -> Optional['carla.Actor']:
        for actor in self._world.get_actors():
            if actor.attributes.get('role_name') == role:
                print(f"[CarlaCapture] Found ego: {actor.type_id}  id={actor.id}")
                return actor
        return None

    def _spawn_ego(self) -> 'carla.Actor':
        bp_lib = self._world.get_blueprint_library()
        bp = bp_lib.filter('vehicle.tesla.model3')[0]
        bp.set_attribute('role_name', 'hero')
        spawn_pts = self._world.get_map().get_spawn_points()
        vehicle = self._world.spawn_actor(bp, spawn_pts[0])
        vehicle.set_autopilot(True)
        print(f"[CarlaCapture] Spawned ego: {vehicle.type_id}  id={vehicle.id}")
        return vehicle

    def _attach_rgb_camera(self, w: int, h: int, fov: float) -> 'carla.Actor':
        bp_lib  = self._world.get_blueprint_library()
        cam_bp  = bp_lib.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(w))
        cam_bp.set_attribute('image_size_y', str(h))
        cam_bp.set_attribute('fov',          str(fov))

        mount = carla.Transform(carla.Location(**_CAM_TRANSFORM_ARGS))
        sensor = self._world.spawn_actor(cam_bp, mount, attach_to=self._ego)
        sensor.listen(self._on_scene_image)
        print(f"[CarlaCapture] RGB sensor attached  id={sensor.id}  {w}x{h} fov={fov}deg")
        return sensor

    def _on_scene_image(self, image: 'carla.Image'):
        """CARLA sensor callback — runs in CARLA's internal thread."""
        # Convert BGRA → BGR
        arr   = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr   = arr.reshape((image.height, image.width, 4))
        frame = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        ts    = time.perf_counter()

        # Gather telemetry in the same callback to keep them in sync
        self._update_telemetry(ts)

        # Drop stale frame if consumer is slow
        if self._scene_queue.full():
            try:
                self._scene_queue.get_nowait()
            except Empty:
                pass
        self._scene_queue.put((frame, ts))

    def _update_telemetry(self, ts: float):
        try:
            v      = self._ego.get_velocity()
            tf     = self._ego.get_transform()
            ctrl   = self._ego.get_control()
            speed  = 3.6 * (v.x**2 + v.y**2 + v.z**2) ** 0.5

            # Nearest traffic light affecting ego
            tl_state = 'unknown'
            tl = self._ego.get_traffic_light()
            if tl is not None:
                tl_state = str(tl.get_state()).split('.')[-1].lower()

            with self._tele_lock:
                self._telemetry = {
                    'speed_kmh':    round(speed, 1),
                    'location':     {'x': round(tf.location.x, 2),
                                     'y': round(tf.location.y, 2),
                                     'z': round(tf.location.z, 2)},
                    'rotation':     {'yaw':   round(tf.rotation.yaw,   1),
                                     'pitch': round(tf.rotation.pitch, 1),
                                     'roll':  round(tf.rotation.roll,  1)},
                    'throttle':     round(ctrl.throttle, 3),
                    'brake':        round(ctrl.brake,    3),
                    'steer':        round(ctrl.steer,    3),
                    'traffic_light': tl_state,
                    'timestamp':    ts
                }
        except Exception:
            pass   # ego may be mid-destroy; silently skip
