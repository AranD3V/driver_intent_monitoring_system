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

import math
import re
import sys
import time
import threading
import numpy as np
import cv2
from queue import Queue, Empty
from typing import Optional, Tuple


# ── Traffic-light cycle constants ────────────────────────────────────── #
TL_GREEN_S  = 8.0
TL_YELLOW_S = 2.0
TL_RED_S    = 8.0
TL_PERIOD_S = TL_GREEN_S + TL_YELLOW_S + TL_RED_S   # 18s

# How far ahead (m) we look for an upcoming light, and the max lateral
# offset from the ego heading we still consider "in our lane".
TL_LOOKAHEAD_M     = 60.0
TL_LATERAL_TOL_M   = 8.0


def _light_status_label(status: str) -> str:
    """Maps MetaDriveType.LIGHT_* string constants to short labels."""
    s = (status or '').lower()
    if 'green'  in s: return 'green'
    if 'yellow' in s: return 'yellow'
    if 'red'    in s: return 'red'
    return 'unknown'


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
#  Traffic-light controller                                            #
# ------------------------------------------------------------------ #

class _TrafficLightController:
    """
    Spawns one BaseTrafficLight per approach lane at every intersection
    block (X / T / Roundabout / Y-junction) in the current map, then cycles
    each light through green->yellow->red on a fixed schedule with a phase
    offset per arm so opposing arms aren't both green at once.

    All methods MUST be called from the main thread (same Panda3D rule
    that applies to env.step()).
    """

    _INTERSECTION_MARKERS = (
        'InterSection', 'Intersection', 'TInter',
        'Round', 'YJunction', 'Junction',
    )

    def __init__(self, env):
        from metadrive.component.traffic_light.base_traffic_light import BaseTrafficLight
        self._env = env
        # List of dicts: {'light', 'lane', 'phase_offset_s', 'block_idx'}
        self._lights = []
        self._t0 = time.time()
        self._BaseTL = BaseTrafficLight
        self._spawn_all()

    # ------------------------------------------------------------------ #

    def _spawn_all(self):
        try:
            blocks = self._env.current_map.blocks
            graph  = self._env.current_map.road_network.graph
        except Exception as e:
            print(f"[TrafficLight] Could not read map: {e}")
            return

        # Which block indices are intersections?
        intersection_blocks = {}
        for blk in blocks:
            name = type(blk).__name__
            if any(m in name for m in self._INTERSECTION_MARKERS):
                intersection_blocks[getattr(blk, 'block_index', -1)] = name

        if not intersection_blocks:
            print("[TrafficLight] No intersection blocks in this map.")
            return

        # Approach lanes: lanes whose to-node sits inside an intersection
        # block AND whose from-node belongs to a different block.
        def _node_block_idx(n: str) -> Optional[int]:
            m = re.match(r'^-?(\d+)', n)
            return int(m.group(1)) if m else None

        approaches: dict = {bi: [] for bi in intersection_blocks}
        for frm in graph:
            for to in graph[frm]:
                from_bi = _node_block_idx(frm)
                to_bi   = _node_block_idx(to)
                if to_bi in intersection_blocks and from_bi != to_bi:
                    lanes = graph[frm][to]
                    if lanes:
                        # Rightmost lane only — one light per approach arm.
                        approaches[to_bi].append(lanes[0])

        # Spawn lights with phase offsets so arms within the same
        # intersection alternate (e.g. NS green when EW red).
        spawned = 0
        for bi, lanes in approaches.items():
            n = max(len(lanes), 1)
            half = TL_PERIOD_S / 2.0
            for i, lane in enumerate(lanes):
                try:
                    light = self._env.engine.spawn_object(self._BaseTL, lane=lane)
                except Exception as e:
                    print(f"[TrafficLight] Spawn failed on block {bi}: {e}")
                    continue
                # Alternate phase between adjacent approach lanes
                phase = (i % 2) * half
                self._lights.append({
                    'light':        light,
                    'lane':         lane,
                    'phase_offset_s': phase,
                    'block_idx':    bi,
                })
                spawned += 1

        if spawned:
            print(f"[TrafficLight] Spawned {spawned} lights across "
                  f"{len(intersection_blocks)} intersections "
                  f"({list(intersection_blocks.values())}).")

    # ------------------------------------------------------------------ #

    def step(self) -> None:
        """Advance each light's state machine. Called once per env.step()."""
        if not self._lights:
            return
        now = time.time() - self._t0
        for entry in self._lights:
            t = (now + entry['phase_offset_s']) % TL_PERIOD_S
            light = entry['light']
            try:
                if t < TL_GREEN_S:
                    if 'GREEN' not in (light.status or ''):
                        light.set_green()
                elif t < TL_GREEN_S + TL_YELLOW_S:
                    if 'YELLOW' not in (light.status or ''):
                        light.set_yellow()
                else:
                    if 'RED' not in (light.status or ''):
                        light.set_red()
            except Exception:
                pass

    # ------------------------------------------------------------------ #

    def find_nearest_ahead(self,
                            ego_pos: Tuple[float, float],
                            ego_heading: float) -> Optional[dict]:
        """
        Returns the closest light in front of the ego within TL_LOOKAHEAD_M:
            {'state': 'red'|'yellow'|'green',
             'distance_m': float,
             'position': (x, y),
             'light_id':  str}
        or None when no light is upcoming.
        """
        if not self._lights:
            return None
        cos_h = math.cos(ego_heading)
        sin_h = math.sin(ego_heading)
        best = None
        for entry in self._lights:
            light = entry['light']
            try:
                lp = light.position
                lx, ly = float(lp[0]), float(lp[1])
            except Exception:
                continue
            dx = lx - ego_pos[0]
            dy = ly - ego_pos[1]
            ahead = dx * cos_h + dy * sin_h      # >0 = in front of ego
            if ahead < -1.0:
                continue
            lateral = abs(-dx * sin_h + dy * cos_h)
            if lateral > TL_LATERAL_TOL_M:
                continue
            dist = math.hypot(dx, dy)
            if dist > TL_LOOKAHEAD_M:
                continue
            if best is None or dist < best['distance_m']:
                best = {
                    'state':      _light_status_label(light.status),
                    'distance_m': round(dist, 2),
                    'position':   (lx, ly),
                    'light_id':   str(id(light)),
                }
        return best

    # ------------------------------------------------------------------ #

    def destroy(self):
        for entry in self._lights:
            try:
                entry['light'].destroy()
            except Exception:
                pass
        self._lights.clear()


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
            # Richer map: a 7-block sequence with intersections, roundabouts
            # and ramps -- so the sim labeler sees broken/solid/yellow lines,
            # traffic cones, divided roads (for wrong-side detection), and
            # right-of-way intersections within a single free-roam session.
            # 'S' straight, 'C' curve, 'r'/'R' on-/off-ramp, 'X' 4-way,
            # 'T' 3-way, 'O' roundabout, 'y' Y-junction.
            "map":               "SCrRXTOC",
            "accident_prob":     0.4,     # spawns cones + barriers on road
            # ── Free-roam termination policy ─────────────────────────────
            # The user wants the sim to keep running through violations and
            # crashes -- the run only ends when they press q/Esc. So we
            # disable every termination MetaDrive lets us disable. The two
            # we can't disable (arrive_dest, crash_building) get caught in
            # read() and trigger a silent in-place reset.
            "out_of_road_done":        False,
            "on_continuous_line_done": False,
            "on_broken_line_done":     False,
            "crash_vehicle_done":      False,
            "crash_object_done":       False,
            "crash_human_done":        False,
            "horizon":                 None,
            "truncate_as_terminate":   False,
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
        # Last-step info dict (populated each read() so sim_labeler can
        # read crash flags, steering, accel, navigation, etc).
        self._last_info: dict = {}
        # Spawn + cycle traffic lights at every intersection block in the
        # current map. No-op if the map has no intersections.
        try:
            self._tl_controller = _TrafficLightController(self._env)
        except Exception as e:
            print(f"[MetaDriveCapture] Traffic-light setup failed: {e}")
            self._tl_controller = None
        # Cached nearest-light reading from the most recent read(), used by
        # get_sim_state() so sim_labeler doesn't need to recompute it.
        self._last_light_ahead: Optional[dict] = None

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
            obs, _reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, _reward, done, info = step_result

        # Cache for get_sim_state(); reset wipes it because the next info
        # only arrives on the following step.
        self._last_info = info if isinstance(info, dict) else {}

        if done:
            # MetaDrive only forces termination on arrive_dest or
            # crash_building (the two we cannot disable via config).
            # Everything else -- vehicle crashes, off-road, line crossings,
            # ego in oncoming lane -- keeps the run alive.
            reason = ('arrive_dest' if info.get('arrive_dest')
                      else 'crash_building' if info.get('crash_building')
                      else 'unknown')
            print(f"[MetaDriveCapture] Auto-reset (reason: {reason}) — "
                  f"respawning ego, keeping run alive.")
            try:
                # Lights must be destroyed BEFORE env.reset() — MetaDrive's
                # base_engine asserts there are no leftover physics bodies
                # when manager.reset() runs.
                if self._tl_controller is not None:
                    try:
                        self._tl_controller.destroy()
                    except Exception:
                        pass
                    self._tl_controller = None
                obs, _ = self._env.reset()
                self._last_info = {}
                # Re-spawn into the freshly-reset map.
                try:
                    self._tl_controller = _TrafficLightController(self._env)
                except Exception as e:
                    print(f"[MetaDriveCapture] TL re-spawn failed: {e}")
                    self._tl_controller = None
            except Exception as e:
                print(f"[MetaDriveCapture] Reset failed: {e}")

        # Cycle traffic lights + cache nearest-ahead reading for sim_state.
        if self._tl_controller is not None:
            try:
                self._tl_controller.step()
                agent = self._env.agent
                self._last_light_ahead = self._tl_controller.find_nearest_ahead(
                    ego_pos=(float(agent.position[0]), float(agent.position[1])),
                    ego_heading=float(getattr(agent, 'heading_theta', 0.0)),
                )
            except Exception as e:
                # Don't let TL bugs break the pipeline
                print(f"[MetaDriveCapture] TL step error: {e}")
                self._last_light_ahead = None
        else:
            self._last_light_ahead = None

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

    def get_sim_state(self) -> dict:
        """
        Oracle ground-truth pulled straight from MetaDrive every tick.
        Used by modules/sim_labeler.py to produce broken/yellow/solid line
        labels, wrong-side detection, current action and violation rows.

        Returns:
          {
            'info':  {... env.step() info dict ...},
            'agent': {
                'velocity': (vx, vy),
                'speed_kmh': float,
                'heading_theta': float,
                'on_lane': bool,
                'position': (x, y),
            },
            'lane':  {
                'left_marking':  (PGLineType_str, color_rgba),
                'right_marking': (PGLineType_str, color_rgba),
                'lane_direction': (dx, dy),
                'lane_direction_deg': float,
                'speed_limit_kmh': float,
                'lateral_offset_m': float,
            }
          }
        Returns {} if the env is closed or the agent has no lane yet.
        """
        try:
            agent = self._env.agent
            lane  = getattr(agent, 'lane', None)
            out: dict = {'info': dict(self._last_info or {})}

            vel = getattr(agent, 'velocity', (0.0, 0.0))
            pos = getattr(agent, 'position', (0.0, 0.0))
            out['agent'] = {
                'velocity':      (float(vel[0]), float(vel[1])),
                'speed_kmh':     float(getattr(agent, 'speed_km_h', 0.0) or 0.0),
                'heading_theta': float(getattr(agent, 'heading_theta', 0.0)),
                'on_lane':       bool(getattr(agent, 'on_lane', True)),
                'position':      (float(pos[0]), float(pos[1])),
            }

            if lane is not None:
                # line_types/line_colors are [left, right] lists.
                lt = list(getattr(lane, 'line_types', []) or [])
                lc = list(getattr(lane, 'line_colors', []) or [])
                left_t  = lt[0] if len(lt) > 0 else 'UNKNOWN_LINE'
                right_t = lt[1] if len(lt) > 1 else 'UNKNOWN_LINE'
                left_c  = lc[0] if len(lc) > 0 else (1, 1, 1, 1)
                right_c = lc[1] if len(lc) > 1 else (1, 1, 1, 1)

                # Direction vector and yaw (degrees)
                d = getattr(lane, 'direction', (1.0, 0.0))
                try:
                    dx, dy = float(d[0]), float(d[1])
                except Exception:
                    dx, dy = 1.0, 0.0
                import math
                yaw_deg = math.degrees(math.atan2(dy, dx))

                # Lateral offset (positive => right of center)
                lateral = 0.0
                try:
                    long_off, lat_off = lane.local_coordinates(pos)
                    lateral = float(lat_off)
                except Exception:
                    pass

                # MetaDrive's speed_limit is sometimes a sentinel like 1000.
                # Cap to the agent's max_speed_km_h when unrealistic.
                lim = float(getattr(lane, 'speed_limit', 0.0) or 0.0)
                max_kmh = float(getattr(agent, 'max_speed_km_h', 80.0) or 80.0)
                if lim <= 0 or lim > 200:
                    lim = max_kmh

                out['lane'] = {
                    'left_marking':       (str(left_t), tuple(left_c)),
                    'right_marking':      (str(right_t), tuple(right_c)),
                    'lane_direction':     (dx, dy),
                    'lane_direction_deg': yaw_deg,
                    'speed_limit_kmh':    lim,
                    'lateral_offset_m':   lateral,
                }
            # Nearest upcoming traffic light (None if no lights in map or
            # none within lookahead/lateral tolerance).
            out['traffic_light_ahead'] = self._last_light_ahead
            return out
        except Exception:
            return {}

    def release(self):
        self._running = False
        self._driver.stop()
        self._driver.join(timeout=2)
        if self._tl_controller is not None:
            try:
                self._tl_controller.destroy()
            except Exception:
                pass
            self._tl_controller = None
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
