"""
Real-time Inference Pipeline (v2)
Multithreaded: Thread-1 gaze, Thread-2 scene, Thread-3 fusion + intent.
"""

import cv2
import time
import threading
import numpy as np
import argparse
from pathlib import Path
from queue import Queue, Empty, Full
from typing import Optional, Dict

# Repo root: lets the tool be launched from any CWD without breaking
# relative paths to config / weights / calibration files.
_REPO_ROOT = Path(__file__).resolve().parent


def _repo_path(*parts: str) -> str:
    """Join one or more parts onto the repo root, returning an absolute str."""
    return str(_REPO_ROOT.joinpath(*parts))

from modules.sync_capture       import SynchronizedCapture
from modules.carla_capture      import CarlaCapture
from modules.metadrive_capture  import MetaDriveCapture
from modules.gaze_estimator   import GazeEstimator
from modules.scene_detector   import SceneDetector
from modules.affordance_engine import AffordanceEngine
from modules.intersection_engine import IntersectionEngine
from modules.temporal_model   import TemporalIntentPredictor
from modules.temporal_smoother import IntentSmoother
from modules.logger            import RollingLogger
from modules.visualize         import visualize_frame, SIDEBAR_W, HEADER_H, FOOTER_H
from modules.gaze_affordance_map import GazeAffordanceMap
from modules.warning_engine      import WarningEngine
from modules.voice_assistant     import VoiceAssistant
from modules.run_session         import RunSession
from modules.sim_labeler         import SimLabeler


# ------------------------------------------------------------------ #
class DriverIntentSystem:
    """
    Three-stage threaded pipeline:

        Thread 1: Gaze estimation   → gaze_out_queue
        Thread 2: Scene detection   → scene_out_queue
        Thread 3: Fusion + intent   → result_queue   (also writes to viz_queue)

    Main thread reads result_queue and displays.
    """

    def __init__(self, config_path: Optional[str] = None,
                 model_path: Optional[str] = None,
                 voice_enabled: bool = True):

        print("Initialising Driver Intent Monitoring System v2...")

        if config_path is None:
            config_path = _repo_path('config', 'affordance_config.json')
        elif not Path(config_path).is_absolute():
            config_path = _repo_path(config_path)

        # model_path can be a single str or a list of paths (ensemble mode).
        # Resolve each relative path against the repo root.
        if isinstance(model_path, (list, tuple)):
            model_path = [
                str(_repo_path(p)) if not Path(p).is_absolute() else p
                for p in model_path
            ]
        elif model_path and not Path(model_path).is_absolute():
            model_path = _repo_path(model_path)

        # ── Modules ──────────────────────────────────────────────────
        self.gaze_est   = GazeEstimator()
        self.scene_det  = SceneDetector()
        self.aff_eng    = AffordanceEngine(config_path)
        self.int_eng    = IntersectionEngine()
        self.intent_pred = TemporalIntentPredictor(window_size=90)
        self.smoother   = IntentSmoother(window_size=7, min_ratio=0.6, confidence_threshold=0.60)
        self.gaze_map   = GazeAffordanceMap()
        self.warning_eng = WarningEngine()
        self.voice       = VoiceAssistant(enabled=voice_enabled)
        # Oracle labeller for the MetaDrive scene source. No-op when running
        # with real cameras / CARLA (the capture object won't expose
        # get_sim_state() and SimLabeler returns {} silently).
        self.sim_labeler = SimLabeler()
        # RunSession's summary expects a string; collapse list-mode into a
        # readable label like "ensemble[5]: weak_v2_fold0.pth, ..."
        if isinstance(model_path, (list, tuple)):
            names = [Path(p).name for p in model_path]
            self._model_path = f"ensemble[{len(names)}]: " + ", ".join(names)
        else:
            self._model_path = model_path

        # logger and run session created lazily once we know the run mode
        self.logger:  Optional[RollingLogger] = None
        self.session: Optional[RunSession]    = None

        loaded = False
        if model_path:
            if isinstance(model_path, (list, tuple)) and len(model_path) > 1:
                loaded = self.intent_pred.load_ensemble(list(model_path))
            elif isinstance(model_path, (list, tuple)):
                loaded = self.intent_pred.load_weights(model_path[0])
            else:
                loaded = self.intent_pred.load_weights(model_path)
            if not loaded:
                # Don't keep an unusable path in the run summary
                self._model_path = None
        if not loaded:
            print("[System] No usable model weights. Using rule-based intent heuristics.")
            print("[System] Train: python scripts/train_intent.py train --data <files>")

        # ── Thread communication queues (maxsize keeps latency low) ──
        self._gaze_queue  : Queue = Queue(maxsize=2)
        self._scene_queue : Queue = Queue(maxsize=2)
        self._result_queue: Queue = Queue(maxsize=2)

        # Shared state (written by sub-threads, read by fusion thread)
        self._latest_gaze:  Optional[Dict] = None
        self._latest_scene: Optional[list] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()

        self._frame_count = 0
        self._running = False
        # Distraction detector — counts consecutive frames with no usable gaze
        self._no_gaze_streak = 0
        self._distraction_threshold = 30   # ~1 s at 30 fps

        print("System ready.")

    # ================================================================ #
    #  Public API                                                        #
    # ================================================================ #

    def _validate_cameras(self, driver_source, scene_source):
        import sys
        for label, src in [('driver', driver_source), ('scene', scene_source)]:
            if isinstance(src, int):
                if sys.platform == 'win32':
                    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        cap.release()
                        cap = cv2.VideoCapture(src)
                else:
                    cap = cv2.VideoCapture(src)
                ok = cap.isOpened()
                cap.release()
                if not ok:
                    raise RuntimeError(
                        f"Cannot open {label} camera (index {src}).\n"
                        f"Run: python scripts/identify_cameras.py"
                    )

    def start(self, driver_source=0, scene_source=2,
              output_video: Optional[str] = None,
              log_enabled: bool = True,
              sim: str = 'none',
              carla_host: Optional[str] = None,
              carla_port: int = 2000,
              manual: bool = False,
              run_root: str = 'runs'):

        self._running = True
        driver_idx = driver_source if isinstance(driver_source, int) else 0

        # Resolve relative paths against the repo root, not CWD
        if not Path(run_root).is_absolute():
            run_root = _repo_path(run_root)
        if output_video and not Path(output_video).is_absolute():
            output_video = _repo_path(output_video)

        if sim == 'metadrive':
            mode_label = "manual control" if manual else "autopilot"
            mode = f"sim_metadrive_{'manual' if manual else 'auto'}"
            print(f"[System] MetaDrive mode: scene from MetaDrive simulator ({mode_label}).")
            self.capture = MetaDriveCapture(driver_source=driver_idx, manual=manual)
            scene_label = f"metadrive({mode_label})"

        elif sim == 'carla' or carla_host:
            host = carla_host or 'localhost'
            mode = "sim_carla"
            print(f"[System] CARLA mode: scene from simulator @ {host}:{carla_port}")
            self.capture = CarlaCapture(
                driver_source=driver_idx,
                host=host,
                port=carla_port,
            )
            scene_label = f"carla://{host}:{carla_port}"

        else:
            mode = "two_cam"
            self._validate_cameras(driver_source, scene_source)
            self.capture = SynchronizedCapture(driver_source, scene_source)
            scene_label = scene_source

        # Per-run output bundle — owns video, heatmap, summary, logs
        self.session = RunSession(
            mode=mode,
            root=run_root,
            model_path=self._model_path,
            driver_source=driver_source,
            scene_source=scene_label,
        )
        self.logger = RollingLogger(
            log_dir=self.session.get_log_dir(),
            flush_interval=500,
            session_id=self.session.run_id,
        )

        # Voice prompt engine
        self.voice.start()

        # Composite video writer always written to the run dir;
        # `output_video` (CLI override) still wins if given.
        self._video_writer = None
        self._video_size: Optional[tuple] = None
        self._scene_size: Optional[tuple] = None
        self._output_video = output_video or self.session.composite_video_path
        self._scene_res_set = False

        # Launch threads
        t_gaze  = threading.Thread(target=self._gaze_worker,  daemon=True, name='gaze')
        t_scene = threading.Thread(target=self._scene_worker, daemon=True, name='scene')
        t_fuse  = threading.Thread(target=self._fusion_worker, daemon=True, name='fusion')

        t_gaze.start()
        t_scene.start()
        t_fuse.start()

        # Main loop: display
        fps_buf = []

        print("Press 'q' or Esc to quit.\n")
        try:
            while self._running:
                t0 = time.perf_counter()

                driver_frame, scene_frame, ts = self.capture.read()
                if driver_frame is None:
                    print("Stream ended.")
                    break
                if scene_frame is None:
                    continue   # sim still warming up — skip this tick

                self._frame_count += 1

                # One-time: detect actual scene resolution and propagate it
                if not self._scene_res_set:
                    sh, sw = scene_frame.shape[:2]
                    self.gaze_est.set_scene_resolution(sw, sh)
                    self.intent_pred.set_scene_resolution(sw, sh)
                    self.gaze_map.set_scene_resolution(sw, sh)
                    # Sim-aware focal length so distance estimates and
                    # bbox-area triggers work without a calibration file.
                    self.scene_det.set_scene_resolution(
                        sw, sh,
                        hfov_deg=(90.0 if sim == 'carla'
                                  else 70.0 if sim == 'metadrive'
                                  else 65.0))
                    self._scene_size = (sw, sh)
                    self._scene_res_set = True
                    self._video_size = (sw + SIDEBAR_W,
                                        sh + HEADER_H + FOOTER_H)
                    # Loud warning if the scene feed is blank — silent black
                    # frames means YOLO sees nothing for the whole run and
                    # warnings/heatmap end up empty (the historical sim bug).
                    mean_val = float(scene_frame.mean())
                    if mean_val < 4.0:
                        print("=" * 64)
                        print("[System] WARNING: scene frame appears blank "
                              f"(mean pixel value {mean_val:.1f}/255).")
                        print("[System] YOLO will detect nothing, "
                              "warnings.csv and heatmap will be empty.")
                        if sim == 'metadrive':
                            print("[System] Check that MetaDrive's "
                                  "image_source is bound to rgb_camera.")
                        print("=" * 64)
                    else:
                        print(f"[System] Scene feed OK "
                              f"(mean={mean_val:.1f}, size={sw}x{sh}).")

                # Push frames to worker threads (drop if busy → stay real-time)
                self._push(self._gaze_queue,  (driver_frame.copy(), ts))
                self._push(self._scene_queue, (scene_frame.copy(),  ts))
                with self._lock:
                    self._latest_frame = scene_frame

                # FPS
                dt = time.perf_counter() - t0
                fps_buf.append(1.0 / dt if dt > 0 else 0)
                fps_buf = fps_buf[-30:]
                avg_fps = float(np.mean(fps_buf))

                # Build composite frame
                try:
                    result: Dict = self._result_queue.get_nowait()
                    vis = visualize_frame(scene_frame, result,
                                         driver_frame=driver_frame, fps=avg_fps)
                    if log_enabled and self.logger is not None:
                        self.logger.log(result)
                    # Driver-assist voice prompts
                    self.voice.dispatch_warnings(result.get('warnings'))
                    self.voice.dispatch_intent(result.get('intent_prediction'))

                    # Distraction nudge — when no gaze for >1s while objects present
                    gd = result.get('gaze_data') or {}
                    if not gd.get('gaze_point'):
                        self._no_gaze_streak += 1
                    else:
                        self._no_gaze_streak = 0
                    if (self._no_gaze_streak >= self._distraction_threshold
                            and result.get('detected_objects')):
                        self.voice.dispatch_distraction(True)
                        self._no_gaze_streak = 0   # don't re-fire every frame
                    # Run session aggregation
                    if self.session is not None:
                        self.session.update(result, scene_frame=scene_frame,
                                            fps=avg_fps)
                except Empty:
                    vis = visualize_frame(scene_frame, {},
                                         driver_frame=driver_frame, fps=avg_fps)

                # Lazy-create the video writer after FPS has stabilised so
                # the saved file plays at the correct speed instead of being
                # locked to a hardcoded value.
                if (self._video_writer is None
                        and self._output_video
                        and self._scene_res_set
                        and len(fps_buf) >= 20):
                    measured_fps = float(np.clip(np.mean(fps_buf), 5.0, 60.0))
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self._video_writer = cv2.VideoWriter(
                        self._output_video, fourcc, measured_fps,
                        self._video_size)
                    print(f"[System] Recording composite video at "
                          f"{measured_fps:.1f} FPS -> {self._output_video}")

                if self._video_writer:
                    self._video_writer.write(vis)

                cv2.imshow('Driver Intent Monitor v2', vis)

                # Stop only on explicit user quit. Crashes / violations /
                # off-road events DO NOT end the run -- MetaDrive is
                # configured to keep stepping through them.
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:   # 27 = Esc
                    break

        except KeyboardInterrupt:
            pass

        finally:
            self._shutdown()

    # ================================================================ #
    #  Worker threads                                                    #
    # ================================================================ #

    def _gaze_worker(self):
        while self._running:
            item = self._safe_get(self._gaze_queue)
            if item is None:
                continue
            driver_frame, ts = item
            try:
                gaze_data = self.gaze_est.process_frame(driver_frame)
            except Exception as e:
                print(f"[GazeWorker] Error: {e}")
                gaze_data = None
            with self._lock:
                self._latest_gaze = gaze_data

    def _scene_worker(self):
        while self._running:
            item = self._safe_get(self._scene_queue)
            if item is None:
                continue
            scene_frame, ts = item
            try:
                detections = self.scene_det.process_frame(scene_frame)
                detections  = self.aff_eng.encode_affordances(detections)
            except Exception as e:
                print(f"[SceneWorker] Error: {e}")
                detections = []
            with self._lock:
                self._latest_scene = detections

    def _fusion_worker(self):
        while self._running:
            time.sleep(0.01)  # ~100 Hz polling

            with self._lock:
                gaze_data  = self._latest_gaze
                detections = self._latest_scene or []
                scene_frame = self._latest_frame

            if scene_frame is None:
                continue

            # Gaze-object intersection
            gaze_affordance = None
            if gaze_data and gaze_data.get('gaze_point'):
                gaze_affordance = self.int_eng.find_gaze_target(
                    gaze_point   = gaze_data['gaze_point'],
                    gaze_speed   = gaze_data.get('gaze_speed', 0),
                    is_fixation  = gaze_data.get('is_fixation', True),
                    detections   = detections
                )

            telemetry_now = (self.capture.get_telemetry()
                             if hasattr(self.capture, 'get_telemetry') else None)

            # Oracle sim labels: lane markings, current action, violations
            # (only populated when the capture is MetaDriveCapture).
            try:
                sim_label = self.sim_labeler.label(self.capture, telemetry_now)
            except Exception as e:
                print(f"[FusionWorker] SimLabel error: {e}")
                sim_label = {}

            frame_data: Dict = {
                'timestamp':        time.time(),
                'frame_number':     self._frame_count,
                'gaze_data':        gaze_data,
                'detected_objects': detections,
                'gaze_affordance':  gaze_affordance,
                'carla_telemetry':  telemetry_now,
                'sim_label':        sim_label,
            }

            # Intent prediction
            try:
                self.intent_pred.add_frame(frame_data)
                raw_intent  = self.intent_pred.predict()
                intent_pred = self.smoother.update(raw_intent)
            except Exception as e:
                print(f"[FusionWorker] Intent error: {e}")
                intent_pred = {'intent': 'normal_forward', 'confidence': 0.0}

            frame_data['intent_prediction'] = intent_pred

            # Warnings -- scene triggers + driver-state triggers
            # (drowsiness / overspeed / rash driving) all in one list.
            telemetry = (self.capture.get_telemetry()
                         if hasattr(self.capture, 'get_telemetry') else None)
            frame_data['warnings'] = self.warning_eng.evaluate(
                detections, gaze_affordance,
                scene_size=self._scene_size,
                gaze_data=gaze_data,
                telemetry=telemetry,
            )

            # Gaze affordance map — update then render in the same thread
            _gp   = gaze_data['gaze_point'] if gaze_data else None
            _aff  = gaze_affordance.get('affordance')   if gaze_affordance else None
            _risk = gaze_affordance.get('risk_level', 'low') if gaze_affordance else 'low'
            _fix  = gaze_data.get('is_fixation', False) if gaze_data else False
            self.gaze_map.update(_gp, _aff, _risk, _fix)
            frame_data['gaze_map_overlay'] = self.gaze_map.render()

            self._push(self._result_queue, frame_data)

    # ================================================================ #
    #  Helpers                                                           #
    # ================================================================ #

    @staticmethod
    def _push(q: Queue, item):
        """Non-blocking push: drop oldest if full."""
        if q.full():
            try:
                q.get_nowait()
            except Empty:
                pass
        try:
            q.put_nowait(item)
        except Full:
            pass

    @staticmethod
    def _safe_get(q: Queue, timeout: float = 0.05):
        try:
            return q.get(timeout=timeout)
        except Empty:
            return None

    def _shutdown(self):
        self._running = False
        try:
            self.capture.release()
        except Exception:
            pass
        if self._video_writer:
            self._video_writer.release()
        cv2.destroyAllWindows()
        if self.logger is not None:
            self.logger.flush_final()
        try:
            self.voice.shutdown()
        except Exception:
            pass
        if self.session is not None:
            try:
                self.session.finalize(
                    gaze_map=self.gaze_map,
                    voice_assistant=self.voice,
                )
            except Exception as e:
                print(f"[System] Run finalize failed: {e}")
        self.gaze_est.cleanup()
        print("System shut down.")


# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description='Driver Intent Monitor v2')
    parser.add_argument('--driver',  default='0',  help='Driver cam index or video path')
    parser.add_argument('--scene',   default='1',  help='Scene cam index or video path (ignored in sim modes)')
    parser.add_argument('--model',   default=None, nargs='+',
                        help='Model checkpoint path(s). Pass 1 for single-model '
                             'inference, or 5 fold checkpoints for ensemble '
                             '(73%% ensemble accuracy vs 46.6%% single best fold).')
    parser.add_argument('--output',  default=None, help='Output video path')
    parser.add_argument('--no-log',  action='store_true', help='Disable logging')
    parser.add_argument('--sim',     default='none',
                        choices=['none', 'metadrive', 'carla'],
                        help='Scene source: none=real camera, metadrive=MetaDrive sim, carla=CARLA sim')
    parser.add_argument('--carla-host', default=None, type=str, help='CARLA server host (required when --sim carla)')
    parser.add_argument('--carla-port', default=2000, type=int, help='CARLA server port')
    parser.add_argument('--manual',     action='store_true', help='MetaDrive: keyboard control instead of autopilot')
    parser.add_argument('--no-voice',   action='store_true', help='Disable voice assistant prompts')
    parser.add_argument('--run-root',   default='runs', help='Parent directory for per-run output bundles')
    args = parser.parse_args()

    def _parse_src(s):
        return int(s) if s.isdigit() else s

    system = DriverIntentSystem(
        model_path=args.model,
        voice_enabled=not args.no_voice,
    )
    system.start(
        driver_source = _parse_src(args.driver),
        scene_source  = _parse_src(args.scene),
        output_video  = args.output,
        log_enabled   = not args.no_log,
        sim           = args.sim,
        carla_host    = args.carla_host,
        carla_port    = args.carla_port,
        manual        = args.manual,
        run_root      = args.run_root,
    )


if __name__ == '__main__':
    main()
