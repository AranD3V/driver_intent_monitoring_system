"""
Headless end-to-end smoke test for inference.py.

Replaces the live camera capture with a synthetic source that produces
640x480 frames (blank driver + scene with a moving red car rectangle),
patches cv2.imshow/waitKey to no-ops, runs N frames through the full
3-thread pipeline, and verifies the per-run output bundle was created.

Usage:
    python tests/smoke_pipeline.py [--frames 60]

Exit codes:
    0  all checks passed
    1  any check failed
"""
from __future__ import annotations

import argparse
import sys
import time
import json
import shutil
from pathlib import Path

import numpy as np
import cv2

# Make sibling imports work when run directly
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from inference import DriverIntentSystem
import inference as _inference_mod


# ──────────────────────────────────────────────────────────────────────── #
class _SyntheticCapture:
    """
    Drop-in substitute for SynchronizedCapture / sim captures.
    Produces deterministic frames with a "car" approaching from far → close,
    so the warning engine fires both 'high' and 'critical' during the run.
    """

    def __init__(self, total_frames: int = 60,
                 size=(640, 480),
                 driver_size=(640, 480)):
        self._n = total_frames
        self._i = 0
        self._scene_size = size
        self._driver_size = driver_size

    def read(self):
        if self._i >= self._n:
            return None, None, 0.0
        # Scene: a bbox that grows wider as i increases (car gets closer)
        scene = np.full((self._scene_size[1], self._scene_size[0], 3),
                        80, dtype=np.uint8)   # grey "road"
        # Bbox starts small (far), grows large (close)
        progress = self._i / max(1, self._n - 1)
        bbox_w = int(40 + 220 * progress)   # 40 px → 260 px
        bbox_h = int(40 + 220 * progress)
        cx = self._scene_size[0] // 2
        cy = self._scene_size[1] // 2
        x1, x2 = cx - bbox_w // 2, cx + bbox_w // 2
        y1, y2 = cy - bbox_h // 2, cy + bbox_h // 2
        # YOLO won't detect a coloured rectangle reliably; we don't rely on
        # YOLO firing here (we'll inject detections via WarningEngine path
        # in a separate test). This frame is just to keep the loop alive.
        cv2.rectangle(scene, (x1, y1), (x2, y2), (50, 50, 200), -1)
        cv2.putText(scene, f'frame {self._i}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        driver = np.full((self._driver_size[1], self._driver_size[0], 3),
                         50, dtype=np.uint8)
        # Add a vague "face" oval so MediaPipe might pick something up
        cv2.ellipse(driver, (320, 240), (120, 160), 0, 0, 360, (180, 160, 140), -1)
        cv2.circle(driver, (280, 220), 12, (255, 255, 255), -1)  # left eye
        cv2.circle(driver, (360, 220), 12, (255, 255, 255), -1)  # right eye
        cv2.circle(driver, (280, 220), 5, (40, 40, 40), -1)     # iris
        cv2.circle(driver, (360, 220), 5, (40, 40, 40), -1)

        ts = time.time()
        self._i += 1
        return driver, scene, ts

    def release(self):
        pass


# ──────────────────────────────────────────────────────────────────────── #
def _patch_display():
    """No-op cv2.imshow / waitKey so the test is fully headless."""
    cv2.imshow      = lambda *a, **kw: None
    cv2.namedWindow = lambda *a, **kw: None
    cv2.destroyAllWindows = lambda: None
    cv2.waitKey     = lambda *a, **kw: 0   # never returns 'q'
    cv2.setWindowProperty = lambda *a, **kw: None


def _patch_capture(total_frames: int):
    """Force inference.py to use _SyntheticCapture regardless of mode."""
    cap = _SyntheticCapture(total_frames=total_frames)
    _inference_mod.SynchronizedCapture = lambda *a, **kw: cap
    _inference_mod.MetaDriveCapture    = lambda *a, **kw: cap
    _inference_mod.CarlaCapture        = lambda *a, **kw: cap
    return cap


# ──────────────────────────────────────────────────────────────────────── #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--keep',   action='store_true',
                        help='Keep the run directory on success')
    args = parser.parse_args()

    print(f"[smoke] Running pipeline for {args.frames} synthetic frames...")
    _patch_display()
    _patch_capture(args.frames)

    # Use a dedicated, ephemeral run root so we don't pollute runs/
    test_runs = _REPO / 'runs' / '_smoke'
    if test_runs.exists():
        shutil.rmtree(test_runs, ignore_errors=True)
    test_runs.mkdir(parents=True, exist_ok=True)

    system = DriverIntentSystem(voice_enabled=False)
    system.start(
        driver_source=0,
        scene_source=0,
        log_enabled=True,
        sim='none',                # forces SynchronizedCapture path
        run_root=str(test_runs),
    )

    # ── Verify output bundle ──────────────────────────────────────────
    run_dirs = list(test_runs.glob('run_*'))
    if not run_dirs:
        print("[smoke] FAIL: no run_* directory created")
        return 1
    rd = run_dirs[0]
    print(f"[smoke] Run dir: {rd}")

    # Required artefacts get written every run.
    # gaze_affordance_map.png and gaze_affordance_overlay.png are only
    # produced when at least one fixation was registered — synthetic
    # frames don't have a real face, so MediaPipe finds no gaze and
    # the heatmap is empty. We mark those optional.
    expected = {
        'composite_video.mp4':          False,   # only if FPS settled
        'gaze_affordance_map.png':      False,
        'gaze_affordance_overlay.png':  False,
        'warnings.csv':                 True,
        'voice_log.json':               True,
        'summary.json':                 True,
        'summary.md':                   True,
        'session_log':                  True,    # directory
    }

    fails = []
    for name, must_exist in expected.items():
        p = rd / name
        if must_exist and not p.exists():
            fails.append(f"missing required artefact: {name}")
        elif p.exists():
            sz = p.stat().st_size if p.is_file() else 0
            print(f"  [OK]   {name}{'  ('+str(sz)+' B)' if p.is_file() else ' (dir)'}")
        else:
            print(f"  [skip] {name}  (optional, FPS may not have settled)")

    # Check summary.json schema
    sj = rd / 'summary.json'
    if sj.exists():
        try:
            with open(sj, encoding='utf-8') as f:
                summary = json.load(f)
            for k in ('mode', 'frame_count', 'avg_fps', 'duration_sec',
                      'intent_distribution', 'warnings_total',
                      'warnings_by_severity', 'artefacts'):
                if k not in summary:
                    fails.append(f"summary.json missing key: {k}")
            if summary.get('frame_count', 0) < 1:
                fails.append("summary.json frame_count is 0")
            print(f"  Frames processed: {summary.get('frame_count')}, "
                  f"avg_fps: {summary.get('avg_fps')}")
        except Exception as e:
            fails.append(f"summary.json parse error: {e}")

    # Check session_log has at least one batch
    sl = rd / 'session_log'
    if sl.exists():
        batches = list(sl.glob('*.json'))
        if not batches:
            print("  [warn] session_log is empty (test was too short to flush)")

    print()
    if fails:
        for f in fails:
            print(f"[smoke] FAIL: {f}")
        return 1

    print(f"[smoke] PASS — all expected artefacts produced.")
    if not args.keep:
        shutil.rmtree(test_runs, ignore_errors=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
