"""
Personal Gaze Calibration Script

Run this once per driver per camera setup. Output:
    calibration/gaze_user.yaml

Procedure (≈30 s):
    1. A fullscreen black window opens on the active monitor.
    2. Nine target dots are shown in a 3×3 grid, one at a time.
    3. For each dot, look at the dot until the progress ring fills.
    4. The script captures iris offset + head pose samples from your
       driver-facing camera while you fixate.
    5. After all 9 targets, a polynomial mapping is fit and saved.

Usage:
    python scripts/calibrate_gaze.py --driver 1
    python scripts/calibrate_gaze.py --driver 1 --width 1920 --height 1080

Keys:
    Space   start the calibration sequence
    Esc / q abort
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Allow running as a script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.gaze_estimator import GazeEstimator
from modules.gaze_calibration import GazeCalibrator


WINDOW = "Gaze Calibration"

# 3 × 3 grid of normalised target positions (10% inset from edges)
TARGETS_NORM: List[Tuple[float, float]] = [
    (0.10, 0.10), (0.50, 0.10), (0.90, 0.10),
    (0.10, 0.50), (0.50, 0.50), (0.90, 0.50),
    (0.10, 0.90), (0.50, 0.90), (0.90, 0.90),
]

DWELL_SECONDS    = 1.6      # how long to fixate each dot
WARMUP_SECONDS   = 0.5      # discarded at the start of each fixation
MIN_SAMPLES_PER  = 12       # per target dot


def _open_camera(idx: int) -> cv2.VideoCapture:
    if sys.platform == 'win32':
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
    else:
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open driver camera index {idx}")
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(5):
        cap.read()
    return cap


def _draw_preview(canvas: np.ndarray, frame: Optional[np.ndarray]) -> None:
    """Blit a small webcam thumbnail in the top-right so the user can
    confirm their face is properly framed."""
    if frame is None:
        return
    h, w = canvas.shape[:2]
    fh, fw = frame.shape[:2]
    pw = 220
    ph = int(pw * fh / fw)
    if ph > h // 3:
        ph = h // 3
        pw = int(ph * fw / fh)
    thumb = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
    pad = 12
    x0, y0 = w - pw - pad, pad + 30
    canvas[y0:y0 + ph, x0:x0 + pw] = thumb
    cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + pw, y0 + ph),
                  (90, 90, 90), 1)


def _draw_target(canvas: np.ndarray, x: int, y: int,
                 progress: float, label: str = "",
                 face_ok: bool = True,
                 eyes_open: bool = True,
                 preview: Optional[np.ndarray] = None) -> None:
    """Draw a fixation target with a ring whose fill reflects progress."""
    canvas[:] = (10, 10, 10)
    cv2.circle(canvas, (x, y), 32, (45, 45, 45), 2, cv2.LINE_AA)
    if progress > 0:
        ang = int(360 * min(progress, 1.0))
        cv2.ellipse(canvas, (x, y), (32, 32), -90, 0, ang,
                    (200, 220, 255), 3, cv2.LINE_AA)
    cv2.circle(canvas, (x, y),  8, (60, 130, 250), -1, cv2.LINE_AA)
    cv2.circle(canvas, (x, y),  3, (235, 235, 235), -1, cv2.LINE_AA)
    if label:
        cv2.putText(canvas, label, (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200),
                    1, cv2.LINE_AA)

    # Status indicator on the right side
    h, w = canvas.shape[:2]
    if not face_ok:
        msg, color = "NO FACE DETECTED", (60, 60, 230)
    elif not eyes_open:
        msg, color = "EYES CLOSED",      (60, 180, 230)
    else:
        msg, color = "TRACKING",         (90, 200, 90)
    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX,
                                  0.55, 1)
    cv2.putText(canvas, msg, (w - tw - 30, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    _draw_preview(canvas, preview)


def _wait_for_start(width: int, height: int,
                    cap: Optional[cv2.VideoCapture] = None) -> bool:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    while True:
        canvas[:] = (10, 10, 10)
        title = "Gaze Calibration"
        body = [
            "You will see 9 dots, one at a time.",
            "Look at each dot until the ring fills.",
            "Keep your head still; eyes do most of the work.",
            "",
            "Press SPACE to start, ESC to abort.",
        ]
        cv2.putText(canvas, title, (60, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (235, 235, 235),
                    2, cv2.LINE_AA)
        for i, line in enumerate(body):
            cv2.putText(canvas, line, (60, 150 + i * 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200),
                        1, cv2.LINE_AA)
        # Live webcam thumbnail so the user can confirm framing
        if cap is not None:
            ret, frame = cap.read()
            if ret and frame is not None:
                _draw_preview(canvas, frame)
        cv2.imshow(WINDOW, canvas)
        k = cv2.waitKey(30) & 0xFF
        if k == 32:    # space
            return True
        if k in (27, ord('q'), ord('Q')):
            return False


def _run_calibration(cap: cv2.VideoCapture,
                     gaze: GazeEstimator,
                     width: int,
                     height: int) -> List[dict]:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    samples: List[dict] = []

    for i, (nx, ny) in enumerate(TARGETS_NORM):
        tx, ty = int(nx * width), int(ny * height)

        # Quick attention-grab pulse so the user knows the dot has moved
        for r in range(48, 14, -3):
            canvas[:] = (10, 10, 10)
            cv2.circle(canvas, (tx, ty), r, (180, 180, 200), 1, cv2.LINE_AA)
            cv2.circle(canvas, (tx, ty), 8, (60, 130, 250), -1, cv2.LINE_AA)
            cv2.imshow(WINDOW, canvas)
            cv2.waitKey(8)

        # Capture
        t0 = time.time()
        captured = 0
        face_ok = True
        eyes_open = True
        last_frame: Optional[np.ndarray] = None
        gaze.set_scene_resolution(width, height)
        while True:
            elapsed = time.time() - t0
            progress = elapsed / DWELL_SECONDS
            _draw_target(canvas, tx, ty, progress,
                         label=f"Target {i + 1}/9   samples {captured}",
                         face_ok=face_ok, eyes_open=eyes_open,
                         preview=last_frame)
            cv2.imshow(WINDOW, canvas)
            if cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')):
                return []   # aborted

            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            last_frame = frame
            data = gaze.process_frame(frame)
            face_ok = data is not None
            if not face_ok:
                # Don't advance the warmup clock while the face is missing
                t0 += 1.0 / 30.0
                continue
            eyes_open = not data.get('eyes_closed', False)
            if elapsed < WARMUP_SECONDS:
                continue
            if not eyes_open:
                continue
            iris = data.get('iris_offset')
            hp   = data.get('head_pose') or {}
            if iris is None:
                continue
            samples.append({
                'iris_dx':  iris[0],
                'iris_dy':  iris[1],
                'yaw':      hp.get('yaw', 0.0),
                'pitch':    hp.get('pitch', 0.0),
                'target_x': tx,
                'target_y': ty,
            })
            captured += 1

            if elapsed >= DWELL_SECONDS and captured >= MIN_SAMPLES_PER:
                break
            if elapsed > DWELL_SECONDS * 3.0:
                # Safety: if face stayed lost, move on to avoid hanging
                break

        # Brief pause between targets
        canvas[:] = (10, 10, 10)
        cv2.imshow(WINDOW, canvas)
        cv2.waitKey(180)

    return samples


def _run_verification(cap: cv2.VideoCapture,
                      gaze: GazeEstimator,
                      width: int,
                      height: int,
                      cal_path: str) -> int:
    """
    Live verification: show 5 random targets, ask the user to fixate each,
    and measure the median pixel error of the loaded calibration's
    prediction. Useful as a quick sanity check after fitting.
    """
    cal = GazeCalibrator.load(cal_path)
    if cal is None or not cal.is_fit:
        print(f"[calibrate_gaze] No calibration to verify at {cal_path}. "
              f"Run a calibration first.")
        return 1

    # Spread 5 targets so each region of the scene is exercised
    targets_norm = [
        (0.20, 0.20), (0.80, 0.20), (0.50, 0.50),
        (0.20, 0.80), (0.80, 0.80),
    ]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    errors_px: List[float] = []

    print("[calibrate_gaze] Verifying. Look at each dot for ~1.5 s.")
    for nx, ny in targets_norm:
        tx, ty = int(nx * width), int(ny * height)
        t0 = time.time()
        last_pred = None
        last_frame = None
        face_ok = True
        eyes_open = True
        while time.time() - t0 < DWELL_SECONDS:
            progress = (time.time() - t0) / DWELL_SECONDS
            _draw_target(canvas, tx, ty, progress,
                         label=f"VERIFY  fixate target",
                         face_ok=face_ok, eyes_open=eyes_open,
                         preview=last_frame)
            if last_pred is not None:
                cv2.circle(canvas, last_pred, 6, (0, 230, 230), 2,
                           cv2.LINE_AA)
            cv2.imshow(WINDOW, canvas)
            if cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')):
                return 1

            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            last_frame = frame
            data = gaze.process_frame(frame)
            face_ok = data is not None
            if not face_ok:
                continue
            eyes_open = not data.get('eyes_closed', False)
            if not eyes_open or progress < WARMUP_SECONDS / DWELL_SECONDS:
                continue
            iris = data.get('iris_offset')
            hp   = data.get('head_pose') or {}
            if iris is None:
                continue
            last_pred = cal.predict(iris[0], iris[1],
                                    hp.get('yaw', 0.0),
                                    hp.get('pitch', 0.0),
                                    width, height)

        if last_pred is not None:
            err = float(np.hypot(last_pred[0] - tx, last_pred[1] - ty))
            errors_px.append(err)
            print(f"  target ({tx:4d},{ty:4d})  pred ({last_pred[0]:4d},"
                  f"{last_pred[1]:4d})  error {err:5.1f} px")

    if not errors_px:
        print("[calibrate_gaze] Verification produced no usable readings.")
        return 1

    median_err = float(np.median(errors_px))
    diag = float(np.hypot(width, height))
    print(f"[calibrate_gaze] Median verification error: "
          f"{median_err:.1f} px ({100*median_err/diag:.1f}% of diagonal)")

    canvas[:] = (10, 10, 10)
    cv2.putText(canvas, f"Median error: {median_err:.1f} px",
                (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Press any key to close.",
                (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (180, 180, 180), 1, cv2.LINE_AA)
    cv2.imshow(WINDOW, canvas)
    cv2.waitKey(0)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="9-point gaze calibration")
    ap.add_argument('--driver', type=int, default=1,
                    help='Driver-facing camera index (default 1)')
    ap.add_argument('--width',  type=int, default=1280,
                    help='Calibration window width  (px)')
    ap.add_argument('--height', type=int, default=720,
                    help='Calibration window height (px)')
    ap.add_argument('--out',    type=str,
                    default='calibration/gaze_user.yaml',
                    help='Output YAML path')
    ap.add_argument('--fullscreen', action='store_true',
                    help='Show calibration window fullscreen on the main display')
    ap.add_argument('--verify', action='store_true',
                    help='Skip fitting; load an existing calibration and '
                         'measure live prediction error against random targets')
    args = ap.parse_args()

    cap = _open_camera(args.driver)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(WINDOW, args.width, args.height)
    # Keep the calibration window in front of any preview windows
    try:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_TOPMOST, 1.0)
    except Exception:
        pass

    print("[calibrate_gaze] Initialising gaze estimator...")
    gaze = GazeEstimator(scene_width=args.width, scene_height=args.height)

    try:
        if args.verify:
            return _run_verification(cap, gaze, args.width, args.height,
                                      args.out)

        if not _wait_for_start(args.width, args.height, cap):
            print("[calibrate_gaze] Aborted before start.")
            return 1

        samples = _run_calibration(cap, gaze, args.width, args.height)
        min_total = GazeCalibrator.MIN_SAMPLES
        if len(samples) < min_total:
            print(f"[calibrate_gaze] Only collected {len(samples)} samples "
                  f"(need >= {min_total}). Calibration not saved.")
            return 1

        cal = GazeCalibrator()
        residual_px, n = cal.fit(samples, args.width, args.height)
        print(f"[calibrate_gaze] Fit complete: "
              f"{n} samples, median residual {residual_px:.1f} px "
              f"on a {args.width}x{args.height} window.")

        cal.save(args.out)
        print(f"[calibrate_gaze] Calibration saved to {args.out}")

        # Final summary screen
        canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        for line, msg in enumerate([
            "Calibration complete.",
            f"  Median residual: {residual_px:.1f} px",
            f"  Samples: {n}",
            f"  Saved to: {args.out}",
            "",
            "Press any key to close.",
        ]):
            cv2.putText(canvas, msg, (60, 100 + line * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235),
                        1, cv2.LINE_AA)
        cv2.imshow(WINDOW, canvas)
        cv2.waitKey(0)
        return 0

    finally:
        cap.release()
        cv2.destroyAllWindows()
        gaze.cleanup()


if __name__ == "__main__":
    sys.exit(main())
