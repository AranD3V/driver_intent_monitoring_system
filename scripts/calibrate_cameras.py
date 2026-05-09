"""
Camera Calibration Script
Run this ONCE before deploying the system.

Usage:
  python scripts/calibrate_cameras.py --driver 0 --scene 1 --size 9x6 --square 0.025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import argparse

from modules.calibration import save_calibration, save_extrinsics


def collect_chessboard_images(source, board_size, n_images=30):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {source}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

    obj_pts, img_pts = [], []
    print(f"Show chessboard to camera {source}. Press SPACE to capture, Q to finish.")

    while len(obj_pts) < n_images:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, board_size)

        display = frame.copy()
        if found:
            cv2.drawChessboardCorners(display, board_size, corners, found)
        cv2.putText(display, f"Captured: {len(obj_pts)}/{n_images}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f"Calibration – cam {source}", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_pts.append(objp)
            img_pts.append(corners2)
            print(f"  Captured {len(obj_pts)}/{n_images}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return obj_pts, img_pts, gray.shape[::-1]


def calibrate_intrinsics(source, board_size, n_images=30):
    obj_pts, img_pts, img_size = collect_chessboard_images(source, board_size, n_images)
    if len(obj_pts) < 10:
        raise RuntimeError("Not enough calibration images (need ≥ 10).")

    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, img_size, None, None)
    print(f"  RMS reprojection error: {ret:.4f} px")
    return K, D


def calibrate_stereo(driver_src, scene_src, board_size, square_size, n_images=30):
    """Simultaneous capture from both cameras for stereo calibration."""
    cap_d = cv2.VideoCapture(driver_src)
    cap_s = cv2.VideoCapture(scene_src)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_pts, img_pts_d, img_pts_s = [], [], []
    img_size = None
    print("Show chessboard to BOTH cameras simultaneously. SPACE=capture, Q=done.")

    while len(obj_pts) < n_images:
        ret_d, fd = cap_d.read()
        ret_s, fs = cap_s.read()
        if not ret_d or not ret_s:
            break

        gd = cv2.cvtColor(fd, cv2.COLOR_BGR2GRAY)
        gs = cv2.cvtColor(fs, cv2.COLOR_BGR2GRAY)

        found_d, corners_d = cv2.findChessboardCorners(gd, board_size)
        found_s, corners_s = cv2.findChessboardCorners(gs, board_size)

        disp = np.hstack([fd, fs])
        status = f"OK [{len(obj_pts)}/{n_images}]" if (found_d and found_s) else "Not found"
        cv2.putText(disp, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Stereo Calibration", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and found_d and found_s:
            c_d = cv2.cornerSubPix(gd, corners_d, (11, 11), (-1, -1), criteria)
            c_s = cv2.cornerSubPix(gs, corners_s, (11, 11), (-1, -1), criteria)
            obj_pts.append(objp)
            img_pts_d.append(c_d)
            img_pts_s.append(c_s)
            img_size = gd.shape[::-1]
            print(f"  Captured {len(obj_pts)}/{n_images}")
        elif key == ord('q'):
            break

    cap_d.release()
    cap_s.release()
    cv2.destroyAllWindows()

    if len(obj_pts) < 10:
        raise RuntimeError("Need at least 10 stereo pairs.")

    # Calibrate each camera
    _, Kd, Dd, _, _ = cv2.calibrateCamera(obj_pts, img_pts_d, img_size, None, None)
    _, Ks, Ds, _, _ = cv2.calibrateCamera(obj_pts, img_pts_s, img_size, None, None)

    # Stereo calibration for R, T
    flags = cv2.CALIB_FIX_INTRINSIC
    _, Kd, Dd, Ks, Ds, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img_pts_d, img_pts_s,
        Kd, Dd, Ks, Ds, img_size,
        criteria=criteria, flags=flags
    )

    return Kd, Dd, Ks, Ds, R, T.ravel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver', type=int, default=0)
    parser.add_argument('--scene',  type=int, default=1)
    parser.add_argument('--size',   type=str, default='9x6',
                        help='Chessboard inner corners (colsxrows)')
    parser.add_argument('--square', type=float, default=0.025,
                        help='Square size in metres')
    parser.add_argument('--n',      type=int, default=25,
                        help='Number of calibration images to capture')
    parser.add_argument('--mode',   choices=['stereo', 'separate'], default='stereo')
    parser.add_argument('--outdir', default='calibration')
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    cols, rows = map(int, args.size.split('x'))
    board = (cols, rows)

    if args.mode == 'stereo':
        print("=== Stereo Calibration ===")
        Kd, Dd, Ks, Ds, R, T = calibrate_stereo(
            args.driver, args.scene, board, args.square, args.n
        )
        save_calibration(Kd, Dd, f"{args.outdir}/driver_cam.yaml")
        save_calibration(Ks, Ds, f"{args.outdir}/scene_cam.yaml")
        save_extrinsics(R, T,     f"{args.outdir}/extrinsics.yaml")
    else:
        print("=== Separate Intrinsic Calibration ===")
        print("Calibrating driver camera...")
        Kd, Dd = calibrate_intrinsics(args.driver, board, args.n)
        save_calibration(Kd, Dd, f"{args.outdir}/driver_cam.yaml")

        print("Calibrating scene camera...")
        Ks, Ds = calibrate_intrinsics(args.scene, board, args.n)
        save_calibration(Ks, Ds, f"{args.outdir}/scene_cam.yaml")
        print("Note: extrinsics not computed in separate mode.")

    print("\nCalibration complete!")
    print(f"Files written to: {args.outdir}/")


if __name__ == '__main__':
    main()
