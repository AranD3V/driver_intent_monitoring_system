"""
Driver Intent Monitor — interactive launcher.

End-to-end entry point that wraps inference.py with:
  1. A menu to pick a setup (two-cam | one-cam + MetaDrive | one-cam + CARLA)
  2. Camera auto-detection (replaces the manual identify_cameras step)
  3. Auto-discovery of the most recent trained intent model
  4. A per-run output directory containing the visualization video, the
     final gaze affordance map (heatmap + overlay), warnings CSV, voice
     log and a Markdown summary report.

Run:
    python launcher.py

Non-interactive shortcuts:
    python launcher.py --mode two-cam
    python launcher.py --mode metadrive
    python launcher.py --mode carla --carla-host 127.0.0.1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from inference import DriverIntentSystem


_MODES = {
    '1': ('two-cam',   'Two-camera setup  (driver cam + scene cam)'),
    '2': ('metadrive', 'One camera + MetaDrive simulator (recommended demo)'),
    '3': ('carla',     'One camera + CARLA simulator'),
    '4': ('replay',    'Replay a recorded video file as the scene feed'),
    '5': ('calibrate', 'Calibrate driver gaze (9-point, do this first)'),
}

_GAZE_CAL_PATH = 'calibration/gaze_user.yaml'


# ─────────────────────────────────────────────────────────────────────────── #
#  Camera discovery                                                            #
# ─────────────────────────────────────────────────────────────────────────── #

def _probe_camera(idx: int, timeout_s: float = 1.5) -> Optional[Tuple[int, int]]:
    """Return (w, h) if the camera at `idx` responds, else None."""
    if sys.platform == 'win32':
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
    else:
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    deadline = time.time() + timeout_s
    frame = None
    ret = False
    while time.time() < deadline:
        ret, frame = cap.read()
        if ret and frame is not None:
            break
    cap.release()
    if not ret or frame is None:
        return None
    h, w = frame.shape[:2]
    return (w, h)


def discover_cameras(max_idx: int = 5) -> List[Tuple[int, int, int]]:
    """Return [(idx, width, height), ...] for working camera indices."""
    found: List[Tuple[int, int, int]] = []
    print("Scanning for cameras...")
    for i in range(max_idx):
        res = _probe_camera(i)
        if res:
            print(f"  [OK] Camera {i}: {res[0]}x{res[1]}")
            found.append((i, res[0], res[1]))
        else:
            print(f"  [--] Camera {i}: not available")
    return found


# ─────────────────────────────────────────────────────────────────────────── #
#  Trained model discovery                                                     #
# ─────────────────────────────────────────────────────────────────────────── #

def _ref_architecture_shapes() -> Optional[dict]:
    """Return {param_name: shape} for the current TemporalIntentModel."""
    try:
        from modules.temporal_model import TemporalIntentPredictor
        ref = TemporalIntentPredictor(window_size=90)
        return {k: tuple(v.shape) for k, v in ref.model.state_dict().items()}
    except Exception:
        return None


def _checkpoint_matches(path: Path, ref_shapes: dict) -> bool:
    """Quick architecture compatibility test (key set + tensor shapes)."""
    try:
        import torch
        ckpt = torch.load(str(path), map_location='cpu')
    except Exception:
        return False
    state = ckpt.get('model_state_dict', ckpt) \
            if isinstance(ckpt, dict) else ckpt
    if not isinstance(state, dict):
        return False
    if set(state.keys()) != set(ref_shapes.keys()):
        return False
    for k, expected in ref_shapes.items():
        try:
            if tuple(state[k].shape) != expected:
                return False
        except Exception:
            return False
    return True


def find_latest_model(models_dir: str = 'models') -> Optional[str]:
    """
    Return the newest .pth in models/ that actually loads cleanly into
    the current TemporalIntentModel. Each candidate's state_dict keys
    and tensor shapes are compared against a freshly-built reference
    model — no announcement happens unless a real match is found.

    Returns None when nothing is compatible (the system then runs in
    rule-based mode with a clear log message).
    """
    p = Path(models_dir)
    if not p.exists():
        return None

    # Skip checkpoints we already know belong to legacy variants.
    INCOMPATIBLE = ('kfold', 'seed', 'native', 'dreyeve', 'scene',
                    'sanity', 'fold', 'gaze_predictor')

    candidates = []
    for f in p.glob('*.pth'):
        name = f.name.lower()
        if any(tok in name for tok in INCOMPATIBLE):
            continue
        if name.startswith('intent_') or 'intent' in name:
            candidates.append(f)

    if not candidates:
        return None

    candidates.sort(key=lambda f: -f.stat().st_mtime)   # newest first

    ref_shapes = _ref_architecture_shapes()
    if ref_shapes is None:
        # Architecture probe failed (torch not importable) — best effort
        return str(candidates[0])

    for f in candidates:
        if _checkpoint_matches(f, ref_shapes):
            return str(f)
    return None


# ─────────────────────────────────────────────────────────────────────────── #
#  Interactive prompts                                                         #
# ─────────────────────────────────────────────────────────────────────────── #

def _ask(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{suffix}: ").strip()
    return val if val else (default or "")


def _ask_int(prompt: str, default: int,
             allowed: Optional[List[int]] = None) -> int:
    while True:
        raw = _ask(prompt, str(default))
        try:
            v = int(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if allowed is not None and v not in allowed:
            print(f"  Must be one of: {allowed}")
            continue
        return v


def _ask_yesno(prompt: str, default_yes: bool = True) -> bool:
    d = 'Y/n' if default_yes else 'y/N'
    while True:
        raw = input(f"{prompt} [{d}]: ").strip().lower()
        if not raw:
            return default_yes
        if raw in ('y', 'yes'):
            return True
        if raw in ('n', 'no'):
            return False


def pick_mode_interactive() -> str:
    print()
    print("=" * 60)
    print("  Driver Intent Monitoring System -- Setup Selector")
    print("=" * 60)
    for k, (_, label) in _MODES.items():
        print(f"  [{k}] {label}")
    print()
    while True:
        choice = _ask("Select setup", "2")
        if choice in _MODES:
            return _MODES[choice][0]
        print("  Invalid choice. Pick one of:", ", ".join(_MODES.keys()))


def _default_driver(indices: List[int]) -> int:
    """Prefer camera index 1 for the driver-facing cam; fall back if absent."""
    if 1 in indices:
        return 1
    return indices[0]


def configure_two_cam() -> dict:
    cams = discover_cameras()
    if len(cams) < 2:
        print("\nERROR: Two-camera setup needs at least 2 working cameras.")
        print("Connect a second camera and re-run, or use the simulator mode.")
        sys.exit(1)
    indices = [c[0] for c in cams]
    print()
    driver = _ask_int("Driver-cam index", _default_driver(indices),
                      allowed=indices)
    remaining = [i for i in indices if i != driver]
    scene  = _ask_int("Scene-cam index",  remaining[0], allowed=remaining)
    return {'sim': 'none', 'driver': driver, 'scene': scene}


def configure_metadrive() -> dict:
    cams = discover_cameras()
    if not cams:
        print("\nERROR: No driver-facing camera detected.")
        sys.exit(1)
    indices = [c[0] for c in cams]
    print()
    driver = _ask_int("Driver-cam index", _default_driver(indices),
                      allowed=indices)
    manual = _ask_yesno("Drive the simulator manually (keyboard)?",
                        default_yes=False)
    return {'sim': 'metadrive', 'driver': driver, 'scene': driver,
            'manual': manual}


def configure_carla() -> dict:
    cams = discover_cameras()
    if not cams:
        print("\nERROR: No driver-facing camera detected.")
        sys.exit(1)
    indices = [c[0] for c in cams]
    print()
    driver = _ask_int("Driver-cam index", _default_driver(indices),
                      allowed=indices)
    host   = _ask("CARLA server host", "localhost")
    port_s = _ask("CARLA server port", "2000")
    try:
        port = int(port_s)
    except ValueError:
        port = 2000
    return {'sim': 'carla', 'driver': driver, 'scene': driver,
            'carla_host': host, 'carla_port': port}


def run_calibration_flow() -> int:
    """Launch the 9-point gaze calibration script directly."""
    cams = discover_cameras()
    if not cams:
        print("\nERROR: No driver-facing camera detected.")
        return 1
    indices = [c[0] for c in cams]
    print()
    driver = _ask_int("Driver-cam index", _default_driver(indices),
                      allowed=indices)
    fullscreen = _ask_yesno("Run fullscreen?", default_yes=True)

    import subprocess
    cmd = [sys.executable,
           str(Path(__file__).parent / 'scripts' / 'calibrate_gaze.py'),
           '--driver', str(driver)]
    if fullscreen:
        cmd.append('--fullscreen')
    print(f"\n[Launcher] Running: {' '.join(cmd)}\n")
    return subprocess.call(cmd)


def maybe_offer_calibration() -> None:
    """Offer to run camera/gaze calibration before each pipeline run."""
    has_cal = Path(_GAZE_CAL_PATH).exists()
    print()
    print("-" * 60)
    if has_cal:
        print(f"  Existing gaze calibration found: {_GAZE_CAL_PATH}")
        print("  Re-calibrate if lighting, seating, or camera position")
        print("  has changed since the last run.")
    else:
        print("  No personal gaze calibration found.")
        print("  Eye tracking will work, but a 30-second calibration")
        print("  dramatically improves accuracy.")
    print("-" * 60)
    if _ask_yesno("Run camera calibration now?", default_yes=not has_cal):
        run_calibration_flow()


def configure_replay() -> dict:
    cams = discover_cameras()
    indices = [c[0] for c in cams]
    print()
    driver_default = str(_default_driver(indices)) if indices else "1"
    driver_raw = _ask("Driver source (camera index OR video path)",
                      driver_default)
    scene_path = _ask("Scene video path", "")
    if not scene_path or not Path(scene_path).exists():
        print(f"ERROR: Scene video '{scene_path}' not found.")
        sys.exit(1)
    driver = int(driver_raw) if driver_raw.isdigit() else driver_raw
    return {'sim': 'none', 'driver': driver, 'scene': scene_path}


_CONFIGURERS = {
    'two-cam':   configure_two_cam,
    'metadrive': configure_metadrive,
    'carla':     configure_carla,
    'replay':    configure_replay,
}


# ─────────────────────────────────────────────────────────────────────────── #
#  Main                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Driver Intent Monitor -- interactive launcher')
    parser.add_argument('--mode', default=None,
                        choices=['two-cam', 'metadrive', 'carla',
                                 'replay', 'calibrate'],
                        help='Skip the menu and use this mode')
    parser.add_argument('--driver', default=None,
                        help='Driver source (camera index or video path); '
                             'overrides the interactive prompt')
    parser.add_argument('--scene', default=None,
                        help='Scene source for two-cam / replay')
    parser.add_argument('--carla-host', default=None)
    parser.add_argument('--carla-port', default=2000, type=int)
    parser.add_argument('--manual', action='store_true',
                        help='MetaDrive: manual keyboard control')
    parser.add_argument('--model', default=None,
                        help='Intent model checkpoint (auto-detected if omitted)')
    parser.add_argument('--no-voice', action='store_true',
                        help='Disable spoken driver-assist prompts')
    parser.add_argument('--run-root', default='runs',
                        help='Parent dir for per-run output bundles')
    args = parser.parse_args()

    # 1. Pick mode
    mode = args.mode or pick_mode_interactive()

    # Calibration is its own end-state — run and exit
    if mode == 'calibrate':
        return sys.exit(run_calibration_flow())

    # Offer camera calibration before each run
    maybe_offer_calibration()

    # 2. Build configuration (CLI args override interactive prompts)
    if args.driver is not None:
        driver_raw = args.driver
        driver = int(driver_raw) if driver_raw.isdigit() else driver_raw
        scene_raw  = args.scene if args.scene is not None else driver_raw
        scene  = int(scene_raw) if isinstance(scene_raw, str) and scene_raw.isdigit() \
                                else scene_raw
        cfg = {'sim': 'metadrive' if mode == 'metadrive'
                       else 'carla' if mode == 'carla' else 'none',
               'driver': driver, 'scene': scene}
        if mode == 'metadrive':
            cfg['manual'] = args.manual
        if mode == 'carla':
            cfg['carla_host'] = args.carla_host or 'localhost'
            cfg['carla_port'] = args.carla_port
    else:
        cfg = _CONFIGURERS[mode]()

    # 3. Trained model auto-discovery
    model_path = args.model
    if model_path:
        print(f"\n[Launcher] Using intent model (from --model): {model_path}")
    else:
        model_path = find_latest_model()
        if model_path:
            print(f"\n[Launcher] Using intent model: {model_path}")
        else:
            print("\n[Launcher] No compatible intent model in models/. "
                  "Running in rule-based mode.")
            print("[Launcher] Train one with: "
                  "python scripts/train_intent.py train --data data/*.json")

    # 4. Banner
    print()
    print("-" * 60)
    print(f"  Mode         : {mode}")
    print(f"  Driver source: {cfg['driver']}")
    print(f"  Scene source : {cfg.get('scene', cfg['driver'])}")
    if mode == 'carla':
        print(f"  CARLA server : "
              f"{cfg.get('carla_host')}:{cfg.get('carla_port')}")
    print(f"  Voice assist : {'disabled' if args.no_voice else 'enabled'}")
    print(f"  Run root     : {Path(args.run_root).resolve()}")
    print("-" * 60)
    print()

    # 5. Hand off to the pipeline
    system = DriverIntentSystem(
        model_path=model_path,
        voice_enabled=not args.no_voice,
    )
    system.start(
        driver_source = cfg['driver'],
        scene_source  = cfg.get('scene', cfg['driver']),
        output_video  = None,
        log_enabled   = True,
        sim           = cfg.get('sim', 'none'),
        carla_host    = cfg.get('carla_host'),
        carla_port    = cfg.get('carla_port', 2000),
        manual        = cfg.get('manual', False),
        run_root      = args.run_root,
    )


if __name__ == '__main__':
    main()
