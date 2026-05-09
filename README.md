# Driver Intent Monitoring System — v2.0

## What's New in v2 (vs. v1)

| Issue from Review       | Fix in v2                                               |
|-------------------------|---------------------------------------------------------|
| Naive gaze projection   | Calibrated stereo camera projection via solvePnP + ray-cast |
| No camera calibration   | Full intrinsic + extrinsic calibration pipeline         |
| No frame sync           | `SynchronizedCapture` with timestamp matching           |
| Single-threaded         | 3-thread pipeline (gaze / scene / fusion)               |
| No object tracking      | IoU-based tracker with persistent IDs                   |
| Jittery predictions     | `IntentSmoother` (majority-vote, confidence gate)       |
| Unbounded memory log    | `RollingLogger` (disk-flush every 500 frames)           |
| Weak feature vector     | 34-dim features (gaze velocity, obj velocity, risk, etc.) |
| Simple LSTM             | BiLSTM + multi-head self-attention + residual norm      |
| No evaluation tooling   | Confusion matrix, classification report, training curves |

---

## Quick Start

### Single-click on Windows
Double-click `RUN.bat` in this folder. It activates the bundled venv,
launches the interactive setup menu, and on exit shows where the run's
output bundle was written.

### Interactive launcher (cross-platform)
```bash
pip install -r requirements.txt
python launcher.py
```
You'll see a menu:
```
[1] Two-camera setup  (driver cam + scene cam)
[2] One camera + MetaDrive simulator   (recommended demo)
[3] One camera + CARLA simulator
[4] Replay a recorded video file as the scene feed
[5] Calibrate driver gaze (9-point, do this first)
```

### Recommended first run

1. **Calibrate gaze** (~30 s) — option `[5]`, or:
   ```bash
   python scripts/calibrate_gaze.py --driver 1 --fullscreen
   ```
   Look at each of nine on-screen dots until its ring fills. Saves to
   `calibration/gaze_user.yaml`. Eye tracking accuracy depends on this.

2. **Run the system** — option `[1]` for two cameras, `[2]` for the
   built-in MetaDrive simulator (no second camera needed).

Each run produces a self-contained output bundle:
```
runs/run_YYYYMMDD_HHMMSS_<mode>/
    composite_video.mp4              HUD video at the measured FPS
    gaze_affordance_map.png          cumulative heatmap with legend
    gaze_affordance_overlay.png      heatmap blended on last scene frame
    warnings.csv                     every fired driver warning
    voice_log.json                   every spoken assistant prompt
    session_log/                     rolling per-frame JSON
    summary.json + summary.md        machine + human readable report
```

### Non-interactive shortcuts
```bash
python launcher.py --mode metadrive
python launcher.py --mode metadrive --manual
python launcher.py --mode two-cam   --driver 1 --scene 0
python launcher.py --mode carla     --carla-host localhost
python launcher.py --mode replay    --driver 1 --scene road.mp4
python launcher.py --no-voice                # silent mode
```

### Optional: chessboard camera calibration
The personal gaze calibration above absorbs camera geometry. The
chessboard step is only needed if you want lens-distortion correction
for the object detector:
```bash
python scripts/calibrate_cameras.py --driver 1 --scene 0 --square 0.025
```

### Optional: train an intent model
The shipped `.pth` files target an older architecture. Retrain:
```bash
python scripts/train_intent.py train --data data/*.json \
    --output models/intent_model.pth --epochs 60
```
The launcher's auto-pick validates each checkpoint's keys and shapes
against the current `TemporalIntentModel` and ignores incompatible files.

---

## Project Structure

```
Final_Project/
├── RUN.bat                     # Single-click Windows launcher
├── launcher.py                 # Cross-platform interactive launcher
├── inference.py                # Main 3-thread pipeline
├── requirements.txt
├── README.md
├── DATASETS_AND_TRAINING.md
├── CARLA_SETUP.md
├── yolov8n.pt                  # Bundled object detector weights
├── config/
│   └── affordance_config.json
├── modules/                       # Core library (importable as modules.X)
│   ├── calibration.py             # Chessboard camera calibration utils
│   ├── sync_capture.py            # Two-cam timestamp-synced capture
│   ├── metadrive_capture.py       # MetaDrive sim adapter
│   ├── carla_capture.py           # CARLA sim adapter
│   ├── gaze_estimator.py          # MediaPipe iris + EAR + hysteresis
│   ├── gaze_calibration.py        # 9-point polynomial regression fit
│   ├── scene_detector.py          # YOLOv8 + IoU tracker, FOV-aware focal
│   ├── affordance_engine.py       # Risk-aware affordance mapping
│   ├── intersection_engine.py     # Fixation -> object matching
│   ├── temporal_model.py          # BiLSTM + attention intent model
│   ├── temporal_smoother.py       # Majority-vote + confidence gate
│   ├── warning_engine.py          # Tiered triggers + bbox-area fallback
│   ├── voice_assistant.py         # pyttsx3 / winsound prompt engine
│   ├── gaze_affordance_map.py     # Per-affordance density heatmap
│   ├── run_session.py             # Per-run output bundle + summary
│   ├── logger.py                  # Rolling JSON frame logger
│   └── visualize.py               # HUD + sidebar dashboard
├── scripts/                       # Run-once / utility scripts
│   ├── calibrate_gaze.py          # 9-point user gaze calibration (do this!)
│   ├── calibrate_cameras.py       # Chessboard intrinsics calibration
│   ├── identify_cameras.py        # Auto-label connected cameras
│   ├── prepare_dataset.py         # Convert your logs to training data
│   ├── prepare_hdd.py             # HDD dataset adapter
│   ├── prepare_dreyeve.py         # DR(eye)VE dataset adapter
│   ├── populate_gaze.py           # Add gaze annotations to scene-only data
│   ├── diagnose_features.py       # Feature distribution sanity check
│   ├── train_intent.py            # Train / k-fold / evaluate intent model
│   └── train_gaze_predictor.py    # Optional regression-based gaze model
├── tests/                         # Automated test suite
│   ├── test_safety_chain.py       # Unit tests (warning + voice + session)
│   ├── smoke_pipeline.py          # Headless end-to-end pipeline run
│   ├── test_cameras.py            # Hardware diagnostics
│   ├── test_camera_read.py
│   └── test_phone_camera.py
├── calibration/                   # Saved calibration files
├── data/                          # Training data
├── logs/                          # Session logs (legacy; runs/ is preferred)
├── models/                        # Intent + gaze model checkpoints
└── runs/                          # Per-run output bundles (auto-created)
```

---

## Safety Warning System

The system fires graduated warnings whenever a tracked object meets a
trigger condition, regardless of where the driver is looking. Warnings
that fire on objects the driver is NOT gazing at are flagged
`gaze_miss=True` in the CSV — those are the safety-critical events.

### Scene-based warnings

| Object        | High-severity trigger          | Critical trigger              |
|---------------|--------------------------------|-------------------------------|
| Vehicle       | distance < 25 m  OR  bbox > 7% | distance < 8 m  OR  bbox > 18% |
| Pedestrian    | distance < 15 m  OR  bbox > 7% | distance < 8 m  OR  bbox > 18% |
| Cyclist       | distance < 20 m  OR  bbox > 7% | distance < 8 m  OR  bbox > 18% |
| Traffic light | red state                       | —                             |
| Stop sign     | distance < 20 m                 | —                             |

Bbox-area fallback ensures the system warns even when the distance
estimate is unreliable (sim scenes, uncalibrated cameras).

### Driver-state warnings

These fire independently of scene objects, monitoring the driver:

| Trigger        | Signal                          | High                 | Critical             |
|----------------|---------------------------------|----------------------|----------------------|
| Drowsiness     | EAR (eye aspect ratio) below threshold | eyes closed ≥ 1.0 s | eyes closed ≥ 2.0 s |
| Overspeeding   | telemetry speed_kmh             | > limit + 10 km/h    | > limit + 25 km/h    |
| Rash driving   | longitudinal acceleration       | \|a\| > 4.5 m/s² over ~0.5 s | (advisory at 3 m/s²) |

Drowsiness uses the **PERCLOS** approach — sustained eye closure flagged
as drowsiness, not blinks (which last < 0.4 s). Overspeed and rash-driving
require telemetry, which the MetaDrive and CARLA capture adapters expose
via `get_telemetry()`. Two-cam mode has no telemetry signal so those
triggers are skipped automatically.

Voice prompts use a priority queue: `critical` preempts `high`, with
per-message cooldowns to prevent spam (4 s for scene warnings, 4 s for
driver-state warnings).

---

## Running the Tests

```bash
# Unit tests — safety-critical pieces (~5 s)
python tests/test_safety_chain.py

# End-to-end pipeline smoke test (headless; runs the full 3-thread
# pipeline on synthetic frames and verifies the output bundle)
python tests/smoke_pipeline.py --frames 60
```

---

## Intent Classes

| Class                | Behaviour | Typical Head Pose |
|----------------------|-----------|-------------------|
| `normal_forward`     | Eyes on road ahead | yaw ≈ 0°  |
| `mirror_check`       | Checking rear/side mirror | yaw > 30° or < -30° |
| `pedestrian_monitor` | Sustained gaze on pedestrian | variable yaw |
| `lane_change_prepare`| Oscillating mirror ↔ road | high yaw variance |
| `intersection_scan`  | Wide horizontal sweep | large yaw range |

---

## Recommended Datasets

See **DATASETS_AND_TRAINING.md** for full guide.

| Dataset    | Priority | Gaze Quality | Access |
|------------|----------|--------------|--------|
| Your logs  | 1st      | ⭐⭐⭐⭐⭐  | Collect yourself |
| DGW        | 2nd      | ⭐⭐⭐⭐    | Request academic |
| DADA-2000  | 3rd      | ⭐⭐⭐⭐    | GitHub |
| DrFixD     | 4th      | ⭐⭐⭐⭐    | GitHub |
| BDD-X      | 5th      | ⭐⭐        | GitHub |

---

## Performance Targets

Development hardware: **GTX 1650, 4 GB VRAM** (Turing, fp16 via AMP supported).
Inference targets are realistic on this card; targets in parentheses are what
the same code reaches on a mid-range RTX (3060-class) GPU.

| Component       | Target latency (GTX 1650) |
|-----------------|---------------------------|
| Gaze (Thread 1) | 25–35 ms                  |
| Scene (Thread 2)| 35–55 ms (YOLOv8n fp16)   |
| Intent (Thread 3)| 5–10 ms                  |
| End-to-end FPS  | 12–18 FPS                 |

Training-time constraints on the 4 GB GTX 1650: `--batch 8`, `stream_hidden ≤ 128`,
AMP fp16 required. Drop to `--batch 4` if backward OOMs.
