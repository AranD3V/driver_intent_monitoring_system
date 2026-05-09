# Driver Intent Monitoring System — User Manual

Version 2.0 · End-to-end guide for installation, calibration, real-time
operation, training, and troubleshooting.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [System Requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [Feature Reference](#4-feature-reference)
5. [End-to-End Workflows](#5-end-to-end-workflows)
6. [The Run Output Bundle](#6-the-run-output-bundle)
7. [Training Your Own Intent Model](#7-training-your-own-intent-model)
8. [Configuration & Tuning](#8-configuration--tuning)
9. [Testing & Diagnostics](#9-testing--diagnostics)
10. [Troubleshooting](#10-troubleshooting)
11. [Glossary](#11-glossary)

---

## 1. What This System Does

The Driver Intent Monitoring System is a real-time perception pipeline
that watches the driver and the road simultaneously and answers four
questions every frame:

- **Where is the driver looking?** — 3-D gaze direction from a
  driver-facing camera.
- **What is in front of the car?** — vehicles, pedestrians, cyclists,
  traffic lights and signs from a scene-facing camera (or simulator).
- **What is the driver about to do?** — one of five intent classes
  predicted from a 90-frame history of gaze and scene features.
- **Is the driver missing a hazard?** — graduated voice + on-screen
  warnings whenever a tracked object is dangerous and gaze is elsewhere.

It runs as a **3-thread pipeline** (gaze / scene / fusion+intent) and
produces a self-contained per-run output bundle that you can review,
share, or feed back into training.

---

## 2. System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU       | GTX 1650 4 GB (Turing, fp16) | RTX 3060 or better |
| CPU       | 4 cores | 8 cores |
| RAM       | 8 GB    | 16 GB   |
| Cameras   | 1 webcam (driver-facing) for sim modes; 2 for two-camera mode | 2 × 1080p USB webcams |
| Storage   | 10 GB free | 50 GB+ for training data |

### Software

- Windows 10/11 (primary platform; Linux/macOS supported via `launcher.py`)
- Python 3.11 (a `.venv311` is bundled in this repo)
- CUDA 11.8 / 12.1 wheels for PyTorch if you want GPU inference

---

## 3. Installation

### 3.1 Using the Bundled Windows venv (fastest)

The repo ships with `.venv311/`. Just double-click [RUN.bat](RUN.bat) —
it activates the venv and launches the menu. No further setup needed.

### 3.2 Fresh Install (cross-platform)

```bash
python -m venv .venv311
.venv311\Scripts\activate          # PowerShell: .venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3.3 Verify the Install

```bash
python tests/test_safety_chain.py     # ~5 s, no hardware needed
python tests/smoke_pipeline.py --frames 60   # headless end-to-end test
```

If both pass, the install is healthy.

---

## 4. Feature Reference

Each feature is listed with **what it does**, **why it matters**, and
**how to invoke it**.

### 4.1 Interactive Launcher

**What:** Menu-driven entry point. Picks a setup, scans cameras,
auto-discovers the latest compatible intent model, and hands off to
[inference.py](inference.py).

**Why:** Removes the need to remember CLI flags. Validates camera
indices and model checkpoints before the run starts.

**How:**
```bash
python launcher.py
```
Menu options:
```
[1] Two-camera setup  (driver cam + scene cam)
[2] One camera + MetaDrive simulator   (recommended demo)
[3] One camera + CARLA simulator
[4] Replay a recorded video file as the scene feed
[5] Calibrate driver gaze (9-point, do this first)
```
Non-interactive shortcuts are listed in
[Section 5](#5-end-to-end-workflows).

---

### 4.2 9-Point Personal Gaze Calibration

**What:** A 30-second routine where you stare at nine on-screen dots
while the system fits a per-user polynomial mapping from raw iris
landmarks to screen coordinates. Saved to
`calibration/gaze_user.yaml`.

**Why:** Eye anatomy varies. Without calibration the system still
runs, but gaze estimates can drift 5–10° per user. Calibration brings
typical error under 2°.

**How:**
```bash
python scripts/calibrate_gaze.py --driver 1 --fullscreen
```
Or pick `[5]` in the launcher menu. Look at each dot until its ring
fills (~3 s per dot). Re-run any time you change camera position or
seating posture.

---

### 4.3 Camera Calibration (optional)

**What:** Computes intrinsic matrix + distortion coefficients for each
camera using a printed chessboard pattern.

**Why:** Improves pinhole-projection-based distance estimates for the
object detector. Personal gaze calibration absorbs most camera geometry,
so this is **optional** unless you see clear lens distortion.

**How:**
```bash
python scripts/calibrate_cameras.py --driver 1 --scene 0 --square 0.025
```
Print a 9×6 chessboard, hold it at multiple angles, press `c` to
capture frames, `q` to compute. Results saved to
`calibration/camera_<index>.yaml`.

---

### 4.4 Camera Auto-Detection

**What:** Probes camera indices 0–4, lists working cameras with their
resolution. Replaces the manual `identify_cameras.py` step.

**Why:** USB camera indices are unstable across reboots and unplug
events. Auto-detect picks the right one every time.

**How:** Runs automatically inside `launcher.py`. Standalone:
```bash
python scripts/identify_cameras.py
```

---

### 4.5 Two-Camera Real-Time Mode

**What:** Reads driver and scene from two physical cameras with
timestamp-synced capture.

**Why:** The intended deployment shape — actual in-vehicle setup with a
dashcam and a driver-facing camera.

**How:**
```bash
python launcher.py --mode two-cam --driver 1 --scene 0
```
Or pick `[1]` in the menu. Press `q` in the HUD window to stop the run.

---

### 4.6 MetaDrive Simulator Mode

**What:** Drives the scene feed from the MetaDrive driving simulator
while a real driver-facing camera tracks your gaze.

**Why:** Repeatable demo without a second camera or a car. The
simulator can be driven on autopilot or manually with the keyboard.

**How:**
```bash
python launcher.py --mode metadrive            # autopilot
python launcher.py --mode metadrive --manual   # WASD keyboard control
```

---

### 4.7 CARLA Simulator Mode

**What:** Same as MetaDrive but uses the CARLA simulator over a TCP
client connection. Higher fidelity, heavier dependencies.

**Why:** More realistic urban scenarios; CARLA worlds export richer
metadata. Use this for sim-to-real benchmarking.

**How:** Start a CARLA server, then:
```bash
python launcher.py --mode carla --carla-host localhost
```
See [CARLA_SETUP.md](CARLA_SETUP.md) for server install steps.

---

### 4.8 Video Replay Mode

**What:** Plays a recorded `.mp4` as the scene feed alongside a live
driver camera (or a second video file).

**Why:** Reproduce a previously logged scenario, demo without any car,
or test changes against a fixed input.

**How:**
```bash
python launcher.py --mode replay --driver 1 --scene path/to/road.mp4
```

---

### 4.9 Gaze Estimation (Thread 1)

**What:** MediaPipe iris landmarks + Eye-Aspect-Ratio (EAR) blink
detection + hysteresis smoothing → 3-D gaze vector + head pose (yaw,
pitch, roll).

**Why:** First half of the "intent" signal. Detects mirror checks,
intersection scans, fixations on pedestrians.

**Tunable via:** [modules/gaze_estimator.py](modules/gaze_estimator.py).
Calibration applied automatically if `calibration/gaze_user.yaml`
exists.

---

### 4.10 Scene Detection & Tracking (Thread 2)

**What:** YOLOv8n detector (`yolov8n.pt`) + IoU-based tracker that
assigns persistent IDs to vehicles, pedestrians, cyclists, traffic
lights, stop signs. FOV-aware focal length gives a per-object distance
estimate.

**Why:** Second half of the intent signal — what's in the world the
driver is reacting to. Persistent IDs let the temporal model see object
velocities, not just per-frame positions.

**Tunable via:** confidence threshold and class filter at the top of
[modules/scene_detector.py](modules/scene_detector.py).

---

### 4.11 Affordance Engine

**What:** Maps each tracked object to a per-class **affordance score**
(navigational relevance) using config-driven weights for distance,
bbox area, lateral position, and class priority.

**Why:** Not every detection deserves equal attention. The affordance
score is what feeds the warning engine and the intent model's risk
features.

**Tunable via:** [config/affordance_config.json](config/affordance_config.json).

---

### 4.12 Intersection Engine (Gaze ↔ Object)

**What:** Casts the gaze ray into the scene image and matches it to
tracked-object bounding boxes; produces fixation events with object IDs
and dwell time.

**Why:** Tells you not just *that* the driver is looking somewhere,
but *what they're looking at*. Powers `pedestrian_monitor` intent and
`gaze_miss` warnings.

---

### 4.13 Temporal Intent Predictor (Thread 3)

**What:** A 90-frame BiLSTM with multi-head self-attention and
residual normalization that consumes a 34-dim feature vector
per frame and outputs a 5-class softmax over driver intent.

**Why:** Captures sequential patterns (e.g. mirror→road→mirror means
"lane change prepare", which a per-frame classifier cannot see).

**Classes:**

| Class                 | Behaviour                          | Typical head pose |
|-----------------------|------------------------------------|-------------------|
| `normal_forward`      | Eyes on road ahead                 | yaw ≈ 0°          |
| `mirror_check`        | Checking rear/side mirror          | yaw > 30° / < -30° |
| `pedestrian_monitor`  | Sustained gaze on pedestrian       | variable yaw     |
| `lane_change_prepare` | Oscillating mirror ↔ road          | high yaw variance|
| `intersection_scan`   | Wide horizontal sweep              | large yaw range  |

---

### 4.14 Intent Smoother

**What:** A 7-frame majority-vote filter with a 0.60 confidence gate.
Drops predictions that don't clear the gate; otherwise emits the modal
class.

**Why:** Raw frame-level predictions flicker. Smoothing makes the HUD
readable and prevents voice warnings from spamming.

**Tunable via:** `IntentSmoother(window_size=7, min_ratio=0.6,
confidence_threshold=0.60)` at the top of
[inference.py](inference.py).

---

### 4.15 Warning Engine

**What:** Tiered threat detector. Each tracked object is checked
against `high` and `critical` triggers (distance + bbox-area
fallback). When a hazard fires on an object the driver is **not**
looking at, the warning is flagged `gaze_miss=True`.

**Why:** Safety-critical layer. Gaze-miss events are the ones that
matter for driver-distraction studies.

**Triggers:**

| Object        | High-severity                  | Critical                       |
|---------------|--------------------------------|--------------------------------|
| Vehicle       | distance < 25 m  OR  bbox > 7% | distance < 8 m  OR  bbox > 18% |
| Pedestrian    | distance < 15 m  OR  bbox > 7% | distance < 8 m  OR  bbox > 18% |
| Cyclist       | distance < 20 m  OR  bbox > 7% | distance < 8 m  OR  bbox > 18% |
| Traffic light | red state                      | —                              |
| Stop sign     | distance < 20 m                | —                              |

Every fired warning is written to `runs/<run>/warnings.csv`.

---

### 4.16 Voice Assistant

**What:** `pyttsx3`/`winsound` text-to-speech with a priority queue:
critical preempts high; 4-second per-message cooldown prevents spam.

**Why:** Hands-free driver-facing prompts. Disable for silent
demos or when running headless.

**How to disable:**
```bash
python launcher.py --no-voice
```
Every spoken prompt is logged to `runs/<run>/voice_log.json`.

---

### 4.17 HUD & Sidebar Visualizer

**What:** A composite OpenCV window with the scene feed plus a
sidebar showing gaze yaw/pitch, head pose, current intent + confidence,
top-3 tracked objects with IDs and distances, and FPS.

**Why:** Live readout for demos and debugging. Recorded into
`runs/<run>/composite_video.mp4` at the measured FPS.

**Controls:** `q` quits; window automatically resizes to source feed.

---

### 4.18 Gaze Affordance Heatmap

**What:** A 2-D density map of where the driver looked, separated per
affordance class. Saved at the end of every run as
`gaze_affordance_map.png` (heatmap + legend) and
`gaze_affordance_overlay.png` (heatmap blended on the last scene
frame).

**Why:** Post-hoc visual summary — "how much of the run did the driver
spend looking at vehicles vs. pedestrians vs. signage?"

---

### 4.19 Rolling Logger

**What:** Per-frame JSON log flushed to disk every 500 frames into
`runs/<run>/session_log/`. Bounded memory footprint.

**Why:** Supports arbitrarily long sessions without unbounded RAM
growth. The flushed batches are the input format for
`scripts/prepare_dataset.py`.

---

### 4.20 Run Session Bundle

**What:** Every run gets a self-contained directory under `runs/`
with video, heatmaps, warnings CSV, voice log, per-frame JSON, and a
human + machine summary. See
[Section 6](#6-the-run-output-bundle).

**Why:** Single artefact you can archive, share, or attach to a bug
report.

---

### 4.21 Training Pipeline

**What:** [scripts/train_intent.py](scripts/train_intent.py) trains
the temporal model with k-fold cross-validation, AMP fp16, and exports
classification reports + confusion matrices.

**Why:** The shipped checkpoint covers the canonical 5-class split
(see `MEMORY.md` → canonical_checkpoint). Retrain whenever you change
the feature vector or add new data.

See [Section 7](#7-training-your-own-intent-model).

---

## 5. End-to-End Workflows

### 5.1 First-Time Demo (recommended path)

This is the fastest way to see everything working.

1. **Install** (Section 3).
2. **Verify** with `python tests/smoke_pipeline.py --frames 60`.
3. **Calibrate gaze** — pick `[5]` in `python launcher.py`, look at
   each of 9 dots.
4. **Run the MetaDrive demo** — pick `[2]`, drive on autopilot.
5. **Quit** with `q`. The launcher prints the run directory.
6. **Open the bundle** — watch `composite_video.mp4`, open
   `summary.md`, inspect `warnings.csv`.

### 5.2 Two-Camera Real Run

```bash
python launcher.py --mode two-cam --driver 1 --scene 0
```
Optional preflight: `python scripts/calibrate_cameras.py --driver 1
--scene 0 --square 0.025` for chessboard calibration.

### 5.3 Replaying a Recorded Drive

```bash
python launcher.py --mode replay --driver 1 --scene drives/run42.mp4
```
The driver source can also be a video file path.

### 5.4 Silent / Headless Demo

```bash
python launcher.py --mode metadrive --no-voice
```
For fully headless CI runs use the smoke test
(`tests/smoke_pipeline.py`) instead — it generates synthetic frames
and asserts the output bundle is well-formed.

### 5.5 CARLA Run

```bash
# 1. Start CARLA server (see CARLA_SETUP.md)
# 2. Launch the monitor
python launcher.py --mode carla --carla-host 127.0.0.1 --carla-port 2000
```

---

## 6. The Run Output Bundle

Each run produces:

```
runs/run_YYYYMMDD_HHMMSS_<mode>/
├── composite_video.mp4         HUD video at the measured FPS
├── gaze_affordance_map.png     cumulative heatmap with legend
├── gaze_affordance_overlay.png heatmap blended on last scene frame
├── warnings.csv                every fired driver warning
├── voice_log.json              every spoken assistant prompt
├── session_log/                rolling per-frame JSON batches
├── summary.json                machine-readable run report
└── summary.md                  human-readable run report
```

### `warnings.csv` columns

| Column           | Meaning |
|------------------|---------|
| `t_seconds`      | seconds since run start |
| `severity`       | `high` or `critical` |
| `class`          | object class (vehicle/pedestrian/…) |
| `track_id`       | persistent tracker ID |
| `distance_m`     | estimated distance |
| `bbox_area_pct`  | bbox as % of frame |
| `gaze_miss`      | `True` when driver was not looking at the object |

### `summary.md`

Top-line metrics: run duration, mean FPS, per-class intent share,
total warnings split by severity and `gaze_miss`, top 5 most-fixated
objects.

---

## 7. Training Your Own Intent Model

### 7.1 Collect Data

Run the system in any mode — every session writes per-frame JSON to
`runs/<run>/session_log/`. Annotate with intent labels (manual review
or use the `mirror_check` boost in `scripts/prepare_hdd.py` as a
template).

### 7.2 Convert Logs to Training Data

```bash
python scripts/prepare_dataset.py --source logs --input runs/ \
    --output data/your_set.json
```

### 7.3 Sanity-Check Features

```bash
python scripts/diagnose_features.py --data data/your_set.json
```
Look for non-degenerate distributions across classes — features that
collapse to a single value are the usual cause of stuck training loss.

### 7.4 Train

```bash
python scripts/train_intent.py train --data data/*.json \
    --output models/intent_combined.pth --epochs 60 --batch 8
```
On a 4 GB GTX 1650 keep `--batch 8` and `stream_hidden ≤ 128` with
AMP fp16 (default). Drop to `--batch 4` if backward OOMs.

### 7.5 Evaluate

```bash
python scripts/train_intent.py evaluate --data data/val.json \
    --model models/intent_combined.pth
```
Outputs confusion matrix and classification report under `runs/`.

### 7.6 Auto-Pickup at Inference

`launcher.py` scans `models/*.pth`, validates each candidate's
`state_dict` keys and tensor shapes against the current
`TemporalIntentModel`, and loads the newest compatible file. No
manual `--model` flag needed unless you want to override.

---

## 8. Configuration & Tuning

### 8.1 Affordance Weights

Edit [config/affordance_config.json](config/affordance_config.json):

```json
{
  "vehicle":    {"distance_weight": 1.0, "area_weight": 0.8, ...},
  "pedestrian": {"distance_weight": 1.4, "area_weight": 1.0, ...}
}
```

Higher weight = more influence on the affordance score for that class.

### 8.2 Smoother Strictness

In [inference.py](inference.py):

```python
self.smoother = IntentSmoother(
    window_size=7,           # frames in the majority vote
    min_ratio=0.6,            # ≥60% must agree
    confidence_threshold=0.60 # softmax floor
)
```
Increase `window_size` for steadier output (laggier); lower
`confidence_threshold` to surface low-certainty predictions.

### 8.3 Warning Triggers

Hard-coded in [modules/warning_engine.py](modules/warning_engine.py).
Edit the `_TRIGGERS` table or override per-class thresholds in a
fork-local config.

### 8.4 Voice Cooldown

Top of [modules/voice_assistant.py](modules/voice_assistant.py):
adjust the per-message cooldown (default 4 s) and the priority queue
behaviour.

---

## 9. Testing & Diagnostics

| Test | Command | What it checks |
|------|---------|----------------|
| Unit tests | `python tests/test_safety_chain.py` | Warning engine + voice queue + run-session writer |
| Smoke test | `python tests/smoke_pipeline.py --frames 60` | Full 3-thread pipeline on synthetic frames; asserts bundle layout |
| Camera probe | `python tests/test_cameras.py` | Confirms each connected camera opens and reads |
| Single-camera read | `python tests/test_camera_read.py` | Reads frames from one index, prints stats |
| Phone-as-webcam | `python tests/test_phone_camera.py` | DroidCam / IP webcam validation |
| Feature audit | `python scripts/diagnose_features.py --data <json>` | Distribution + collapse check on a training set |

---

## 10. Troubleshooting

**Launcher says "No driver-facing camera detected."**
Plug in a USB webcam. On Windows, close any app that may be holding
the device (Teams, OBS, browser). Re-run.

**HUD shows gaze stuck at one corner.**
You skipped or moved after gaze calibration. Re-run option `[5]` while
sitting in your normal driving posture.

**No voice prompts.**
You launched with `--no-voice`, or `pyttsx3`'s SAPI driver isn't
installed. Try `python -c "import pyttsx3; pyttsx3.init().say('hi');
pyttsx3.init().runAndWait()"`.

**`[Launcher] No compatible intent model in models/`.**
Either train one (Section 7) or proceed in rule-based mode — the
system still runs, with intent estimated from heuristics over gaze
yaw/pitch.

**OOM during training.**
Lower `--batch` (8 → 4), and keep `stream_hidden ≤ 128`. AMP fp16 is
already on by default; verify with `nvidia-smi` that another process
isn't holding VRAM.

**FPS below 10 in two-camera mode.**
You're likely running on CPU. Confirm with
`python -c "import torch; print(torch.cuda.is_available())"`. If
`False`, install the CUDA-matched PyTorch wheel.

**MetaDrive window won't open.**
First launch downloads assets. Re-run; check stdout for download
errors. Behind a corporate proxy, set `HTTPS_PROXY` before launching.

**CARLA: connection refused.**
The server isn't running, or you're on the wrong port. Start CARLA
with `./CarlaUE4.sh -carla-rpc-port=2000` first; see
[CARLA_SETUP.md](CARLA_SETUP.md).

---

## 11. Glossary

- **Affordance** — class-specific score expressing how navigationally
  relevant a tracked object is right now.
- **Fixation** — a sustained gaze on a single object; emitted by the
  intersection engine with object ID + dwell time.
- **Gaze miss** — a high/critical warning fired on an object the
  driver was not looking at; the safety-critical event class.
- **Intent** — one of five labels (`normal_forward`, `mirror_check`,
  `pedestrian_monitor`, `lane_change_prepare`, `intersection_scan`)
  predicted from a 90-frame window.
- **Run bundle** — the per-run output directory under `runs/`.
- **Smoother** — majority-vote + confidence-gated filter applied to
  raw per-frame intent predictions.
- **Two-camera mode** — real-world deployment shape: a driver-facing
  webcam plus a forward-facing dashcam, captured with
  timestamp-synced reads.

---

For dataset-specific guidance see
[DATASETS_AND_TRAINING.md](DATASETS_AND_TRAINING.md). For CARLA
server setup see [CARLA_SETUP.md](CARLA_SETUP.md). For an overview
of v2 changes vs. v1 see [README.md](README.md).
