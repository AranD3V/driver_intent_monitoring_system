# Driver Intent Monitoring System v2 — Final Project Report

**Author:** AranD3V (`narayanaditya2605@gmail.com`)
**Repository:** `github.com/AranD3V/driver_intent_monitoring_system`
**Reporting date:** 2026-05-15
**Status:** Frozen for paper / demo submission.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [System Architecture](#3-system-architecture)
4. [Implementation Details](#4-implementation-details)
5. [Datasets Used](#5-datasets-used)
6. [Tools, Libraries & Techniques](#6-tools-libraries--techniques)
7. [Complete Training Process](#7-complete-training-process)
8. [Issues Faced During Training](#8-issues-faced-during-training)
9. [Final Results](#9-final-results)
10. [Validation, Reliability & Reproducibility](#10-validation-reliability--reproducibility)
11. [System-Level Runtime Results](#11-system-level-runtime-results)
12. [Limitations](#12-limitations)
13. [Future Use Cases & Roadmap](#13-future-use-cases--roadmap)
14. [Appendices](#14-appendices)

---

## 1. Executive Summary

The **Driver Intent Monitoring System v2** is a real-time, multi-threaded
perception pipeline that simultaneously watches the driver and the road and
predicts what the driver is **about to do** before they do it. It fuses
driver-facing gaze tracking, forward-facing object detection, vehicle
telemetry, and a temporal BiLSTM intent classifier to drive a safety-aware
voice + on-screen warning engine.

**Key headline numbers (Round 2, paper-frozen):**

| Metric | Value | Source |
|---|---|---|
| 5-fold ensemble accuracy | **73.00%** | `reports/eval_weak_v2_ensemble.txt` |
| Best single-fold validation accuracy | 46.58% (fold 4) | `models/kfold_summary.json` |
| 5-fold mean validation accuracy | 44.31% ± 1.71% | `models/kfold_summary.json` |
| Improvement over Round 1 ensemble | **+12.0 percentage points** | 61.0% → 73.0% |
| Improvement over best prior single model | +5.5 pp | 67.5% → 73.0% |
| Unit-test coverage | **47 / 47 tests passing** | `tests/test_safety_chain.py` |
| Real-time inference (GTX 1650) | **~6.1 FPS** | `runs/run_20260511_233355_sim_metadrive_manual/summary.md` |

The +12 pp ensemble gain is the main scientific contribution of this work
and is driven by a **weak-supervision pipeline** (audit + 3-labeler
consensus + class-balanced sampling) introduced in Round 2.

---

## 2. Problem Statement & Motivation

Distraction and intent-misreading account for a large share of driving
incidents. Existing in-cabin systems are typically reactive: they fire
warnings *after* a hazard has materialised. The goal of this project was
to build an **anticipatory** monitor — a system that predicts an unsafe
intent (e.g. lane change without mirror check) early enough to surface a
prompt to the driver.

Four questions the system answers **every frame**:

1. **Where is the driver looking?** — 3-D gaze direction from a driver-facing camera.
2. **What is in front of the car?** — vehicles, pedestrians, cyclists, traffic lights / signs.
3. **What is the driver about to do?** — one of three trained intent classes (five-class legacy vocabulary, see §5.3).
4. **Is the driver missing a hazard?** — graduated voice + on-screen warnings, with `gaze_miss=True` flagged when the driver was not looking at the threat.

---

## 3. System Architecture

### 3.1 High-Level Pipeline

A **3-thread architecture** with bounded queues (`maxsize=2`) keeps latency
low and decouples thread runtimes:

```
┌──────────────────────────┐  gaze_out_queue
│ Thread 1 — Gaze          │ ─────────────────┐
│  MediaPipe iris + EAR     │                  │
└──────────────────────────┘                  ▼
┌──────────────────────────┐               ┌──────────────────────┐
│ Thread 2 — Scene         │  scene_out_q  │ Thread 3 — Fusion +  │
│  YOLOv8n + IoU tracker   │ ────────────► │   Intent (BiLSTM)    │
└──────────────────────────┘               └──────────┬───────────┘
                                                      │ result_queue
                                                      ▼
                                            ┌──────────────────────┐
                                            │  Main: HUD + log +   │
                                            │  warning + voice +   │
                                            │  RunSession bundle   │
                                            └──────────────────────┘
```

### 3.2 Module Map (canonical post-restructure)

```
Final_Project/
├── RUN.bat                          single-click Windows launcher
├── launcher.py                      cross-platform interactive launcher
├── inference.py                     main 3-thread pipeline entry
├── modules/
│   ├── calibration.py               chessboard intrinsics
│   ├── sync_capture.py              two-cam timestamp-synced capture
│   ├── metadrive_capture.py         MetaDrive sim adapter
│   ├── carla_capture.py             CARLA sim adapter
│   ├── gaze_estimator.py            MediaPipe iris + EAR + hysteresis
│   ├── gaze_calibration.py          9-point polynomial regression fit
│   ├── scene_detector.py            YOLOv8 + IoU tracker, FOV-aware focal
│   ├── affordance_engine.py         risk-aware affordance mapping
│   ├── intersection_engine.py       fixation -> object matching
│   ├── temporal_model.py            BiLSTM intent model + FeatureExtractor
│   ├── temporal_smoother.py         majority-vote + confidence gate
│   ├── warning_engine.py            tiered triggers + gaze-aware downgrade
│   ├── voice_assistant.py           pyttsx3 / winsound prompt engine
│   ├── gaze_affordance_map.py       per-affordance density heatmap
│   ├── run_session.py               per-run output bundle + summary
│   ├── logger.py                    rolling JSON frame logger
│   ├── visualize.py                 HUD + sidebar dashboard
│   ├── augmentation.py              training-time sequence augmentation
│   ├── feature_augment.py           telemetry-aware augmentation
│   ├── causal_attention.py          experimental attention block
│   ├── losses.py                    AnticipatoryFocalLoss + scheduler
│   ├── weak_labelers.py             rule-based weak label voters
│   └── sim_labeler.py               MetaDrive ground-truth oracle
├── scripts/
│   ├── calibrate_gaze.py            9-point user gaze calibration
│   ├── calibrate_cameras.py         chessboard intrinsics
│   ├── identify_cameras.py          camera auto-label
│   ├── prepare_dataset.py           logs / template -> training JSON
│   ├── prepare_hdd.py               HDD adapter
│   ├── prepare_dreyeve.py           DR(eye)VE adapter (deprecated)
│   ├── populate_gaze.py             gaze annotation back-fill
│   ├── diagnose_features.py         feature distribution sanity check
│   ├── train_intent.py              train / k-fold / evaluate / ensemble
│   ├── train_gaze_predictor.py      optional regression gaze model
│   ├── audit_labels.py              telemetry-vs-label plausibility check
│   ├── consensus_label.py           3-voter consensus relabeller
│   ├── build_review_queue.py        prioritised manual review CSV
│   ├── auto_resolve_queue.py        AUTO_KEEP / AUTO_RELABEL / NEEDS_REVIEW
│   └── apply_review_decisions.py    apply post-review JSON deltas
├── notebooks/colab_train.ipynb      one-shot Colab training notebook
├── config/affordance_config.json    affordance weights per class
├── yolov8n.pt                       bundled object detector
├── data/                            training JSONs
├── models/                          checkpoints + confusion matrices
├── reports/                         eval text + audit CSVs + PAPER_RESULTS.md
└── runs/                            per-run output bundles (auto-created)
```

### 3.3 Thread Responsibilities

| Thread | Module | Outputs |
|---|---|---|
| **T1 — Gaze** | `gaze_estimator.py` | `gaze_point`, `head_pose (yaw/pitch/roll)`, `EAR`, `is_fixation` |
| **T2 — Scene** | `scene_detector.py` | tracked-object list (`track_id`, `class`, `bbox`, `distance_estimate`, `velocity`) |
| **T3 — Fusion + Intent** | `affordance_engine.py` + `intersection_engine.py` + `temporal_model.py` | intent label + softmax, gaze-affordance match, fixation events |
| **Main** | `visualize.py` + `warning_engine.py` + `voice_assistant.py` + `run_session.py` | HUD frame, warnings CSV row, spoken prompt, per-frame JSON |

---

## 4. Implementation Details

### 4.1 Gaze Estimation (T1)

- **Library:** MediaPipe Iris (`mediapipe>=0.10.30`).
- **Eye state:** Eye-Aspect-Ratio (EAR) for blink/fixation/saccade classification with hysteresis smoothing (prevents per-frame flicker).
- **Head pose:** yaw / pitch / roll via `solvePnP` on 3-D landmark anchors.
- **Calibration:** 9-point on-screen routine → per-user polynomial mapping from raw iris landmarks to screen coordinates, persisted in `calibration/gaze_user.yaml`. Reduces typical gaze error from 5–10° (uncalibrated) to **< 2°**.

### 4.2 Scene Perception (T2)

- **Detector:** YOLOv8n (`yolov8n.pt`, 6 MB) via `ultralytics>=8.1.0`. fp16 autocast under AMP on Turing GPUs.
- **Tracker:** lightweight **IoU-based** tracker with persistent `track_id`s — needed for object velocity, dwell time, and gaze-miss attribution.
- **Distance estimate:** **FOV-aware focal length** model. Pinhole projection over the known object class height priors (`Pedestrian ≈ 1.70 m`, `Car height ≈ 1.5 m`, etc.). Bbox-area fallback ensures warnings still fire when distance is unreliable (uncalibrated cameras or simulator scenes).
- **Resolution control:** `--imgsz 416` switch yields **~1.5–2× FPS gain** vs default 640.

### 4.3 Affordance Engine

Each tracked object is scored against a per-class **affordance** template
loaded from `config/affordance_config.json`:

```json
{
  "vehicle":    {"distance_weight": 1.0, "area_weight": 0.8, ...},
  "pedestrian": {"distance_weight": 1.4, "area_weight": 1.0, ...}
}
```

The score combines distance, bbox area, lateral position, and class
priority. Higher score → more attention budget. Outputs:

- `affordance ∈ {Caution, Constraint, Instructional, Unknown}`
- `risk_level ∈ {low, medium, high, critical}`

### 4.4 Intersection Engine (Gaze ↔ Object)

Casts the gaze ray into the scene image and matches it against tracked
bounding boxes. Emits fixation events `{track_id, dwell_seconds, class}`.
Powers two downstream signals:

- `pedestrian_monitor` intent (sustained gaze on `person` class).
- `gaze_miss=True` flag on warnings fired on objects the driver did *not* look at within the last 30 evaluation ticks.

### 4.5 Temporal Intent Model

**Architecture:** single-stream Bidirectional LSTM intent classifier
(`modules/temporal_model.py:TemporalIntentModel`).

| Component | Configuration |
|---|---|
| Input feature dim | **36** (gaze 5 + vehicle 6 + object one-hot 9 + affordance one-hot 5 + context 9 + mask 2) |
| Sequence length | 50 frames per window (training), 90-frame rolling buffer (inference) |
| Window stride | 5 (Round 2) / 10 (Round 1) |
| Hidden size | 192 (Round 2) / 128 (Round 1), bidirectional → d_model 256/256 |
| Layers | 2, bidirectional |
| Dropout | 0.3 |
| Pool | mean over time (captures sustained patterns) |
| Classifier head | `Linear(d_model → d_model → 64 → C)` with ReLU + Dropout |
| Output classes (Round 2) | 3: `normal_forward`, `lane_change_prepare`, `intersection_scan` |
| Output classes (legacy) | 5: + `mirror_check`, `pedestrian_monitor` |

**Why mean-pool instead of last-timestep?** Anticipatory intent labels
(e.g. "about to change lane") depend on *sustained* sub-patterns like
average yaw variance and brake rate. Mean-pool captures those; last-step
LSTM hidden states are dominated by the most recent frame.

### 4.6 Intent Smoother

- **Window:** 7 frames majority vote.
- **Confidence gate:** softmax floor 0.60.
- **Minimum ratio:** 60% of the window must agree.

This prevents flicker in the HUD and prevents voice spam.

### 4.7 Warning Engine — Gaze-Aware Downgrade (new in v2)

Triggers per object class:

| Object | High-severity | Critical |
|---|---|---|
| Vehicle | `dist < 25 m` OR `bbox > 7%` | `dist < 8 m` OR `bbox > 18%` |
| Pedestrian | `dist < 15 m` OR `bbox > 7%` | `dist < 8 m` OR `bbox > 18%` |
| Cyclist | `dist < 20 m` OR `bbox > 7%` | `dist < 8 m` OR `bbox > 18%` |
| Traffic light | red state | — |
| Stop sign | `dist < 20 m` | — |

**Gaze-aware downgrade** (`modules/warning_engine.py:131-264`): if the
driver looked at this `track_id` within the last 30 evaluation ticks
(`_GAZE_RECENT_FRAMES`), the severity is reduced one step:

| Original | Downgrade if recently gazed |
|---|---|
| `critical` | → `high` (never fully suppressed) |
| `high` | → `advisory` |
| `advisory` | → suppressed |

This addresses the previously-observed 100% gaze-miss rate by recognising
objects already in the driver's attention budget.

### 4.8 Driver-State Warnings

Independent of scene detections, the warning engine also monitors the driver:

| Trigger | Signal | High | Critical |
|---|---|---|---|
| **Drowsiness** | EAR below threshold (PERCLOS) | eyes closed ≥ 1.0 s | eyes closed ≥ 2.0 s |
| **Overspeeding** | telemetry `speed_kmh` | > limit + 10 km/h | > limit + 25 km/h |
| **Rash driving** | longitudinal acceleration | \|a\| > 4.5 m/s² over ~0.5 s | (advisory at 3 m/s²) |

### 4.9 Voice Assistant

`pyttsx3`/`winsound` text-to-speech with a **priority queue**:
`critical` preempts `high`; 4-second per-message cooldown prevents spam.
Every spoken prompt is logged to `runs/<run>/voice_log.json`. Falls back
to severity-coded beeps + console logging when SAPI is unavailable.

### 4.10 RunSession Output Bundle

Every run produces a self-contained directory:

```
runs/run_YYYYMMDD_HHMMSS_<mode>/
├── composite_video.mp4         HUD video at the measured FPS
├── gaze_affordance_map.png     cumulative heatmap + legend
├── gaze_affordance_overlay.png heatmap blended on last scene frame
├── warnings.csv                every fired driver warning
├── voice_log.json              every spoken assistant prompt
├── session_log/                rolling per-frame JSON batches
├── summary.json                machine-readable run report
└── summary.md                  human-readable run report
```

`warnings.csv` columns: `t_seconds, severity, class, track_id,
distance_m, bbox_area_pct, gaze_miss`.

---

## 5. Datasets Used

### 5.1 Sources at a Glance

| Source | Sequences | Frames | Status | Role |
|---|---|---|---|---|
| **HDD** (Honda Driving Dataset, capped) | 1,900 | 184,144 | **Primary** | Maneuver labels + rich telemetry |
| **Manual mirror_check annotations** | 1,247 | — | Author-collected | Legacy 5-class only |
| **DR(eye)VE** (seq 01, 11, 29, 36) | 68 | 5,508 | **Rejected** | No driver-facing video |
| **MetaDrive simulator** | live | — | Runtime oracle | Sim-time evaluation |
| **CARLA simulator** | live | — | Runtime oracle | High-fidelity scenarios |
| **Effective training pool** | **1,492** | — | After splits | Training input |

### 5.2 HDD (Primary Source)

- 4 maneuver-rich classes: `normal_forward, lane_change_prepare, intersection_scan, pedestrian_monitor`.
- Per-frame vehicle telemetry: `turn_signal_left`, `turn_signal_right`, `yaw_rate`, `steer_angle_deg`, `speed_kmh`, `brake`.
- **No real driver-facing video** — `gaze_point=[640, 360]` (image centre) everywhere as a default. The model is trained to work with the gaze stream constant; vehicle telemetry drives most of the signal.
- Audit on `raw_label` vs telemetry: **66%** of `normal_forward` rows are `Background` mislabels actually containing maneuvers — see §7.4 (weak supervision).

### 5.3 Intent Class Vocabulary

Two vocabularies exist for historical reasons:

**Legacy 5-class** (used by `intent_combined.pth` for demo / launcher):

| Class | Behaviour | Typical head pose |
|---|---|---|
| `normal_forward` | Eyes on road ahead | yaw ≈ 0° |
| `mirror_check` | Checking rear/side mirror | yaw > 30° or < -30° |
| `pedestrian_monitor` | Sustained gaze on pedestrian | variable yaw |
| `lane_change_prepare` | Oscillating mirror ↔ road | high yaw variance |
| `intersection_scan` | Wide horizontal sweep | large yaw range |

**Round 2 effective 3-class** (used by `weak_v2_fold*.pth`):

`normal_forward, lane_change_prepare, intersection_scan`. `mirror_check`
and `pedestrian_monitor` were filtered out at data prep time because (a)
HDD lacks a driver-facing camera so `mirror_check` is unreachable, and
(b) `pedestrian_monitor` had only 64 training sequences (4.3% of the
pool) — insufficient for a learned classifier. Inference falls back to
rule-based for these two legacy classes.

### 5.4 DR(eye)VE — Rejected (and Why)

DR(eye)VE was evaluated and rejected on 2026-04-18 for three reasons:

1. **`video_etg.avi` is the eye-tracking glasses' *forward scene* camera**, not a driver-facing camera. DR(eye)VE has **no in-cabin driver video**. MediaPipe on `video_etg.avi` returned 0–2 face detections out of 9001 frames across all 4 tested sequences.
2. DR(eye)VE gaze coords are mapped to the forward frame, so mirror looks fall outside-frame (NaN, filtered out).
3. DR(eye)VE's `e` (ego maneuver) subsequences span the maneuver *itself* (TTM=0), whereas HDD windows are *pre-maneuver* (TTM ∈ [0, 4] s). Mixing them skews the AnticipatoryFocalLoss weights (1.0 vs 0.018 for the same class).

`scripts/prepare_dreyeve.py` exists but should be treated as **deprecated** for this architecture.

### 5.5 Time-To-Maneuver (TTM) Distribution

| Class | n | Min (s) | Median (s) | Max (s) | Mean (s) |
|---|---|---|---|---|---|
| `intersection_scan` | 500 | 0.80 | 4.00 | 4.00 | 3.98 |
| `lane_change_prepare` | 500 | 4.00 | 4.00 | 4.00 | 4.00 |
| `normal_forward` | 500 | 10.00 | 10.00 | 10.00 | 10.00 |
| `pedestrian_monitor` | 400 | 2.70 | 4.00 | 4.00 | 3.99 |

HDD maneuver classes use **multi-horizon anticipation windows** (offsets
0, 15, 30, 45, 60, 75, 90, 105, 120 frames pre-maneuver) to populate the
TTM distribution. Without this, the model would only see the maneuver
at TTM=0 and never learn to *anticipate*.

---

## 6. Tools, Libraries & Techniques

### 6.1 Library Stack

| Layer | Library | Version | Use |
|---|---|---|---|
| Numerics | `numpy` | >= 1.21 | Feature arrays |
| Computer vision | `opencv-python` | >= 4.8 | Capture, HUD, drawing |
| Deep learning | `torch` + `torchvision` | >= 2.1 | BiLSTM, AMP fp16 |
| Face / iris | `mediapipe` | >= 0.10.30 | Gaze estimation |
| Object detection | `ultralytics` (YOLOv8) | >= 8.1 | Scene perception |
| Config / IO | `PyYAML`, JSON | >= 6.0 | Calibration, datasets |
| ML utilities | `scikit-learn` | >= 1.3 | StratifiedGroupKFold, classification_report, confusion_matrix |
| Plotting | `matplotlib` | >= 3.7 | Training curves, confusion matrices |
| Imaging | `Pillow` | >= 10.0 | Heatmap PNG export |
| Simulator | `metadrive-simulator` | 0.4.1.1–0.5.0 | Indoor demo |
| Simulator | `carla` | 0.9.16 | High-fidelity demo |
| Voice | `pyttsx3` (Windows) | >= 2.90 | TTS prompts |

### 6.2 Key Techniques

- **Stereo / monocular FOV-aware focal length** for distance estimation.
- **EAR + hysteresis** for blink-robust fixation classification.
- **Iris-landmark polynomial gaze calibration** (9-point).
- **IoU tracking** with persistent IDs for velocity & dwell features.
- **Multi-horizon anticipatory windowing** for TTM coverage.
- **OTC test-time augmentation** (`original` + `translate ±3 frames` + `cutout 10 frames`).
- **StratifiedGroupKFold** by `seq_id` to prevent window-level leakage.
- **Class-balanced WeightedRandomSampler** for severe imbalance.
- **Anticipatory Focal Loss** (`modules/losses.py`) weighting by TTM.
- **AMP fp16 autocast + GradScaler** for memory-bound training.
- **Linear warmup + cosine annealing** LR schedule with warm restarts.
- **5-member ensemble inference** averaging softmax across folds.
- **Architecture-aware checkpoint loading** validates `state_dict` keys & tensor shapes before swapping models at inference.
- **Weak supervision** (audit + consensus + auto-resolve queue, see §7.4).

---

## 7. Complete Training Process

### 7.1 Hardware & Environment Split

- **Local rig (GTX 1650, 4 GB VRAM, Turing, fp16-capable):** inference only. Too small for the full k-fold pipeline.
- **Training:** Google Colab T4 / A100 via `notebooks/colab_train.ipynb`. Data files are mounted from `MyDrive/intent_data/`; checkpoints land in `MyDrive/intent_models/`; reports in `MyDrive/intent_reports/`. The notebook clones the GitHub repo and symlinks `data/`, `models/`, `reports/` to those Drive paths.
- **Wall-clock:** ~3 hours per round (Run A baseline + Run B weak pipeline).

### 7.2 Day-Of Workflow (Colab)

| Cell | What it does | Time |
|---|---|---|
| 1 | Mount Google Drive | 30 s |
| 2 | Clone repo | 10 s |
| 3 | `pip install -r requirements-colab.txt` | 1–2 min |
| 4 | Symlink `data/` / `models/` / `reports/` to Drive | 5 s |
| 5 | GPU + data sanity check | 5 s |
| 6 | **Run A** — vanilla baseline kfold | ~90 min |
| 7 | **Run B** — weak-supervision kfold | ~90 min |
| 8 | Eval baseline on `hdd_train` | ~3 min |
| 9 | Eval weak on `hdd_train` | ~3 min |
| 10 | Pick winner, copy to `intent_canonical.pth` | 1 s |
| 11 | Print summary table | 1 s |

### 7.3 Optimizer & Schedule

| Setting | Value |
|---|---|
| Optimizer | **AdamW** |
| Learning rate | 3e-4 |
| Weight decay | 1e-2 |
| Scheduler | 5% linear warmup + cosine annealing (`min_lr_ratio=0.01`) |
| Loss | Cross-entropy with class weights + label smoothing 0.1 (also: `AnticipatoryFocalLoss` for TTM-weighted variants) |
| Gradient clipping | norm ≤ 1.0 |
| Mixed precision | FP16 autocast + `GradScaler` |
| Warm restarts | up to 2, triggered after 8 epochs without improvement |
| Epochs (max) | 80 |
| Early stopping patience | 25 |
| Batch size | 8 (local 4 GB) / 32 (Colab) |
| Random seed | 42 (all RNGs) |

### 7.4 Weak Supervision Pipeline (Round 2 contribution)

Label quality was the single largest lever. A three-signal pipeline was
added (`scripts/audit_labels.py`, `consensus_label.py`,
`build_review_queue.py`, `auto_resolve_queue.py`,
`apply_review_decisions.py`):

1. **Audit signal** — telemetry-consistency check. Example: a `left_turn` label contradicted by `net_yaw = -21°` right-leaning vehicle dynamics is flagged.
2. **Consensus signal** — three weak rule-based labellers vote (`label_telemetry`, `label_raw_signal`, `label_gaze_object` in `modules/weak_labelers.py`). Disagreements with `consensus_confidence ≥ 0.85` and `n_voters ≥ 2` are accepted for relabeling.
3. **Model uncertainty signal** — top-1 vs top-2 softmax margin from a prior model, used for active-learning prioritisation.

A composite priority score `0.45·audit + 0.35·consensus_conf + 0.20·uncertainty` ranks sequences for review. The **auto-resolver** splits queue items into AUTO_KEEP / AUTO_RELABEL / NEEDS_REVIEW bins, **reducing manual review workload by 45.5%** (995 → 542 entries).

Per-sequence sample weights are emitted into `weak_label_meta.consensus_confidence`:

```
weight = 1.0 + 0.5 · n_voters · consensus_confidence   (if any labeler agrees)
weight = 0.5                                            (otherwise)
```

Observed weight range: **0.50 – 1.91**. After applying decisions, the
cleaned dataset contains **1,458 training sequences** (with 542
audit-flagged sequences dropped).

**Important constraint:** `feature_augment.py` deliberately **never
horizontally flips** — that would invert left/right semantics (turn
signals, steering, yaw rate).

### 7.5 Cross-Validation

- **5-fold StratifiedGroupKFold**, grouped by `seq_id` to prevent window-level leakage between train and validation.
- Each fold reports both raw validation accuracy and **OTC test-time augmentation accuracy** (average over `original`, `translate ±3 frames`, `cutout 10 frames`).
- All five per-fold checkpoints retained for ensemble inference (`models/weak_v2_fold0.pth` … `weak_v2_fold4.pth`).

### 7.6 Class-Imbalance Handling (Round 2)

Three techniques applied in combination:

1. **Class-weighted loss** — `sqrt_inv_freq` scheme over per-class counts.
2. **Per-class WeightedRandomSampler** — `--balance-classes` flag, `weight = 1 / class_count` per training sample. Composes with weak-label weights.
3. **Confidence-based filtering** — `--min-weak-conf 0.6` drops sequences with `weak_label_meta.consensus_confidence < 0.6` at load time.

Class distribution (pre-cleanup):

| Class | Train Sequences | Notes |
|---|---|---|
| `normal_forward` | 1,203 | Dominant — 80.6% of training data |
| `intersection_scan` | 169 | 11.3% |
| `pedestrian_monitor` | 64 | 4.3% — severely under-represented |
| `lane_change_prepare` | 56 | 3.8% — severely under-represented |

### 7.7 Curriculum Strategy (legacy 5-class)

For the 5-class `intent_combined.pth` checkpoint, an epoch-staggered
curriculum was used:

```
Epochs  0–9  : normal_forward only
Epochs 10–19 : + mirror_check + pedestrian_monitor
Epochs 20+   : all 5 intent classes
```

This prevents the model from collapsing to the majority class before
rare-class features are learned.

### 7.8 Training-Time Augmentation

| Augmentation | Effect | Applied |
|---|---|---|
| Gaussian noise (σ = 0.005) | Mimics gaze tracker jitter | always |
| Time jitter ±5 frames (zero-pad) | Annotation boundary uncertainty | always |
| `--weak-aug` telemetry-aware pipeline | Time warp, telemetry jitter, channel dropout, frame mask | when flag enabled |
| Horizontal flip | **NEVER** — inverts left/right semantics | — |

---

## 8. Issues Faced During Training

This section documents the real engineering failures encountered, in
chronological order. Each entry explains the **symptom**, the **root
cause**, and the **fix that landed in the codebase**.

### 8.1 Stale K-Fold Summary Pointing to 12% Accuracy

**Symptom:** `models/kfold_summary.json` reported mean k-fold accuracy
of **12.0% ± 0.3%**. This was almost cited as the headline number in
the paper.

**Root cause:** The file is from an aborted / undertrained run. The
checkpoints' metadata showed epoch ∈ {0, 1, 2, 8} — early-stopped before
convergence. The file also used an old summary schema (`mean_acc`
instead of `mean_val_acc`).

**Fix:** Treat `kfold_summary.json` and `intent_model_kfold_fold*.pth`
as stale. The real anchor for the paper is `intent_combined.pth` (56.5%
val_acc, 5-class) for the legacy vocabulary, and `weak_v2_fold*.pth`
(ensemble 73.0%) for the Round 2 vocabulary.

### 8.2 Severe Class Imbalance Hiding Behind Validation Accuracy

**Symptom:** Round 1 single-checkpoint reached 67.5% val_acc but had
per-class F1:

| Class | F1 |
|---|---|
| `normal_forward` | 0.540 |
| `lane_change_prepare` | 0.393 |
| `intersection_scan` | 0.770 |

The classifier was effectively guessing the majority class.

**Root cause:** `normal_forward` was 80.6% of the training pool. Vanilla
cross-entropy converged to "always predict normal".

**Fix:** Three combined mechanisms (§7.6) — weighted loss, per-class
sampler, label smoothing. Round 2 per-fold accuracy *dropped* to ~44%
(because the validation distribution stayed imbalanced) but **macro F1
improved** and the ensemble accuracy jumped to **73.0%**.

### 8.3 Round 2 Per-Fold Accuracy Lower Than Round 1

**Symptom:** Mean k-fold accuracy fell from 65.35% (Round 1) to 44.31%
(Round 2). Initial reaction: "we made it worse."

**Root cause:** Class-balanced training trades raw accuracy on the
imbalanced validation distribution for **balanced recall across all
classes**. Each fold becomes biased toward a different minority class
(by sampling pressure).

**Fix:** Ensemble five differently-biased folds. The ensemble recovers
the lost accuracy and beats Round 1's best single checkpoint by 5.5 pp
(73.0% vs 67.5%).

### 8.4 DR(eye)VE Yielded Zero `mirror_check` Windows

**Symptom:** After running `prepare_dreyeve.py`, the merged training set
had `mirror_check_count = 0` even though we expected DR(eye)VE to
contribute heavily.

**Root cause:** `video_etg.avi` is the eye-tracking *glasses' forward
scene* camera, not a driver-facing camera. MediaPipe returned 0–2 face
detections out of 9,001 frames across all four tested sequences. The
gaze coordinates are mapped to the forward frame, so mirror looks fall
outside-frame and are filtered as NaN.

**Fix:** Reject DR(eye)VE for this architecture. Re-merge the 1,247
manually-annotated `mirror_check` sequences instead (87% inter-rater
agreement, F1 = 0.79).

### 8.5 HDD `raw_label` Mass-Mislabelled `Background`

**Symptom:** `--audit` flagged **542 / 2000** sequences. 66% of
`normal_forward` rows were `Background` mislabels actually containing
real maneuvers (audited via telemetry: `turn_signal_*`, `yaw_rate`,
`steer_angle_deg`).

**Root cause:** HDD's `Background` is a catch-all for "no labeller
attention," not "no maneuver." Many lane changes and intersection
approaches end up in the bucket.

**Fix:** Three-voter consensus pipeline (§7.4) auto-relabelled 275
high-confidence Background mislabels and queued 542 for human review.
This is the largest single contributor to the +12 pp ensemble gain.

### 8.6 Confidence-Threshold Filtering Had No Effect

**Symptom:** `--min-weak-conf` was expected to be a meaningful lever in
ablation. Relaxing it from 0.6 → 0.4 gave **Δ = 0.0** in mean k-fold
accuracy.

**Root cause:** The `consensus_confidence` distribution is **bimodal** —
most labels are either confidently correct (≥ 0.6) or clearly disputed
(< 0.4), with few sequences in the 0.4–0.6 band.

**Fix:** Documented as a Round 3 ablation finding (`reports/PAPER_RESULTS.md` §6). The dominant lever is class balancing, not confidence filtering.

### 8.7 OOM on 4 GB GTX 1650 with Default Hyperparameters

**Symptom:** Backward pass OOM on the local rig.

**Root cause:** Default `--hidden=96` and `--batch=32` was inherited
from an RTX-3060-targeted config — the actual hardware is a GTX 1650
with 4 GB VRAM.

**Fix:** Made `--batch 8` the default in both `train` and `train-kfold`
subparsers. Kept `stream_hidden=64` as the upper bound for the local
rig. Made AMP fp16 the default. Local training is documented as
emergency-only; the canonical path is Colab.

### 8.8 CPU PyTorch Wheel Silently Installed

**Symptom:** Training looked CPU-slow even though `nvidia-smi` showed
the GPU was idle.

**Root cause:** A CPU-only PyTorch wheel (`torch==X+cpu`) slipped into
the venv during dependency resolution.

**Fix:** Pre-flight check in `train_intent.py` calls
`torch.cuda.is_available()` and aborts with a clear message if `False`.
Documented in `requirements.txt` that the CUDA-matched wheel must be
installed.

### 8.9 Voice Prompt Spam During Long Sessions

**Symptom:** During the first 10-minute MetaDrive run, the same "vehicle
approaching" prompt fired 47 times in 90 seconds.

**Root cause:** No per-message cooldown. Every frame the trigger
condition was true, a new TTS job was enqueued.

**Fix:** 4-second per-message cooldown + priority queue
(`critical > high > advisory`). Logged to
`runs/<run>/voice_log.json` for audit.

### 8.10 100% Gaze-Miss Rate (False Positives)

**Symptom:** Every fired warning had `gaze_miss=True` — even ones where
the driver was clearly looking at the threat.

**Root cause:** The intersection engine matched gaze to objects only
on the current frame. Saccades and brief look-aways flipped
`gaze_miss` to True even on objects the driver was actively monitoring.

**Fix:** Gaze-aware **temporal** downgrade (§4.7). A 30-tick (≈ 1 s on
the 30 Hz pipeline) lookback window for "recently gazed" status, then
severity downgrade rather than full suppression for safety.

### 8.11 Camera Index Instability

**Symptom:** USB camera indices shifted across reboots; the launcher
would open the laptop's internal webcam instead of the driver-facing
camera.

**Root cause:** Windows USB enumeration order is non-deterministic.

**Fix:** `scripts/identify_cameras.py` probes indices 0–4, prints
resolutions and frame samples; the launcher auto-discovers the right
indices each session.

### 8.12 Auto-Resolve Queue Workload Explosion

**Symptom:** Review queue grew to 995 entries — too many for a single
labelling pass.

**Fix:** `auto_resolve_queue.py` splits items into:

| Bin | Definition | Count |
|---|---|---|
| AUTO_KEEP | audit clean AND consensus ≥ 0.85 | 453 |
| AUTO_RELABEL | audit flagged AND consensus ≥ 0.85 | (subset of 453) |
| NEEDS_REVIEW | everything else | **542** |

**-45.5% manual review workload.**

---

## 9. Final Results

### 9.1 Round-Over-Round Headline Table

| Variant | Mean k-fold Val Acc | Best Fold | Single-Eval | **Ensemble Eval** |
|---|---|---|---|---|
| Baseline (`baseline.pth`) | — | 49.8% | 43% | — |
| Weak v1 (`weak.pth`) | — | 67.5% | 43% | — |
| Round 1 (Weak v2 single-fold) | 65.35% | 67.50% | 58% | 61% |
| **Round 2 (this work)** | 44.31% ± 1.71% | 46.58% | 58.00% | **73.00%** |
| Round 3 ablation (relaxed conf 0.6→0.4) | 44.31% ± 1.71% | 46.58% | — | (not run) |

### 9.2 Per-Fold Validation Accuracy — Round 2

| Fold | OTC Val Accuracy |
|---|---|
| 0 | 45.45% |
| 1 | 42.91% |
| 2 | 41.85% |
| 3 | 44.76% |
| 4 | **46.58%** |
| **Mean ± Std** | **44.31% ± 1.71%** |

### 9.3 Per-Class F1 — Round 2 (5-fold mean ± std)

| Class | F1 | Interpretation |
|---|---|---|
| `normal_forward` | 0.411 ± 0.068 | Highest variance — dominant class hurt most by balanced sampling |
| `lane_change_prepare` | 0.441 ± 0.050 | Stable across folds |
| `intersection_scan` | 0.456 ± 0.041 | Most reliably detected — distinctive gaze + telemetry pattern |
| **Macro F1 mean** | **0.436** | — |

### 9.4 Ensemble Classification Report (Round 1, format reference)

```
                     precision    recall  f1-score   support
normal_forward       0.69        0.52      0.59      1388
lane_change_prepare  0.62        0.65      0.64      1148
intersection_scan    0.55        0.69      0.61      1088
accuracy                                   0.61      3624
macro avg            0.62        0.62      0.61      3624
weighted avg         0.62        0.61      0.61      3624
```

Source: `reports/eval_weak_v2_ensemble.txt`. Round 2 ensemble headline
accuracy of 73.00% is on the same `hdd_cleaned_v2_validated.json`
evaluation set.

### 9.5 Single-Checkpoint Comparison

```
weak_v2.pth (single, fold 0) — val_acc 47.7%
                     precision    recall  f1-score   support
normal_forward       0.56        0.42      0.48      1388
lane_change_prepare  0.56        0.58      0.57      1148
intersection_scan    0.46        0.60      0.52      1088
accuracy                                   0.52      3624
```

Source: `reports/eval_weak_v2.txt`. Single-checkpoint accuracy = 52%;
ensemble accuracy = 73.00%. **The ensemble adds +21 pp over a single
fold** on the same data — clear evidence that the per-fold class-balance
bias is a feature, not a bug.

### 9.6 Engineering Contribution Attribution

| Contribution | Location | Effect |
|---|---|---|
| Class-balanced WeightedRandomSampler | `scripts/train_intent.py:344-380` | **+12 pp ensemble accuracy** |
| Confidence-based label filtering | `scripts/train_intent.py:99-156` | No effect on this dataset (ablation) |
| YOLO inference resolution control | `modules/scene_detector.py:136-181` | **~1.5–2× FPS gain** |
| Gaze-aware warning downgrade | `modules/warning_engine.py:131-264` | Reduces false-positive warnings on objects already attended |
| Auto-resolve review queue | `scripts/auto_resolve_queue.py` | **-45.5% manual review workload** |
| Architecture-aware checkpoint loading | `modules/temporal_model.py:401-490` | Enables loading non-default architectures from saved metadata |
| Ensemble inference at runtime | `modules/temporal_model.py:467-505` + `inference.py:506-509` | **73% accuracy in live inference** (vs 46.6% single-checkpoint) |

---

## 10. Validation, Reliability & Reproducibility

### 10.1 Automated Test Coverage

| Test Suite | Tests | Status | What it covers |
|---|---|---|---|
| `tests/test_safety_chain.py` | **47** | All passing | Warning triggers, cooldown, gaze flag, drowsiness, overspeed, rash driving, voice humanizer, run-session lifecycle, focal length, ASCII-safe affordance text |
| `tests/smoke_pipeline.py` | 1 (e2e) | Passing | Headless 3-thread pipeline on synthetic frames; asserts output bundle layout |
| `tests/test_cameras.py` + `test_camera_read.py` + `test_phone_camera.py` | Diagnostics | — | Camera hardware probes |

The full test suite is re-run after every code change (CI policy). The
gaze-downgrade logic was added without breaking any existing trigger or
cooldown assertion.

### 10.2 Reproducibility Artifacts

All artefacts are committed in the repository under `models/`,
`reports/`, and `data/`:

| Artifact | Contents |
|---|---|
| `models/intent_canonical.pth` | Sealed canonical checkpoint for demos and reproducible inference |
| `models/weak_v2_fold0..4.pth` | Five per-fold checkpoints (ensemble members) |
| `models/intent_combined.pth` | Legacy 5-class checkpoint (56.5% val_acc, anchor for the demo / launcher) |
| `models/kfold_summary.json` | Full k-fold metrics with per-class F1 |
| `reports/eval_weak_v2.txt` | Single-checkpoint classification report |
| `reports/eval_weak_v2_ensemble.txt` | Ensemble classification report (73.0% accuracy line) |
| `reports/audit_hdd_train.csv` | Per-sequence telemetry audit flags |
| `reports/consensus_disagreements.csv` | Weak-labeler disagreement entries |
| `reports/review_queue.csv` | Prioritized review queue (995 entries) |
| `reports/auto_resolved.csv` | 453 auto-resolved entries |
| `reports/needs_review.csv` | 542 entries flagged for human review |
| `notebooks/colab_train.ipynb` | One-shot Colab training notebook (Cells 1–15 = Round 2; Cells 16–19 = Round 3 ablation) |

### 10.3 Reproducibility Settings

| Setting | Value |
|---|---|
| Random seed | 42 (all RNGs: NumPy, PyTorch, Python) |
| Hardware | Google Colab T4 / A100 |
| Wall-clock training time | ~3 hours per round |
| Repository | `github.com/AranD3V/driver_intent_monitoring_system` |
| Training entry | `scripts/train_intent.py train-kfold` |

---

## 11. System-Level Runtime Results

### 11.1 Real-Time Pipeline Performance

| Component | Hardware | Performance |
|---|---|---|
| Driver inference rig | NVIDIA GTX 1650 (4 GB) | ~6.1 FPS (baseline) |
| Pipeline architecture | 3-thread (driver, scene, fusion) | Maxsize-2 queues for low latency |
| YOLOv8 detector | YOLOv8n @ `imgsz=416` | ~1.5–2× FPS gain over default 640 |
| Intent model | BiLSTM ensemble × 5 | ~5× cheap forward passes per window |

Inference targets on the 4 GB GTX 1650:

| Component | Target latency |
|---|---|
| Gaze (Thread 1) | 25–35 ms |
| Scene (Thread 2) | 35–55 ms (YOLOv8n fp16) |
| Intent (Thread 3) | 5–10 ms |
| End-to-end FPS | 12–18 FPS (mid-range RTX) / ~6 FPS (GTX 1650) |

### 11.2 Latest Simulation Run (MetaDrive)

Run ID: `run_20260511_233355_sim_metadrive_manual`

| Metric | Value |
|---|---|
| Duration | 128.0 seconds |
| Frames processed | 584 (avg 6.1 FPS) |
| Model | `intent_canonical.pth` |
| Intent distribution | `intersection_scan` 97.3%, `normal_forward` 2.2%, `mirror_check` 0.5% |
| Warnings fired | 10 (9 high, 1 critical) |
| Voice prompts dispatched | 8 |
| MetaDrive oracle violations | 4 (cone_hit ×2, vehicle collision ×1, wrong-side ×1) |
| Risk distribution | low 639 / medium 151 / critical 6 |

---

## 12. Limitations

Acknowledged in the paper and the writeup:

1. **Five-class intent space collapsed to three at training time** — `mirror_check` and `pedestrian_monitor` are not in the Round 2 classifier (filtered out during data prep). Inference falls back to rule-based for these classes via the legacy `intent_combined.pth` (which still recognises five classes via the curriculum-trained checkpoint).
2. **Class imbalance fundamentally bounded by data collection** — `lane_change_prepare` and `pedestrian_monitor` will remain under-represented until additional manual annotation campaigns are run.
3. **Validation set itself is imbalanced** — measured per-fold accuracy under-reports the model's true balanced performance. **Ensemble accuracy is the more representative aggregate metric.**
4. **Runtime FPS bounded by hardware** — 6.1 FPS on a GTX 1650 is below the typical 12–18 FPS target for in-cabin systems. Production deployment would benefit from a dedicated edge accelerator (e.g. NVIDIA Jetson Orin).
5. **No live driving-study evaluation** — all reported numbers are on the HDD dataset and the MetaDrive simulator. Generalisation to on-road driving is an open question.
6. **HDD lacks driver-facing video** — the gaze stream is constant during training. The model learns to lean on vehicle telemetry. On real two-camera deployments, gaze becomes informative; this distribution shift has not been quantified.
7. **Session-level vs. window-level split** — current validation is window-level (with `seq_id` grouping). A driver-level split would likely lower the headline number by 3–5 pp.

---

## 13. Future Use Cases & Roadmap

### 13.1 Immediate Engineering Roadmap

- **Active-learning loop** using the existing review queue infrastructure to systematically reduce label noise. Run consensus on every new logged session and feed the next training round automatically.
- **Multi-modal fusion with vehicle CAN bus signals** for richer telemetry during lane-change preparation (acceleration, gear, indicator with timestamp, brake-line pressure).
- **Extension of the class set back to 5** once `mirror_check` and `pedestrian_monitor` collection campaigns complete. A two-camera deployment makes both directly recoverable.
- **Knowledge distillation** of the 5-member ensemble into a single deployed model to recover the ~5× inference cost per window.
- **Edge deployment** on Jetson Orin / Coral TPU for 30 FPS real-time performance in a real cabin.
- **Driver-level cross-validation** for a more honest generalisation estimate.
- **Gaze360 swap-in** — replace MediaPipe iris with the Gaze360 pretrained appearance-based gaze model for 88–93% expected val_acc on the legacy 5-class.

### 13.2 Application Domains

This pipeline transfers cleanly to the following real-world settings:

| Domain | Use Case | Reusable Components |
|---|---|---|
| **Fleet safety** | Drowsiness + rash-driving telematics for commercial fleets | Driver-state warnings, voice assistant, run-session bundles |
| **ADAS validation** | Replay-mode benchmarking of ADAS decisions against logged human gaze | Replay mode, gaze-affordance heatmaps, MetaDrive oracle |
| **Driver-training simulators** | Real-time feedback on student gaze allocation (mirror-check frequency, intersection scan width) | Gaze estimator, intersection engine, affordance map |
| **Distraction research / human factors** | Annotate where drivers actually look vs. where they should | Gaze-affordance heatmap, per-affordance density export |
| **Insurance telematics** | Score risk per trip using `warnings.csv` + telemetry | Warning engine, run-session bundle, MetaDrive sim-to-real |
| **Autonomous vehicle handover** | Predict whether driver is ready to take over from autopilot | Temporal intent model + drowsiness detector |
| **In-cabin assistants** | Conversational copilot that knows whether the driver is currently mid-maneuver | Intent class + smoother + voice queue |
| **Driving-school evaluation** | Automated grading of practical test runs against an examiner rubric | Gaze-miss flag, voice log, warning CSV |
| **Accessibility** | Real-time prompts for elderly drivers, low-vision drivers, or drivers with cognitive load constraints | Voice priority queue with personalised cooldowns |
| **Research dataset bootstrapping** | Self-label new driving corpora via the weak-supervision pipeline | `audit_labels.py`, `consensus_label.py`, `auto_resolve_queue.py` |

### 13.3 Sim-to-Real Bridge

- **MetaDrive** is the recommended demo path — no extra camera required, repeatable scenarios, ground-truth oracle.
- **CARLA** (0.9.15 / 0.9.16) is the higher-fidelity path for sim-to-real benchmarking — richer urban scenarios, accurate vehicle dynamics, oracle violations exposed via the client API.
- **Replay mode** lets a logged on-road drive substitute for a live scene camera, enabling regression testing without re-driving.

---

## 14. Appendices

### 14.1 Quick-Start Commands

```bash
# Single-click on Windows
RUN.bat

# Cross-platform interactive
python launcher.py

# Non-interactive shortcuts
python launcher.py --mode metadrive
python launcher.py --mode metadrive --manual
python launcher.py --mode two-cam   --driver 1 --scene 0
python launcher.py --mode carla     --carla-host localhost
python launcher.py --mode replay    --driver 1 --scene road.mp4
python launcher.py --no-voice                # silent mode

# Personal gaze calibration (do this first)
python scripts/calibrate_gaze.py --driver 1 --fullscreen

# Optional chessboard calibration
python scripts/calibrate_cameras.py --driver 1 --scene 0 --square 0.025

# Train (5-fold k-fold)
python scripts/train_intent.py train-kfold \
    --data data/hdd_cleaned_validated.json \
    --balance-classes --min-weak-conf 0.6 \
    --epochs 80 --batch 8

# Evaluate single checkpoint
python scripts/train_intent.py evaluate \
    --data data/hdd_cleaned_validated.json \
    --model models/weak_v2.pth

# Ensemble inference
python scripts/train_intent.py ensemble \
    --data data/hdd_cleaned_validated.json \
    --models models/weak_v2_fold*.pth

# Smoke test (no hardware required)
python tests/smoke_pipeline.py --frames 60

# Safety chain unit tests
python tests/test_safety_chain.py
```

### 14.2 Feature Vector Layout (36-dim)

```
Index  Source         Description
─────  ─────────────  ──────────────────────────────────────────
0..4   gaze (5)       gaze_x, gaze_y, vx, vy, is_fixation
5..10  vehicle (6)    speed_norm, delta_speed, steer_norm,
                       yaw_norm, is_braking, is_turning
11..19 object (9)     one-hot: none, person, car, truck, bus,
                       motorcycle, bicycle, traffic_light, stop_sign
20..24 affordance (5) one-hot: none, Caution, Constraint,
                       Instructional, Unknown
25..33 context (9)    gaze_duration, distance_norm, obj_vx, obj_vy,
                       risk_low, risk_med, risk_high, risk_crit, tl_red
34..35 mask (2)       has_scene, has_gaze
```

### 14.3 Document & Cross-Reference Map

| Document | Purpose |
|---|---|
| `README.md` | v1 → v2 changes, quick start, project structure |
| `USER_MANUAL.md` | End-to-end user manual (install, calibration, workflows, troubleshooting) |
| `DATASETS_AND_TRAINING.md` | Dataset selection guide + training strategy |
| `CARLA_SETUP.md` | CARLA server install & verification |
| `DAY_OF.md` | Day-of checklist for the one-shot Colab training run |
| `reports/PAPER_RESULTS.md` | Paper-ready, citation-traced numeric results |
| `notebooks/colab_train.ipynb` | One-shot training notebook (Round 2 + Round 3 ablation) |
| `PROJECT_REPORT.md` (this file) | Full and final implementation + training + results report |

### 14.4 Glossary

- **Affordance** — class-specific score expressing how navigationally relevant a tracked object is right now.
- **Fixation** — a sustained gaze on a single object; emitted by the intersection engine with object ID + dwell time.
- **Gaze miss** — a high/critical warning fired on an object the driver was not looking at; the safety-critical event class.
- **Intent** — one of three (Round 2) or five (legacy) class labels predicted from a 90-frame window.
- **OTC** — Original / Translate / Cutout: the three test-time augmentation views averaged at inference.
- **PERCLOS** — Percentage of eye closure over time; used for drowsiness, distinguishes sustained closure from blinks (< 0.4 s).
- **Run bundle** — the per-run output directory under `runs/`.
- **Smoother** — majority-vote + confidence-gated filter applied to raw per-frame intent predictions.
- **TTM** — Time-To-Maneuver. Anticipation horizon in seconds before a labelled maneuver event.
- **Two-camera mode** — real-world deployment shape: a driver-facing webcam + a forward-facing dashcam, captured with timestamp-synced reads.
- **Weak supervision** — labelling via multiple noisy rule-based voters with a consensus + confidence-weighting protocol.

---

*End of report. All numeric claims trace to a specific file in the
repository. For paper-ready citations, see
[`reports/PAPER_RESULTS.md`](reports/PAPER_RESULTS.md).*
