# Datasets & Training Strategy
## Driver Intent Monitoring System

---

## Part 1 — Which Datasets to Use

### Recommended Stack (Ordered by Priority)

---

### 1. Your Own Logs (Primary Source — Do This First)

**Why:** Your system logs in exactly the format the model needs.
No format conversion. No domain gap. Ground truth is your actual deployment scenario.

**How to collect:**
1. Mount cameras, drive for 30–60 minutes across varied scenarios
2. System auto-logs all frames to `logs/`
3. Run annotation template generator:
   ```bash
   python scripts/prepare_dataset.py \
       --source template \
       --input  logs/session_xyz.json \
       --output data/annotated/session_xyz.json
   ```
4. Open the template in any JSON editor, set `intent_label` for each 3-second block
5. Repeat until you have 500–1000 labelled sequences

**Minimum for first training run:** 200 sequences × 5 intent classes = 1000 sequences

---

### 2. DADA-2000 (Driver Attention in Driving Accident)

| Attribute       | Value |
|-----------------|-------|
| URL             | https://github.com/JWFangit/LOTVS-DADA |
| Paper           | DADA: Driver Attention in Driving Accident Scenarios (TITS 2022) |
| Size            | 2000 driving clips, 658,476 frames |
| Gaze data       | ✅ Eye-tracking (2D gaze coordinates) |
| Scene events    | ✅ 54 accident types labelled |
| Driver actions  | ✅ Steering, braking, gaze |
| License         | Academic research only |

**What it gives you:**
- Real eye-tracking gaze traces mapped to scene video
- High-risk scenarios: pedestrian crossings, intersections, sudden stops
- Labelled accident precursors → maps well to `intersection_scan` and `pedestrian_monitor`

**How to use:**
```bash
python scripts/prepare_dataset.py \
    --source dada \
    --input  /data/DADA-2000 \
    --output data/dada_sequences.json
```

**Mapping DADA labels → your intent classes:**
| DADA Event          | Your Intent Class      |
|---------------------|------------------------|
| Pedestrian          | pedestrian_monitor     |
| Intersection events | intersection_scan      |
| Vehicle merging     | lane_change_prepare    |
| Normal driving      | normal_forward         |
| Mirror regions      | mirror_check           |

---

### 3. DrFixD (Driver Fixation Dataset)

| Attribute   | Value |
|-------------|-------|
| URL         | https://github.com/taodeng/drfixd |
| Paper       | DrFixD: Driver Fixation Dataset (TIV 2023) |
| Size        | 1,280 driving clips with eye-tracking |
| Gaze data   | ✅ Fixation points, saccade flags |
| Scenarios   | Urban, highway, suburban |
| License     | Academic research only |

**What it gives you:**
- Clean fixation/saccade separation (critical for your `IntersectionEngine`)
- Diverse lighting and weather
- Good coverage of `mirror_check` and `intersection_scan`

---

### 4. BDD-X (Berkeley DeepDrive Explanations)

| Attribute   | Value |
|-------------|-------|
| URL         | https://github.com/JinkyuKimUCB/BDD-X-dataset |
| Paper       | Textual Explanations for Self-Driving (ECCV 2018) |
| Size        | 6,984 driving clips |
| Gaze data   | ❌ No eye-tracking |
| Proxy gaze  | ✅ Attention maps (GradCAM from model) |
| Actions     | ✅ Lane change, turn, stop, etc. |
| License     | CC BY-NC 4.0 |

**What it gives you:**
- Large-scale diverse driving data
- Strong vehicle/pedestrian annotation → `lane_change_prepare`, `pedestrian_monitor`
- Use attention maps as proxy for gaze (imperfect but supplemental)

---

### 5. DGW (Driver Gaze in the Wild)

| Attribute   | Value |
|-------------|-------|
| URL         | https://sites.google.com/view/drivergazeinwild |
| Paper       | Generalizing Gaze Estimation with Weak-Supervision (CVPR 2021) |
| Size        | 237,000 frames with gaze labels |
| Gaze data   | ✅ Gaze zone labels (9 zones: forward, mirrors, instruments, etc.) |
| Driver      | ✅ Real driver faces |
| License     | Academic |

**What it gives you:**
- Gaze zone → intent mapping directly:
  | DGW Zone            | Your Intent              |
  |---------------------|--------------------------|
  | Forward             | normal_forward           |
  | Left mirror         | mirror_check             |
  | Right mirror        | mirror_check             |
  | Left shoulder       | lane_change_prepare      |
  | Right shoulder      | lane_change_prepare      |
  | Instrument cluster  | mirror_check (interior)  |
- Large driver face corpus → helps gaze estimator generalization

---

### 6. Gaze360 (Pretrained Gaze Model)

| Attribute   | Value |
|-------------|-------|
| URL         | https://github.com/erkil1452/gaze360 |
| Paper       | Gaze360 (ICCV 2019) |
| Use         | Pretrained gaze estimator weights |
| Benefit     | Replaces iris heuristic with learned appearance model |

**Drop-in replacement for GazeEstimator:**
```python
# In gaze_estimator.py, swap to Gaze360 for better accuracy
from gaze360 import Gaze360Model
self.gaze_model = Gaze360Model.from_pretrained('gaze360_resnet50.pth')
gaze_vector = self.gaze_model(face_crop)  # returns 3D unit vector
```

---

### 2b. Dr(eye)ve (Available Now)

| Attribute       | Value |
|-----------------|-------|
| Location        | `D:\TW\dr(eye)ve` |
| Paper           | DR(eye)VE: A Dataset for Attention-Based Tasks with Applications to Autonomous and Assisted Driving (TPAMI 2018) |
| Size            | 74 sequences total, 4 available locally (seq 01, 11, 29, 36) |
| Gaze data       | ✅ Real eye-tracking — fixation/saccade/blink labels + (X, Y) on scene |
| Scene video     | ✅ Garmin dashcam (`video_garmin.avi`) |
| Saliency maps   | ✅ Ground-truth visual saliency (`video_saliency.avi`, `mean_gt.png`) |
| Telemetry       | ✅ Speed (km/h), heading (deg), GPS per frame |
| License         | Academic research only |

**Sequences available locally:**

| Seq | Time | Weather | Road | Driver |
|-----|------|---------|------|--------|
| 01 | Evening | Sunny | Countryside | D8 |
| 11 | Evening | Cloudy | Downtown | D5 |
| 29 | Night | Cloudy | Countryside | D8 |
| 36 | Evening | Cloudy | Countryside | D1 |

**Label mapping (subsequences.txt → intent classes):**

| Dr(eye)ve label | Meaning | Your Intent Class |
|-----------------|---------|-------------------|
| `k` | Keep lane / straight driving | `normal_forward` |
| `e` | Ego maneuver (turn, overtake) | `lane_change_prepare` |
| `i` | Intersection approach | `intersection_scan` |

> **Note:** `mirror_check` and `pedestrian_monitor` have no direct dr(eye)ve equivalent.
> Head pose is not recorded in dr(eye)ve — those features are zeroed out in extracted vectors.

**How to prepare:**
```bash
# Run the offline pipeline (YOLO + affordance + gaze alignment)
python scripts/prepare_dreyeve.py \
    --dataset "D:/TW/dr(eye)ve" \
    --output  data/dreyeve_train.json

# Or use the all-in-one launcher (prepare + train + evaluate)
bash scripts/train_dreyeve.sh
```

**Flags:**

| Flag | Default | Effect |
|------|---------|--------|
| `--frame-skip N` | 1 | Process every Nth frame (2 = 2× faster, small accuracy drop) |
| `--min-frames N` | 30 | Discard subsequences shorter than N frames |
| `--seq 11 36` | all | Only process specific sequence IDs |
| `--no-scene` | off | Skip YOLO (gaze-only, very fast; no object features) |

---

### Dataset Priority Matrix

| Dataset      | Size    | Gaze Quality | Intent Labels | Effort to Use | Priority |
|--------------|---------|--------------|---------------|---------------|----------|
| Your Logs    | Custom  | ⭐⭐⭐⭐⭐   | Manual        | Low           | 1st      |
| Dr(eye)ve    | 4 seq   | ⭐⭐⭐⭐     | Subseq labels | **Ready now** | 2nd      |
| DGW          | 237K fr | ⭐⭐⭐⭐     | Zone-based    | Medium        | 3rd      |
| DADA-2000    | 658K fr | ⭐⭐⭐⭐     | Event-based   | Medium        | 4th      |
| DrFixD       | 1280 cl | ⭐⭐⭐⭐     | Fixation      | Medium        | 5th      |
| BDD-X        | 6984 cl | ⭐⭐         | Action-based  | High          | 6th      |

---

## Part 2 — Training Strategy

### Overview

Training happens in four progressive stages:

```
Stage 0: Baseline (rule-based, no ML)
Stage 1: Binary classifiers per intent (1 vs rest)
Stage 2: Full 5-class LSTM on clean data
Stage 3: Fine-tuning on your deployment environment
```

---

### Stage 0 — Rule-Based Baseline

Before training any model, implement a rule-based intent predictor.
This gives you a performance floor to beat and helps debug the pipeline.

```python
def rule_based_intent(gaze_data, gaze_affordance, gaze_history):
    if gaze_data is None:
        return 'normal_forward'

    yaw   = gaze_data['head_pose']['yaw']
    pitch = gaze_data['head_pose']['pitch']

    # Mirror check: head turned significantly to side
    if abs(yaw) > 30:
        return 'mirror_check'

    # Looking at pedestrian
    if gaze_affordance and gaze_affordance.get('looked_object', {}).get('class') == 'person':
        return 'pedestrian_monitor'

    # Wide horizontal gaze sweep
    if len(gaze_history) >= 30:
        x_range = max(h[0] for h in gaze_history) - min(h[0] for h in gaze_history)
        if x_range > 600:
            return 'intersection_scan'

    return 'normal_forward'
```

**Target accuracy:** 55–65%
**If your LSTM can't beat this, something is wrong.**

---

### Stage 1 — Data Collection & Annotation Protocol

**Minimum annotation requirements per class:**

| Intent Class        | Minimum Sequences | Recommended |
|---------------------|-------------------|-------------|
| normal_forward      | 200               | 500         |
| mirror_check        | 150               | 400         |
| pedestrian_monitor  | 150               | 400         |
| lane_change_prepare | 100               | 300         |
| intersection_scan   | 100               | 300         |
| **Total**           | **700**           | **1900**    |

**Recording protocol:**
- Record in sessions of 20–30 minutes
- Include: urban, suburban, highway
- Include: day, dusk, artificial lighting
- Include: multiple drivers (ideally 5–10 different people)
- For each intent, have the driver deliberately perform it and mark timestamps

**Annotation tool (simple):**
```python
# Use your own system logs + annotation template
python scripts/prepare_dataset.py --source template \
    --input logs/session_001.json \
    --output data/annotated/session_001.json

# Then open in VS Code / any editor, search "FILL_ME_IN", replace with:
# "normal_forward" | "mirror_check" | "pedestrian_monitor" |
# "lane_change_prepare" | "intersection_scan"
```

---

### Stage 2 — Training Protocol

#### Phase 2a: Curriculum Learning (Epochs 0–20)

Start simple, add complexity.

```
Epochs  0–9  : normal_forward only
Epochs 10–19 : + mirror_check + pedestrian_monitor
Epochs 20+   : all 5 intent classes
```

**Why curriculum?**
- Prevents model from getting stuck in majority-class local minima
- Each class properly learned before multi-class confusion introduced
- Mirrors how a human annotator would annotate: easy cases first

This is already implemented in `train_intent.py`.

#### Phase 2b: Class Imbalance Handling

Three mechanisms used in parallel:

1. **Weighted CrossEntropy**
   ```python
   # Automatic inverse-frequency weighting
   weights = dataset.class_weights()   # [0.4, 1.2, 1.2, 1.8, 1.8]
   criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
   ```

2. **Weighted Random Sampler**
   ```python
   # Oversample rare classes so each batch is more balanced
   sampler = WeightedRandomSampler(dataset.sample_weights(), num_samples=len(dataset))
   ```

3. **Label Smoothing (ε = 0.1)**
   ```python
   nn.CrossEntropyLoss(label_smoothing=0.1)
   # Prevents overconfident predictions, improves calibration
   ```

#### Phase 2c: Augmentation Strategy

Three augmentation types applied during training:

| Augmentation        | Effect                                       | Probability |
|---------------------|----------------------------------------------|-------------|
| Gaussian noise      | Mimics gaze tracker jitter                   | Always      |
| Time jitter ±5fr    | Accounts for annotation boundary uncertainty | Always      |
| Horizontal flip     | Simulates right-hand / left-hand traffic     | 50%         |

All implemented in `IntentSequenceDataset._augment()`.

#### Phase 2d: Optimiser Configuration

```python
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)
```

- **AdamW**: better generalisation than Adam via decoupled weight decay
- **Cosine annealing**: smooth LR decay without sharp drops
- **Gradient clipping** (max_norm=1.0): prevents LSTM exploding gradients
- **Mixed precision** (fp16): 1.5–2× faster on RTX GPUs

---

### Stage 3 — Fine-tuning for Your Environment

After training on public datasets, fine-tune with your own logs.

```bash
# Initial training (all datasets)
python train_intent.py train \
    --data data/dada_sequences.json data/drfixd_sequences.json \
    --output models/intent_pretrained.pth \
    --epochs 60

# Fine-tune on your environment (lower LR, fewer epochs)
python train_intent.py train \
    --data data/annotated/my_logs_session_*.json \
    --output models/intent_final.pth \
    --epochs 20 \
    --lr 1e-4
```

**Why two-stage?**
- Public datasets provide a strong prior across diverse scenarios
- Your environment fine-tunes for your specific camera placement, FOV, driving context
- Prevents catastrophic forgetting by using 10× lower learning rate

---

### Stage 4 — Evaluation Protocol

Run this after every training run:

```bash
# Full evaluation with confusion matrix
python train_intent.py evaluate \
    --data data/annotated/test_set.json \
    --output models/intent_final.pth
```

**Metrics to track:**

| Metric             | Target    | Description |
|--------------------|-----------|-------------|
| Overall Accuracy   | ≥ 80%     | All classes |
| Per-class F1       | ≥ 0.75    | Especially rare classes |
| Temporal Consistency| ≥ 90%    | % of frames where smoothed output doesn't flip between adjacent frames |
| Confusion diagonal | High       | Main confusion pairs reveal labelling issues |

**Temporal consistency check:**
```python
def temporal_consistency(predictions, window=10):
    flips = 0
    for i in range(1, len(predictions)):
        if predictions[i] != predictions[i-1]:
            flips += 1
    return 1.0 - (flips / len(predictions))
```

---

### Common Training Failure Modes

| Symptom                            | Likely Cause                          | Fix |
|------------------------------------|---------------------------------------|-----|
| All predictions = normal_forward   | Class imbalance dominates             | Increase sample weights for rare classes |
| Accuracy < rule-based baseline     | Feature extraction bug                | Log and inspect feature vectors |
| High train acc, low val acc        | Overfitting                           | Add dropout, more augmentation, more data |
| Oscillating predictions at runtime | Insufficient temporal smoothing       | Increase smoother window size |
| mirror_check confused with lane_change | Both involve sideward gaze        | Add gaze_duration as discriminating feature |
| intersection_scan ↔ normal_forward | Intersection scan too brief           | Reduce sequence length or add scene context features |

---

### Expected Performance Trajectory

| Stage               | Data Size   | Expected Val Acc |
|---------------------|-------------|------------------|
| Rule-based baseline | —           | 55–65%           |
| Stage 2 (clean data)| 700 seq     | 65–75%           |
| Stage 2 (full data) | 1500+ seq   | 75–83%           |
| Stage 3 (fine-tuned)| +200 own    | 83–90%           |
| With Gaze360 model  | +200 own    | 88–93%           |

---

### Quick Start Training Commands

```bash
# 1. Record your own sessions with the system running
python inference.py --driver 0 --scene 1 --no-output

# 2. Generate annotation templates
python scripts/prepare_dataset.py \
    --source template \
    --input  logs/session_001_batch_0000500.json \
    --output data/annotated/session_001.json

# 3. Open data/annotated/session_001.json, fill in intent labels

# 4. Convert any public datasets
python scripts/prepare_dataset.py \
    --source dada \
    --input  /data/DADA-2000 \
    --output data/dada.json

# 5. Train
python train_intent.py train \
    --data  data/annotated/*.json data/dada.json \
    --output models/intent_model.pth \
    --epochs 60 --batch 32 --lr 1e-3

# 6. Run with trained model
python inference.py \
    --driver 0 --scene 1 \
    --model  models/intent_model.pth
```

---

### Hardware Recommendations for Training

| Spec        | Minimum          | Recommended       |
|-------------|------------------|-------------------|
| GPU         | RTX 3060 (12GB)  | RTX 3090 / A6000  |
| RAM         | 16 GB            | 32 GB             |
| Storage     | 50 GB SSD        | 200 GB NVMe       |
| Training time (60 epochs, 1500 seq) | ~2hr | ~40 min |

**Cloud option:** Google Colab Pro / Lambda Labs for training,
deploy on local RTX 3060 for inference.
