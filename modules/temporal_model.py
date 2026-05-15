"""
Temporal Intent Model — single-stream BiLSTM intent classifier.

Input:  (B, T, 36) hand-crafted feature vector
  -> BiLSTM (64*2) -> mean-pool -> classifier (128->64->C)

Feature layout (36-dim):
  gaze(5) + vehicle(6) + object_onehot(9) + affordance_onehot(5)
  + context(9) + mask(2)
"""

from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
#  Feature engineering constants                                       #
# ------------------------------------------------------------------ #

OBJECT_CLASSES = [
    'none', 'person', 'car', 'truck', 'bus',
    'motorcycle', 'bicycle', 'traffic_light', 'stop_sign'
]

AFFORDANCE_TYPES = ['none', 'Caution', 'Constraint', 'Instructional', 'Unknown']

INTENT_CLASSES = [
    'normal_forward',
    'lane_change_prepare',
    'intersection_scan',
]

NATIVE_CLASSES = [
    'normal_forward',
    'intersection_passing',
    'left_turn',
    'right_turn',
    'U-turn',
    'left_lane_change',
    'right_lane_change',
    'left_lane_branch',
    'right_lane_branch',
    'merge',
    'crosswalk_passing',
    'pedestrian_monitor',
    'lane_change_prepare',
    'intersection_scan',
]

NATIVE_TO_INTENT = {
    'normal_forward':        'normal_forward',
    'intersection_passing':  'intersection_scan',
    'left_turn':             'intersection_scan',
    'right_turn':            'intersection_scan',
    'U-turn':                'intersection_scan',
    'intersection_scan':     'intersection_scan',
    'left_lane_change':      'lane_change_prepare',
    'right_lane_change':     'lane_change_prepare',
    'left_lane_branch':      'lane_change_prepare',
    'right_lane_branch':     'lane_change_prepare',
    'merge':                 'lane_change_prepare',
    'lane_change_prepare':   'lane_change_prepare',
    'crosswalk_passing':     'normal_forward',
    'pedestrian_monitor':    'normal_forward',
}

NATIVE_LABEL_ALIASES = {
    'Background':          'normal_forward',
    'crosswalk_passing':   'normal_forward',
}

# Feature layout (total = 36):
#   gaze (5):       x, y, vx, vy, is_fixation                        [0..4]
#   vehicle (6):    speed_norm, delta_speed, steer_norm, yaw_norm,   [5..10]
#                   is_braking, is_turning
#   object (9):     one-hot                                          [11..19]
#   affordance (5): one-hot                                          [20..24]
#   context (9):    gaze_duration, distance, obj_vx, obj_vy,         [25..33]
#                   risk_low, risk_med, risk_high, risk_crit, tl_red
#   mask (2):       has_scene, has_gaze                              [34..35]
FEATURE_DIM = 36
SCENE_W, SCENE_H = 1280, 720

# Stream split of the 36-dim vector:
#   in-cabin (12) = gaze(5) + vehicle(6) + has_gaze(1)
#   scene    (24) = object(9) + affordance(5) + context(9) + has_scene(1)
#
# Vehicle telemetry goes in-cabin so it drives the CRE query even when gaze
# is absent (HDD has no driver camera — gaze dims are constant defaults).
_CABIN_SLICE    = slice(0, 11)   # gaze(5) + vehicle(6)
_HAS_GAZE_SLICE = slice(35, 36)
_SCENE_SLICE    = slice(11, 35)  # object(9) + affordance(5) + context(9) + has_scene(1)

IN_CABIN_DIM = 12   # cabin(11) + has_gaze(1)
SCENE_DIM    = 24   # 35 - 11 = 24


# ------------------------------------------------------------------ #
#  Model                                                               #
# ------------------------------------------------------------------ #

class TemporalIntentModel(nn.Module):
    """
    Single-stream BiLSTM intent classifier.

    Processes the full FEATURE_DIM vector directly — no stream splitting,
    no cross-attention. This is the reliable baseline for datasets that lack
    a driver-facing camera (e.g. HDD), where the gaze stream is always zero
    and the CRE cross-attention query degenerates.

    Constructor signature is backward-compatible with the dual-stream version
    so train_intent.py needs no changes.
    """

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        stream_hidden: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,            # kept for call-site compat, unused
        num_classes: int = len(INTENT_CLASSES),
        dropout: float = 0.3,
        in_cabin_dim: int = IN_CABIN_DIM,   # unused
        scene_dim: int = SCENE_DIM,          # unused
    ):
        super().__init__()
        d_model = stream_hidden * 2    # bidirectional -> 128

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=stream_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    @staticmethod
    def _split_streams(x: torch.Tensor):
        """Stub kept for call-site compat — not used in forward."""
        return x, x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T, FEATURE_DIM) -> logits : (B, num_classes)"""
        out, _ = self.lstm(x)
        # Mean pool over time: captures sustained patterns (avg speed, avg
        # brake rate) better than last-timestep for anticipatory intent labels.
        return self.classifier(self.norm(out.mean(dim=1)))


# ------------------------------------------------------------------ #
#  Feature extractor                                                   #
# ------------------------------------------------------------------ #

class FeatureExtractor:
    """Converts a frame_data packet → fixed-length numpy vector."""

    def __init__(self, scene_w: int = SCENE_W, scene_h: int = SCENE_H):
        self._prev_vehicle = None   # (speed, steer_norm) from previous frame
        self.scene_w = scene_w
        self.scene_h = scene_h

    def extract(self, fd: Dict) -> np.ndarray:
        """Returns np.ndarray of shape (FEATURE_DIM,)."""
        feats = []

        # ── Gaze (5) ─────────────────────────────────────────────────
        gd = fd.get('gaze_data') or {}
        has_gaze = 1.0 if (gd and 'gaze_point' in gd) else 0.0
        gp = gd.get('gaze_point', (self.scene_w / 2, self.scene_h / 2))
        gv = gd.get('gaze_velocity', (0, 0))

        feats += [
            gp[0] / self.scene_w,
            gp[1] / self.scene_h,
            np.clip(gv[0] / self.scene_w, -1, 1),
            np.clip(gv[1] / self.scene_h, -1, 1),
            float(gd.get('is_fixation', True))
        ]

        # ── Vehicle telemetry (6) ────────────────────────────────────
        veh = fd.get('vehicle') or {}
        speed = veh.get('speed_kmh', 0.0)

        speed_norm = np.clip(speed / 130.0, 0, 1)

        steer_deg = veh.get('steer_angle_deg')
        if steer_deg is not None:
            steer_norm = np.clip(steer_deg / 500.0, -1, 1)      # HDD: ±500° range
        else:
            heading = veh.get('heading_deg', 0.0)
            steer_norm = (heading / 180.0) - 1.0                # dr(eye)ve fallback

        yaw_rate = veh.get('yaw_rate')
        if yaw_rate is not None:
            yaw_norm = np.clip(yaw_rate / 40.0, -1, 1)          # HDD: ±40 deg/s
        else:
            yaw_norm = 0.0

        if self._prev_vehicle is not None:
            prev_spd, prev_steer = self._prev_vehicle
            delta_speed = np.clip((speed - prev_spd) / 5.0, -1, 1)
            delta_steer = np.clip((steer_norm - prev_steer) * 5.0, -1, 1)
        else:
            delta_speed = 0.0
            delta_steer = 0.0

        brake = veh.get('brake', 0.0)
        brake_norm = np.clip(brake / 2000.0, 0, 1) if brake else 0.0
        is_braking = float((brake_norm > 0.05) or (delta_speed < -0.1))

        ts_left  = veh.get('turn_signal_left',  0.0)
        ts_right = veh.get('turn_signal_right', 0.0)
        signal_active = float(ts_left > 0.5 or ts_right > 0.5)
        is_turning = float(signal_active or abs(yaw_norm) > 0.1 or abs(delta_steer) > 0.1)

        self._prev_vehicle = (speed, steer_norm)
        feats += [speed_norm, delta_speed, steer_norm, yaw_norm,
                  is_braking, is_turning]

        # ── Object class one-hot (9) ─────────────────────────────────
        ga = fd.get('gaze_affordance') or {}
        lo = ga.get('looked_object') or {}

        # Fall back to detected_objects when gaze_affordance is absent.
        # Picks the closest high-confidence detection as the primary object.
        if not lo:
            _dets = fd.get('detected_objects') or []
            if _dets:
                _best = min(_dets, key=lambda o: o.get('distance_estimate', 999.0))
                _dist = float(_best.get('distance_estimate', 50.0))
                _cls  = _best.get('class', 'none')
                _AFF  = {
                    'person': 'Caution', 'pedestrian': 'Caution',
                    'motorcycle': 'Caution', 'bicycle': 'Caution',
                    'traffic_light': 'Instructional', 'stop_sign': 'Instructional',
                    'car': 'Constraint', 'truck': 'Constraint', 'bus': 'Constraint',
                }
                lo = {'class': _cls, 'velocity': _best.get('velocity', [0, 0])}
                ga = {
                    'looked_object': lo,
                    'affordance':    _AFF.get(_cls, 'none'),
                    'risk_level':    ('critical' if _dist < 5  else
                                      'high'     if _dist < 15 else
                                      'medium'   if _dist < 30 else 'low'),
                    'gaze_duration': 0.0,
                    'distance':      _dist,
                }

        has_scene = 1.0 if lo else 0.0
        obj_cls = lo.get('class', 'none')

        obj_cls = obj_cls.replace(' ', '_')
        one_hot = [0.0] * len(OBJECT_CLASSES)
        if obj_cls in OBJECT_CLASSES:
            one_hot[OBJECT_CLASSES.index(obj_cls)] = 1.0
        feats += one_hot

        # ── Affordance one-hot (5) ───────────────────────────────────
        aff = ga.get('affordance', 'none')
        aff_hot = [0.0] * len(AFFORDANCE_TYPES)
        if aff in AFFORDANCE_TYPES:
            aff_hot[AFFORDANCE_TYPES.index(aff)] = 1.0
        feats += aff_hot

        # ── Context (9) ──────────────────────────────────────────────
        duration = min(ga.get('gaze_duration', 0) / 60.0, 1.0)
        dist     = ga.get('distance', 50)
        dist_n   = np.clip(1.0 - (dist or 50) / 50.0, 0, 1) if dist else 0.0

        vel = lo.get('velocity', [0, 0])
        obj_vx = np.clip((vel[0] if vel else 0) / self.scene_w, -1, 1)
        obj_vy = np.clip((vel[1] if vel else 0) / self.scene_h, -1, 1)

        risk  = ga.get('risk_level', 'low')
        risks = [
            float(risk == 'low'),
            float(risk == 'medium'),
            float(risk == 'high'),
            float(risk == 'critical')
        ]

        tl_red = 0.0
        if lo.get('class') == 'traffic light':
            tl_red = float(lo.get('traffic_light_state', '') == 'red')

        feats += [duration, dist_n, obj_vx, obj_vy] + risks + [tl_red]

        # ── Modality mask (2) ────────────────────────────────────────
        feats += [has_scene, has_gaze]

        assert len(feats) == FEATURE_DIM, f"Feature dim mismatch: {len(feats)}"
        vec = np.array(feats, dtype=np.float32)
        return np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)


# ------------------------------------------------------------------ #
#  Predictor wrapper                                                   #
# ------------------------------------------------------------------ #

class TemporalIntentPredictor:

    def __init__(self, window_size: int = 90, device: str = 'auto'):
        self.window_size = window_size
        self.device = (
            'cuda' if torch.cuda.is_available() else 'cpu'
        ) if device == 'auto' else device

        self.feature_buffer = deque(maxlen=window_size)
        self.extractor = FeatureExtractor()
        self._weights_loaded = False

        self.model = TemporalIntentModel().to(self.device)
        self.model.eval()
        # Ensemble mode: when populated, predict() averages softmax across all.
        self.ensemble_models: list = []
        self._classes = list(INTENT_CLASSES)

    # -- public --------------------------------------------------------

    def add_frame(self, frame_data: Dict):
        vec = self.extractor.extract(frame_data)
        self.feature_buffer.append(vec)

    def _rule_based_predict(self) -> Dict:
        window   = list(self.feature_buffer)[-30:]
        gaze_x       = np.array([f[0]  for f in window])
        delta_hdg    = np.array([f[8]  for f in window])   # delta_heading
        is_braking   = np.array([f[9]  for f in window])
        is_turning   = np.array([f[10] for f in window])
        person       = np.array([f[12] for f in window])

        turn_ratio   = float(is_turning.mean())
        brake_ratio  = float(is_braking.mean())
        person_ratio = float(person.mean())
        gaze_range   = float(gaze_x.max() - gaze_x.min())
        std_gaze_x   = float(gaze_x.std())
        abs_delta_hdg = float(np.abs(delta_hdg).mean())

        if person_ratio > 0.40:
            intent, conf = 'pedestrian_monitor', min(0.80, person_ratio * 1.5)
        elif gaze_range > 0.40 and (brake_ratio > 0.3 or turn_ratio > 0.3):
            intent, conf = 'intersection_scan',  min(0.75, gaze_range)
        elif abs_delta_hdg > 0.15 or turn_ratio > 0.4:
            intent, conf = 'lane_change_prepare', min(0.75, max(abs_delta_hdg, turn_ratio))
        elif std_gaze_x > 0.15:
            intent, conf = 'mirror_check',       min(0.70, std_gaze_x * 2)
        else:
            intent, conf = 'normal_forward',      0.70

        probs = {c: (1 - conf) / (len(INTENT_CLASSES) - 1) for c in INTENT_CLASSES}
        probs[intent] = conf
        return {'intent': intent, 'confidence': conf, 'probabilities': probs,
                'ready': True, 'rule_based': True, 'warmup_progress': 1.0}

    def predict(self) -> Dict:
        progress = len(self.feature_buffer) / self.window_size
        enough   = len(self.feature_buffer) >= 30

        if not self._weights_loaded:
            return self._rule_based_predict() if enough else {
                'intent': 'normal_forward', 'confidence': 0.0,
                'probabilities': {c: 0.0 for c in INTENT_CLASSES},
                'ready': False, 'warmup_progress': progress
            }

        if len(self.feature_buffer) < self.window_size:
            return self._rule_based_predict() if enough else {
                'intent': 'normal_forward', 'confidence': 0.0,
                'probabilities': {c: 0.0 for c in INTENT_CLASSES},
                'ready': False, 'warmup_progress': progress
            }

        seq = np.stack(list(self.feature_buffer))                         # (T, F)
        t   = torch.FloatTensor(seq).unsqueeze(0).to(self.device)        # (1, T, F)

        with torch.no_grad():
            if self.ensemble_models:
                # Average softmax across all ensemble models
                acc = None
                for m in self.ensemble_models:
                    p = torch.softmax(m(t), dim=1)
                    acc = p if acc is None else acc + p
                probs = (acc / len(self.ensemble_models))[0]
            else:
                logits = self.model(t)
                probs  = torch.softmax(logits, dim=1)[0]
            idx    = int(probs.argmax())

        classes = self._classes
        return {
            'intent':        classes[idx],
            'confidence':    float(probs[idx]),
            'probabilities': {classes[i]: float(probs[i]) for i in range(len(classes))},
            'ready':         True,
            'warmup_progress': 1.0
        }

    def _build_model_from_ckpt(self, ckpt: dict):
        """Rebuild a TemporalIntentModel matching the checkpoint's saved arch."""
        arch    = ckpt.get('arch', {}) or {}
        classes = ckpt.get('classes') or list(INTENT_CLASSES)
        # Infer stream_hidden from weight shape if not saved in arch
        if 'stream_hidden' not in arch:
            state = ckpt.get('model_state_dict', ckpt)
            w = state.get('lstm.weight_ih_l0')
            arch['stream_hidden'] = int(w.shape[0] // 4) if w is not None else 128
        m = TemporalIntentModel(
            input_size    = FEATURE_DIM,
            stream_hidden = arch.get('stream_hidden', 128),
            num_layers    = arch.get('num_layers', 2),
            num_heads     = arch.get('num_heads', 4),
            num_classes   = len(classes),
            dropout       = arch.get('dropout', 0.3),
        ).to(self.device)
        m.eval()
        return m, classes

    def load_weights(self, path: str) -> bool:
        """
        Try to load checkpoint weights. Returns True on success.
        On any failure (missing file, architecture mismatch, class-count
        mismatch) prints a clear diagnostic, leaves the predictor in
        rule-based mode and returns False — never raises.

        Rebuilds the model to match the checkpoint's saved arch, so
        non-default hidden sizes (e.g. Round 2's hidden=192) load cleanly.
        """
        try:
            ckpt = torch.load(path, map_location=self.device)
        except FileNotFoundError:
            print(f"[TemporalPredictor] Checkpoint not found: {path}")
            return False
        except Exception as e:
            print(f"[TemporalPredictor] Failed to read checkpoint "
                  f"{path}: {e}")
            return False

        if not isinstance(ckpt, dict):
            print(f"[TemporalPredictor] Unexpected checkpoint format at {path}")
            return False

        try:
            model, classes = self._build_model_from_ckpt(ckpt)
            state = ckpt.get('model_state_dict', ckpt)
            model.load_state_dict(state, strict=True)
            self.model         = model
            self._classes      = classes
            self.ensemble_models = []
            self._weights_loaded = True
            print(f"[TemporalPredictor] Loaded weights from {path} "
                  f"(classes={classes}, val_acc={ckpt.get('val_acc', -1.0):.1f}%)")
            return True
        except RuntimeError as e:
            print(f"[TemporalPredictor] Checkpoint at {path} does not "
                  f"match the current model architecture.")
            msg = str(e)
            head = msg.splitlines()[0] if msg else ''
            print(f"[TemporalPredictor]   {head}")
            print("[TemporalPredictor] Falling back to rule-based intent.")
            self._weights_loaded = False
            return False

    def load_ensemble(self, paths: list) -> bool:
        """
        Load multiple checkpoints for ensemble inference. predict() averages
        softmax across all loaded models. Returns True if >= 1 loaded.
        All checkpoints must share the same class list (sanity check).
        """
        loaded = []
        ref_classes = None
        for p in paths:
            try:
                ckpt = torch.load(p, map_location=self.device)
                model, classes = self._build_model_from_ckpt(ckpt)
                model.load_state_dict(
                    ckpt.get('model_state_dict', ckpt), strict=True)
                if ref_classes is None:
                    ref_classes = classes
                elif classes != ref_classes:
                    print(f"[TemporalPredictor] Class mismatch in {p}: "
                          f"{classes} vs {ref_classes} — skipping")
                    continue
                loaded.append((p, model, float(ckpt.get('val_acc', -1.0))))
            except Exception as e:
                print(f"[TemporalPredictor] Could not load {p}: {e}")
                continue

        if not loaded:
            print("[TemporalPredictor] Ensemble: no checkpoints loaded.")
            self._weights_loaded = False
            return False

        self.ensemble_models = [m for _, m, _ in loaded]
        self._classes        = ref_classes or list(INTENT_CLASSES)
        self._weights_loaded = True
        print(f"[TemporalPredictor] Ensemble: loaded {len(loaded)} checkpoints")
        for p, _, acc in loaded:
            print(f"  - {p} (val_acc={acc:.1f}%)")
        return True

    def set_scene_resolution(self, w: int, h: int):
        """Update scene dimensions for correct feature normalization."""
        self.extractor.scene_w = w
        self.extractor.scene_h = h

    def reset(self):
        self.feature_buffer.clear()
