"""
Feature-vector augmentation for intent training.

Operates on (T, 36) feature sequences as produced by `FeatureExtractor`.
All operations preserve the directional meaning of the data (no horizontal
flip — that would swap left/right semantics).

Feature layout (36-dim):
  gaze(5)        : x, y, vx, vy, is_fixation              [0..4]
  vehicle(6)     : speed_norm, delta_speed, steer_norm,
                   yaw_norm, is_braking, is_turning       [5..10]
  object(9)      : one-hot                                [11..19]
  affordance(5)  : one-hot                                [20..24]
  context(9)     : gaze_dur, dist, vx, vy, risk_l/m/h/c,
                   tl_red                                 [25..33]
  mask(2)        : has_scene, has_gaze                    [34..35]
"""

import random
import numpy as np


# Channel index groups (matching modules.temporal_model layout)
GAZE_IDX        = list(range(0, 5))
VEHICLE_IDX     = list(range(5, 11))
OBJECT_IDX      = list(range(11, 20))
AFFORDANCE_IDX  = list(range(20, 25))
CONTEXT_IDX     = list(range(25, 34))
MASK_IDX        = [34, 35]


# ── Individual ops ────────────────────────────────────────────────────────

def time_warp(seq: np.ndarray, factor_range=(0.85, 1.15)) -> np.ndarray:
    """
    Resample along time at a random speed factor.
    Linear interpolation. Preserves shape (T, F).
    """
    T, F = seq.shape
    factor = random.uniform(*factor_range)
    src_T = max(2, int(round(T / factor)))
    src_idx = np.linspace(0, T - 1, src_T)
    out_idx = np.linspace(0, src_T - 1, T)
    src = np.empty((src_T, F), dtype=seq.dtype)
    for i, s in enumerate(src_idx):
        a = int(s); b = min(T - 1, a + 1); w = s - a
        src[i] = (1 - w) * seq[a] + w * seq[b]
    out = np.empty_like(seq)
    for i, o in enumerate(out_idx):
        a = int(o); b = min(src_T - 1, a + 1); w = o - a
        out[i] = (1 - w) * src[a] + w * src[b]
    return out


def telemetry_jitter(seq: np.ndarray,
                     gaze_std: float = 0.01,
                     vehicle_std: float = 0.02) -> np.ndarray:
    """
    Calibration-drift simulation: small Gaussian noise on continuous channels.
    One-hot, mask, and is_fixation channels are left unchanged.
    """
    seq = seq.copy()
    # Continuous gaze channels: x, y, vx, vy (skip is_fixation at idx 4)
    seq[:, 0:4] += np.random.normal(0, gaze_std, (seq.shape[0], 4)).astype(seq.dtype)
    # Continuous vehicle channels: speed, dspeed, steer, yaw (skip is_braking,is_turning)
    seq[:, 5:9] += np.random.normal(0, vehicle_std, (seq.shape[0], 4)).astype(seq.dtype)
    # Context continuous: first 4 entries (duration, dist, obj_vx, obj_vy)
    seq[:, 25:29] += np.random.normal(0, vehicle_std, (seq.shape[0], 4)).astype(seq.dtype)
    return np.clip(seq, -1.0, 1.0)


def channel_dropout(seq: np.ndarray, prob: float = 0.10) -> np.ndarray:
    """
    Randomly zero an entire channel group for the whole window. Forces the
    model to reason from redundant features when one stream is missing.
    """
    seq = seq.copy()
    if random.random() < prob:
        # Drop scene stream entirely (objects + affordance + context + has_scene)
        seq[:, 11:35] = 0.0
        seq[:, 34] = 0.0
    elif random.random() < prob:
        # Drop gaze stream (gaze coords + has_gaze flag)
        seq[:, 0:5] = 0.0
        seq[:, 35] = 0.0
    elif random.random() < prob:
        # Drop vehicle stream (rare but useful for gaze-only inference paths)
        seq[:, 5:11] = 0.0
    return seq


def frame_mask(seq: np.ndarray, max_run: int = 8, prob: float = 0.30) -> np.ndarray:
    """
    Zero out a contiguous window of frames — simulates sensor dropout
    (camera glitch, frame drop). Encourages temporal robustness.
    """
    if random.random() >= prob:
        return seq
    T, _ = seq.shape
    run = random.randint(2, max_run)
    start = random.randint(0, max(0, T - run))
    seq = seq.copy()
    seq[start:start + run] = 0.0
    return seq


def temporal_shift(seq: np.ndarray, max_shift: int = 5) -> np.ndarray:
    """Zero-padded temporal shift (replaces train_intent._jitter shift logic)."""
    T, F = seq.shape
    shift = random.randint(-max_shift, max_shift)
    if shift > 0:
        return np.concatenate([np.zeros((shift, F), seq.dtype), seq[:-shift]])
    if shift < 0:
        return np.concatenate([seq[-shift:], np.zeros((-shift, F), seq.dtype)])
    return seq


# ── Composed pipeline ────────────────────────────────────────────────────

def augment_train(seq: np.ndarray,
                  enable: dict | None = None) -> np.ndarray:
    """
    Default training-time augmentation pipeline. Tuned conservative — fires
    each op with moderate probability so a window typically receives 1-2
    augmentations.

    Pass `enable={'time_warp': False, ...}` to disable individual ops for
    ablation.
    """
    enable = enable or {}

    if enable.get('temporal_shift', True):
        seq = temporal_shift(seq, max_shift=5)
    if enable.get('time_warp', True) and random.random() < 0.40:
        seq = time_warp(seq)
    if enable.get('telemetry_jitter', True):
        seq = telemetry_jitter(seq)
    if enable.get('channel_dropout', True):
        seq = channel_dropout(seq, prob=0.10)
    if enable.get('frame_mask', True):
        seq = frame_mask(seq, max_run=8, prob=0.20)
    return np.clip(seq.astype(np.float32), -1.0, 1.0)
