"""
Temporally-consistent augmentation for dual-stream driving video.

Usage
-----
    from modules.augmentation import SpatioTemporalAugmenter

    augmenter = SpatioTemporalAugmenter().to(device)
    augmenter.train()

    for batch in train_loader:
        a = batch["stream_a"].to(device)        # (B, T, C, H, W), [0, 1]
        b = batch["stream_b"].to(device)
        y = batch["label"].to(device)

        a, b, y = augmenter(a, b, y)            # label is returned unchanged
        a = imagenet_normalize(a)               # apply AFTER augmentation
        b = imagenet_normalize(b)
        ...
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalAugmenter(nn.Module):
    """
    Inputs
    ------
    stream_a, stream_b : (B, T, C, H, W) float tensors in [0, 1]
    label              : (B,) long tensor -- returned UNCHANGED
                         (5-class taxonomy is direction-agnostic).

    Design
    ------
    * One set of augmentation params per sample, reused across all T frames
      to preserve motion / temporal coherence.
    * Cutout is a spatio-temporal *tube*: the same rectangular mask is
      applied to every frame of a window.
    * AugMix ops are photometric only -- per-frame geometric distortions
      would corrupt pose / depth cues.
    * FlipLR applies a pure spatial mirror. `sync_streams_flip=True` keeps
      the scene and cabin cameras spatially consistent with each other.

    Apply BEFORE ImageNet normalization.
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        cutout_prob: float = 0.5,
        cutout_size_frac: float = 0.15,
        cutout_n_holes: int = 1,
        augmix_prob: float = 0.5,
        augmix_width: int = 3,
        augmix_depth: int = -1,          # -1 -> random in [1, 3]
        augmix_severity: int = 3,        # 1..5
        augmix_alpha: float = 1.0,
        sync_streams_flip: bool = True,
    ):
        super().__init__()
        self.flip_prob = flip_prob
        self.cutout_prob = cutout_prob
        self.cutout_size_frac = cutout_size_frac
        self.cutout_n_holes = cutout_n_holes
        self.augmix_prob = augmix_prob
        self.augmix_width = augmix_width
        self.augmix_depth = augmix_depth
        self.augmix_severity = augmix_severity
        self.augmix_alpha = augmix_alpha
        self.sync_streams_flip = sync_streams_flip

        self._ops = (
            self._identity,
            self._brightness, self._contrast, self._saturation,
            self._gaussian_noise, self._sharpness,
            self._blur, self._posterize,
        )

    # -------- photometric ops: x is (T, C, H, W) in [0, 1] --------

    @staticmethod
    def _identity(x, _sev):
        return x

    @staticmethod
    def _brightness(x, sev):
        f = 1.0 + (sev / 10.0) * (random.random() * 2 - 1)
        return (x * f).clamp_(0, 1)

    @staticmethod
    def _contrast(x, sev):
        f = 1.0 + (sev / 10.0) * (random.random() * 2 - 1)
        m = x.mean(dim=(-3, -2, -1), keepdim=True)
        return ((x - m) * f + m).clamp_(0, 1)

    @staticmethod
    def _saturation(x, sev):
        f = 1.0 + (sev / 10.0) * (random.random() * 2 - 1)
        gray = x.mean(dim=-3, keepdim=True)
        return ((x - gray) * f + gray).clamp_(0, 1)

    @staticmethod
    def _gaussian_noise(x, sev):
        std = sev / 100.0
        return (x + torch.randn_like(x) * std).clamp_(0, 1)

    @staticmethod
    def _posterize(x, sev):
        bits = max(1, 8 - sev)
        levels = 2 ** bits
        return (torch.round(x * (levels - 1)) / (levels - 1)).clamp_(0, 1)

    @staticmethod
    def _blur(x, sev):
        k = 2 * max(1, sev // 2) + 1
        C = x.shape[1]
        kernel = torch.ones(C, 1, k, k, device=x.device, dtype=x.dtype) / (k * k)
        x_pad = F.pad(x, [k // 2] * 4, mode="reflect")
        return F.conv2d(x_pad, kernel, groups=C)

    @staticmethod
    def _sharpness(x, sev):
        f = 1.0 + (sev / 5.0) * random.random()
        C = x.shape[1]
        k = torch.ones(C, 1, 3, 3, device=x.device, dtype=x.dtype) / 9.0
        blurred = F.conv2d(F.pad(x, [1, 1, 1, 1], mode="reflect"), k, groups=C)
        return ((x - blurred) * f + blurred).clamp_(0, 1)

    # -------- per-sample primitives --------

    def _augmix_single(self, x):
        alpha = self.augmix_alpha
        ws = torch.distributions.Dirichlet(
            torch.full((self.augmix_width,), alpha)
        ).sample()
        m = torch.distributions.Beta(
            torch.tensor(alpha), torch.tensor(alpha)
        ).sample().item()

        mix = torch.zeros_like(x)
        for i in range(self.augmix_width):
            aug = x
            depth = (self.augmix_depth if self.augmix_depth > 0
                     else random.randint(1, 3))
            for _ in range(depth):
                op = random.choice(self._ops)
                aug = op(aug, self.augmix_severity)
            mix = mix + ws[i].item() * aug
        return ((1.0 - m) * x + m * mix).clamp_(0, 1)

    def _cutout_tube(self, x):
        _, _, H, W = x.shape
        hh = max(1, int(H * self.cutout_size_frac))
        hw = max(1, int(W * self.cutout_size_frac))
        x = x.clone()
        for _ in range(self.cutout_n_holes):
            cy = random.randint(0, H - 1)
            cx = random.randint(0, W - 1)
            y1, y2 = max(0, cy - hh // 2), min(H, cy + hh // 2)
            x1, x2 = max(0, cx - hw // 2), min(W, cx + hw // 2)
            x[:, :, y1:y2, x1:x2] = 0.0
        return x

    # -------- forward --------

    def forward(self, stream_a, stream_b, label):
        if not self.training:
            return stream_a, stream_b, label

        B = stream_a.shape[0]
        a_out = torch.empty_like(stream_a)
        b_out = torch.empty_like(stream_b)

        for i in range(B):
            a, b = stream_a[i], stream_b[i]

            # 1. FlipLR -- pure spatial mirror, labels UNCHANGED
            flip_a = random.random() < self.flip_prob
            flip_b = (flip_a if self.sync_streams_flip
                      else random.random() < self.flip_prob)
            if flip_a:
                a = torch.flip(a, dims=[-1])
            if flip_b:
                b = torch.flip(b, dims=[-1])

            # 2. AugMix (photometric, independent per stream)
            if random.random() < self.augmix_prob:
                a = self._augmix_single(a)
            if random.random() < self.augmix_prob:
                b = self._augmix_single(b)

            # 3. Cutout tube (independent per stream)
            if random.random() < self.cutout_prob:
                a = self._cutout_tube(a)
            if random.random() < self.cutout_prob:
                b = self._cutout_tube(b)

            a_out[i] = a
            b_out[i] = b

        return a_out, b_out, label
