"""
Gaze Affordance Map
Per-affordance Gaussian density accumulator that builds a spatiotemporal
heatmap of where the driver's gaze has dwelled and on what type of object.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple

# Affordance channel keys
_AFFORDANCES = ('Caution', 'Constraint', 'Instructional', 'Opportunity', 'Unknown', 'background')

# BGR colours per affordance
_COLORS: Dict[str, Tuple[int, int, int]] = {
    'Caution':       (  0, 165, 255),   # orange
    'Constraint':    (  0,  40, 210),   # red
    'Instructional': ( 30, 215, 215),   # yellow-cyan
    'Opportunity':   (200, 230,  30),   # lime-green
    'Unknown':       (130, 120, 100),   # grey
    'background':    (  0, 190,  55),   # green  (road / empty space)
}

# How much each risk level amplifies the Gaussian splat
_RISK_WEIGHTS: Dict[str, float] = {
    'critical': 1.00,
    'high':     0.70,
    'medium':   0.45,
    'low':      0.20,
}


class GazeAffordanceMap:
    """
    Maintains one float32 density channel per affordance category.

    update() — call every fusion frame (from the fusion worker thread).
    render() — call after update() in the same thread; returns BGRA uint8
               overlay ready to be alpha-blended onto the scene frame, or
               None when the map is still empty.
    """

    def __init__(self,
                 width:     int   = 1280,
                 height:    int   = 720,
                 sigma:     int   = 35,
                 decay:     float = 0.97,
                 max_alpha: float = 0.50):
        self.sigma     = sigma
        self.decay     = decay
        self.max_alpha = max_alpha
        self._kernel:  Optional[np.ndarray] = None
        self._init(width, height)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def set_scene_resolution(self, w: int, h: int) -> None:
        self._init(w, h)

    def update(self,
               gaze_point:  Optional[Tuple[int, int]],
               affordance:  Optional[str],
               risk_level:  str  = 'low',
               is_fixation: bool = True) -> None:
        """
        Call once per fusion tick.
        - Decays all channels by self.decay regardless of fixation state.
        - On fixation: splats a risk-weighted Gaussian on the matching channel.
        - Always also accumulates into a non-decaying channel for end-of-run
          cumulative heatmap dumps.
        """
        for ch in self._ch.values():
            ch *= self.decay

        if gaze_point is None or not is_fixation:
            return

        gx, gy = int(gaze_point[0]), int(gaze_point[1])
        if not (0 <= gx < self.w and 0 <= gy < self.h):
            return

        key    = affordance if affordance in self._ch else 'background'
        weight = _RISK_WEIGHTS.get(risk_level, 0.20)
        k      = self._build_kernel()
        ks     = k.shape[0]
        half   = ks // 2

        y1 = max(0,      gy - half)
        y2 = min(self.h, gy + half + 1)
        x1 = max(0,      gx - half)
        x2 = min(self.w, gx + half + 1)

        ky1 = y1 - (gy - half);  ky2 = ky1 + (y2 - y1)
        kx1 = x1 - (gx - half);  kx2 = kx1 + (x2 - x1)

        splat = k[ky1:ky2, kx1:kx2] * weight
        self._ch[key][y1:y2, x1:x2]      += splat
        self._cum_ch[key][y1:y2, x1:x2]  += splat
        self._cum_fixations              += 1

    def render(self) -> Optional[np.ndarray]:
        """
        Returns a BGRA uint8 ndarray (H, W, 4) where channel 3 is the
        per-pixel alpha (0–255), or None when the map is still empty.
        """
        total = sum(self._ch.values())      # (H, W) float32
        peak  = float(total.max())
        if peak < 0.02:
            return None

        inv_peak = 1.0 / (peak + 1e-8)

        # Weighted colour accumulation:  Σ (channel / peak) * color
        rgb = np.zeros((self.h, self.w, 3), dtype=np.float32)
        for aff, ch in self._ch.items():
            c    = np.array(_COLORS[aff], dtype=np.float32)
            norm = (ch * inv_peak)[:, :, np.newaxis]   # (H,W,1)
            rgb += norm * c

        # Rescale so the brightest pixel hits 255
        rgb_max = rgb.max()
        if rgb_max > 1e-4:
            rgb *= 255.0 / rgb_max

        # Alpha: proportional to total intensity, capped at max_alpha
        alpha_f  = np.clip(total * inv_peak * self.max_alpha, 0.0, self.max_alpha)
        alpha_u8 = (alpha_f * 255).astype(np.uint8)

        bgra = np.dstack([rgb.clip(0, 255).astype(np.uint8), alpha_u8])
        return bgra                         # (H, W, 4)

    # ------------------------------------------------------------------ #
    #  Persistence — dump heatmap state at end of a run                    #
    # ------------------------------------------------------------------ #

    def render_full(self,
                    accumulator: Optional[Dict[str, np.ndarray]] = None
                    ) -> Optional[np.ndarray]:
        """
        Like render() but ignores the alpha cap and produces a fully opaque
        BGR uint8 image suitable for saving as a standalone PNG.
        Pass an explicit accumulator to render a snapshot other than the
        live decaying one (e.g. the cumulative session-long density).
        """
        ch_dict = accumulator if accumulator is not None else self._ch
        total = sum(ch_dict.values())
        peak  = float(total.max())
        if peak < 1e-4:
            return None

        inv_peak = 1.0 / (peak + 1e-8)
        rgb = np.zeros((self.h, self.w, 3), dtype=np.float32)
        for aff, ch in ch_dict.items():
            c    = np.array(_COLORS[aff], dtype=np.float32)
            norm = (ch * inv_peak)[:, :, np.newaxis]
            rgb += norm * c

        rgb_max = rgb.max()
        if rgb_max > 1e-4:
            rgb *= 255.0 / rgb_max
        return rgb.clip(0, 255).astype(np.uint8)

    def save_heatmap(self,
                     path: str,
                     accumulator: Optional[Dict[str, np.ndarray]] = None,
                     legend: bool = True) -> bool:
        """Write a standalone heatmap PNG. Returns True on success."""
        img = self.render_full(accumulator)
        if img is None:
            return False
        if legend:
            img = _draw_legend_block(img)
        return bool(cv2.imwrite(path, img))

    def save_overlay(self,
                     path: str,
                     scene_frame: np.ndarray,
                     accumulator: Optional[Dict[str, np.ndarray]] = None,
                     alpha: float = 0.55) -> bool:
        """Blend the heatmap on a scene frame and write a PNG."""
        if scene_frame is None:
            return False
        heat = self.render_full(accumulator)
        if heat is None:
            return False
        if heat.shape[:2] != scene_frame.shape[:2]:
            heat = cv2.resize(heat, (scene_frame.shape[1], scene_frame.shape[0]))
        mask  = (heat.sum(axis=2) > 4).astype(np.float32)[:, :, None]
        blend = scene_frame.astype(np.float32) * (1 - alpha * mask) \
              + heat.astype(np.float32) * (alpha * mask)
        out = blend.clip(0, 255).astype(np.uint8)
        out = _draw_legend_block(out)
        return bool(cv2.imwrite(path, out))

    def get_accumulator_snapshot(self) -> Dict[str, np.ndarray]:
        """Return a copy of the current per-affordance density channels."""
        return {k: v.copy() for k, v in self._ch.items()}

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _init(self, w: int, h: int) -> None:
        self.w  = w
        self.h  = h
        self._ch: Dict[str, np.ndarray] = {
            a: np.zeros((h, w), dtype=np.float32) for a in _AFFORDANCES
        }
        # Non-decaying accumulator for end-of-run cumulative heatmap dump.
        self._cum_ch: Dict[str, np.ndarray] = {
            a: np.zeros((h, w), dtype=np.float32) for a in _AFFORDANCES
        }
        self._cum_fixations = 0
        self._kernel = None     # invalidate cached kernel on resize

    def _build_kernel(self) -> np.ndarray:
        if self._kernel is None:
            ks = self.sigma * 4 + 1
            k  = cv2.getGaussianKernel(ks, self.sigma)
            self._kernel = (k @ k.T).astype(np.float32)
        return self._kernel


# ── Legend metadata (exported so visualize.py can draw it) ──────────────────
LEGEND_ITEMS = [
    ('Caution',       _COLORS['Caution']),
    ('Constraint',    _COLORS['Constraint']),
    ('Instructional', _COLORS['Instructional']),
    ('Opportunity',   _COLORS['Opportunity']),
    ('Unknown',       _COLORS['Unknown']),
    ('Background',    _COLORS['background']),
]


def _draw_legend_block(img: np.ndarray) -> np.ndarray:
    """Burn the affordance colour legend into the bottom-right of an image."""
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    pad, item_h, swatch = 8, 18, 12
    title_h = 16
    legend_h = pad + title_h + len(LEGEND_ITEMS) * item_h + pad
    legend_w = 168
    x0 = w - legend_w - pad
    y0 = h - legend_h - pad
    if x0 < 0 or y0 < 0:
        return img

    # Translucent dark backdrop
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + legend_w, y0 + legend_h),
                  (16, 14, 12), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + legend_w, y0 + legend_h),
                  (90, 88, 80), 1)

    cv2.putText(img, "GAZE AFFORDANCE MAP",
                (x0 + pad, y0 + pad + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1, cv2.LINE_AA)

    for i, (label, color) in enumerate(LEGEND_ITEMS):
        cy = y0 + pad + title_h + i * item_h + item_h // 2
        cv2.rectangle(img,
                      (x0 + pad, cy - swatch // 2),
                      (x0 + pad + swatch, cy + swatch // 2),
                      color, -1)
        cv2.putText(img, label,
                    (x0 + pad + swatch + 6, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235),
                    1, cv2.LINE_AA)
    return img
