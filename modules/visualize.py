"""
Visualization Dashboard (v2)
Single-window composite: header | scene + sidebar (driver cam, intent, status) | footer.
"""

import cv2
import numpy as np
from typing import Dict, Optional

from modules.gaze_affordance_map import LEGEND_ITEMS

# ── Public palette (BGR) ────────────────────────────────────────────────────
INTENT_COLORS = {
    'normal_forward':      (  0, 200,   0),
    'mirror_check':        (  0, 165, 255),
    'pedestrian_monitor':  (255, 165,   0),
    'lane_change_prepare': (230, 230,   0),
    'intersection_scan':   (255,   0, 200),
}

RISK_COLORS = {
    'low':      (  0, 210,   0),
    'medium':   (  0, 220, 200),
    'high':     (  0, 140, 255),
    'critical': (  0,   0, 255),
}

# ── Layout constants (exported so inference.py can size the VideoWriter) ────
SIDEBAR_W = 300
HEADER_H  = 44
FOOTER_H  = 54

# ── Internal palette ────────────────────────────────────────────────────────
_BG     = ( 28,  25,  22)
_PANEL  = ( 42,  38,  35)
_BAR    = ( 16,  14,  12)
_BORDER = ( 65,  62,  55)
_BORDA  = (200, 170, 100)   # active border
_PRI    = (245, 242, 235)
_SEC    = (158, 152, 132)
_DIM    = ( 95,  90,  75)
_ACCENT = (255, 200, 100)   # light-blue accent (BGR)
_PAD    = 8
_GAP    = 6


# ═══════════════════════════════════════════════════════════════════════════ #
#  Main compositor
# ═══════════════════════════════════════════════════════════════════════════ #

def visualize_frame(
    scene_frame:  np.ndarray,
    frame_data:   Dict,
    driver_frame: Optional[np.ndarray] = None,
    fps:          float = 0.0,
    show_gaze_map: bool = True,
) -> np.ndarray:
    sh, sw = scene_frame.shape[:2]
    tw = sw + SIDEBAR_W
    th = HEADER_H + sh + FOOTER_H

    canvas = np.full((th, tw, 3), _BG, dtype=np.uint8)

    # ── Header ──────────────────────────────────────────────────────────── #
    _draw_header(canvas, tw, fps, frame_data.get('frame_number', 0),
                 frame_data.get('intent_prediction'), sw)

    # ── Scene area (with overlays baked onto a copy) ─────────────────────  #
    sc = scene_frame.copy()
    if show_gaze_map:
        gmap_overlay = frame_data.get('gaze_map_overlay')
        if gmap_overlay is not None:
            sc = _blend_affordance_map(sc, gmap_overlay)
            sc = _draw_affordance_legend(sc)
    sc = _draw_objects(sc, frame_data.get('detected_objects', []))
    sc = _draw_gaze(sc, frame_data.get('gaze_data'),
                    frame_data.get('gaze_affordance'))
    warnings = frame_data.get('warnings', [])
    if warnings:
        sc = _draw_warning_banner(sc, warnings, frame_data.get('frame_number', 0))
    ip = frame_data.get('intent_prediction')
    if ip and not ip.get('ready', True) and not ip.get('rule_based', False):
        sc = _warmup_overlay(sc, ip.get('warmup_progress', 0.0))
    canvas[HEADER_H:HEADER_H + sh, :sw] = sc
    cv2.line(canvas, (sw, HEADER_H), (sw, HEADER_H + sh), _BORDA, 1)

    # ── Sidebar ──────────────────────────────────────────────────────────  #
    sx  = sw + _PAD
    sy  = HEADER_H + _PAD
    spw = SIDEBAR_W - _PAD * 2          # panel outer width
    max_y = HEADER_H + sh - _PAD        # sidebar must not exceed this

    sy = _driver_panel (canvas, sx, sy, spw, driver_frame,
                        frame_data.get('gaze_data'), max_y)
    sy += _GAP
    sy = _intent_panel (canvas, sx, sy, spw,
                        frame_data.get('intent_prediction'), max_y)
    sy += _GAP
    sy = _warnings_panel(canvas, sx, sy, spw, warnings, max_y)
    sy += _GAP
    _info_panel(canvas, sx, sy, spw, frame_data, max_y - sy)

    # ── Footer ───────────────────────────────────────────────────────────  #
    _draw_footer(canvas, sw, tw, th, frame_data)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════ #
#  Header
# ═══════════════════════════════════════════════════════════════════════════ #

def _draw_header(canvas, total_w, fps, frame_num, intent_pred, scene_w):
    canvas[:HEADER_H, :] = _BAR
    cv2.line(canvas, (0, HEADER_H - 1), (total_w, HEADER_H - 1), _BORDER, 1)

    cv2.putText(canvas, "DRIVER INTENT MONITOR",
                (_PAD + 4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, _ACCENT, 1, cv2.LINE_AA)

    if intent_pred and intent_pred.get('intent'):
        intent = intent_pred['intent']
        color  = INTENT_COLORS.get(intent, _PRI)
        conf   = intent_pred.get('confidence', 0.0)
        label  = f"{intent.replace('_',' ').upper()}  {conf:.0%}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
        mid = scene_w // 2
        cv2.putText(canvas, label, (mid - tw // 2, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)

    fps_c = (0, 200, 60) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 60, 255)
    rtext = f"FPS {fps:5.1f}   #{frame_num:6d}"
    (rw, _), _ = cv2.getTextSize(rtext, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    cv2.putText(canvas, rtext, (total_w - rw - _PAD - 4, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, fps_c, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Scene overlays (drawn on the scene copy before blitting)
# ═══════════════════════════════════════════════════════════════════════════ #

def _draw_objects(frame, detections):
    for det in detections:
        bbox  = det['bbox']
        color = RISK_COLORS.get(det.get('risk_level', 'low'), (255, 255, 255))
        tid   = det.get('track_id', '?')
        dist  = det.get('distance_estimate')

        cv2.rectangle(frame,
                      (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), color, 2)

        label = f"#{tid} {det['class']}"
        if dist and dist != float('inf'):
            label += f" {dist:.0f}m"
        _bbox_label(frame, label, (bbox['x1'], bbox['y1'] - 4), color)

        vel = det.get('velocity', [0, 0])
        if vel and (abs(vel[0]) > 2 or abs(vel[1]) > 2):
            cx, cy = bbox['center_x'], bbox['center_y']
            cv2.arrowedLine(frame, (cx, cy),
                            (cx + int(vel[0] * 3), cy + int(vel[1] * 3)),
                            color, 2, tipLength=0.4)
    return frame


def _draw_gaze(frame, gaze_data, gaze_affordance):
    if not gaze_data or not gaze_data.get('gaze_point'):
        return frame

    gx, gy = gaze_data['gaze_point']
    is_fix = gaze_data.get('is_fixation', True)
    calib  = gaze_data.get('calibrated', False)
    color  = (220, 220, 0) if is_fix else (255, 80, 180)   # yellow | purple

    for dx0, dy0, dx1, dy1 in [(-24, 0, -7, 0), (7, 0, 24, 0),
                                (0, -24, 0, -7), (0, 7, 0, 24)]:
        cv2.line(frame, (gx + dx0, gy + dy0), (gx + dx1, gy + dy1),
                 color, 2, cv2.LINE_AA)
    cv2.circle(frame, (gx, gy), 5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, "CAL" if calib else "EST",
                (gx + 10, gy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    if gaze_affordance and gaze_affordance.get('looked_object'):
        obj  = gaze_affordance['looked_object']
        bbox = obj['bbox']
        dur  = gaze_affordance.get('gaze_duration', 0)
        cv2.rectangle(frame,
                      (bbox['x1'] - 5, bbox['y1'] - 5),
                      (bbox['x2'] + 5, bbox['y2'] + 5), color, 3)
        bx, by = bbox['x1'], bbox['y2'] + 8
        bw = min(int(dur / 60.0 * 100), 100)
        cv2.rectangle(frame, (bx, by), (bx + 100, by + 7), (40, 38, 32), -1)
        if bw > 0:
            cv2.rectangle(frame, (bx, by), (bx + bw, by + 7), color, -1)
        cv2.putText(frame, f"{dur}f", (bx + 104, by + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, _SEC, 1, cv2.LINE_AA)
    return frame


def _blend_affordance_map(frame: np.ndarray, bgra: np.ndarray) -> np.ndarray:
    """Alpha-blend the BGRA affordance overlay onto the scene frame."""
    if bgra.shape[:2] != frame.shape[:2]:
        return frame
    alpha = bgra[:, :, 3:].astype(np.float32) / 255.0   # (H,W,1)
    src   = bgra[:, :, :3].astype(np.float32)
    dst   = frame.astype(np.float32)
    out   = dst * (1.0 - alpha) + src * alpha
    return out.clip(0, 255).astype(np.uint8)


def _draw_affordance_legend(frame: np.ndarray) -> np.ndarray:
    """Draw a compact affordance colour legend in the bottom-right corner."""
    h, w    = frame.shape[:2]
    item_h  = 16
    title_h = 14
    pad     = 6
    swatch  = 10
    label_x = pad + swatch + 5
    legend_h = pad + title_h + len(LEGEND_ITEMS) * item_h + pad
    legend_w = 130
    x0 = w - legend_w - pad
    y0 = h - legend_h - pad

    # Semi-transparent background
    bg = frame.copy()
    cv2.rectangle(bg, (x0, y0), (x0 + legend_w, y0 + legend_h), _BAR, -1)
    cv2.addWeighted(bg, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + legend_w, y0 + legend_h), _BORDER, 1)

    cv2.putText(frame, "GAZE MAP", (x0 + pad, y0 + pad + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, _DIM, 1, cv2.LINE_AA)

    for i, (label, color) in enumerate(LEGEND_ITEMS):
        row_y = y0 + pad + title_h + i * item_h
        cy    = row_y + item_h // 2
        sx    = x0 + pad
        cv2.rectangle(frame, (sx, cy - swatch // 2),
                      (sx + swatch, cy + swatch // 2), color, -1)
        cv2.putText(frame, label, (x0 + label_x, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, _PRI, 1, cv2.LINE_AA)
    return frame


# ── Warning severity colours (BGR) ──────────────────────────────────────────
_WARN_COLORS = {
    'critical': (  0,   0, 220),
    'high':     (  0, 120, 255),
    'advisory': (180, 200,  30),
}


def _draw_warning_banner(frame: np.ndarray, warnings: list, frame_number: int) -> np.ndarray:
    """
    Draws stacked warning banners at the top of the scene frame.
    Critical banners flash on alternate 10-frame windows.
    Each banner shows the message and a GAZE MISS tag when the driver
    is not looking at the triggering object.
    """
    if not warnings:
        return frame

    banner_h = 26
    pad      = 6

    for i, w in enumerate(warnings[:3]):          # cap at 3 banners
        sev   = w['severity']
        color = _WARN_COLORS.get(sev, _SEC)
        y0    = i * (banner_h + 2)
        y1    = y0 + banner_h

        # Critical banners flash: visible on even 10-frame windows
        if sev == 'critical' and (frame_number // 10) % 2 == 1:
            color = tuple(max(0, c - 80) for c in color)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y0), (frame.shape[1], y1), color, -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # Severity tag
        sev_label = sev.upper()
        cv2.putText(frame, sev_label, (pad, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA)
        (sw, _), _ = cv2.getTextSize(sev_label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)

        # Message
        cv2.putText(frame, w['message'], (pad + sw + 10, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # GAZE MISS tag — driver is not looking at this object
        if w.get('gaze_miss'):
            miss_label = '[ GAZE MISS ]'
            (mw, _), _ = cv2.getTextSize(miss_label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.putText(frame, miss_label,
                        (frame.shape[1] - mw - pad, y0 + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def _warnings_panel(canvas, x, y, w, warnings: list, max_y) -> int:
    """Sidebar panel listing active warnings with affordance and severity."""
    if not warnings:
        return y

    LINE_H = 18
    n      = min(len(warnings), 4)
    ph     = _LBL_H + _PAD + n * LINE_H + _PAD
    if y + ph > max_y:
        return y

    _panel(canvas, x, y, w, ph, "WARNINGS")
    cy = y + _LBL_H + _PAD

    for warn in warnings[:n]:
        sev   = warn['severity']
        color = _WARN_COLORS.get(sev, _SEC)
        # Severity dot
        cv2.circle(canvas, (x + _PAD + 4, cy + 6), 4, color, -1)
        # Message text (truncated to fit panel width)
        msg = warn['message']
        cv2.putText(canvas, msg, (x + _PAD + 14, cy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, _PRI, 1, cv2.LINE_AA)
        # Affordance badge
        aff = warn.get('affordance', '')
        if aff:
            (aw, _), _ = cv2.getTextSize(aff, cv2.FONT_HERSHEY_SIMPLEX, 0.28, 1)
            ax = x + w - aw - _PAD - 2
            cv2.putText(canvas, aff, (ax, cy + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, _DIM, 1, cv2.LINE_AA)
        cy += LINE_H

    return y + ph


def _warmup_overlay(frame, progress):
    h, w = frame.shape[:2]
    bx, by = w - 264, 10
    cv2.rectangle(frame, (bx - 4, by - 4), (bx + 260, by + 72), _BAR, -1)
    cv2.rectangle(frame, (bx - 4, by - 4), (bx + 260, by + 72), _BORDA, 1)
    cv2.putText(frame, "INTENT MODEL",
                (bx, by + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, _SEC, 1, cv2.LINE_AA)
    cv2.putText(frame, "WARMING UP",
                (bx, by + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.70, _ACCENT, 2, cv2.LINE_AA)
    bar_w = int(min(progress, 1.0) * 252)
    cv2.rectangle(frame, (bx, by + 50), (bx + 252, by + 62), (40, 38, 32), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (bx, by + 50), (bx + bar_w, by + 62), _ACCENT, -1)
    cv2.putText(frame, f"{int(min(progress, 1) * 100)}%",
                (bx, by + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.40, _SEC, 1, cv2.LINE_AA)
    return frame


# ═══════════════════════════════════════════════════════════════════════════ #
#  Sidebar panels  (each receives outer x,y,w; returns outer bottom y)
# ═══════════════════════════════════════════════════════════════════════════ #

_LBL_H = 20   # section-label strip height inside each panel


def _driver_panel(canvas, x, y, w, driver_frame, gaze_data, max_y) -> int:
    if driver_frame is None:
        ph = _LBL_H + _PAD + 46 + _PAD
        if y + ph > max_y:
            return y
        _panel(canvas, x, y, w, ph, "DRIVER CAM")
        cv2.putText(canvas, "No camera feed",
                    (x + _PAD, y + _LBL_H + _PAD + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, _DIM, 1, cv2.LINE_AA)
        return y + ph

    dh, dw = driver_frame.shape[:2]
    scale  = min((w - _PAD * 2) / dw, 180 / dh)
    tw, th = int(dw * scale), int(dh * scale)
    thumb  = cv2.resize(driver_frame, (tw, th), interpolation=cv2.INTER_AREA)
    _annotate_driver(thumb, gaze_data)

    ph = _LBL_H + _PAD + th + _PAD
    if y + ph > max_y:
        return y
    _panel(canvas, x, y, w, ph, "DRIVER CAM")

    off_x = x + _PAD + ((w - _PAD * 2) - tw) // 2
    off_y = y + _LBL_H + _PAD
    canvas[off_y:off_y + th, off_x:off_x + tw] = thumb
    return y + ph


def _annotate_driver(thumb, gaze_data):
    h, w = thumb.shape[:2]
    if not gaze_data:
        cv2.putText(thumb, "No face detected", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 60, 200), 1, cv2.LINE_AA)
        return

    hp    = gaze_data.get('head_pose', {})
    yaw   = hp.get('yaw',   0.0)
    pitch = hp.get('pitch', 0.0)
    conf  = gaze_data.get('confidence', 0.0)
    is_fix = gaze_data.get('is_fixation', False)
    calib  = gaze_data.get('calibrated', False)
    speed  = gaze_data.get('gaze_speed', 0.0)

    strip_h = 40
    overlay = thumb.copy()
    cv2.rectangle(overlay, (0, h - strip_h), (w, h), (10, 8, 6), -1)
    cv2.addWeighted(overlay, 0.70, thumb, 0.30, 0, thumb)

    fix_c = (0, 200, 50) if is_fix else (0, 140, 255)
    cv2.putText(thumb, f"Y{yaw:+.0f}  P{pitch:+.0f}  C{conf:.2f}",
                (5, h - strip_h + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.34, _PRI, 1, cv2.LINE_AA)
    cv2.putText(thumb,
                f"{'FIX' if is_fix else 'SAC'}  spd:{speed:.0f}  {'[CAL]' if calib else '[EST]'}",
                (5, h - strip_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.34, fix_c, 1, cv2.LINE_AA)


def _intent_panel(canvas, x, y, w, intent_pred, max_y) -> int:
    BAR_W = w - _PAD * 2

    if not intent_pred:
        ph = _LBL_H + _PAD + 30 + _PAD
        if y + ph > max_y:
            return y
        _panel(canvas, x, y, w, ph, "INTENT")
        cv2.putText(canvas, "Initialising...",
                    (x + _PAD, y + _LBL_H + _PAD + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, _DIM, 1, cv2.LINE_AA)
        return y + ph

    # Warmup state
    if not intent_pred.get('ready', True) and not intent_pred.get('rule_based', False):
        ph = _LBL_H + _PAD + 20 + 16 + _PAD
        if y + ph > max_y:
            return y
        _panel(canvas, x, y, w, ph, "INTENT")
        cy = y + _LBL_H + _PAD
        cv2.putText(canvas, "WARMING UP",
                    (x + _PAD, cy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.60, _ACCENT, 2, cv2.LINE_AA)
        cy += 20
        progress = intent_pred.get('warmup_progress', 0.0)
        bw = int(min(progress, 1.0) * BAR_W)
        cv2.rectangle(canvas, (x + _PAD, cy), (x + _PAD + BAR_W, cy + 12), (35, 32, 28), -1)
        if bw > 0:
            cv2.rectangle(canvas, (x + _PAD, cy), (x + _PAD + bw, cy + 12), _ACCENT, -1)
        cv2.rectangle(canvas, (x + _PAD, cy), (x + _PAD + BAR_W, cy + 12), _BORDER, 1)
        return y + ph

    intent   = intent_pred.get('intent', '-')
    conf     = intent_pred.get('confidence', 0.0)
    uncertain = intent_pred.get('uncertain', False)
    probs    = intent_pred.get('probabilities', {})
    is_rb    = intent_pred.get('rule_based', False)
    color    = INTENT_COLORS.get(intent, _PRI)

    n_probs  = len(probs)
    ph = _LBL_H + _PAD + 26 + 16 + 14 + n_probs * 20 + _PAD

    # Shrink probability rows if panel would overflow
    while y + ph > max_y and n_probs > 0:
        n_probs -= 1
        ph -= 20

    if y + ph > max_y:
        return y

    _panel(canvas, x, y, w, ph, "INTENT")
    cy = y + _LBL_H + _PAD

    # Intent name
    disp = intent.replace('_', ' ').upper() + ('  ?' if uncertain else '')
    cv2.putText(canvas, disp, (x + _PAD, cy + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.66, color, 2, cv2.LINE_AA)
    cy += 26

    # Confidence bar
    bfill = int(min(conf, 1.0) * BAR_W)
    cv2.rectangle(canvas, (x + _PAD, cy), (x + _PAD + BAR_W, cy + 13), (35, 32, 28), -1)
    if bfill > 0:
        cv2.rectangle(canvas, (x + _PAD, cy), (x + _PAD + bfill, cy + 13), color, -1)
    cv2.rectangle(canvas, (x + _PAD, cy), (x + _PAD + BAR_W, cy + 13), _BORDER, 1)
    cy += 16

    suffix = "  rule-based" if is_rb else ""
    cv2.putText(canvas, f"{conf:.0%} confidence{suffix}",
                (x + _PAD, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34, _SEC, 1, cv2.LINE_AA)
    cy += 14

    for cls, prob in sorted(probs.items(), key=lambda kv: -kv[1])[:n_probs]:
        c  = INTENT_COLORS.get(cls, (100, 100, 100))
        mw = int(prob * BAR_W)
        cv2.rectangle(canvas, (x + _PAD, cy), (x + _PAD + BAR_W, cy + 9), (30, 28, 24), -1)
        if mw > 0:
            cv2.rectangle(canvas, (x + _PAD, cy), (x + _PAD + mw, cy + 9), c, -1)
        cv2.putText(canvas, f"{cls.replace('_',' ')}  {prob:.2f}",
                    (x + _PAD + 3, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.30, _PRI, 1, cv2.LINE_AA)
        cy += 20

    return y + ph


def _info_panel(canvas, x, y, w, frame_data, max_h):
    if max_h < _LBL_H + _PAD * 2 + 18:
        return

    gd   = frame_data.get('gaze_data') or {}
    ga   = frame_data.get('gaze_affordance') or {}
    objs = frame_data.get('detected_objects', [])
    lo   = ga.get('looked_object')

    tracking = bool(gd)
    calib    = gd.get('calibrated', False)

    rows = [
        ("Objects",  str(len(objs)),                          _PRI),
        ("Gaze",     "Tracking" if tracking else "No face",
                     (0, 200, 50) if tracking else (0, 60, 200)),
        ("Mode",     "[CAL]" if calib else "[EST]",
                     (0, 200, 50) if calib else (0, 140, 255)),
    ]
    if lo:
        rows.append(("Looking", f"{lo['class']} #{lo.get('track_id','?')}", _PRI))
        aff_risk = ga.get('risk_level', 'low')
        rows.append(("Afford",
                     f"{ga.get('affordance','-')} ({aff_risk})",
                     RISK_COLORS.get(aff_risk, _PRI)))
    if gd.get('head_pose'):
        hp = gd['head_pose']
        rows.append(("Head", f"Y{hp['yaw']:+.0f}  P{hp['pitch']:+.0f}", _PRI))

    LINE_H   = 18
    max_rows = max(0, (max_h - _LBL_H - _PAD * 2 - 4) // LINE_H)
    rows     = rows[:max_rows]
    if not rows:
        return

    ph = _LBL_H + _PAD + len(rows) * LINE_H + _PAD
    _panel(canvas, x, y, w, ph, "STATUS")

    cy = y + _LBL_H + _PAD
    for lbl, val, vc in rows:
        cv2.putText(canvas, lbl + ":", (x + _PAD, cy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, _SEC, 1, cv2.LINE_AA)
        cv2.putText(canvas, val, (x + _PAD + 62, cy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, vc, 1, cv2.LINE_AA)
        cy += LINE_H


# ═══════════════════════════════════════════════════════════════════════════ #
#  Footer
# ═══════════════════════════════════════════════════════════════════════════ #

def _draw_footer(canvas, scene_w, total_w, total_h, frame_data):
    fy = total_h - FOOTER_H
    canvas[fy:, :] = _BAR
    cv2.line(canvas, (0, fy), (total_w, fy), _BORDER, 1)

    objs = frame_data.get('detected_objects', [])
    ga   = frame_data.get('gaze_affordance') or {}
    gd   = frame_data.get('gaze_data') or {}
    lo   = ga.get('looked_object')

    risk_counts: Dict = {}
    for o in objs:
        r = o.get('risk_level', 'low')
        risk_counts[r] = risk_counts.get(r, 0) + 1

    segments = [
        ("OBJECTS",   str(len(objs)),                       _PRI),
        ("CRITICAL",  str(risk_counts.get('critical', 0)),  RISK_COLORS['critical']),
        ("HIGH RISK", str(risk_counts.get('high',     0)),  RISK_COLORS['high']),
        ("LOOKING",   (lo or {}).get('class', '-'),         _PRI),
        ("AFFORD",    ga.get('affordance', '-'),             _PRI),
        ("TRACKING",  "ON" if gd else "OFF",
                      (0, 200, 50) if gd else (0, 60, 200)),
    ]

    n     = len(segments)
    seg_w = scene_w // n
    for i, (lbl, val, vc) in enumerate(segments):
        cx = _PAD + i * seg_w
        cv2.putText(canvas, lbl, (cx, fy + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, _SEC, 1, cv2.LINE_AA)
        cv2.putText(canvas, val, (cx, fy + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, vc,  1, cv2.LINE_AA)
        if i < n - 1:
            dx = cx + seg_w - _PAD
            cv2.line(canvas, (dx, fy + 8), (dx, fy + FOOTER_H - 8), _BORDER, 1)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Low-level helpers
# ═══════════════════════════════════════════════════════════════════════════ #

def _panel(canvas, x, y, w, h, label=""):
    """Solid dark panel with border and a section-label strip."""
    h2, w2 = canvas.shape[:2]
    x2 = min(x + w, w2)
    y2 = min(y + h, h2)
    if x2 <= x or y2 <= y:
        return
    cv2.rectangle(canvas, (x, y),  (x2, y2), _PANEL,  -1)
    cv2.rectangle(canvas, (x, y),  (x2, y2), _BORDER,  1)
    if label:
        cv2.putText(canvas, label, (x + 5, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, _DIM, 1, cv2.LINE_AA)


def _bbox_label(frame, text, pos, color):
    (lw, lh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    x, y = pos
    cv2.rectangle(frame, (x, y - lh - 4), (x + lw + 6, y + 2), color, -1)
    cv2.putText(frame, text, (x + 3, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)
