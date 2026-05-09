"""
Run Session — per-execution output bundle.

Every invocation of inference.py creates one RunSession that owns a
single output directory:

    runs/run_YYYYMMDD_HHMMSS_<mode>/
        composite_video.mp4          ← visualization video (with sidebar/HUD)
        scene_video.mp4              ← raw scene feed (optional)
        gaze_affordance_map.png      ← cumulative density heatmap
        gaze_affordance_overlay.png  ← heatmap blended on last scene frame
        warnings.csv                 ← every fired driver warning
        voice_log.json               ← every spoken assistant prompt
        session_log/                 ← rolling JSON batches (frame data)
        summary.json                 ← machine-readable run summary
        summary.md                   ← human-readable run report

Construct one and pass it to the pipeline; on shutdown, call finalize().
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


_SEV_RANK = {'critical': 0, 'high': 1, 'advisory': 2, 'coaching': 3}


class RunSession:
    """
    Tracks per-run output paths and aggregates session statistics.
    The pipeline calls update() on every frame; finalize() writes
    artefacts to disk and returns a summary dict.
    """

    def __init__(self,
                 mode: str,
                 root: str = 'runs',
                 model_path: Optional[str] = None,
                 driver_source: Optional[object] = None,
                 scene_source:  Optional[object] = None):

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        slug = mode.replace(' ', '_').lower()
        self.run_id = f"{ts}_{slug}"
        self.dir: Path = Path(root) / f"run_{self.run_id}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / 'session_log').mkdir(exist_ok=True)

        self.start_time = time.time()
        self.mode       = mode
        self.metadata = {
            'mode':           mode,
            'model':          model_path or 'rule-based',
            'driver_source':  str(driver_source),
            'scene_source':   str(scene_source),
            'started_at':     datetime.now().isoformat(timespec='seconds'),
            'platform':       _platform_short(),
        }

        # Aggregate counters
        self._frame_count       = 0
        self._intent_counts:    Counter = Counter()
        self._risk_counts:      Counter = Counter()
        self._affordance_counts: Counter = Counter()
        self._warning_count = 0
        self._warning_log: List[Dict] = []
        self._sev_counts:  Counter = Counter()
        self._gaze_miss_count = 0
        self._fixation_frames = 0
        self._gaze_object_dwell: Dict[str, int] = defaultdict(int)
        self._fps_samples: List[float] = []
        self._last_scene_frame: Optional[np.ndarray] = None

        # Paths
        self.composite_video_path = str(self.dir / 'composite_video.mp4')
        self.scene_video_path     = str(self.dir / 'scene_video.mp4')
        self.heatmap_path         = str(self.dir / 'gaze_affordance_map.png')
        self.overlay_path         = str(self.dir / 'gaze_affordance_overlay.png')
        self.warnings_csv_path    = str(self.dir / 'warnings.csv')
        self.voice_log_path       = str(self.dir / 'voice_log.json')
        self.summary_json_path    = str(self.dir / 'summary.json')
        self.summary_md_path      = str(self.dir / 'summary.md')

        print(f"[RunSession] Output directory: {self.dir.resolve()}")

    # ------------------------------------------------------------------ #
    #  Per-frame ingestion                                                 #
    # ------------------------------------------------------------------ #

    def update(self,
               frame_data: Dict,
               scene_frame: Optional[np.ndarray] = None,
               fps: float = 0.0) -> None:
        self._frame_count += 1

        if scene_frame is not None:
            self._last_scene_frame = scene_frame

        if fps > 0:
            self._fps_samples.append(fps)
            if len(self._fps_samples) > 600:
                self._fps_samples.pop(0)

        # Intent
        ip = frame_data.get('intent_prediction')
        if ip and ip.get('intent'):
            self._intent_counts[ip['intent']] += 1

        # Object risks
        for det in frame_data.get('detected_objects', []) or []:
            r = det.get('risk_level', 'low')
            self._risk_counts[r] += 1
            a = det.get('affordance', 'Unknown')
            self._affordance_counts[a] += 1

        # Gaze
        gd = frame_data.get('gaze_data')
        if gd and gd.get('is_fixation'):
            self._fixation_frames += 1

        ga = frame_data.get('gaze_affordance')
        if ga and ga.get('looked_object'):
            cls = ga['looked_object'].get('class', '?')
            self._gaze_object_dwell[cls] += 1

        # Warnings
        warns = frame_data.get('warnings') or []
        for w in warns:
            sev = w.get('severity', 'advisory')
            self._sev_counts[sev] += 1
            self._warning_count   += 1
            if w.get('gaze_miss'):
                self._gaze_miss_count += 1
            self._warning_log.append({
                'frame':      self._frame_count,
                'time':       round(time.time() - self.start_time, 2),
                'severity':   sev,
                'class':      w.get('class'),
                'track_id':   w.get('track_id'),
                'affordance': w.get('affordance'),
                'message':    w.get('message'),
                'gaze_miss':  bool(w.get('gaze_miss')),
            })

    # ------------------------------------------------------------------ #
    #  Finalization                                                        #
    # ------------------------------------------------------------------ #

    def finalize(self,
                 gaze_map=None,
                 voice_assistant=None) -> Dict:
        """
        Write all output artefacts. Returns the in-memory summary dict.
        Safe to call multiple times.
        """
        duration = time.time() - self.start_time

        # ── Gaze affordance heatmap dumps ──────────────────────────────
        heatmap_saved, overlay_saved = False, False
        if gaze_map is not None:
            try:
                snap = gaze_map.get_accumulator_snapshot()
                heatmap_saved = gaze_map.save_heatmap(
                    self.heatmap_path, accumulator=snap, legend=True)
                if self._last_scene_frame is not None:
                    overlay_saved = gaze_map.save_overlay(
                        self.overlay_path, self._last_scene_frame,
                        accumulator=snap)
            except Exception as e:
                print(f"[RunSession] Heatmap dump failed: {e}")

        # ── Warnings CSV ───────────────────────────────────────────────
        try:
            with open(self.warnings_csv_path, 'w', newline='',
                      encoding='utf-8') as f:
                cols = ['frame', 'time', 'severity', 'class', 'track_id',
                        'affordance', 'gaze_miss', 'message']
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for row in self._warning_log:
                    w.writerow({k: row.get(k, '') for k in cols})
        except Exception as e:
            print(f"[RunSession] Warnings CSV failed: {e}")

        # ── Voice log ──────────────────────────────────────────────────
        # Always write the file (even empty) so downstream tooling has a
        # stable contract — every run produces voice_log.json.
        history = []
        if voice_assistant is not None:
            try:
                history = list(voice_assistant.history)
            except Exception as e:
                print(f"[RunSession] Voice history read failed: {e}")
        spoken_count = len(history)
        try:
            with open(self.voice_log_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, default=str)
        except Exception as e:
            print(f"[RunSession] Voice log dump failed: {e}")

        # ── Summary ────────────────────────────────────────────────────
        avg_fps = float(np.mean(self._fps_samples)) if self._fps_samples else 0.0
        gaze_miss_pct = (
            100.0 * self._gaze_miss_count / max(1, self._warning_count)
        )
        fixation_pct = (
            100.0 * self._fixation_frames / max(1, self._frame_count)
        )

        summary = {
            **self.metadata,
            'duration_sec':         round(duration, 1),
            'frame_count':          self._frame_count,
            'avg_fps':              round(avg_fps, 1),
            'intent_distribution':  dict(self._intent_counts),
            'risk_distribution':    dict(self._risk_counts),
            'affordance_distribution': dict(self._affordance_counts),
            'gaze_object_dwell_frames': dict(self._gaze_object_dwell),
            'fixation_pct':         round(fixation_pct, 1),
            'warnings_total':       self._warning_count,
            'warnings_by_severity': dict(self._sev_counts),
            'gaze_miss_count':      self._gaze_miss_count,
            'gaze_miss_pct':        round(gaze_miss_pct, 1),
            'voice_prompts_spoken': spoken_count,
            'artefacts': {
                'composite_video':   str(self.composite_video_path),
                'gaze_heatmap':      self.heatmap_path if heatmap_saved else None,
                'gaze_overlay':      self.overlay_path if overlay_saved else None,
                'warnings_csv':      self.warnings_csv_path,
                'voice_log':         self.voice_log_path,
            },
        }

        try:
            with open(self.summary_json_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as e:
            print(f"[RunSession] summary.json failed: {e}")

        try:
            self._write_markdown_summary(summary)
        except Exception as e:
            print(f"[RunSession] summary.md failed: {e}")

        print(f"[RunSession] Run complete: {self._frame_count} frames, "
              f"{duration:.1f}s, {self._warning_count} warnings.")
        print(f"[RunSession] Artefacts in: {self.dir.resolve()}")
        return summary

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def get_log_dir(self) -> str:
        """Path passed to RollingLogger so frame batches land here."""
        return str(self.dir / 'session_log')

    def _write_markdown_summary(self, s: Dict) -> None:
        lines = [
            f"# Driver Intent Monitor — Run Report",
            "",
            f"- **Mode:** `{s['mode']}`",
            f"- **Started:** {s['started_at']}",
            f"- **Duration:** {s['duration_sec']} s",
            f"- **Frames processed:** {s['frame_count']} "
            f"(avg {s['avg_fps']} FPS)",
            f"- **Model:** `{s['model']}`",
            "",
            f"## Driver Attention",
            f"- Fixation frames: **{s['fixation_pct']}%**",
            f"- Voice prompts issued: **{s['voice_prompts_spoken']}**",
            "",
            f"## Warnings",
            f"- Total: **{s['warnings_total']}**",
        ]
        for sev in ('critical', 'high', 'advisory'):
            lines.append(f"- {sev.title()}: "
                         f"**{s['warnings_by_severity'].get(sev, 0)}**")
        lines.append(f"- Gaze-miss rate: **{s['gaze_miss_pct']}%** "
                     f"(driver was not looking at the warned object)")

        if s['intent_distribution']:
            lines += ["", f"## Intent Distribution"]
            total = sum(s['intent_distribution'].values()) or 1
            for k, v in sorted(s['intent_distribution'].items(),
                               key=lambda kv: -kv[1]):
                lines.append(f"- {k}: {v} frames "
                             f"({100*v/total:.1f}%)")

        if s['gaze_object_dwell_frames']:
            lines += ["", f"## Gaze Dwell on Objects"]
            for k, v in sorted(s['gaze_object_dwell_frames'].items(),
                               key=lambda kv: -kv[1]):
                lines.append(f"- {k}: {v} frames")

        lines += [
            "",
            f"## Artefacts",
            f"- Composite video: `{Path(s['artefacts']['composite_video']).name}`",
        ]
        if s['artefacts']['gaze_heatmap']:
            lines.append(f"- Gaze affordance heatmap: "
                         f"`{Path(s['artefacts']['gaze_heatmap']).name}`")
        if s['artefacts']['gaze_overlay']:
            lines.append(f"- Heatmap overlay (last scene): "
                         f"`{Path(s['artefacts']['gaze_overlay']).name}`")
        lines.append(f"- Warnings CSV: "
                     f"`{Path(s['artefacts']['warnings_csv']).name}`")
        lines.append(f"- Voice log: "
                     f"`{Path(s['artefacts']['voice_log']).name}`")

        with open(self.summary_md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")


def _platform_short() -> str:
    import platform
    return f"{platform.system()} {platform.release()} / Py{platform.python_version()}"
