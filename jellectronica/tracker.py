"""
Jellyfish Tracker — Persistent tracking with position smoothing

Port of trackJellyfish() from main.ts with improvements:
- Exponential moving average (EMA) on positions for smooth movement
- Larger grace period for lost tracks (15 frames)
- Stable IDs across frames
- Cell transition detection triggers notes
"""

import math
import time

# ── Grid config (must match music_engine) ──────────────────────
GRID_COLS = 8
GRID_ROWS = 4
TRACK_MAX_DIST = 0.20       # normalized distance threshold (increased for persistence)
SMOOTHING = 0.4              # EMA factor (0=no smoothing, 1=ignore new data)
LOST_GRACE_FRAMES = 15       # Keep lost tracks alive for this many frames


def position_to_cell(x: float, y: float) -> tuple[int, int]:
    """Map normalized (0-1) position to grid cell (row, col)."""
    col = min(GRID_COLS - 1, max(0, int(x * GRID_COLS)))
    row = min(GRID_ROWS - 1, max(0, int(y * GRID_ROWS)))
    return row, col


class TrackedJellyfish:
    __slots__ = ("id", "x", "y", "smooth_x", "smooth_y",
                 "cell_row", "cell_col", "age", "confidence", "bbox")

    def __init__(self, jf_id: int, x: float, y: float, row: int, col: int,
                 confidence: float = 0.0, bbox: dict | None = None):
        self.id = jf_id
        self.x = x                 # raw position
        self.y = y
        self.smooth_x = x          # smoothed position
        self.smooth_y = y
        self.cell_row = row
        self.cell_col = col
        self.age = 0               # frames since last matched
        self.confidence = confidence
        self.bbox = bbox or {"x1": 0, "y1": 0, "x2": 0, "y2": 0}


class Tracker:
    """Track jellyfish across frames with smoothing and persistence."""

    def __init__(self):
        self._next_id = 0
        self._tracked: list[TrackedJellyfish] = []

    def update(self, detections: list[dict]) -> list[tuple[int, int]]:
        """
        Update tracks with new detections.
        Returns list of (row, col) cell transitions that should trigger notes.
        """
        unmatched = list(detections)
        updated: list[TrackedJellyfish] = []
        triggers: list[tuple[int, int]] = []

        # Sort existing tracks by confidence (match best ones first)
        sorted_tracks = sorted(self._tracked, key=lambda t: t.confidence, reverse=True)

        for jf in sorted_tracks:
            best_idx = -1
            best_dist = TRACK_MAX_DIST

            for i, det in enumerate(unmatched):
                cx = det["centroid"]["x"]
                cy = det["centroid"]["y"]
                # Use smoothed position for matching (more stable)
                dx = cx - jf.smooth_x
                dy = cy - jf.smooth_y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                det = unmatched.pop(best_idx)
                cx = det["centroid"]["x"]
                cy = det["centroid"]["y"]

                # Smooth position with EMA
                sx = jf.smooth_x * SMOOTHING + cx * (1 - SMOOTHING)
                sy = jf.smooth_y * SMOOTHING + cy * (1 - SMOOTHING)

                row, col = position_to_cell(sx, sy)

                # Cell transition → trigger note
                if row != jf.cell_row or col != jf.cell_col:
                    triggers.append((row, col))

                track = TrackedJellyfish(jf.id, cx, cy, row, col,
                                        det["confidence"], det["bbox"])
                track.smooth_x = sx
                track.smooth_y = sy
                updated.append(track)
            else:
                # Lost — keep alive with grace period, fade position
                jf.age += 1
                if jf.age < LOST_GRACE_FRAMES:
                    jf.confidence *= 0.85  # Fade confidence
                    updated.append(jf)

        # New detections → new tracks + initial trigger
        for det in unmatched:
            cx = det["centroid"]["x"]
            cy = det["centroid"]["y"]
            row, col = position_to_cell(cx, cy)
            triggers.append((row, col))
            updated.append(TrackedJellyfish(
                self._next_id, cx, cy, row, col,
                det["confidence"], det["bbox"]
            ))
            self._next_id += 1

        self._tracked = updated
        return triggers

    @property
    def count(self) -> int:
        """Count only actively matched tracks (not stale ones)."""
        return sum(1 for t in self._tracked if t.age == 0)

    @property
    def tracks(self) -> list[TrackedJellyfish]:
        """All tracked jellyfish (including stale ones fading out)."""
        return self._tracked
