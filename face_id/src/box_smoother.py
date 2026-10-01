# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Frame-to-frame bounding-box smoothing for the on-screen overlay only.

Reduces visual jitter/flicker in the drawn rectangle (and anything derived
from it, like the face-card thumbnail crop) without touching the raw boxes
used for keypoint/embedding extraction or any enrollment/recognition slot
matching - those all keep using the original, frame-accurate detections.

Position (x, y) and size (w, h) are smoothed differently on purpose: a face
walking across frame has genuine velocity, so position uses a constant-
velocity predictor. Box size does not have real "velocity" - a stationary
face's box shouldn't be growing/shrinking - so applying the same velocity
model to size let raw measurement noise drive false size momentum and made
the box shape flicker worse, not better. Size is instead smoothed with plain
exponential averaging against the previous smoothed size, no prediction term.
"""

from __future__ import annotations

import numpy as np


class BoxSmoother:
    """Constant-velocity position smoothing + plain exponential size smoothing."""

    def __init__(
        self, alpha_position=0.45, alpha_size=0.25,
        max_missed_frames=2, max_center_distance=80.0,
    ):
        self.alpha_position = alpha_position
        self.alpha_size = alpha_size
        self.max_missed_frames = max_missed_frames
        self.max_center_distance = max_center_distance
        self._tracks = []  # each: {"pos": np.float32[2], "vel": np.float32[2], "size": np.float32[2], "missed": int}

    def clear(self):
        self._tracks = []

    def smooth(self, detections):
        """Return detections (same length/order) with jitter-smoothed boxes."""
        if not detections:
            self._tracks = []
            return []

        raw_positions = [np.asarray(box[:2], dtype=np.float32) for _label, _confidence, box in detections]
        raw_sizes = [np.asarray(box[2:4], dtype=np.float32) for _label, _confidence, box in detections]
        old_tracks = self._tracks
        predicted_positions = [track["pos"] + track["vel"] for track in old_tracks]

        # Greedy nearest-center association: repeatedly take the globally
        # closest still-unmatched (track, detection) pair within the distance
        # gate, one-to-one - this is the "nearest points" part.
        candidates = []
        for track_index, (track, predicted_pos) in enumerate(zip(old_tracks, predicted_positions)):
            predicted_center = predicted_pos + track["size"] / 2.0
            for box_index, (raw_pos, raw_size) in enumerate(zip(raw_positions, raw_sizes)):
                raw_center = raw_pos + raw_size / 2.0
                distance = float(np.linalg.norm(predicted_center - raw_center))
                if distance <= self.max_center_distance:
                    candidates.append((distance, track_index, box_index))
        candidates.sort(key=lambda item: item[0])

        matched_tracks, matched_boxes, assignments = set(), set(), []
        for _distance, track_index, box_index in candidates:
            if track_index in matched_tracks or box_index in matched_boxes:
                continue
            matched_tracks.add(track_index)
            matched_boxes.add(box_index)
            assignments.append((track_index, box_index))

        smoothed_boxes = [None] * len(detections)
        new_tracks = []

        # Matched: position uses velocity prediction (the "history based
        # velocity" part); size uses plain exponential averaging only,
        # deliberately with no velocity/prediction term.
        for track_index, box_index in assignments:
            track = old_tracks[track_index]
            new_pos = self.alpha_position * raw_positions[box_index] + (1.0 - self.alpha_position) * predicted_positions[track_index]
            new_size = self.alpha_size * raw_sizes[box_index] + (1.0 - self.alpha_size) * track["size"]
            new_tracks.append({"pos": new_pos, "vel": new_pos - track["pos"], "size": new_size, "missed": 0})
            smoothed_boxes[box_index] = np.concatenate([new_pos, new_size])

        # Unmatched detections: brand-new track, nothing to smooth against yet.
        for box_index in range(len(detections)):
            if box_index in matched_boxes:
                continue
            new_tracks.append({
                "pos": raw_positions[box_index].copy(), "vel": np.zeros(2, dtype=np.float32),
                "size": raw_sizes[box_index].copy(), "missed": 0,
            })
            smoothed_boxes[box_index] = np.concatenate([raw_positions[box_index], raw_sizes[box_index]])

        # Unmatched tracks: keep alive (position/velocity intact) for a short
        # grace period to bridge a single missed detection frame, then drop -
        # never surfaced as an extra "ghost" box, just kept warm internally.
        for track_index, track in enumerate(old_tracks):
            if track_index in matched_tracks:
                continue
            missed = track["missed"] + 1
            if missed > self.max_missed_frames:
                continue
            new_tracks.append({
                "pos": predicted_positions[track_index], "vel": track["vel"],
                "size": track["size"], "missed": missed,
            })

        self._tracks = new_tracks
        return [
            (label, confidence, smoothed_box)
            for (label, confidence, _box), smoothed_box in zip(detections, smoothed_boxes)
        ]

