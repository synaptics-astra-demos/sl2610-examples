# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Persistent face embedding enrollment and cosine matching."""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class FaceDatabase:
    # No hardcoded cap: `embeddings` is a plain dict keyed by name, sized only by
    # how many identities have been enrolled. Nothing in this class or in
    # RecognitionTracker truncates it. The MAX_FACES constant in
    # batch_enrollment.py bounds simultaneous ENROLLMENT card slots only and is
    # never read from here.
    def __init__(self, path):
        self.path = Path(path)
        self.embeddings = {}
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.embeddings = {
                str(name): np.asarray(values, dtype=np.float32)
                for name, values in data.get("faces", {}).items()
            }

    def describe(self):
        """Return per-identity diagnostics: name, embedding dimension, and norm."""
        return [
            {"name": name, "dimension": int(vector.shape[0]), "norm": float(np.linalg.norm(vector))}
            for name, vector in self.embeddings.items()
        ]

    def log_summary(self):
        """Print a debug dump of the loaded database - identity count, dims, norms."""
        entries = self.describe()
        logger.info("[FACE DB] path=%s total enrolled identities=%d", self.path, len(entries))
        for entry in entries:
            logger.info(
                "[FACE DB] identity=%s embedding_dimension=%d norm=%.4f",
                entry["name"], entry["dimension"], entry["norm"],
            )

    def enroll(self, name, embedding):
        self.enroll_many(name, [embedding])

    def enroll_many(self, name, embeddings):
        vectors = [self._normalize(embedding) for embedding in embeddings]
        if not vectors:
            raise ValueError("Cannot enroll without embeddings")
        vector = self._normalize(np.mean(vectors, axis=0))
        self.embeddings[name] = vector
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"version": 1, "faces": {key: value.tolist() for key, value in self.embeddings.items()}},
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

    def match(self, embedding):
        if not self.embeddings:
            return None, 0.0
        vector = self._normalize(embedding)
        name, score = max(
            ((name, float(np.dot(vector, stored))) for name, stored in self.embeddings.items()),
            key=lambda item: item[1],
        )
        return name, score

    def match_top2(self, embedding):
        """Return (best_name, best_score, second_name, second_score) for diagnostics.

        second_name/second_score are None/-1.0 when fewer than two identities are enrolled.
        """
        if not self.embeddings:
            return None, 0.0, None, -1.0
        vector = self._normalize(embedding)
        scored = sorted(
            ((name, float(np.dot(vector, stored))) for name, stored in self.embeddings.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        best_name, best_score = scored[0]
        if len(scored) == 1:
            return best_name, best_score, None, -1.0
        second_name, second_score = scored[1]
        return best_name, best_score, second_name, second_score

    @staticmethod
    def _normalize(embedding):
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Cannot store or match a zero embedding")
        return vector / norm


class EmbeddingHistory:
    """Smooth live embeddings before matching against the database."""

    def __init__(self, size):
        self.values = deque(maxlen=max(int(size), 1))

    def add(self, embedding):
        self.values.append(np.asarray(embedding, dtype=np.float32).reshape(-1))
        return FaceDatabase._normalize(np.mean(self.values, axis=0))

    def clear(self):
        self.values.clear()


class RecognitionTracker:
    """Keep independent embedding histories and identity hysteresis per tracked face.

    Each track carries its own displayed label plus a pending-candidate label and
    consecutive-frame counter, so a single-frame confusion with another enrolled
    identity can never immediately override an already-stable label. Track
    continuity across frames requires BOTH box overlap (IoU) AND appearance
    continuity (the new embedding must resemble the track's own recent history) -
    box overlap alone is never enough to let a brand-new person inherit a
    previous person's stale label. Tracks not seen for several frames are pruned
    so a departed person's identity can never be "revived" by someone new
    standing in the same spot later.
    """

    # Cosine similarity a fresh detection's raw embedding must have against a
    # track's own recent (pre-update) smoothed embedding to be considered "the
    # same physical face" for track continuity. This is independent of, and
    # deliberately looser than, the enrolled-identity match threshold - it only
    # answers "is this plausibly still the same person as last frame", not
    # "who is this person".
    CONTINUITY_THRESHOLD = 0.35
    # Frames a track may go unmatched before being pruned, so a vacated track's
    # label/hold state can never be picked up by an unrelated later detection.
    STALE_TRACK_TTL = 5

    def __init__(
        self, window, hold_frames=8, iou_threshold=0.2, hysteresis=0.05,
        switch_confirm_frames=2, acquire_confirm_frames=5, min_margin=0.03, debug=False,
    ):
        self.window = window
        self.hold_frames = hold_frames
        self.iou_threshold = iou_threshold
        self.hysteresis = max(float(hysteresis), 0.0)
        self.switch_confirm_frames = max(int(switch_confirm_frames), 1)
        # A brand-new/never-tracked face needs MORE consecutive confirming frames
        # than switching between two already-known tracks: a genuinely unknown
        # person's borderline similarity to an enrolled identity tends to be
        # noisy frame-to-frame, while a real enrolled person's similarity stays
        # consistently high. Requiring more frames on first acquisition sharply
        # cuts false accepts without punishing already-confirmed identities.
        self.acquire_confirm_frames = max(int(acquire_confirm_frames), self.switch_confirm_frames)
        self.min_margin = max(float(min_margin), 0.0)
        self.debug = debug
        self.tracks = []
        self._next_track_id = 0
        self._frame_counter = 0

    def begin_frame(self):
        """Call once per frame before processing its detections.

        Advances the frame counter and prunes tracks that have not been matched
        recently, so a departed person's track/label can never be reused by an
        unrelated face that later appears in roughly the same screen position.
        """
        self._frame_counter += 1
        self.tracks = [
            track for track in self.tracks
            if self._frame_counter - track["last_seen"] <= self.STALE_TRACK_TTL
        ]

    def update(self, box, embedding, database, threshold):
        # Rank all tracks by box overlap, best first, then walk down that ranking
        # looking for the first candidate that also passes the appearance-
        # continuity gate. This matters when a stale track (rejected on
        # continuity) still has the best raw IoU: without trying the next-best
        # candidate too, the real continuing track for this face would never be
        # found and a brand-new track would be spawned every single frame.
        raw_vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        raw_vector = raw_vector / (np.linalg.norm(raw_vector) + 1e-6)

        ranked = sorted(
            (
                (self._iou(box, track["box"]), index)
                for index, track in enumerate(self.tracks)
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        best_index = -1
        for overlap, index in ranked:
            if overlap <= self.iou_threshold:
                break
            candidate = self.tracks[index]
            anchor = candidate.get("last_raw_embedding")
            if anchor is not None:
                # Compare against the single MOST RECENT frame, not the multi-frame
                # smoothed history average. A gradual change (e.g. walking farther
                # from the camera) drifts slowly frame-to-frame, so consecutive-frame
                # similarity stays high even though the average of several older,
                # closer-up frames would have drifted noticeably by now. Using the
                # smoothed average here previously caused legitimate, gradually
                # changing faces to trip the continuity gate and lose their track.
                continuity = float(np.dot(anchor, raw_vector))
                if continuity < self.CONTINUITY_THRESHOLD:
                    # Appearance-continuity gate: even with a good box-IoU match,
                    # only reuse the existing track (and therefore its
                    # label/hold state) if the new raw embedding actually
                    # resembles what that track has been seeing. This is what
                    # stops a different person who steps into the same screen
                    # position as a previous track from inheriting its identity.
                    if self.debug:
                        logger.debug(
                            "[RECOG] track continuity broken (iou=%.2f continuity=%.3f < %.3f) - "
                            "skipping reuse of track_id=%s",
                            overlap, continuity, self.CONTINUITY_THRESHOLD, candidate["track_id"],
                        )
                    continue
            best_index = index
            break

        if best_index < 0:
            track = {
                "track_id": self._next_track_id,
                "box": np.asarray(box, dtype=np.float32).copy(),
                "history": EmbeddingHistory(self.window),
                "label": None,
                "hold": 0,
                "pending_label": None,
                "pending_count": 0,
                "last_seen": self._frame_counter,
                "last_raw_embedding": None,
            }
            self._next_track_id += 1
            self.tracks.append(track)
        else:
            track = self.tracks[best_index]
            track["box"] = np.asarray(box, dtype=np.float32).copy()
            track["last_seen"] = self._frame_counter

        stable_embedding = track["history"].add(embedding)
        track["last_raw_embedding"] = raw_vector
        best_name, best_score, second_name, second_score = database.match_top2(stable_embedding)
        acquire_threshold = float(threshold)
        keep_threshold = max(0.0, acquire_threshold - self.hysteresis)
        margin = (best_score - second_score) if second_name is not None else 1.0
        matched_name, similarity = best_name, best_score
        confident = (
            matched_name is not None
            and similarity >= acquire_threshold
            and margin >= self.min_margin
        )

        displayed = track["label"]
        previous = displayed
        classification = "UNKNOWN"

        # Currently stable on `displayed` and the match still agrees (within hysteresis
        # slack): keep it, reset any pending-switch candidate.
        if displayed is not None and matched_name == displayed and similarity >= keep_threshold:
            track["hold"] = 0
            track["pending_label"] = None
            track["pending_count"] = 0
            result_name, classification = displayed, "KNOWN"
        # A different (or no) identity was matched this frame. Never switch on a
        # single frame - require several consecutive frames agreeing on the same
        # candidate before the displayed label changes.
        elif confident and matched_name != displayed:
            if track["pending_label"] == matched_name:
                track["pending_count"] += 1
            else:
                track["pending_label"] = matched_name
                track["pending_count"] = 1

            required_frames = self.switch_confirm_frames if displayed is not None else self.acquire_confirm_frames
            if track["pending_count"] >= required_frames:
                track["label"] = matched_name
                track["pending_label"] = None
                track["pending_count"] = 0
                track["hold"] = 0
                result_name, classification = matched_name, "KNOWN"
            elif displayed is not None:
                # Candidate not yet confirmed: keep showing the previous stable label.
                track["hold"] = min(track["hold"] + 1, self.hold_frames)
                result_name, classification = displayed, "PENDING_SWITCH"
            else:
                # First-ever candidate for this track, not yet confirmed for enough
                # consecutive frames - stay unlabeled rather than guess early.
                result_name, classification = None, "PENDING_ACQUIRE"
        else:
            # No confident candidate this frame: reset pending switch, apply hold/drop.
            track["pending_label"] = None
            track["pending_count"] = 0
            if displayed is None:
                result_name = None
                classification = "UNKNOWN" if matched_name is None or similarity < acquire_threshold else "AMBIGUOUS"
            elif track["hold"] < self.hold_frames:
                track["hold"] += 1
                result_name, classification = displayed, "HOLD"
            else:
                track["label"] = None
                track["hold"] = 0
                result_name, classification = None, "UNKNOWN"

        if self.debug:
            logger.debug(
                "[RECOG] frame=%d track_id=%d bbox=(%.0f,%.0f,%.0f,%.0f) candidates=%d "
                "best=%s(%.3f) second=%s(%.3f) threshold=%.3f margin=%.3f "
                "-> class=%s displayed=%s previous=%s",
                self._frame_counter, track["track_id"], box[0], box[1], box[2], box[3],
                len(database.embeddings),
                best_name, best_score, second_name, second_score if second_name is not None else -1.0,
                acquire_threshold, margin, classification, result_name, previous,
            )

        return result_name, similarity

    def clear(self):
        self.tracks.clear()

    @staticmethod
    def _iou(first, second):
        first = np.asarray(first, dtype=np.float32)
        second = np.asarray(second, dtype=np.float32)
        first_x2, first_y2 = first[0] + first[2], first[1] + first[3]
        second_x2, second_y2 = second[0] + second[2], second[1] + second[3]
        intersection_width = max(0.0, min(first_x2, second_x2) - max(first[0], second[0]))
        intersection_height = max(0.0, min(first_y2, second_y2) - max(first[1], second[1]))
        intersection = intersection_width * intersection_height
        union = first[2] * first[3] + second[2] * second[3] - intersection
        return float(intersection / union) if union > 0 else 0.0