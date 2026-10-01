# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Batch enrollment state for up to five faces."""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


class BatchEnrollmentSession:
    MAX_FACES = 4
    DUPLICATE_SIMILARITY_THRESHOLD = 0.50
    PREVIEW_WINDOW = 5
    IOV_MATCHING_THRESHOLD = 0.3  # Increased from 0.2 for stricter matching

    def __init__(self, database, target_frames=10):
        self.database = database
        self.target_frames = max(int(target_frames), 1)
        self._lock = threading.RLock()
        self.status = "idle"
        self.message = "Ready"
        self.slots = []
        self.armed = False
        self.active_slot_id = None  # Use slot ID instead of index (stable across removals)
        self._next_slot_id = 0
        self._session_id = 0  # Bumped on every fresh Add Face/Register Faces flow; slots carry their creation session
        self._enrolled_names_cache = set()  # Cache of enrolled names to avoid repeated DB lookups
        self._cache_stale = True  # Flag to refresh cache when needed
        self._frames_since_refresh = 0  # Track frames since last refresh, to stabilize new slots
        self._embedding_validated = set()  # Track slot IDs that have been validated with embeddings

    def _is_face_already_registered(self, embedding):
        matched_name, similarity = self.database.match(embedding)
        if matched_name is None:
            return False, None
        if similarity >= self.DUPLICATE_SIMILARITY_THRESHOLD:
            return True, matched_name
        return False, None

    def _refresh_enrolled_names_cache(self):
        """Update cache of all enrolled names in database."""
        if self.database and hasattr(self.database, 'embeddings'):
            self._enrolled_names_cache = {name.casefold() for name in self.database.embeddings.keys()}
        self._cache_stale = False
        logger.debug(f"[CACHE] Enrolled names: {len(self._enrolled_names_cache)}")

    def _enrollable_slot_indices(self):
        """Indices of slots that are ready to enroll (validated, current session, not already registered)."""
        return [i for i, s in enumerate(self.slots)
                if not s.get("already_registered") and not s.get("enrolled")
                and s.get("session_id") == self._session_id  # Reject any slot leaked from a prior session
                and s.get("id") in self._embedding_validated]  # Only show if validated

    def _get_slot_by_id(self, slot_id):
        """Find slot by ID (safe across list removals)."""
        for slot in self.slots:
            if slot.get("id") == slot_id:
                return slot
        return None

    def _remove_enrolled_and_registered_faces(self):
        """Dynamically remove face cards that are already enrolled or already registered in the database."""
        if not self.slots:
            return
        
        original_count = len(self.slots)
        removed_ids = []
        
        # Track removed IDs
        for s in self.slots:
            if s.get("already_registered") or s.get("enrolled"):
                removed_ids.append(s.get("id"))
        
        # Keep only faces that are NOT already registered/enrolled
        self.slots = [s for s in self.slots if not s.get("already_registered") and not s.get("enrolled")]
        
        # Clean up validation tracking for removed slots
        for removed_id in removed_ids:
            self._embedding_validated.discard(removed_id)
        
        # If active enrollment target was removed, abort capture
        if self.active_slot_id in removed_ids:
            self.active_slot_id = None
            self.status = "selecting"
        
        removed_count = original_count - len(self.slots)
        if removed_count > 0:
            # Mark cache as stale since database enrollment happened
            self._cache_stale = True
            logger.info(f"[CLEANUP] Removed {removed_count} already registered/enrolled face card(s). Remaining: {len(self.slots)}")

    def _merge_duplicate_slots(self):
        """Aggressively merge slots with the same face - run this frequently during preview."""
        if len(self.slots) < 2:
            return
        
        merged_indices = set()
        for i in range(len(self.slots)):
            if i in merged_indices:
                continue
            slot_i = self.slots[i]
            
            # CRITICAL: Never merge the actively-capturing slot
            if slot_i.get("id") == self.active_slot_id:
                continue
                
            emb_i = slot_i.get("preview_embedding")
            
            # Skip slots without embeddings (yet)
            if emb_i is None:
                continue
            
            slot_i_name = slot_i.get("name", f"face_{slot_i.get('id')}")
            
            # Compare with all subsequent slots for duplicates
            for j in range(i + 1, len(self.slots)):
                if j in merged_indices:
                    continue
                slot_j = self.slots[j]
                slot_j_id = slot_j.get("id")
                
                # CRITICAL: Never merge the actively-capturing slot
                if slot_j_id == self.active_slot_id:
                    continue
                
                emb_j = slot_j.get("preview_embedding")
                if emb_j is None:
                    continue
                
                # Aggressive duplicate detection: 0.50 threshold (high confidence)
                similarity = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-6)
                if similarity >= 0.50:
                    merged_indices.add(j)
                    slot_j_name = slot_j.get("name", f"face_{slot_j_id}")
                    logger.info(f"[DEDUP] Removing duplicate: '{slot_j_name}' (ID:{slot_j_id}) = '{slot_i_name}' (ID:{slot_i.get('id')}) sim:{similarity:.4f}")
        
        # Remove merged slots in reverse order to avoid index shifting
        if merged_indices:
            for idx in sorted(merged_indices, reverse=True):
                removed_slot = self.slots.pop(idx)
            logger.info(f"[DEDUP] Merged {len(merged_indices)} duplicate(s). Remaining: {len(self.slots)}")

    def update_faces(self, detections):
        with self._lock:
            if not self.armed:
                return
            
            # CRITICAL: During capturing, ONLY update box positions of existing slots
            # DO NOT create new slots - this prevents stale slot accumulation
            if self.status == "capturing":
                if self.active_slot_id:
                    # Keep slot boxes current so IoU matching stays accurate during capture
                    # But never create new slots during active enrollment
                    for _label, _confidence, box in detections[:self.MAX_FACES]:
                        best, best_iou = None, 0.1
                        for i, slot in enumerate(self.slots):
                            iou = self._iou(box, slot["box"])
                            if iou > best_iou:
                                best_iou, best = iou, i
                        if best is not None:
                            self.slots[best]["box"] = box
                    logger.debug(f"[UPDATE] During capture: updated {len(self.slots)} slots, detected {len(detections)} faces, active_id={self.active_slot_id}")
                return
            
            # During PREVIEW (not capturing): Match and create/remove slots
            old_slots = self.slots
            detection_boxes = [box for _label, _confidence, box in detections[: self.MAX_FACES]]

            # Match detections to existing slots by IoU, update matched slots
            matched_slots = set()
            for detection_box in detection_boxes:
                best_slot_index = None
                best_iou = self.IOV_MATCHING_THRESHOLD
                for slot_index, slot in enumerate(old_slots):
                    if slot_index in matched_slots:
                        continue
                    iou = self._iou(detection_box, slot["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_slot_index = slot_index
                if best_slot_index is not None:
                    old_slots[best_slot_index]["box"] = detection_box
                    matched_slots.add(best_slot_index)

            # Remove slots that weren't matched to any detection
            self.slots = [slot for i, slot in enumerate(old_slots) if i in matched_slots]

            # Add new slots for unmatched detections
            for detection_box in detection_boxes:
                is_matched = any(self._iou(detection_box, slot["box"]) >= self.IOV_MATCHING_THRESHOLD for slot in self.slots)
                if not is_matched and len(self.slots) < self.MAX_FACES:
                    self.slots.append({
                        "id": self._next_slot_id,
                        "session_id": self._session_id,  # Tag with current session so stale slots can never render
                        "box": detection_box,
                        "name": "",
                        "vectors": [],
                        "count": 0,
                        "preview_embedding": None,
                        "preview_vectors": [],
                        "already_registered": False,
                        "registered_name": None,
                        "enrolled": False,
                    })
                    self._next_slot_id += 1

            # Dynamically remove already registered/enrolled faces and duplicates
            self._remove_enrolled_and_registered_faces()
            self._merge_duplicate_slots()

            # Update status message
            if len(detection_boxes) > self.MAX_FACES:
                self.message = "Maximum 5 faces supported"
            elif not self.slots:
                self.message = "Show faces to the camera"
            else:
                count = len(self.slots)
                self.message = f"{count} face{'s' if count != 1 else ''} registered"
            if len(detections) > self.MAX_FACES:
                self.message = "Maximum 5 faces supported"
            elif not self.slots:
                self.message = "Show faces to the camera"
            else:
                self.message = f"{len(self.slots)} faces ready"

    def order_detections(self, detections):
        """Return detections aligned with slot order; returns a partial list if fewer faces visible."""
        with self._lock:
            if not self.slots:
                return list(detections)

            remaining = list(detections)
            ordered = []
            for slot in self.slots:
                if not remaining:
                    break
                best_index = max(
                    range(len(remaining)),
                    key=lambda index: self._iou(slot["box"], remaining[index][2]),
                )
                ordered.append(remaining.pop(best_index))
            return ordered

    def start_selection(self):
        with self._lock:
            # Refresh cache when starting selection to have latest enrolled names
            self._refresh_enrolled_names_cache()
            # New session: any slot tagged with an older session_id is now unreachable/invalid
            self._session_id += 1
            self.slots = []
            self.active_slot_id = None
            # Clear validation state for fresh start
            self._embedding_validated.clear()
            self.armed = True
            self.status = "selecting"
            self.message = "Faces captured: enter names"
            return self.snapshot()

    @staticmethod
    def _iou(first, second):
        first_x2, first_y2 = first[0] + first[2], first[1] + first[3]
        second_x2, second_y2 = second[0] + second[2], second[1] + second[3]
        width = max(0.0, min(first_x2, second_x2) - max(first[0], second[0]))
        height = max(0.0, min(first_y2, second_y2) - max(first[1], second[1]))
        intersection = width * height
        union = first[2] * first[3] + second[2] * second[3] - intersection
        return intersection / union if union > 0 else 0.0

    def set_name(self, ui_index, name):
        with self._lock:
            indices = self._enrollable_slot_indices()
            if 0 <= ui_index < len(indices):
                self.slots[indices[ui_index]]["name"] = str(name).strip()

    def update_preview_embeddings(self, embeddings):
        with self._lock:
            if not self.armed:
                return
            
            # CRITICAL: Refresh cache if stale BEFORE checking embeddings
            if self._cache_stale:
                self._refresh_enrolled_names_cache()
            
            # During CAPTURING: Skip full validation but still update active slot embeddings
            if self.status == "capturing":
                # Only update the active slot's preview embedding for later submission
                if self.active_slot_id is not None:
                    for i, slot in enumerate(self.slots):
                        if slot.get("id") == self.active_slot_id and i < len(embeddings):
                            # Keep preview embedding current
                            vector = np.asarray(embeddings[i], dtype=np.float32).reshape(-1)
                            slot["preview_embedding"] = vector
                return  # Don't validate other slots during capture
            
            # During PREVIEW (not capturing): Process all embeddings for validation
            for slot, embedding in zip(self.slots, embeddings):
                vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
                slot["preview_embedding"] = vector
                history = slot.setdefault("preview_vectors", [])
                history.append(vector)
                if len(history) > self.PREVIEW_WINDOW:
                    del history[:-self.PREVIEW_WINDOW]

                # AGGRESSIVE: Single-frame check for immediate detection
                already_registered, matched_name = self._is_face_already_registered(vector)
                if not already_registered and len(history) >= 3:
                    # Use average of recent history for more robust check (after 3 frames)
                    preview_candidate = np.mean(history, axis=0)
                    already_registered, matched_name = self._is_face_already_registered(preview_candidate)
                
                slot["already_registered"] = already_registered
                slot["registered_name"] = matched_name
                
                # CRITICAL: Mark slot as validated once embeddings processed
                slot_id = slot.get("id")
                if slot_id not in self._embedding_validated:
                    self._embedding_validated.add(slot_id)
                    if already_registered:
                        logger.info(f"[VALIDATED] Slot {slot_id}: Already-enrolled face detected: '{matched_name}'")
                    else:
                        logger.info(f"[VALIDATED] Slot {slot_id}: New face - ready to enroll")
            
            # CRITICAL: Aggressively merge duplicates during preview
            self._merge_duplicate_slots()
            
            # CRITICAL: Remove already-registered faces IMMEDIATELY
            # This prevents ghost cards from ever showing in UI
            self._remove_enrolled_and_registered_faces()

            if self.slots and all(slot.get("already_registered") for slot in self.slots):
                self.message = "All detected faces already registered"

    def start_capture(self, ui_slot_index=None):
        with self._lock:
            indices = self._enrollable_slot_indices()
            if not indices:
                self.message = "No faces to enroll"
                return False
            if ui_slot_index is None:
                ui_slot_index = 0
            if not 0 <= int(ui_slot_index) < len(indices):
                self.message = "Select a face to enroll"
                return False
            internal_index = indices[int(ui_slot_index)]
            slot = self.slots[internal_index]
            name = slot["name"]
            if not name:
                self.message = "Enter a name for this face"
                return False
            if slot.get("enrolled"):
                self.message = "Face already enrolled"
                return False
            preview = slot.get("preview_embedding")
            if preview is None:
                self.message = "Hold face steady before enrolling"
                return False
            already_registered, _matched_name = self._is_face_already_registered(preview)
            if already_registered:
                self.message = "Face already registered"
                return False
            existing = {stored_name.casefold() for stored_name in self.database.embeddings.keys()}
            if name.casefold() in existing:
                self.message = "Face already registered"
                return False
            slot["vectors"] = []
            slot["count"] = 0
            self.active_slot_id = slot["id"]  # Store slot ID, not index
            self.status = "capturing"
            self.message = f"Capturing {name}"
            return True

    def submit(self, embeddings):
        with self._lock:
            if self.status != "capturing":
                return self.snapshot()
            
            # Find slot by ID (handles index shifts from removals)
            if self.active_slot_id is None:
                self.status = "selecting"
                return self.snapshot()
            
            slot = self._get_slot_by_id(self.active_slot_id)
            if slot is None:
                # Slot was removed (e.g., duplicate merge)
                self.active_slot_id = None
                self.status = "selecting"
                self.message = "Face enrollment cancelled (slot no longer exists)"
                logger.warning(f"[SUBMIT] Active slot {self.active_slot_id} was removed")
                return self.snapshot()
            
            # Find slot index for embeddings alignment
            slot_index = None
            for i, s in enumerate(self.slots):
                if s.get("id") == self.active_slot_id:
                    slot_index = i
                    break
            
            # DEFENSIVE: Validate slot still exists and embeddings available
            if slot_index is None:
                logger.warning(f"[SUBMIT] Could not find slot index for active ID {self.active_slot_id}")
                self.status = "selecting"
                self.active_slot_id = None
                return self.snapshot()
            
            if slot_index >= len(embeddings):
                # This can happen briefly when embedding extraction is slow
                logger.debug(f"[SUBMIT] Embedding not ready for slot {slot_index}/{len(embeddings)}")
                return self.snapshot()
            
            # Validate embedding exists and is valid
            embedding = embeddings[slot_index]
            if embedding is None:
                logger.debug(f"[SUBMIT] Null embedding at index {slot_index}")
                return self.snapshot()
            
            try:
                vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            except (ValueError, TypeError) as e:
                logger.warning(f"[SUBMIT] Failed to convert embedding to vector: {e}")
                return self.snapshot()
            
            slot["vectors"].append(vector)
            slot["count"] = len(slot["vectors"])
            
            logger.debug(f"[SUBMIT] Captured frame {slot['count']}/{self.target_frames} for '{slot['name']}'")
            
            if slot["count"] >= self.target_frames:
                candidate = np.mean(slot["vectors"], axis=0)
                already_registered, _matched_name = self._is_face_already_registered(candidate)
                if already_registered:
                    self.status = "selecting"
                    self.message = "Face already registered"
                    self.active_slot_id = None
                    self._remove_enrolled_and_registered_faces()
                    return self.snapshot()
                
                self.database.enroll_many(slot["name"], slot["vectors"])
                slot["enrolled"] = True
                enrolled_name = slot["name"]
                enrolled_count = slot["count"]
                self.active_slot_id = None
                self.status = "selecting"
                self.message = f"{enrolled_name} enrolled successfully ({enrolled_count} frames)"
                
                # Immediately remove enrolled face card
                self._remove_enrolled_and_registered_faces()
                logger.info(f"[ENROLLMENT] Successfully enrolled '{enrolled_name}' with {enrolled_count} frames. Remaining slots: {len(self.slots)}")
            
            return self.snapshot()

    def refresh_for_new_faces(self):
        """Complete refresh: clear ALL state for fresh detection session."""
        with self._lock:
            # Step 1: Refresh cache FIRST to have latest database state
            self._refresh_enrolled_names_cache()
            
            # Step 2: AGGRESSIVE CLEAR - remove everything including partial/incomplete slots
            # This prevents any stale slot from recreating a ghost card
            logger.info(f"[REFRESH] Clearing {len(self.slots)} slot(s) for complete reset")
            self.slots = []
            self.active_slot_id = None
            self._embedding_validated.clear()
            self._frames_since_refresh = 0
            self._next_slot_id = 0  # CRITICAL: Reset slot ID counter for fresh detection
            # New generation: any slot tagged with an older session_id can never pass
            # _enrollable_slot_indices() again, even if a stray reference to it leaks somewhere
            self._session_id += 1
            
            # Step 3: Ensure status allows fresh detection
            self.status = "selecting"
            self.message = "Ready to detect new faces..."
            
            logger.info(f"[REFRESH] Complete reset done (session #{self._session_id}). Cache: {len(self._enrolled_names_cache)} enrolled. Ready for fresh detection.")
            return self.snapshot()

    def snapshot(self):
        with self._lock:
            indices = self._enrollable_slot_indices()
            visible = [self.slots[i] for i in indices]
            return {
                "status": self.status,
                "message": self.message,
                "count": len(visible),
                "target": self.target_frames,
                "session_id": self._session_id,
                "slots": [
                    {
                        "id": slot["id"],
                        "name": slot["name"],
                        "count": slot["count"],
                        "already_registered": False,
                        "registered_name": None,
                        "enrolled": False,
                        "_order_index": idx,
                    }
                    for idx, slot in zip(indices, visible)
                ],
            }
