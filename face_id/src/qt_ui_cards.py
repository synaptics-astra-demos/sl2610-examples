# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Face-card widget lifecycle, rendering, and selection for the Face ID UI.

CardsMixin is combined into QtEnrollmentUI (see qt_ui.py); it shares that
class's instance state (self._cards, self.cards_layout, self._selected_slot_id,
etc.) rather than owning it independently.
"""

from __future__ import annotations

import numpy as np

from face_id.src.qt_compat import button_style, set_button_icon


class CardsMixin:
    """Face-card widget lifecycle, rendering, and selection."""

    def _set_name(self, index, text):
        self.batch_session.set_name(index, text)

    def _clear_cards(self):
        """Tear down every face-card widget completely: detach, disconnect, delete, and drop references."""
        for entry in self._cards:
            self._teardown_card(entry)

        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                self._teardown_widget(widget)
            child_layout = item.layout()
            if child_layout is not None:
                self._teardown_layout(child_layout)

        self._cards = []
        assert self.cards_layout.count() <= 1, "Face-card container must be empty after clearing"

    def _teardown_widget(self, widget):
        """Recursively detach a widget's child layout/widgets, then schedule the widget for deletion."""
        layout = widget.layout()
        if layout is not None:
            self._teardown_layout(layout)
        widget.setParent(None)
        widget.deleteLater()

    def _teardown_layout(self, layout):
        """Recursively empty a layout, tearing down any nested widgets/layouts."""
        while layout.count():
            item = layout.takeAt(0)
            child_widget = item.widget()
            if child_widget is not None:
                self._teardown_widget(child_widget)
            nested_layout = item.layout()
            if nested_layout is not None:
                self._teardown_layout(nested_layout)

    def _build_card(self, index):
        qt = self._qt
        card = qt.QWidget()
        card.setStyleSheet(
            "background: #111a24; border: 1px solid #3a475a; border-radius: 8px;"
        )
        layout = qt.QVBoxLayout(card)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        head = qt.QHBoxLayout()
        head.setSpacing(4)
        layout.addLayout(head)
        head.addStretch(1)

        state = qt.QLabel("READY")
        state.setStyleSheet("color: #9fb2c9; font-size: 10px; font-weight: 700;")
        head.addWidget(state)

        body = qt.QHBoxLayout()
        body.setSpacing(6)
        layout.addLayout(body)

        thumb_container = qt.QWidget()
        thumb_container.setFixedSize(70, 70)
        thumb_container.setStyleSheet("background: transparent; border: none;")

        thumb = qt.QLabel("FACE IMAGE", thumb_container)
        thumb.setGeometry(0, 0, 70, 70)
        thumb.setAlignment(qt.Qt.AlignmentFlag.AlignCenter)
        thumb.setStyleSheet(
            "background: #05080c; color: #90a4bf; border: 1px solid #4b607b; border-radius: 6px;"
        )
        body.addWidget(thumb_container)

        right = qt.QVBoxLayout()
        right.setSpacing(3)
        body.addLayout(right, stretch=1)

        status = qt.QLabel("Ready to register")
        status.setStyleSheet("color: #62d287; font-size: 10px;")
        right.addWidget(status)

        name_edit = qt.QLineEdit()
        name_edit.setPlaceholderText("Enter name")
        name_edit.setStyleSheet(
            "QLineEdit { background: #0d141d; color: #e8eef7; border: 1px solid #40566f;"
            "border-radius: 5px; padding: 2px 6px; }"
            "QLineEdit:focus { border: 1px solid #6faeff; }"
            "QLineEdit:disabled { background: #1a2330; color: #8ca0bc; border: 1px solid #3a5272; }"
        )
        name_edit.textEdited.connect(lambda text, i=index: self._set_name(i, text))
        right.addWidget(name_edit)
        right.addStretch(1)

        select_btn = qt.QPushButton("SELECT")
        select_btn.setFixedHeight(26)
        select_btn.setStyleSheet(button_style())
        set_button_icon(qt, self._app, select_btn, "select")
        select_btn.clicked.connect(lambda _checked=False, i=index: self._select_card(i))
        layout.addWidget(select_btn)

        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
        self._cards.append(
            {
                "card": card,
                "state": state,
                "thumb": thumb,
                "status": status,
                "name_edit": name_edit,
                "select_btn": select_btn,
            }
        )

    def _ensure_cards(self, count):
        current = len(self._cards)
        if current > count:
            # Only tear down the excess trailing cards - NOT a full rebuild.
            # A full clear-and-rebuild here would destroy (and recreate) the
            # widget the user is currently typing into whenever the card count
            # merely dips for a frame (e.g. an unrelated new face is briefly
            # shown then filtered out as already-enrolled), stealing keyboard
            # focus from the card the user actually selected.
            excess = self._cards[count:]
            self._cards = self._cards[:count]
            for entry in excess:
                self._teardown_card(entry)
            current = count
        for index in range(current, count):
            self._build_card(index)

    def _teardown_card(self, entry):
        """Disconnect signals and schedule deletion for a single card entry."""
        name_edit = entry.get("name_edit")
        select_btn = entry.get("select_btn")
        card = entry.get("card")
        if name_edit is not None:
            try:
                name_edit.textEdited.disconnect()
            except (TypeError, RuntimeError):
                pass
        if select_btn is not None:
            try:
                select_btn.clicked.disconnect()
            except (TypeError, RuntimeError):
                pass
        if card is not None:
            self._teardown_widget(card)

    def _select_card(self, index):
        self._selected_index = max(0, int(index))
        state = self.batch_session.snapshot()
        slots = state.get("slots", [])
        # Remember the SLOT's stable id, not just its current array position, so
        # the selection keeps following this same face even if an unrelated slot
        # elsewhere in the list is added/removed and shifts everyone's position.
        self._selected_slot_id = slots[self._selected_index]["id"] if self._selected_index < len(slots) else None
        self._render_cards(state, self._last_detections)
        if self._selected_index < len(self._cards):
            edit = self._cards[self._selected_index]["name_edit"]
            if edit.isEnabled():
                edit.setFocus()
                edit.setCursorPosition(len(edit.text()))

    def _extract_crop(self, frame, detection):
        if frame is None or detection is None:
            return None
        box = detection.get("bounding_box", {})
        origin = box.get("origin", {})
        size = box.get("size", {})
        x = int(origin.get("x", 0))
        y = int(origin.get("y", 0))
        w = int(size.get("x", 0))
        h = int(size.get("y", 0))
        if w <= 0 or h <= 0:
            return None
        x2 = min(x + w, frame.shape[1])
        y2 = min(y + h, frame.shape[0])
        x = max(0, x)
        y = max(0, y)
        if x2 <= x or y2 <= y:
            return None
        return frame[y:y2, x:x2].copy()

    def _show_crop_on_label(self, label, crop):
        if crop is None or crop.size == 0:
            label.setText("FACE IMAGE")
            label.setPixmap(self._qt.QPixmap())
            return
        rgb = np.ascontiguousarray(crop[:, :, ::-1])
        h, w, _ = rgb.shape
        image = self._qt.QImage(rgb.data, w, h, w * 3, self._rgb888_format).copy()
        pix = self._qt.QPixmap.fromImage(image)
        scaled = pix.scaled(
            label.width(),
            label.height(),
            self._qt.Qt.AspectRatioMode.KeepAspectRatio,
            self._qt.Qt.TransformationMode.SmoothTransformation,
        )
        label.setText("")
        label.setPixmap(scaled)

    def _render_cards(self, state, detections):
        # Defense-in-depth: if the backend has moved on to a new enrollment session
        # (Add Face / Register Faces was clicked), force a full teardown before
        # rendering so no widget from a previous session can remain on screen.
        incoming_session_id = state.get("session_id")
        if incoming_session_id != self._rendered_session_id:
            self._clear_cards()
            self._rendered_session_id = incoming_session_id

        slots = state.get("slots", [])
        # Always derive count from enrollable slots; never use detections count as a fallback
        display_count = len(slots)
        self._ensure_cards(display_count)

        if display_count == 0:
            self._selected_slot_id = None
            self._clear_cards()
            return

        # Re-resolve the selected POSITION from the selected slot's stable id every
        # render, since an unrelated slot being added/removed elsewhere in the list
        # can shift everyone's array position without the user's selection actually
        # having changed. Falls back to clamping the previous index only if that
        # slot id truly no longer exists (e.g. it just got enrolled).
        matched_index = next(
            (i for i, s in enumerate(slots) if s.get("id") == self._selected_slot_id),
            None,
        )
        if matched_index is not None:
            self._selected_index = matched_index
        else:
            self._selected_index = max(0, min(self._selected_index, display_count - 1))
            self._selected_slot_id = slots[self._selected_index].get("id")

        # Use _order_index to pick the correct detection even when registered slots are hidden
        live_crops = [
            self._extract_crop(
                self._last_frame,
                detections[s.get("_order_index", i)] if s.get("_order_index", i) < len(detections) else None,
            )
            for i, s in enumerate(slots)
        ]

        for index in range(display_count):
            card = self._cards[index]
            selected = index == self._selected_index
            slot = slots[index] if index < len(slots) else None

            border = "#2f89ff" if selected else "#3a475a"
            width = "2px" if selected else "1px"
            card["card"].setStyleSheet(
                "background: #111a24; "
                f"border: {width} solid {border}; border-radius: 8px;"
            )

            state_text = "SELECTED" if selected else "READY"
            state_color = "#2f89ff" if selected else "#9fb2c9"
            status_text = "Ready to register"
            status_color = "#62d287"
            disabled = False

            if slot is not None:
                name = slot.get("name", "")
                if slot.get("enrolled"):
                    state_text = "ENROLLED"
                    state_color = "#62d287"
                    status_text = "Enrolled successfully"
                    status_color = "#62d287"
                    disabled = True
                elif state.get("status") == "capturing":
                    state_text = "CAPTURING"
                    state_color = "#62d287"
                    status_text = f"Capturing {slot.get('count', 0)}/{state.get('target', 0)}"
                    status_color = "#62d287"

                # Do not overwrite text while user is actively typing.
                current_text = card["name_edit"].text()
                if (not card["name_edit"].hasFocus()) and current_text != name:
                    card["name_edit"].blockSignals(True)
                    card["name_edit"].setText(name)
                    card["name_edit"].blockSignals(False)

            card["state"].setText(state_text)
            card["state"].setStyleSheet(f"color: {state_color}; font-size: 12px; font-weight: 700;")
            card["status"].setText(status_text)
            card["status"].setStyleSheet(f"color: {status_color}; font-size: 12px;")
            card["name_edit"].setEnabled(not disabled and bool(slots) and selected)
            card["select_btn"].setEnabled(bool(slots))
            card["select_btn"].setText("ENROLLED" if slot and slot.get("enrolled") else "SELECT")

            crop = None
            if self._captured_crops and index < len(self._captured_crops):
                crop = self._captured_crops[index]
            elif index < len(live_crops):
                crop = live_crops[index]
            self._show_crop_on_label(card["thumb"], crop)
