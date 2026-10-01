# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Qt-based FACE ID enrollment UI."""

from __future__ import annotations

import numpy as np

from app_utils.display import display_resolution
from face_id.src.qt_compat import button_style, ensure_utf8_locale, import_qt, resolve_rgb888_format, set_button_icon
from face_id.src.qt_ui_cards import CardsMixin


class QtEnrollmentUI(CardsMixin):
    """Widget-based enrollment UI that reuses BatchEnrollmentSession state."""

    def __init__(self, batch_session, window_title=""):
        self.batch_session = batch_session
        ensure_utf8_locale()
        self._qt = import_qt()
        self._rgb888_format = resolve_rgb888_format(self._qt)
        self._app = self._qt.QApplication.instance() or self._qt.QApplication([])

        self._selected_index = 0
        self._selected_slot_id = None  # Stable slot id the selection should keep following
        self._updating_name = False
        self._last_frame = None
        self._last_detections = []
        self._captured_crops = []
        self._cards = []
        self._rendered_session_id = None  # Session ID currently reflected in the card widgets
        self._resetting = False  # Guards against overlapping Add Face / Register Faces clicks

        self.window = self._qt.QWidget()
        self.window.setWindowTitle(window_title)
        display_width, display_height = display_resolution()
        portrait = display_height > display_width
        self.window.setWindowFlags(self._qt.Qt.WindowType.FramelessWindowHint)
        self.window.setFixedSize(display_width, display_height)
        self.window.move(0, 0)
        self.window.setStyleSheet("background: #0b1118;")

        root = self._qt.QVBoxLayout(self.window)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        header = self._qt.QWidget()
        header.setStyleSheet("background: #0d151f; border: 1px solid #1f2a38; border-radius: 8px;")
        header_layout = self._qt.QHBoxLayout(header)
        header_layout.setContentsMargins(14, 8, 14, 8)
        header_layout.setSpacing(8)

        title = self._qt.QLabel("FACE ID")
        title.setStyleSheet("color: #e6eef8; font-size: 28px; font-weight: 700;")
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        root.addWidget(header)

        content = self._qt.QVBoxLayout() if portrait else self._qt.QHBoxLayout()
        content.setSpacing(10)
        root.addLayout(content, stretch=1)

        left = self._qt.QWidget()
        left.setStyleSheet("background: #0d151f; border: 1px solid #1f2a38; border-radius: 8px;")
        left_layout = self._qt.QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        self.face_count = self._qt.QLabel("No face detected")
        self.face_count.setStyleSheet("color: #5ddc83; font-size: 16px; font-weight: 600;")
        left_layout.addWidget(self.face_count)

        self.video_label = self._qt.QLabel("Waiting for camera frames...")
        self.video_label.setAlignment(self._qt.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(160, 120)
        self.video_label.setSizePolicy(
            self._qt.QSizePolicy.Policy.Ignored, self._qt.QSizePolicy.Policy.Ignored,
        )
        self.video_label.setStyleSheet("background: #06090d; color: #9aa8ba; border: 1px solid #243446; border-radius: 8px;")
        left_layout.addWidget(self.video_label, stretch=1)
        content.addWidget(left, stretch=3 if portrait else 5)

        toggle_column = self._qt.QHBoxLayout() if portrait else self._qt.QVBoxLayout()
        toggle_column.setContentsMargins(0, 0, 0, 0)
        toggle_column.setSpacing(0)
        toggle_column.addStretch(1)

        self.panel_toggle_button = self._qt.QPushButton("<")
        self.panel_toggle_button.setFixedSize(36, 36)
        self.panel_toggle_button.setStyleSheet(button_style(round_px=18))
        self.panel_toggle_button.clicked.connect(self._toggle_panel)
        toggle_column.addWidget(self.panel_toggle_button)
        content.addLayout(toggle_column)

        self.right_panel = self._qt.QWidget()
        self.right_panel.setMinimumWidth(0 if portrait else 280)
        self.right_panel.setStyleSheet("background: #0d151f; border: 1px solid #1f2a38; border-radius: 8px;")
        right_layout = self._qt.QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(6)

        panel_head = self._qt.QHBoxLayout()
        panel_head.setSpacing(8)
        right_layout.addLayout(panel_head)

        panel_title = self._qt.QLabel("ENROLLMENT")
        panel_title.setStyleSheet("color: #2f89ff; font-size: 22px; font-weight: 700;")
        panel_head.addWidget(panel_title)
        panel_head.addStretch(1)

        self.instructions_label = self._qt.QLabel("Select a card and enter a name")
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setStyleSheet("color: #d4deeb; font-size: 12px;")
        right_layout.addWidget(self.instructions_label)

        self.notice_label = self._qt.QLabel("")
        self.notice_label.setWordWrap(True)
        self.notice_label.setStyleSheet("color: #f3b08f; font-size: 12px;")
        right_layout.addWidget(self.notice_label)

        self.scroll_area = self._qt.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.cards_container = self._qt.QWidget()
        self.cards_layout = self._qt.QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(5)
        self.cards_layout.addStretch(1)
        self.scroll_area.setWidget(self.cards_container)
        right_layout.addWidget(self.scroll_area, stretch=1)

        self.enroll_all_button = self._qt.QPushButton("ENROLL SELECTED FACE")
        self.enroll_all_button.setStyleSheet(button_style())
        self.enroll_all_button.clicked.connect(self._on_enroll)
        right_layout.addWidget(self.enroll_all_button)

        self.register_faces_button = self._qt.QPushButton("REGISTER FACES")
        self.register_faces_button.setStyleSheet(button_style())
        self.register_faces_button.clicked.connect(self._on_register)
        right_layout.addWidget(self.register_faces_button)

        set_button_icon(self._qt, self._app, self.enroll_all_button, "enroll")
        set_button_icon(self._qt, self._app, self.register_faces_button, "register")

        content.addWidget(self.right_panel, stretch=3 if portrait else 2)
        self.window.show()

    def _toggle_panel(self):
        visible = self.right_panel.isVisible()
        self.right_panel.setVisible(not visible)
        self.panel_toggle_button.setText(">" if visible else "<")

    def stop(self):
        if self.window is not None:
            self.window.close()
            self.window = None

    def render(self, _canvas):
        """Compatibility hook for display overlay mode; Qt paints its own window."""

    def _on_register(self):
        if not self._last_detections:
            self.notice_label.setText("No face detected")
            return

        # Step 1: Guard against overlapping resets (e.g. double-clicks) - only one
        # Add Face / Register Faces flow may run at a time.
        if self._resetting:
            return
        self._resetting = True
        try:
            # Disable the button immediately so no further clicks can queue up
            # while this reset is in progress.
            self.register_faces_button.setEnabled(False)

            # Step 2: Completely tear down every existing face-card widget
            # (detach from layout, disconnect signals, deleteLater, drop refs).
            self._clear_cards()
            assert self.cards_layout.count() <= 1, "Card container must be empty before starting a new session"

            # Step 3: Clear all temporary UI-side session state. The enrolled
            # face database itself is untouched - only in-progress UI state.
            self._captured_crops = []
            self._selected_index = 0
            self._selected_slot_id = None
            self._last_detections = []
            self._last_frame = None
            self.face_count.setText("Detecting new faces...")
            self.notice_label.setText("")
            self.instructions_label.setText("Show new faces to the camera")

            # Step 4: Ask the backend to start a brand-new session. This bumps
            # the session_id, clears slots/validation/active capture state, and
            # refreshes the enrolled-names cache. Any slot tagged with an older
            # session_id can never pass _enrollable_slot_indices() again.
            if getattr(self.batch_session, "armed", False):
                new_state = self.batch_session.refresh_for_new_faces()
            else:
                new_state = self.batch_session.start_selection() or self.batch_session.snapshot()

            # Step 5: Adopt the new session id immediately so the very next
            # frame's render pass is already aligned with the fresh session.
            self._rendered_session_id = new_state.get("session_id") if isinstance(new_state, dict) else self._rendered_session_id
        finally:
            self._resetting = False
            self.register_faces_button.setEnabled(True)

    def _on_enroll(self):
        # Ensure any edited values are synced before capture starts.
        state = self.batch_session.snapshot()
        slots = state.get("slots", [])
        for index, slot in enumerate(slots):
            if index < len(self._cards):
                self.batch_session.set_name(index, self._cards[index]["name_edit"].text())
            else:
                self.batch_session.set_name(index, slot.get("name", ""))
        self.batch_session.start_capture(self._selected_index)

    def update_source_frame(self, frame, detections):
        self._last_frame = frame.copy()
        self._last_detections = list(detections)
        state = self.batch_session.snapshot()

        face_count = len(detections)
        if face_count == 0:
            self.face_count.setText("No faces detected")
        elif face_count == 1:
            self.face_count.setText("1 face detected")
        else:
            self.face_count.setText(f"{face_count} faces detected")

        message = state.get("message", "")
        hidden_messages = {
            "Ready",
            "Show faces to the camera",
            "Faces captured: enter names",
            "Keep all faces visible and separated",
        }
        self.notice_label.setText("" if message in hidden_messages else message)

        armed = bool(getattr(self.batch_session, "armed", False))
        capturing = state.get("status") == "capturing"
        self.register_faces_button.setEnabled(face_count > 0 and not capturing)
        self.register_faces_button.setText("ADD FACE" if armed else "REGISTER FACES")

        slots = state.get("slots", [])
        selected_slot = slots[self._selected_index] if slots and self._selected_index < len(slots) else None
        can_enroll = (
            armed and selected_slot is not None
            and bool(selected_slot.get("name"))
            and not selected_slot.get("already_registered")
            and not selected_slot.get("enrolled")
            and not capturing
        )
        self.enroll_all_button.setEnabled(can_enroll)
        self.enroll_all_button.setText("ENROLL SELECTED FACE")

        self.instructions_label.setText(
            "Select a card and enter a name" if armed else "Press REGISTER FACES to capture"
        )

        self._render_cards(state, detections)

        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
        height, width, _channels = rgb_frame.shape
        image = self._qt.QImage(
            rgb_frame.data, width, height, width * 3, self._rgb888_format,
        ).copy()
        pixmap = self._qt.QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            self._qt.Qt.AspectRatioMode.KeepAspectRatio,
            self._qt.Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)
        self._app.processEvents()
