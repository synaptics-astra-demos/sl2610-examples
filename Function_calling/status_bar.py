"""Top status bar: app title, connection dot, Reset, subtitle, model pill.

Three states for the connection dot, set via ``set_state``:
    "on"      green dot, Reset enabled.
    "off"     muted-gray dot, Reset enabled.
    "loading" warning-orange dot with pulsing animation, Reset disabled.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

import icons
from theme import PALETTE, TYPE

P = PALETTE
T = TYPE


class _Dot(QFrame):
    """8px circular dot with optional pulsing animation in the loading state."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(8, 8)
        self._state = "on"
        self._pulse_step = 0
        self._timer = QTimer(self)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._tick)
        self._apply()

    def set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        if state == "loading":
            self._pulse_step = 0
            self._timer.start()
        else:
            self._timer.stop()
        self._apply()

    def _apply(self) -> None:
        if self._state == "on":
            color = P.success
        elif self._state == "loading":
            color = P.warning
        else:
            color = P.text_muted
        self.setStyleSheet(f"background: {color}; border-radius: 4px;")

    def _tick(self) -> None:
        self._pulse_step = (self._pulse_step + 1) % 10
        scale = 0.4 + abs(5 - self._pulse_step) / 5.0 * 0.6
        alpha = int(60 + scale * 195)
        self.setStyleSheet(
            f"background: {P.warning}; border-radius: 4px;"
            f" border: 1px solid rgba(217, 119, 6, {alpha});"
        )


class StatusBar(QFrame):
    clicked_reset = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StatusBar")
        self.setFrameShape(QFrame.StyledPanel)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 9, 12, 9)
        root.setSpacing(5)

        row1 = QHBoxLayout()
        row1.setSpacing(8)

        title_cluster = QHBoxLayout()
        title_cluster.setSpacing(8)
        title_cluster.setContentsMargins(0, 0, 0, 0)
        self._title = QLabel("Coral FunctionGemma")
        self._title.setStyleSheet(
            f"background: transparent; color: {P.text_primary};"
            f" font-size: 16px; font-weight: 700; letter-spacing: -0.01em;"
        )
        self._title.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self._dot = _Dot()
        title_cluster.addWidget(self._title)
        title_cluster.addWidget(self._dot, alignment=Qt.AlignVCenter)

        icons.ensure_loaded()
        self._reset_btn = QPushButton(icons.RESET)
        self._reset_btn.setObjectName("GhostButton")
        self._reset_btn.setCursor(Qt.PointingHandCursor)
        self._reset_btn.setFixedSize(32, 32)
        self._reset_btn.setStyleSheet(
            "QPushButton#GhostButton {"
            "  padding: 0; min-height: 32px;"
            "  font-family: 'lucide'; font-size: 14px;"
            "  font-weight: 400; border-radius: 16px;"
            "}"
        )
        self._reset_btn.setToolTip("Reset conversation")
        self._reset_btn.clicked.connect(self.clicked_reset.emit)

        row1.addLayout(title_cluster)
        row1.addStretch(1)
        row1.addWidget(self._reset_btn)

        row2 = QHBoxLayout()
        row2.setSpacing(6)
        self._subtitle_full = "Physical AI demo · SL2619 · metrics=psutil"
        self._subtitle = QLabel(self._subtitle_full)
        self._subtitle.setStyleSheet(
            f"background: transparent; color: {P.text_secondary}; font-size: 11px;"
        )
        self._subtitle.setMinimumWidth(40)
        self._subtitle.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._pill = QLabel("model: functiongemma-270m")
        self._pill.setStyleSheet(
            f"background: {P.bg_tertiary}; color: {P.text_secondary};"
            f" border: 1px solid {P.border}; border-radius: 999px;"
            f" padding: 2px 8px; font-family: {T.mono};"
            f" font-size: 10px; font-weight: 500; letter-spacing: -0.01em;"
        )
        self._pill.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        row2.addWidget(self._subtitle, stretch=1)
        row2.addWidget(self._pill, alignment=Qt.AlignVCenter)

        root.addLayout(row1)
        root.addLayout(row2)

    def resizeEvent(self, event) -> None:  # noqa: D401 - Qt override
        super().resizeEvent(event)
        self._elide_subtitle()

    def _elide_subtitle(self) -> None:
        fm = QFontMetrics(self._subtitle.font())
        avail = max(20, self._subtitle.width())
        self._subtitle.setText(fm.elidedText(self._subtitle_full, Qt.ElideRight, avail))

    def set_state(self, state: str) -> None:
        self._dot.set_state(state)
        self._reset_btn.setEnabled(state != "loading")

    def set_model_name(self, name: str) -> None:
        self._pill.setText(f"model: {name}")
