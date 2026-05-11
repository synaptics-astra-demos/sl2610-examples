"""Loading panel shown inside the conversation-log card while the model warms up.

Replaces the empty-state title during the initial ~50s prefill. ``start()`` /
``stop()`` control the spinner + progress-bar animations; the parent window
ticks ``set_remaining_seconds`` every second.
"""

from __future__ import annotations

from PyQt5.QtCore import QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar, QSizePolicy, QVBoxLayout, QWidget,
)

from theme import PALETTE, TYPE

P = PALETTE
T = TYPE


class _Spinner(QWidget):
    """36x36 rotating arc spinner. Track circle + 20% accent-blue arc."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(36, 36)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 fps
        self._timer.timeout.connect(self._advance)

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _advance(self) -> None:
        self._angle = (self._angle + 11) % 360
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(3, 3, 30, 30)

        track = QPen(QColor("#e2e8f0"))
        track.setWidth(3)
        painter.setPen(track)
        painter.drawEllipse(rect)

        arc = QPen(QColor(P.accent))
        arc.setWidth(3)
        arc.setCapStyle(Qt.RoundCap)
        painter.setPen(arc)
        start_angle = (-self._angle) * 16
        span_angle = -72 * 16  # 20% of 360
        painter.drawArc(rect, start_angle, span_angle)


class ModelLoadingPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignCenter)

        self._spinner = _Spinner()
        layout.addWidget(self._spinner, alignment=Qt.AlignHCenter)

        title = QLabel("Loading model")
        title.setAlignment(Qt.AlignHCenter)
        title.setStyleSheet(
            f"background: transparent; color: {P.text_primary};"
            f" font-size: 14px; font-weight: 600;"
        )
        title.setContentsMargins(0, 4, 0, 0)
        layout.addWidget(title)

        sub = QLabel("functiongemma-270m · llamacpp")
        sub.setAlignment(Qt.AlignHCenter)
        sub.setStyleSheet(
            f"background: transparent; color: {P.text_muted};"
            f" font-family: {T.mono}; font-size: 10px; letter-spacing: -0.01em;"
        )
        layout.addWidget(sub)

        self._bar = QProgressBar()
        self._bar.setRange(0, 0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(4)
        self._bar.setMaximumWidth(240)
        self._bar.setStyleSheet(
            f"QProgressBar {{ background: {P.bg_tertiary};"
            f" border: none; border-radius: 2px; }}"
            f"QProgressBar::chunk {{ background: {P.accent}; border-radius: 2px; }}"
        )
        bar_wrap = QHBoxLayout()
        bar_wrap.setContentsMargins(0, 8, 0, 0)
        bar_wrap.addStretch(1)
        bar_wrap.addWidget(self._bar)
        bar_wrap.addStretch(1)
        layout.addLayout(bar_wrap)

        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 2, 0, 0)
        meta_inner = QHBoxLayout()
        meta_inner.setSpacing(0)
        self._remaining = QLabel("~ 50s remaining")
        self._remaining.setStyleSheet(
            f"background: transparent; color: {P.text_secondary};"
            f" font-family: {T.mono}; font-size: 10px;"
        )
        self._remaining.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        stage = QLabel("mmap weights · 312 MB")
        stage.setStyleSheet(
            f"background: transparent; color: {P.text_muted};"
            f" font-family: {T.mono}; font-size: 10px;"
        )
        stage.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        meta_inner_widget = QWidget()
        meta_inner_widget.setMinimumWidth(260)
        meta_inner_widget.setMaximumWidth(300)
        meta_inner_widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        inner_layout = QHBoxLayout(meta_inner_widget)
        inner_layout.setContentsMargins(0, 0, 0, 0)
        inner_layout.setSpacing(10)
        inner_layout.addWidget(self._remaining)
        inner_layout.addStretch(1)
        inner_layout.addWidget(stage)
        meta_row.addStretch(1)
        meta_row.addWidget(meta_inner_widget)
        meta_row.addStretch(1)
        layout.addLayout(meta_row)

        hint = QLabel("Input is disabled until the model is ready.")
        hint.setAlignment(Qt.AlignHCenter)
        hint.setWordWrap(True)
        hint.setMaximumWidth(280)
        hint.setStyleSheet(
            f"background: transparent; color: {P.text_muted};"
            f" font-size: 11px;"
        )
        hint.setContentsMargins(0, 10, 0, 0)
        hint_wrap = QHBoxLayout()
        hint_wrap.addStretch(1)
        hint_wrap.addWidget(hint)
        hint_wrap.addStretch(1)
        layout.addLayout(hint_wrap)

    def start(self) -> None:
        self._spinner.start()

    def stop(self) -> None:
        self._spinner.stop()

    def set_remaining_seconds(self, seconds: int) -> None:
        seconds = max(0, int(seconds))
        self._remaining.setText(f"~ {seconds}s remaining")
