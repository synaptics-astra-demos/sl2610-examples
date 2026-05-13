"""Circular mic-toggle button. Lucide mic glyph + accent-blue pulse halo."""

from __future__ import annotations

from PyQt5.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFontMetrics, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

import icons
from theme import PALETTE

P = PALETTE


class MicButton(QWidget):
    toggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(44, 44)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._icon_font = icons.icon_font(20)
        self._checked = False
        self._enabled = True
        self._hover = False
        self._pulse_step = 0
        self._timer = QTimer(self)
        self._timer.setInterval(70)
        self._timer.timeout.connect(self._tick)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, value: bool) -> None:
        if value == self._checked:
            return
        self._checked = value
        if value:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def setEnabled(self, value: bool) -> None:  # noqa: N802 - Qt override
        self._enabled = value
        super().setEnabled(value)
        self.update()

    def enterEvent(self, _event) -> None:
        self._hover = True
        self.update()

    def leaveEvent(self, _event) -> None:
        self._hover = False
        self.update()

    def mousePressEvent(self, event) -> None:
        if not self._enabled or event.button() != Qt.LeftButton:
            return
        self.setChecked(not self._checked)
        self.toggled.emit(self._checked)

    def _tick(self) -> None:
        self._pulse_step = (self._pulse_step + 1) % 24
        self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)

        if not self._enabled:
            bg = QColor(P.bg_secondary)
            border = QColor(P.border)
            ink = QColor(P.text_muted)
        elif self._checked:
            bg = QColor(P.accent)
            border = QColor(P.accent)
            ink = QColor(P.text_inverse)
        elif self._hover:
            bg = QColor(P.bg_tertiary)
            border = QColor(P.border_strong)
            ink = QColor(P.text_secondary)
        else:
            bg = QColor(P.bg_secondary)
            border = QColor(P.border_strong)
            ink = QColor(P.text_secondary)

        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        p.setPen(QPen(border, 1))
        p.setBrush(bg)
        p.drawEllipse(rect)

        if self._checked:
            phase = abs(12 - self._pulse_step) / 12.0
            ripple = QColor(P.text_inverse)
            ripple.setAlphaF(0.18 * (1.0 - phase))
            p.setPen(Qt.NoPen)
            p.setBrush(ripple)
            shrink = 2 + phase * 8
            p.drawEllipse(rect.adjusted(shrink, shrink, -shrink, -shrink))

        p.setFont(self._icon_font)
        p.setPen(ink)
        fm = QFontMetrics(self._icon_font)
        glyph = icons.MIC
        text_rect = fm.tightBoundingRect(glyph)
        x = (self.width() - text_rect.width()) / 2 - text_rect.x()
        y = (self.height() + text_rect.height()) / 2 - text_rect.y() - text_rect.height()
        p.drawText(int(x), int(y), glyph)
