"""Top-half metrics panel: 2x2 tile grid with sparklines.

Tiles: CPU, Memory, TTFT (time-to-first-token), Tok/s (decode throughput).
Power was replaced with TTFT and total Inference time replaced with Tok/s per
Google feedback — the Power sysfs path never resolved on the Coralboard, and
TTFT + Tok/s are the inference numbers that match how the gemma_translate
demo presents itself.
"""

from __future__ import annotations

from collections import deque

from PyQt6.QtCore import QPointF, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from metrics_provider import MetricsPump, MetricsSnapshot
from theme import CHART_COLORS, PALETTE, TYPE


SPARK_POINTS = 60
SPARK_HEIGHT = 16


class Sparkline(QWidget):
    """Tiny line chart with area fill. Values clamped to [0, 100]."""

    def __init__(self, color: str = PALETTE.accent, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(SPARK_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._values: deque[float] = deque(maxlen=SPARK_POINTS)
        self._color = QColor(color)

    def push(self, v: float) -> None:
        self._values.append(max(0.0, min(100.0, v)))
        self.update()

    def paintEvent(self, event) -> None:
        if len(self._values) < 2:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        step = w / max(1, SPARK_POINTS - 1)

        pad = SPARK_POINTS - len(self._values)
        points: list[QPointF] = []
        for i in range(len(self._values)):
            x = (i + pad) * step
            y = h - (self._values[i] / 100.0) * h
            points.append(QPointF(x, y))

        area_color = QColor(self._color)
        area_color.setAlphaF(0.10)
        area_pts: list[QPointF] = [QPointF(points[0].x(), h)]
        area_pts.extend(points)
        area_pts.append(QPointF(points[-1].x(), h))
        p.setBrush(area_color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawPolygon(QPolygonF(area_pts))

        pen = QPen(self._color)
        pen.setWidth(2)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(1, len(points)):
            p.drawLine(points[i - 1], points[i])


_VALUE_STYLE = (
    f"color: {PALETTE.text_primary}; font-family: {TYPE.mono};"
    f" font-size: 16px; font-weight: 600; letter-spacing: -0.02em;"
)
_VALUE_MUTED_STYLE = (
    f"color: {PALETTE.text_muted}; font-family: {TYPE.mono};"
    f" font-size: 16px; font-weight: 500; letter-spacing: -0.02em;"
)
_LABEL_STYLE = (
    f"color: {PALETTE.text_secondary}; font-size: 11px; font-weight: 500;"
)


class MetricTile(QFrame):
    def __init__(self, label: str, key: str, color: str, unit: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MetricTile")
        self._unit = unit
        self._key = key

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        head = QHBoxLayout()
        head.setSpacing(8)
        self._label = QLabel(label)
        self._label.setObjectName("MetricLabel")
        self._label.setStyleSheet(_LABEL_STYLE)
        self._label.setMinimumHeight(18)
        self._value = QLabel(f"-- {unit}")
        self._value.setObjectName("MetricValue")
        self._value.setStyleSheet(_VALUE_STYLE)
        self._value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._value.setMinimumHeight(20)
        head.addWidget(self._label, alignment=Qt.AlignmentFlag.AlignBottom)
        head.addStretch(1)
        head.addWidget(self._value, alignment=Qt.AlignmentFlag.AlignBottom)

        self.spark = Sparkline(color=color)

        layout.addLayout(head)
        layout.addWidget(self.spark)

    def set_value(self, text: str, percent: float) -> None:
        self._value.setText(text)
        self._value.setStyleSheet(_VALUE_STYLE)
        self.spark.push(percent)

    def set_muted(self, text: str = "n/a") -> None:
        self._value.setText(text)
        self._value.setStyleSheet(_VALUE_MUTED_STYLE)


class MetricsPanel(QFrame):
    def __init__(self, pump: MetricsPump, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MetricsPanel")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.pump = pump
        self._tiles: dict[str, MetricTile] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 11, 12, 11)
        root.setSpacing(8)

        header = QLabel("SYSTEM METRICS")
        header.setObjectName("SectionHeader")
        root.addWidget(header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(7)
        grid.setVerticalSpacing(7)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        root.addLayout(grid, stretch=1)

        tiles = [
            ("CPU",    "cpu",    CHART_COLORS["cpu"],    "%"),
            ("Memory", "memory", CHART_COLORS["memory"], "%"),
            ("TTFT",   "ttft",   CHART_COLORS["power"],  "ms"),
            ("Tok/s",  "tps",    CHART_COLORS["infer"],  "tok/s"),
        ]
        for i, (label, key, color, unit) in enumerate(tiles):
            tile = MetricTile(label, key, color, unit)
            row, col = divmod(i, 2)
            grid.addWidget(tile, row, col)
            grid.setRowStretch(row, 1)
            self._tiles[key] = tile

        self.timer = QTimer(self)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._drain)
        self.timer.start()

    def _drain(self) -> None:
        latest: MetricsSnapshot | None = None
        q = self.pump.queue
        while not q.empty():
            try:
                latest = q.get_nowait()
            except Exception:
                break
        if latest:
            self._apply(latest)

    def _apply(self, snap: MetricsSnapshot) -> None:
        self._tiles["cpu"].set_value(f"{snap.cpu_percent:.0f}%", snap.cpu_percent)
        self._tiles["memory"].set_value(
            f"{snap.memory_percent:.0f}%", snap.memory_percent,
        )

    def report_inference(self, *, ttft_ms: float, tps: float) -> None:
        # Sparkline domain: TTFT 0-1500ms -> 0-100%, tok/s 0-20 -> 0-100%.
        # Octopus-v2 cold prefill on this CPU is ~500ms warm / ~1500ms cold;
        # decode hovers around 7-10 tok/s.
        self._tiles["ttft"].set_value(
            f"{ttft_ms:.0f}ms", min(100.0, ttft_ms / 15.0),
        )
        self._tiles["tps"].set_value(
            f"{tps:.1f} tok/s", min(100.0, tps * 5.0),
        )
