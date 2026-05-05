"""Main PyQt5 window: metrics on top, scrolling tool log on bottom, input row.

Designed for an 800x1280 portrait 7" panel — same target size used in
``astra-2610-assistant-UI-app/src/app_voice_pyqt.py``.

Run via ``app_pyqt.py``, which injects the model + dispatcher.
"""

from __future__ import annotations

import datetime
import os
import threading
from dataclasses import dataclass
from typing import Any

from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton,
    QShortcut, QSizePolicy, QVBoxLayout, QWidget,
)

from command_log import CommandLog, LogTurn, ToolStep
from compact_codec import ToolCall
from dispatcher import DispatchResult, Dispatcher
from llamacpp import FunctionGemmaModel
from metrics_panel import MetricsPanel
from metrics_provider import MetricsPump, PsutilProvider
from theme import PALETTE, TYPE

SUGGESTED_PROMPTS: tuple[str, ...] = (
    "turn on the lights",
    "beep",
    "take a photo",
    "turn off the lights",
    "system status",
    "play rainbow",
)

_CHIP_QSS = (
    f"QPushButton {{"
    f"  background: {PALETTE.bg_tertiary};"
    f"  color: {PALETTE.text_secondary};"
    f"  border: 1px solid {PALETTE.border};"
    f"  border-radius: 999px;"
    f"  padding: 8px 16px;"
    f"  font-size: {TYPE.sm}px;"
    f"  font-weight: 500;"
    f"}}"
    f"QPushButton:hover {{"
    f"  background: {PALETTE.border};"
    f"  color: {PALETTE.text_primary};"
    f"  border-color: {PALETTE.border_strong};"
    f"}}"
    f"QPushButton:pressed {{"
    f"  background: {PALETTE.border_strong};"
    f"}}"
    f"QPushButton:disabled {{"
    f"  color: {PALETTE.text_muted};"
    f"  background: {PALETTE.bg_tertiary};"
    f"}}"
)


@dataclass(frozen=True)
class Turn:
    user: str
    steps: tuple[ToolStep, ...]
    latency_ms: float


def _build_step(call: ToolCall, result: DispatchResult) -> ToolStep:
    if result.status == "error":
        return ToolStep(
            name=call.name,
            args=dict(call.arguments),
            status="error",
            error=result.message or "Tool failed.",
        )
    return ToolStep(
        name=call.name,
        args=dict(call.arguments),
        status="ok",
        message=result.message,
    )


class InferenceWorker(QObject):
    finished = pyqtSignal(Turn)
    failed = pyqtSignal(str)

    def __init__(self, model: FunctionGemmaModel, dispatcher: Dispatcher) -> None:
        super().__init__()
        self.model = model
        self.dispatcher = dispatcher

    def run(self, user_text: str) -> None:
        threading.Thread(target=self._run, args=(user_text,), daemon=True).start()

    def _run(self, user_text: str) -> None:
        try:
            result = self.model.generate(user_text)
            calls = result.tool_calls
            if not calls:
                fallback = ToolStep(
                    name="(no tool call)", status="error",
                    error=f"Model did not produce a parseable call. Raw: {result.raw_text!r}",
                )
                self.finished.emit(Turn(
                    user=user_text, steps=(fallback,),
                    latency_ms=result.latency_ms,
                ))
                return
            results = self.dispatcher.dispatch_all(calls)
            steps = tuple(_build_step(c, r) for c, r in zip(calls, results))
            self.finished.emit(Turn(
                user=user_text, steps=steps,
                latency_ms=result.latency_ms,
            ))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ChatWindow(QMainWindow):
    def __init__(self, model: FunctionGemmaModel, dispatcher: Dispatcher) -> None:
        super().__init__()
        self.setWindowTitle("FunctionGemma Physical AI Demo")
        self.resize(800, 1280)

        self.pump = MetricsPump(PsutilProvider(), interval_s=0.5)
        self.pump.start()

        self.metrics = MetricsPanel(self.pump)
        self.metrics.setFixedHeight(320)
        self.log = CommandLog()

        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask or command...")
        self.input.returnPressed.connect(self._on_send)

        self.send_btn = QPushButton("Send")
        self.send_btn.setObjectName("PrimaryButton")
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.clicked.connect(self._on_send)

        input_row = QHBoxLayout()
        input_row.setSpacing(12)
        input_row.addWidget(self.input, stretch=1)
        input_row.addWidget(self.send_btn)

        self.status = QLabel("Ready.")
        self.status.setStyleSheet(
            f"color: {PALETTE.text_muted}; padding: 0 4px; font-size: {TYPE.sm}px;"
        )

        self._prompt_chips: list[QPushButton] = []
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        for i, prompt in enumerate(SUGGESTED_PROMPTS):
            chip = QPushButton(prompt)
            chip.setCursor(Qt.PointingHandCursor)
            chip.setStyleSheet(_CHIP_QSS)
            chip.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            chip.clicked.connect(lambda _checked, p=prompt: self._on_prompt_chip(p))
            self._prompt_chips.append(chip)
            (row1 if i < 3 else row2).addWidget(chip)
        row1.addStretch(1)
        row2.addStretch(1)

        input_area = QVBoxLayout()
        input_area.setSpacing(8)
        input_area.setContentsMargins(0, 0, 0, 0)
        input_area.addLayout(row1)
        input_area.addLayout(row2)
        input_area.addLayout(input_row)
        input_area.addWidget(self.status)

        root = QVBoxLayout()
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)
        root.addWidget(self.metrics)
        root.addWidget(self.log, stretch=1)
        root.addLayout(input_area)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        QShortcut(QKeySequence("Ctrl+P"), self, activated=self._screenshot)
        QShortcut(QKeySequence("Esc"), self, activated=self.close)

        self.worker = InferenceWorker(model, dispatcher)
        self.worker.finished.connect(self._on_turn_done)
        self.worker.failed.connect(self._on_failed)

    def _set_status(self, text: str, state: str = "idle") -> None:
        colors = {
            "idle": PALETTE.text_muted,
            "thinking": PALETTE.warning,
            "error": PALETTE.danger,
        }
        color = colors.get(state, PALETTE.text_muted)
        bold = "font-weight: 500; " if state == "thinking" else ""
        self.status.setStyleSheet(
            f"color: {color}; padding: 0 4px; font-size: {TYPE.sm}px; {bold}"
        )
        self.status.setText(text)

    def _on_prompt_chip(self, text: str) -> None:
        self.input.setText(text)
        self._on_send()

    def _set_chips_enabled(self, enabled: bool) -> None:
        for chip in self._prompt_chips:
            chip.setEnabled(enabled)

    def _on_send(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.log.append_user_bubble(text)
        self._set_status("Running tools...", "thinking")
        self.send_btn.setEnabled(False)
        self._set_chips_enabled(False)
        self.log.show_thinking()
        self.worker.run(text)

    def _on_turn_done(self, turn: Turn) -> None:
        self.log.hide_thinking()
        self.log.append_turn(LogTurn(
            user=turn.user, steps=turn.steps, latency_ms=turn.latency_ms,
        ))
        self.metrics.report_inference(turn.latency_ms)
        actions = [s for s in turn.steps if not s.is_respond]
        n = len(actions)
        if n:
            self._set_status(
                f"Done ({n} action{'s' if n != 1 else ''}). {turn.latency_ms:.0f} ms"
            )
        else:
            self._set_status(f"Done. {turn.latency_ms:.0f} ms")
        self.send_btn.setEnabled(True)
        self._set_chips_enabled(True)

    def _on_failed(self, error: str) -> None:
        self.log.hide_thinking()
        self.log.append_error(error)
        self._set_status("Last action failed - see log.", "error")
        self.send_btn.setEnabled(True)
        self._set_chips_enabled(True)

    def _screenshot(self) -> None:
        out_dir = os.environ.get("CORAL_SCREENSHOT_DIR", "/tmp")
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(out_dir, f"functiongemma-{ts}.png")
        pix = self.grab()
        ok = pix.save(path, "PNG")
        msg = f"screenshot saved: {path}" if ok else f"screenshot FAILED: {path}"
        self._set_status(msg)
        self.log.append_system(msg)

    def closeEvent(self, event: Any) -> None:
        self.pump.stop()
        super().closeEvent(event)
