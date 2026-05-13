"""Main PyQt5 window: status bar, metrics, scrolling tool log, input row.

Designed for a 480x800 portrait 7" panel — same target size used in the
Claude Design handoff for the Coralboard. Visual structure:

    [StatusBar]            app title + connection dot + Reset + model pill
    [MetricsPanel]         4 tile sparklines (CPU/Memory/Power/Inference)
    [CommandLog]           scrolling YOU/TOOL/ASSISTANT/THINKING bubbles
    [Input area]           horizontal chip strip + text input + mic + Send
    [Status line]          animated dot + status text

Run via ``app_pyqt.py``, which injects the model + dispatcher.
"""

from __future__ import annotations

import datetime
import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton,
    QScrollArea, QShortcut, QSizePolicy, QVBoxLayout, QWidget,
)

import icons
from command_log import CommandLog, LogTurn, ToolStep
from compact_codec import ToolCall
from dispatcher import DispatchResult, Dispatcher
from llamacpp import FunctionGemmaModel
from metrics_panel import MetricsPanel
from metrics_provider import MetricsPump, PsutilProvider
from mic_button import MicButton
from status_bar import StatusBar
from theme import PALETTE, TYPE
from voice import VoicePipeline

PROMPT_CATEGORIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("status LEDs", (
        "set the red light on",
        "turn the green LED on",
        "turn all LEDs off",
        "blue LED at 50%",
        "blink the green light",
        "flash the red light 5 times fast",
        "blink all LEDs twice slowly",
    )),
    ("neopixels", (
        "aurora on the neopixels",
        "plasma the neopixels with ocean palette",
        "fireworks on the neopixels",
        "comet on the neopixels in red",
        "pulse the neopixels blue",
        "police effect on the neopixels",
        "twinkle the neopixels softly",
        "intense fire on the neopixels",
        "turn off the neopixels",
    )),
    ("status", (
        "system status",
        "what's the cpu usage",
        "how much memory am I using",
        "show me the temperature",
        "what's the NPU at",
        "give me a system report",
    )),
    ("alarms", (
        "set an alarm for 7am",
        "wake me up in 5 minutes",
        "remind me in 30 seconds",
        "set a timer for 1 minute",
        "cancel my alarm",
        "remove the alarm",
    )),
    ("buzzer", (
        "beep",
        "play a chirp",
        "double beep",
        "play the siren",
        "play success sound",
        "make an alarm sound",
        "play the error sound",
    )),
)

WARMUP_TARGET_S = 50

CHIP_HEIGHT = 32
_CHIP_QSS = (
    f"QPushButton {{"
    f"  background: {PALETTE.bg_secondary};"
    f"  color: {PALETTE.text_secondary};"
    f"  border: 1px solid {PALETTE.border_strong};"
    f"  border-radius: {CHIP_HEIGHT // 2}px;"
    f"  padding: 0 14px;"
    f"  font-size: 12px;"
    f"  font-weight: 500;"
    f"}}"
    f"QPushButton:hover {{"
    f"  background: {PALETTE.accent_light};"
    f"  color: {PALETTE.accent_dark};"
    f"  border-color: #93c5fd;"
    f"}}"
    f"QPushButton:pressed {{"
    f"  background: {PALETTE.accent_light};"
    f"}}"
    f"QPushButton:disabled {{"
    f"  color: {PALETTE.text_muted};"
    f"  background: {PALETTE.bg_secondary};"
    f"  border-color: {PALETTE.border};"
    f"}}"
)



class _AnimatedStatusLine(QWidget):
    """One-line status indicator. Optional pulsing dot + text label.

    States: "idle" (muted, no dot), "listening" (accent, dot), "thinking"
    (warning, dot), "loading" (warning, dot), "error" (danger, no dot).
    """

    _DOT_STATES = {"listening", "thinking", "loading"}

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 0, 3, 0)
        layout.setSpacing(6)

        self._dot = QFrame()
        self._dot.setFixedSize(6, 6)
        self._dot.hide()
        layout.addWidget(self._dot, alignment=Qt.AlignVCenter)

        self._label = QLabel("Ready.")
        self._label.setStyleSheet(
            f"background: transparent; color: {PALETTE.text_muted}; font-size: 11px;"
        )
        layout.addWidget(self._label, stretch=1)

        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._tick)
        self._pulse_step = 0
        self._state = "idle"

    def set_status(self, text: str, state: str = "idle") -> None:
        self._state = state
        self._label.setText(text)

        color = PALETTE.text_muted
        weight = "400"
        if state == "listening":
            color = PALETTE.accent
            weight = "500"
        elif state == "thinking" or state == "loading":
            color = PALETTE.warning
            weight = "500"
        elif state == "error":
            color = PALETTE.danger
            weight = "500"
        self._label.setStyleSheet(
            f"background: transparent; color: {color}; font-size: 11px;"
            f" font-weight: {weight};"
        )

        if state in self._DOT_STATES:
            self._dot_color = color
            self._dot.show()
            if not self._timer.isActive():
                self._timer.start()
            self._apply_dot(1.0)
        else:
            self._timer.stop()
            self._dot.hide()

    def _tick(self) -> None:
        self._pulse_step = (self._pulse_step + 1) % 12
        phase = abs(6 - self._pulse_step) / 6.0
        opacity = 0.3 + (1.0 - phase) * 0.7
        self._apply_dot(opacity)

    def _apply_dot(self, opacity: float) -> None:
        color = self._dot_color
        self._dot.setStyleSheet(
            f"background: {color}; border-radius: 3px;"
        )
        self._dot.setWindowOpacity(opacity)


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


class VoiceSignals(QObject):
    """Marshal voice-pipeline emits (worker thread) onto the Qt main thread."""

    text = pyqtSignal(str)
    state = pyqtSignal(str)


class WarmupSignals(QObject):
    """Marshal background-warmup completion onto the Qt main thread."""

    done = pyqtSignal(float)  # elapsed seconds
    failed = pyqtSignal(str)


class AlarmSignals(QObject):
    """Marshal alarm-fire events from the daemon Timer thread to Qt.

    HardwareDevice._fire_alarm runs on a threading.Timer worker; we cannot
    touch widgets there. The owner of this object wires
    ``hardware.on_async_event = alarm_signals.fired.emit`` and ChatWindow
    connects ``fired`` to its UI handler.
    """

    fired = pyqtSignal(str)  # event text (e.g. "ALARM FIRED: <label>")


class ChatWindow(QMainWindow):
    def __init__(
        self,
        model: FunctionGemmaModel,
        dispatcher: Dispatcher,
        voice: VoicePipeline | None = None,
        screenshot_dir: str | Path = "/tmp",
        alarm_signals: AlarmSignals | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("FunctionGemma Physical AI Demo")
        self.resize(480, 800)
        self.setMaximumWidth(480)
        self._screenshot_dir = Path(screenshot_dir)

        self.pump = MetricsPump(PsutilProvider(), interval_s=0.5)
        self.pump.start()

        self.status_bar = StatusBar()
        self.status_bar.clicked_reset.connect(self._on_reset)

        self.metrics = MetricsPanel(self.pump)
        self.log = CommandLog()

        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask or command...")
        self.input.returnPressed.connect(self._on_send)
        self.input.setFixedHeight(44)

        icons.ensure_loaded()
        self.send_btn = QPushButton(icons.SEND)
        self.send_btn.setObjectName("PrimaryButton")
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.setFixedSize(44, 44)
        self.send_btn.setStyleSheet(
            "QPushButton#PrimaryButton {"
            "  padding: 0; font-family: 'lucide'; font-size: 20px;"
            "  font-weight: 400;"
            "}"
        )
        self.send_btn.clicked.connect(self._on_send)

        self._voice = voice
        self._voice_signals: VoiceSignals | None = None
        self.mic_btn: MicButton | None = None
        if voice is not None:
            self._voice_signals = VoiceSignals()
            self._voice_signals.text.connect(self._on_voice_text)
            voice.set_callback(self._voice_signals.text.emit)
            self.mic_btn = MicButton()
            self.mic_btn.toggled.connect(self._on_mic_toggle)

        input_row = QHBoxLayout()
        input_row.setSpacing(7)
        input_row.addWidget(self.input, stretch=1)
        if self.mic_btn is not None:
            input_row.addWidget(self.mic_btn, alignment=Qt.AlignVCenter)
        input_row.addWidget(self.send_btn, alignment=Qt.AlignVCenter)

        self.status = _AnimatedStatusLine()

        self._prompt_chips: list[QPushButton] = []
        chip_row = QHBoxLayout()
        chip_row.setContentsMargins(0, 0, 0, 0)
        chip_row.setSpacing(6)
        for category, pool in PROMPT_CATEGORIES:
            chip = QPushButton(category)
            chip.setCursor(Qt.PointingHandCursor)
            chip.setStyleSheet(_CHIP_QSS)
            chip.setFixedHeight(CHIP_HEIGHT)
            chip.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            chip.clicked.connect(
                lambda _checked, pool=pool: self._on_category_chip(pool)
            )
            self._prompt_chips.append(chip)
            chip_row.addWidget(chip)
        chip_row.addStretch(1)

        chip_container = QWidget()
        chip_container.setStyleSheet("background: transparent;")
        chip_container.setLayout(chip_row)

        chip_scroll = QScrollArea()
        chip_scroll.setWidget(chip_container)
        chip_scroll.setWidgetResizable(False)
        chip_scroll.setFrameShape(QFrame.NoFrame)
        chip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        chip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        chip_scroll.setFixedHeight(46)
        chip_scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        chip_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
        )
        chip_container.adjustSize()

        input_area = QVBoxLayout()
        input_area.setSpacing(5)
        input_area.setContentsMargins(0, 0, 0, 0)
        input_area.addWidget(chip_scroll)
        input_area.addLayout(input_row)
        input_area.addWidget(self.status)

        root = QVBoxLayout()
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(9)
        root.addWidget(self.status_bar)
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

        self._alarm_signals = alarm_signals
        if alarm_signals is not None:
            alarm_signals.fired.connect(self._on_alarm_fired)

        # Loading state is shown during the ~5s model load + warmup so the
        # user sees what's happening instead of facing a frozen Send button.
        self._warmup_signals = WarmupSignals()
        self._warmup_signals.done.connect(self._on_warmup_done)
        self._warmup_signals.failed.connect(self._on_warmup_failed)
        self._warmup_start = time.time()
        self._loading_timer = QTimer(self)
        self._loading_timer.setInterval(1000)
        self._loading_timer.timeout.connect(self._on_loading_tick)
        self._enter_loading()
        threading.Thread(
            target=self._run_warmup, args=(model,), daemon=True, name="warmup",
        ).start()

    def _enter_loading(self) -> None:
        self.status_bar.set_state("loading")
        self.log.show_loading()
        self.log.update_loading_remaining(WARMUP_TARGET_S)
        self._set_status("Loading model - please wait...", "loading")
        self.send_btn.setEnabled(False)
        self.input.setEnabled(False)
        self._set_chips_enabled(False)
        if self.mic_btn is not None:
            self.mic_btn.setEnabled(False)
        self._warmup_start = time.time()
        if not self._loading_timer.isActive():
            self._loading_timer.start()

    def _exit_loading(self) -> None:
        self._loading_timer.stop()
        self.log.hide_loading()
        self.status_bar.set_state("on")
        self.input.setEnabled(True)
        self.send_btn.setEnabled(True)
        self._set_chips_enabled(True)
        if self.mic_btn is not None:
            self.mic_btn.setEnabled(True)

    def _on_loading_tick(self) -> None:
        elapsed = time.time() - self._warmup_start
        remaining = max(0, int(WARMUP_TARGET_S - elapsed))
        self.log.update_loading_remaining(remaining)

    def _run_warmup(self, model: FunctionGemmaModel) -> None:
        t0 = time.time()
        try:
            model.generate("hello")
        except Exception as exc:
            logger.exception("warmup failed")
            self._warmup_signals.failed.emit(str(exc))
            return
        self._warmup_signals.done.emit(time.time() - t0)

    def _on_warmup_done(self, elapsed: float) -> None:
        self._exit_loading()
        self._set_status(f"Ready. Warmed up in {elapsed:.1f}s.")

    def _on_warmup_failed(self, error: str) -> None:
        self._exit_loading()
        self._set_status(f"Warmup failed: {error}", "error")

    def _set_status(self, text: str, state: str = "idle") -> None:
        self.status.set_status(text, state)

    def _on_category_chip(self, pool: tuple[str, ...]) -> None:
        prompt = random.choice(pool)
        self.input.setText(prompt)
        self._on_send()

    def _set_chips_enabled(self, enabled: bool) -> None:
        for chip in self._prompt_chips:
            chip.setEnabled(enabled)

    def _on_mic_toggle(self, checked: bool) -> None:
        if self._voice is None or self.mic_btn is None:
            return
        if checked:
            self._voice.start()
            self._set_status("Listening...", "listening")
            self.send_btn.setEnabled(False)
            self._set_chips_enabled(False)
        else:
            self._voice.stop()
            self._set_status("Ready.")
            self.send_btn.setEnabled(True)
            self._set_chips_enabled(True)

    def _on_voice_text(self, text: str) -> None:
        if self.mic_btn is not None and self.mic_btn.isChecked():
            self.mic_btn.setChecked(False)
            self._on_mic_toggle(False)
        self.input.setText(text)
        self._on_send()

    def _on_reset(self) -> None:
        self.log.clear()
        self._set_status("Ready.")

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

    def _on_alarm_fired(self, event: str) -> None:
        # Hardware emits "ALARM FIRED: <label>"; show just the label in the
        # bubble and status line — the chip already says "ALARM".
        label = event.split(": ", 1)[1] if ": " in event else event
        self.log.append_alarm(label)
        self._set_status(f"ALARM: {label}", "error")

    def _screenshot(self) -> None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        path = self._screenshot_dir / f"functiongemma-{ts}.png"
        pix = self.grab()
        ok = pix.save(str(path), "PNG")
        msg = f"screenshot saved: {path}" if ok else f"screenshot FAILED: {path}"
        self._set_status(msg)
        self.log.append_system(msg)

    def closeEvent(self, event: Any) -> None:
        if self._voice is not None:
            self._voice.stop()
        self.pump.stop()
        super().closeEvent(event)
