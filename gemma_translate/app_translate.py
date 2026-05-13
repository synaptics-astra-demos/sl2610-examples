from __future__ import annotations

import argparse
from collections import deque
import logging
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "UI"))

from PyQt5.QtCore import QEvent, QObject, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from mic_button import MicButton

from gemma_translate.common_args import (
    LANGUAGES,
    LanguageOption,
    add_common_translation_args,
    resolve_gemma_model_path,
)
from gemma_translate.translation import GemmaTranslationService, TranslationResult
from utils.log import configure_logging
from utils.npu import enable_npu_clock
from utils.stats import InferenceStats

if TYPE_CHECKING:
    from utils.speech import SpeechRecognizer, SpeechTranscript


logger = logging.getLogger("Translate App")

SAMPLING_RATE = 16_000
CHUNK_SIZE = 512
BUBBLE_FONT_SIZE = 24
STATS_FONT_SIZE = 18
DEFAULT_MAX_CHAT_BUBBLES = 16
DEFAULT_PARTIAL_UPDATE_MS = 80

_ORIENTATION = os.environ.get("ORIENTATION", "landscape")
CHAT_WIDTH = int(os.environ.get("DISPLAY_WIDTH", 800 if _ORIENTATION == "landscape" else 480))
CHAT_HEIGHT = int(os.environ.get("DISPLAY_HEIGHT", 480 if _ORIENTATION == "landscape" else 800))

_THIS_DIR = Path(__file__).resolve().parent
NON_LATIN_LANGUAGE_FONTS = {
    "Chinese": (_THIS_DIR / "fonts" / "NotoSansSC-VariableFont_wght.ttf",),
    "Simplified Chinese": (_THIS_DIR / "fonts" / "NotoSansSC-VariableFont_wght.ttf",),
    "Hindi": (_THIS_DIR / "fonts" / "NotoSansDevanagari-VariableFont_wdth,wght.ttf",),
    "Thai": (_THIS_DIR / "fonts" / "NotoSansThai-VariableFont_wdth,wght.ttf",),
}
_LOADED_FONT_PATHS: set[str] = set()


def ensure_language_fonts_loaded(language: LanguageOption):
    font_paths = (
        *NON_LATIN_LANGUAGE_FONTS.get(language.display_name, ()),
        *NON_LATIN_LANGUAGE_FONTS.get(language.prompt_name, ()),
    )
    for font_path in font_paths:
        key = str(font_path)
        if key in _LOADED_FONT_PATHS:
            continue
        font_id = QFontDatabase.addApplicationFont(key)
        if font_id < 0:
            logger.warning("Failed to load font for %s: %s", language.display_name, key)
            continue
        _LOADED_FONT_PATHS.add(key)


class LanguageState:
    def __init__(self, initial: LanguageOption):
        self._lock = threading.Lock()
        self._language = initial

    @property
    def current(self) -> LanguageOption:
        with self._lock:
            return self._language

    def set_language(self, language: LanguageOption):
        with self._lock:
            self._language = language


class PartialUpdateThrottle:
    """Keep streamed token updates from flooding the Qt event queue."""

    def __init__(self, window: "ChatWindow", *, min_interval_s: float):
        self.window = window
        self.min_interval_s = max(0.0, min_interval_s)
        self._last_emit_s = 0.0
        self._last_text = ""

    def __call__(self, text: str):
        text = str(text)
        if text == self._last_text:
            return
        now = time.monotonic()
        if self._last_emit_s and now - self._last_emit_s < self.min_interval_s:
            self._last_text = text
            return
        self.flush(text)

    def flush(self, text: str | None = None):
        if text is not None:
            self._last_text = str(text)
        if not self._last_text:
            return
        self._last_emit_s = time.monotonic()
        self.window.update_response_text(self._last_text, replace=True)


class TranslationWorker(threading.Thread):
    def __init__(
        self,
        *,
        recognizer: "SpeechRecognizer",
        translator: GemmaTranslationService,
        language_state: LanguageState,
        window: "ChatWindow",
        min_words: int,
        partial_update_ms: int,
        show_stats: bool,
    ):
        super().__init__(name="translation-worker", daemon=True)
        self.recognizer = recognizer
        self.translator = translator
        self.language_state = language_state
        self.window = window
        self.min_words = min_words
        self.partial_update_s = max(0, partial_update_ms) / 1000
        self.show_stats = show_stats
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()
        try:
            self.recognizer.source.stop()
        except Exception:
            logger.debug("Failed to stop speech recognizer source", exc_info=True)

    def run(self):
        final_status = "Stopped"
        try:
            with self.recognizer:
                self.window.update_status("Active")
                logger.info("Audio worker initialized")
                while not self.stop_event.is_set():
                    transcript = self.recognizer.listen_once(stop_event=self.stop_event)
                    if transcript is None:
                        break
                    if not self._accept_transcript(transcript):
                        continue
                    self._translate(transcript)
        except Exception as exc:
            logger.exception("GUI translation worker failed")
            final_status = f"Error: {exc}"
        finally:
            self.window.update_status(final_status)

    def _accept_transcript(self, transcript: "SpeechTranscript") -> bool:
        if not getattr(self.window, "voice_active", False):
            return False
        text = transcript.text.strip()
        if not text:
            return False
        if len(text.split()) < self.min_words:
            logger.debug("Ignoring short transcript: %s", text)
            return False
        return True

    def _translate(self, transcript: "SpeechTranscript"):
        self.window.signals.update_user.emit("", False)
        self.window.signals.update_resp.emit("", False)
        self.window.update_user_text(
            transcript.text,
            stats=transcript.stats if self.show_stats else None,
        )
        if self.show_stats:
            self.window.update_stt_stats(transcript.stats)

        self.window.signals.mic_deactivate.emit()
        self.window.update_response_text("...", replace=False)
        partials = PartialUpdateThrottle(
            self.window,
            min_interval_s=self.partial_update_s,
        )
        language = self.language_state.current

        try:
            result = self.translator.translate(
                transcript.text,
                target_language=language.prompt_name,
                on_partial=partials,
            )
            self.window.update_translation_result(
                result,
                stats=result.stats if self.show_stats else None,
            )
            if self.show_stats:
                self.window.update_llm_stats(result.stats)
            logger.info("Translated to %s", language.display_name)
        except Exception as exc:
            logger.exception("Translation failed")
            self.window.update_response_text(f"Error: {exc}", replace=True)
        finally:
            self.recognizer.drain()


class CommSignals(QObject):
    update_user = pyqtSignal(str, bool)
    update_resp = pyqtSignal(str, bool)
    update_stat = pyqtSignal(str, str)
    update_status = pyqtSignal(str)
    mic_deactivate = pyqtSignal()


class MessageBubble(QFrame):
    def __init__(self, text: str, *, is_user: bool):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)
        self.setMaximumWidth(max(240, int(CHAT_WIDTH * 0.82)))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        bg_color = "#007AFF" if is_user else "#E1E1E1"
        text_color = "white" if is_user else "#1C1C1E"
        self.setStyleSheet(
            f"""
            MessageBubble {{
                background-color: {bg_color};
                border-radius: 14px;
                margin: 4px;
            }}
            QLabel {{
                color: {text_color};
                font-size: {BUBBLE_FONT_SIZE}px;
                font-family: 'Segoe UI', sans-serif;
            }}
            """
        )
        layout.addWidget(self.label)
        self.set_text(text)

    def set_text(self, text: str):
        self.label.setText(str(text))


class ChatWindow(QWidget):
    def __init__(
        self,
        *,
        language_state: LanguageState,
        max_chat_bubbles: int,
        show_stats: bool,
        verbose_stats: bool,
    ):
        super().__init__()
        self.language_state = language_state
        self.show_stats = show_stats
        self.verbose_stats = verbose_stats
        self.voice_active = False
        self._translator = None
        self._close_handler = None

        self.signals = CommSignals()
        self.signals.update_user.connect(self._handle_user_update)
        self.signals.update_resp.connect(self._handle_resp_update)
        self.signals.update_stat.connect(self._handle_stat_update)
        self.signals.update_status.connect(self._handle_status_update)
        self.signals.mic_deactivate.connect(self._deactivate_mic)

        self._thinking_timer = QTimer(self)
        self._thinking_timer.setInterval(400)
        self._thinking_timer.timeout.connect(self._tick_thinking)
        self._thinking_step = 0

        self._init_ui()

    def set_close_handler(self, handler):
        self._close_handler = handler

    def set_translator(self, translator):
        self._translator = translator

    def _make_panel(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #dadce0;
            }
        """)
        return frame

    def _make_divider(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Plain)
        line.setStyleSheet("color: #dadce0; border: none; border-top: 1px solid #dadce0;")
        line.setFixedHeight(1)
        return line

    def _init_ui(self):
        self.setWindowTitle("Astra SL2610 Voice Translation Engine")
        self.setGeometry(0, 0, CHAT_WIDTH, CHAT_HEIGHT)
        self.setStyleSheet("background-color: #f1f3f4;")

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 8)
        root.setSpacing(8)

        title = QLabel("Language Translation with Gemma3 270M")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 18px; font-weight: 700; color: #202124; border: none;"
        )
        root.addWidget(title)

        # Orientation-aware split: portrait=vertical, landscape=horizontal
        if _ORIENTATION == "portrait":
            content = QVBoxLayout()
        else:
            content = QHBoxLayout()
        content.setSpacing(8)

        # ── Input panel (English, pinned) ──────────────────────────
        input_frame = self._make_panel()
        input_vbox = QVBoxLayout(input_frame)
        input_vbox.setContentsMargins(14, 10, 14, 10)
        input_vbox.setSpacing(6)

        input_header = QHBoxLayout()
        lang_label = QLabel("English")
        lang_label.setStyleSheet(
            "font-size: 15px; font-weight: 600; color: #5f6368; border: none;"
        )
        input_header.addWidget(lang_label)
        input_header.addStretch()
        self.mic_button = MicButton()
        self.mic_button.toggled.connect(self._on_mic_toggled)
        input_header.addWidget(self.mic_button)
        input_vbox.addLayout(input_header)
        input_vbox.addWidget(self._make_divider())

        self.input_text = QPlainTextEdit()
        self.input_text.setPlaceholderText("Enter text")
        self.input_text.setStyleSheet("""
            QPlainTextEdit {
                font-size: 24px;
                color: #202124;
                border: none;
                background: transparent;
            }
        """)
        self.input_text.setFrameShape(QFrame.NoFrame)
        self.input_text.installEventFilter(self)
        input_vbox.addWidget(self.input_text, stretch=1)

        content.addWidget(input_frame, stretch=1)

        # ── Output panel (translated language) ─────────────────────
        output_frame = self._make_panel()
        output_vbox = QVBoxLayout(output_frame)
        output_vbox.setContentsMargins(14, 10, 14, 10)
        output_vbox.setSpacing(6)

        output_header = QHBoxLayout()
        self.language_dropdown = QComboBox()
        self.language_dropdown.addItems(
            [language.display_name for language in LANGUAGES.values()]
        )
        self.language_dropdown.setCurrentText(self.language_state.current.display_name)
        self.language_dropdown.setMinimumWidth(200)
        self.language_dropdown.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.language_dropdown.setStyleSheet("""
            QComboBox {
                font-size: 15px;
                font-weight: 600;
                color: #1a73e8;
                border: 1px solid #dadce0;
                border-right: none;
                border-top-left-radius: 4px;
                border-bottom-left-radius: 4px;
                padding: 4px 8px;
                background: white;
            }
            QComboBox::drop-down { width: 0; border: none; }
        """)
        self.language_dropdown.currentTextChanged.connect(self.on_language_change)

        dropdown_arrow_btn = QPushButton("▼")
        dropdown_arrow_btn.setStyleSheet("""
            QPushButton {
                font-size: 10px;
                color: #1a73e8;
                background: white;
                border: 1px solid #dadce0;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
                padding: 4px 8px;
            }
            QPushButton:hover { background: #f8f9fa; }
        """)
        dropdown_arrow_btn.setCursor(Qt.PointingHandCursor)
        dropdown_arrow_btn.clicked.connect(self.language_dropdown.showPopup)

        # Force both to the same height so they align as one unit
        self.language_dropdown.setFixedHeight(34)
        dropdown_arrow_btn.setFixedHeight(34)

        output_header.addWidget(self.language_dropdown)
        output_header.addWidget(dropdown_arrow_btn)
        output_header.addStretch()
        self.clear_button = QPushButton()
        self.clear_button.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        self.clear_button.setFixedSize(36, 36)
        self.clear_button.setFlat(True)
        self.clear_button.setStyleSheet("border: none; background: transparent;")
        self.clear_button.clicked.connect(self.clear_chat)
        output_header.addWidget(self.clear_button)
        output_vbox.addLayout(output_header)
        output_vbox.addWidget(self._make_divider())

        self.output_text = QLabel()
        self.output_text.setWordWrap(True)
        self.output_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.output_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.output_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.output_text.setStyleSheet("""
            QLabel {
                font-size: 24px;
                color: #202124;
                border: none;
                background: transparent;
                padding: 2px;
            }
        """)
        output_vbox.addWidget(self.output_text, stretch=1)

        content.addWidget(output_frame, stretch=1)
        root.addLayout(content, stretch=1)

        # ── Bottom bar ──────────────────────────────────────────────
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("font-size: 13px; color: #5f6368; border: none;")
        root.addWidget(self.status_label, alignment=Qt.AlignLeft)

        stats_style = f"font-size: {STATS_FONT_SIZE}px; color: #5f6368; border: none;"
        self.stats_stt_label = QLabel("")
        self.stats_llm_label = QLabel("")
        self.stats_stt_label.setStyleSheet(stats_style)
        self.stats_llm_label.setStyleSheet(stats_style)
        if self.show_stats:
            root.addWidget(self.stats_stt_label, alignment=Qt.AlignRight)
            root.addWidget(self.stats_llm_label, alignment=Qt.AlignRight)

    # ── Event handling ──────────────────────────────────────────────

    def eventFilter(self, obj, event):
        if obj is self.input_text and event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                if not (event.modifiers() & Qt.ShiftModifier):
                    self._submit_text_input()
                    return True
        return super().eventFilter(obj, event)

    def _on_mic_toggled(self, active: bool):
        self.voice_active = active
        self.update_status("Listening (voice)" if active else "Listening (text)")

    def _deactivate_mic(self):
        self.voice_active = False
        self.mic_button.setChecked(False)
        self.mic_button.setEnabled(False)
        self.update_status("Listening (text)")

    def _start_thinking_anim(self):
        self._thinking_step = 0
        self._thinking_timer.start()

    def _stop_thinking_anim(self):
        self._thinking_timer.stop()
        self.mic_button.setEnabled(True)

    def _tick_thinking(self):
        dots = "." * ((self._thinking_step % 3) + 1)
        self.output_text.setText(dots)
        self._thinking_step += 1

    def _submit_text_input(self):
        text = self.input_text.toPlainText().strip()
        if not text or self._translator is None:
            return
        self.input_text.clear()
        self.output_text.clear()
        self.update_user_text(text)
        self.update_response_text("...", replace=False)
        language = self.language_state.current
        translator = self._translator
        throttle = PartialUpdateThrottle(self, min_interval_s=DEFAULT_PARTIAL_UPDATE_MS / 1000)

        def _run():
            try:
                result = translator.translate(
                    text,
                    target_language=language.prompt_name,
                    on_partial=throttle,
                )
                self.update_translation_result(
                    result,
                    stats=result.stats if self.show_stats else None,
                )
                if self.show_stats:
                    self.update_llm_stats(result.stats)
            except Exception as exc:
                self.update_response_text(f"Error: {exc}", replace=True)

        threading.Thread(target=_run, daemon=True, name="text-translate").start()

    # ── Public update API (called from worker thread) ───────────────

    def on_language_change(self, selected_language: str):
        language = language_by_display_name(selected_language)
        ensure_language_fonts_loaded(language)
        self.language_state.set_language(language)
        logger.info("Language changed to: %s", language.display_name)

    def update_user_text(
        self,
        text: str,
        *,
        stats: InferenceStats | None = None,
        replace: bool = False,
    ):
        suffix = self._fmt_stats(stats)
        if replace:
            print(f"\r[You] {text}{suffix}", end="", flush=True)
        else:
            print(f"[You] {text}{suffix}", flush=True)
        self.signals.update_user.emit(str(text), replace)

    def update_response_text(
        self,
        text: str,
        *,
        stats: InferenceStats | None = None,
        replace: bool = False,
    ):
        suffix = self._fmt_stats(stats)
        line = f"[Translation] {text}{suffix}"
        if replace:
            print(f"\r{line:<120}", end="", flush=True)
        else:
            print(line, flush=True)
        self.signals.update_resp.emit(str(text), replace)

    def update_translation_result(
        self,
        result: TranslationResult,
        *,
        stats: InferenceStats | None = None,
    ):
        self.update_response_text(result.text, stats=stats, replace=True)
        print(flush=True)

    def update_stt_stats(self, stats: InferenceStats):
        self.signals.update_stat.emit(
            "stt",
            f"Moonshine: {stats.fmt(verbose=self.verbose_stats)}",
        )

    def update_llm_stats(self, stats: InferenceStats):
        self.signals.update_stat.emit(
            "llm",
            f"Gemma: {stats.fmt(verbose=self.verbose_stats)}",
        )

    def update_status(self, text: str):
        self.signals.update_status.emit(text)

    def clear_chat(self):
        self.input_text.clear()
        self.output_text.clear()
        self.stats_stt_label.setText("")
        self.stats_llm_label.setText("")

    # ── Signal handlers (Qt main thread) ───────────────────────────

    def _handle_user_update(self, text: str, replace: bool):
        self.input_text.setPlainText(text)

    def _handle_resp_update(self, text: str, replace: bool):
        if text == "...":
            self._start_thinking_anim()
        else:
            self._stop_thinking_anim()
            self.output_text.setText(text)

    def _handle_stat_update(self, kind: str, text: str):
        if kind == "stt":
            self.stats_stt_label.setText(text)
        elif kind == "llm":
            self.stats_llm_label.setText(text)

    def _handle_status_update(self, text: str):
        self.status_label.setText(text)

    def _fmt_stats(self, stats: InferenceStats | None) -> str:
        if not self.show_stats or stats is None:
            return ""
        return f"  ({stats.fmt(verbose=self.verbose_stats)})"

    def closeEvent(self, event):
        if self._close_handler is not None:
            self._close_handler()
        super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Astra language translation GUI using Moonshine and Gemma"
    )
    add_common_translation_args(parser)
    add_gui_args(parser)
    return parser.parse_args()


def add_gui_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("GUI output options")
    group.add_argument(
        "--hide-stats",
        action="store_true",
        help="Do not show STT/LLM inference stats.",
    )
    group.add_argument(
        "--verbose-stats",
        action="store_true",
        help="Show token counts, prefill/static rates, and total inference latency.",
    )
    group.add_argument(
        "--max-chat-bubbles",
        type=int,
        default=DEFAULT_MAX_CHAT_BUBBLES,
        help="Maximum message bubbles retained in the chat view.",
    )
    group.add_argument(
        "--partial-update-ms",
        type=int,
        default=DEFAULT_PARTIAL_UPDATE_MS,
        help="Minimum interval between streamed translation UI updates.",
    )
    group.add_argument(
        "--windowed",
        action="store_true",
        help="Run in a window instead of fullscreen.",
    )


def choose_audio_device(device_arg: str | None) -> int | str | None:
    from utils.speech import query_input_devices

    if device_arg is None:
        print("List of Audio input devices:")
        print(query_input_devices())
        device_arg = input("Enter input device to listen on [default]: ").strip()

    if device_arg == "":
        return None
    try:
        return int(device_arg)
    except ValueError:
        return device_arg


def language_by_display_name(name: str) -> LanguageOption:
    for language in LANGUAGES.values():
        if language.display_name == name:
            return language
    raise ValueError(f"Unsupported language: {name}")


def build_translation_service(args: argparse.Namespace) -> GemmaTranslationService:
    from utils.gemma import load_gemma

    backend = load_gemma(
        use_llama=args.use_llama_gemma,
        model_path=resolve_gemma_model_path(args),
        instruct_model=not args.non_instruct_model,
    )
    return GemmaTranslationService(backend)


def build_speech_recognizer(
    args: argparse.Namespace,
    *,
    audio_device: int | str | None,
) -> "SpeechRecognizer":
    from utils.speech import (
        MoonshineTranscriber,
        SileroSpeechSegmenter,
        SoundDeviceAudioSource,
        SpeechRecognizer,
    )

    suppress_native_logs = not args.show_native_logs
    transcriber = MoonshineTranscriber(
        args.moonshine_model,
        suppress_native_logs=suppress_native_logs,
    )
    source = SoundDeviceAudioSource(
        device=audio_device,
        sample_rate=SAMPLING_RATE,
        chunk_size=CHUNK_SIZE,
        suppress_native_logs=suppress_native_logs,
    )
    segmenter = SileroSpeechSegmenter(
        sample_rate=SAMPLING_RATE,
        chunk_size=CHUNK_SIZE,
        threshold=args.vad_threshold,
        min_silence_duration_ms=args.silence_ms,
        max_speech_secs=args.max_speech_secs,
        min_segment_secs=args.min_segment_secs,
    )
    return SpeechRecognizer(
        transcriber=transcriber,
        source=source,
        segmenter=segmenter,
    )


def configure_qt_environment():
    # Preserve a correct XDG_RUNTIME_DIR set by PAM/systemd; only fall back
    # to the root-user path if nothing is set.
    xdg = os.environ.get("XDG_RUNTIME_DIR", "/var/run/user/0")
    os.environ["XDG_RUNTIME_DIR"] = xdg
    os.environ["WESTON_DISABLE_GBM_MODIFIERS"] = "true"

    # Prevent X11/Xwayland from being selected over the Wayland backend even
    # when DISPLAY is set in the environment (e.g. by Xwayland).
    os.environ.pop("DISPLAY", None)

    # Disable xdg-decoration protocol negotiation. Older Weston builds and some
    # embedded compositor configs don't support it and will kill the client
    # connection during the handshake, causing a silent startup crash.
    os.environ["QT_QPA_WAYLAND_DISABLE_WINDOWDECORATION"] = "1"
    os.environ["QT_WAYLAND_DISABLE_WINDOWDECORATION"] = "1"

    wayland_display = os.environ.get("WAYLAND_DISPLAY", "")
    if not wayland_display:
        for candidate in ("wayland-1", "wayland-0"):
            if os.path.exists(os.path.join(xdg, candidate)):
                wayland_display = candidate
                break

    if not wayland_display:
        raise RuntimeError(
            f"No Wayland socket found in {xdg} — is Weston running?"
        )

    os.environ["WAYLAND_DISPLAY"] = wayland_display
    os.environ["QT_QPA_PLATFORM"] = "wayland"
    print(f"[Qt] WAYLAND_DISPLAY={wayland_display}", flush=True)


def main() -> int:
    args = parse_args()
    configure_logging(args.logging)

    if not args.no_npu_clock:
        ok, message = enable_npu_clock()
        print(f"[NPU] {message}" if ok else f"[NPU] {message}", flush=True)

    translator = build_translation_service(args)
    audio_device = choose_audio_device(args.audio_device)
    recognizer = build_speech_recognizer(args, audio_device=audio_device)

    configure_qt_environment()
    app = QApplication([sys.argv[0]])

    language_state = LanguageState(language_by_display_name(args.language))
    ensure_language_fonts_loaded(language_state.current)
    window = ChatWindow(
        language_state=language_state,
        max_chat_bubbles=args.max_chat_bubbles,
        show_stats=not args.hide_stats,
        verbose_stats=args.verbose_stats,
    )

    window.set_translator(translator)

    worker = TranslationWorker(
        recognizer=recognizer,
        translator=translator,
        language_state=language_state,
        window=window,
        min_words=args.min_words,
        partial_update_ms=args.partial_update_ms,
        show_stats=not args.hide_stats,
    )
    window.set_close_handler(worker.stop)
    app.aboutToQuit.connect(worker.stop)
    signal.signal(signal.SIGINT, lambda *_: (worker.stop(), app.quit()))

    if args.windowed:
        window.show()
    else:
        window.showFullScreen()

    worker.start()
    exit_code = app.exec_()
    worker.stop()
    worker.join(timeout=2.0)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
