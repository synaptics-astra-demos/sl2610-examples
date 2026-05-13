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

from PyQt5.QtCore import QObject, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

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
                self.window.update_status("Listening")
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
        text = transcript.text.strip()
        if not text:
            return False
        if len(text.split()) < self.min_words:
            logger.debug("Ignoring short transcript: %s", text)
            return False
        return True

    def _translate(self, transcript: "SpeechTranscript"):
        self.window.update_user_text(
            transcript.text,
            stats=transcript.stats if self.show_stats else None,
        )
        if self.show_stats:
            self.window.update_stt_stats(transcript.stats)

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
        self.max_chat_bubbles = max(2, int(max_chat_bubbles))
        self.show_stats = show_stats
        self.verbose_stats = verbose_stats
        self.active_user_bubble: MessageBubble | None = None
        self.active_resp_bubble: MessageBubble | None = None
        self._message_bubbles: deque[MessageBubble] = deque()
        self._close_handler = None

        self.signals = CommSignals()
        self.signals.update_user.connect(self._handle_user_update)
        self.signals.update_resp.connect(self._handle_resp_update)
        self.signals.update_stat.connect(self._handle_stat_update)
        self.signals.update_status.connect(self._handle_status_update)

        self._init_ui()

    def set_close_handler(self, handler):
        self._close_handler = handler

    def _init_ui(self):
        self.setWindowTitle("Astra SL2610 Voice Translation Engine")
        self.setGeometry(0, 0, CHAT_WIDTH, CHAT_HEIGHT)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)

        self.container = QWidget()
        self.chat_layout = QVBoxLayout(self.container)
        self.chat_layout.setContentsMargins(6, 6, 6, 6)
        self.chat_layout.addStretch()
        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(
            "font-size: 16px; color: #333; padding: 2px 6px;"
        )
        layout.addWidget(self.status_label, alignment=Qt.AlignLeft)

        self.stats_stt_label = QLabel("")
        self.stats_llm_label = QLabel("")
        stats_style = (
            f"font-size: {STATS_FONT_SIZE}px; color: #333; "
            "background: #f5f5f5; padding: 6px; border-radius: 8px;"
        )
        self.stats_stt_label.setStyleSheet(stats_style)
        self.stats_llm_label.setStyleSheet(stats_style)
        if self.show_stats:
            layout.addWidget(self.stats_stt_label, alignment=Qt.AlignRight)
            layout.addWidget(self.stats_llm_label, alignment=Qt.AlignRight)

        button_row = QHBoxLayout()
        self.language_dropdown = QComboBox()
        self.language_dropdown.addItems([language.display_name for language in LANGUAGES.values()])
        self.language_dropdown.setCurrentText(self.language_state.current.display_name)
        self.language_dropdown.setMinimumHeight(48)
        self.language_dropdown.setStyleSheet("font-size: 16px; padding: 5px;")
        self.language_dropdown.currentTextChanged.connect(self.on_language_change)
        button_row.addWidget(self.language_dropdown)
        button_row.addStretch()

        self.settings_button = QPushButton()
        self.settings_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.settings_button.setMinimumSize(80, 48)
        self.settings_button.setMaximumSize(80, 48)
        self.settings_button.setIconSize(QSize(40, 40))
        self.settings_button.clicked.connect(self.open_settings)
        button_row.addWidget(self.settings_button)

        self.clear_button = QPushButton()
        self.clear_button.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        self.clear_button.setMinimumSize(80, 48)
        self.clear_button.setMaximumSize(80, 48)
        self.clear_button.setIconSize(QSize(40, 40))
        self.clear_button.clicked.connect(self.clear_chat)
        button_row.addWidget(self.clear_button)
        layout.addLayout(button_row)

    def on_language_change(self, selected_language: str):
        language = language_by_display_name(selected_language)
        ensure_language_fonts_loaded(language)
        self.language_state.set_language(language)
        logger.info("Language changed to: %s", language.display_name)

    def open_settings(self):
        QMessageBox.information(self, "Settings", "Settings not implemented yet.")

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
        while self._message_bubbles:
            self._delete_bubble(self._message_bubbles.popleft())
        self.active_user_bubble = None
        self.active_resp_bubble = None
        self.stats_stt_label.setText("")
        self.stats_llm_label.setText("")

    def _handle_user_update(self, text: str, replace: bool):
        if replace and self.active_user_bubble is not None:
            self.active_user_bubble.set_text(text)
        else:
            self.active_user_bubble = self._insert_bubble(text, is_user=True)
        self._scroll_to_bottom()

    def _handle_resp_update(self, text: str, replace: bool):
        if replace and self.active_resp_bubble is not None:
            self.active_resp_bubble.set_text(text)
        else:
            self.active_resp_bubble = self._insert_bubble(text, is_user=False)
        self._scroll_to_bottom()

    def _handle_stat_update(self, kind: str, text: str):
        if kind == "stt":
            self.stats_stt_label.setText(text)
        elif kind == "llm":
            self.stats_llm_label.setText(text)

    def _handle_status_update(self, text: str):
        self.status_label.setText(text)

    def _insert_bubble(self, text: str, *, is_user: bool) -> MessageBubble:
        bubble = MessageBubble(text, is_user=is_user)
        alignment = Qt.AlignRight if is_user else Qt.AlignLeft
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble, alignment=alignment)
        self._message_bubbles.append(bubble)
        self._prune_old_bubbles()
        return bubble

    def _prune_old_bubbles(self):
        while len(self._message_bubbles) > self.max_chat_bubbles:
            bubble = self._message_bubbles.popleft()
            if bubble is self.active_user_bubble:
                self.active_user_bubble = None
            if bubble is self.active_resp_bubble:
                self.active_resp_bubble = None
            self._delete_bubble(bubble)

    def _delete_bubble(self, bubble: MessageBubble):
        self.chat_layout.removeWidget(bubble)
        bubble.setParent(None)
        bubble.deleteLater()

    def _scroll_to_bottom(self):
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

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
    os.environ["XDG_RUNTIME_DIR"] = "/var/run/user/0"
    os.environ["WESTON_DISABLE_GBM_MODIFIERS"] = "true"
    os.environ["WAYLAND_DISPLAY"] = "wayland-1"
    os.environ["QT_QPA_PLATFORM"] = "wayland"


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
