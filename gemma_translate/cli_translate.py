from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from typing import TYPE_CHECKING

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gemma_translate.common_args import (
    LANGUAGES,
    LanguageOption,
    add_common_translation_args,
    resolve_gemma_model_path,
)
from gemma_translate.translation import GemmaTranslationService, TranslationResult
from app_utils.cli import TerminalMode, install_cli_shutdown_handlers
from app_utils.log import configure_logging
from app_utils.npu import enable_npu_clock
from app_utils.stats import Gemma3InferenceStats, MoonshineInferenceStats

if TYPE_CHECKING:
    from app_utils.speech import SpeechRecognizer, SpeechTranscript


SAMPLING_RATE = 16_000
CHUNK_SIZE = 512


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


class CliPrinter:
    def __init__(self, *, show_stats: bool, verbose_stats: bool):
        self.show_stats = show_stats
        self.verbose_stats = verbose_stats
        self._lock = threading.Lock()
        self._translation_line_width = 0

    def show_header(self, text_mode: bool = False):
        with self._lock:
            print("\n=== Astra SL2610 Voice Translation Engine ===")
            if text_mode:
                print("Type a phrase and press Enter to translate.")
                print("Use /1-/6 to switch language, /q to quit:")
            else:
                print("Press a listed number to change language:")
            for key, language in LANGUAGES.items():
                print(f"  /{key}: {language.display_name}" if text_mode else f"  {key}: {language.display_name}")
            if not text_mode:
                print("Speak to translate. Press Ctrl+C to exit.")
            print()

    def status(self, message: str):
        with self._lock:
            print(message, flush=True)

    def ready(self, language: LanguageOption):
        with self._lock:
            print(f"[Ready] Listening...", flush=True)

    def user(self, transcript: SpeechTranscript):
        suffix = self._fmt_stats(transcript.stats)
        with self._lock:
            print(f"[You] {transcript.text}{suffix}", flush=True)

    def ignored(self, transcript: SpeechTranscript, reason: str):
        suffix = self._fmt_stats(transcript.stats)
        with self._lock:
            print(f"[Ignored] {transcript.text}{suffix} ({reason})", flush=True)

    def translation_partial(self, text: str):
        line = f"[Translation] {text.strip() or '...'}"
        with self._lock:
            self._translation_line_width = max(self._translation_line_width, len(line))
            print(f"\r{line:<{self._translation_line_width}}", end="", flush=True)

    def translation_final(self, result: TranslationResult):
        suffix = self._fmt_stats(result.stats)
        line = f"[Translation] {result.text}{suffix}"
        with self._lock:
            width = max(self._translation_line_width, len(line))
            print(f"\r{line:<{width}}", flush=True)
            print(flush=True)
            self._translation_line_width = 0

    def error(self, message: str):
        with self._lock:
            print(f"\n[Error] {message}", flush=True)
            self._translation_line_width = 0

    def _fmt_stats(self, stats: MoonshineInferenceStats | Gemma3InferenceStats) -> str:
        return f"  ({stats.fmt(verbose=self.verbose_stats)})" if self.show_stats else ""


class KeyboardLanguageController:
    def __init__(
        self,
        *,
        state: LanguageState,
        printer: CliPrinter,
        stop_event: threading.Event,
    ):
        self.state = state
        self.printer = printer
        self.stop_event = stop_event
        self._terminal = TerminalMode()
        self._thread: threading.Thread | None = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name="language-keys", daemon=True)
        self._thread.start()

    def stop(self):
        self.stop_event.set()
        self.restore_terminal()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def restore_terminal(self):
        self._terminal.restore()

    def _run(self):
        if not self._terminal.enter_cbreak():
            return

        try:
            while not self.stop_event.is_set():
                ch = self._terminal.read_key(timeout=0.1)
                if ch is None:
                    continue
                if ch in LANGUAGES:
                    language = LANGUAGES[ch]
                    self.state.set_language(language)
                    self.printer.status(f"\n[Language changed to: {language.display_name}]")
                elif ch == "\x03":
                    self.stop_event.set()
                    os.kill(os.getpid(), signal.SIGINT)
                    return
        finally:
            self.restore_terminal()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Astra language translation CLI using Moonshine and Gemma"
    )
    add_common_translation_args(parser)
    add_cli_output_args(parser)
    return parser.parse_args()


def add_cli_output_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("CLI output options")
    group.add_argument(
        "--text",
        action="store_true",
        help="Text input mode: type phrases instead of using voice.",
    )
    group.add_argument(
        "--hide-stats",
        action="store_true",
        help="Do not print STT/LLM inference stats.",
    )
    group.add_argument(
        "--verbose-stats",
        action="store_true",
        help="Print decoded token count, prefill/static rates, and total inference latency.",
    )


def choose_audio_device(device_arg: str | None) -> int | str | None:
    from app_utils.speech import query_input_devices

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


def initial_language(name: str) -> LanguageOption:
    for language in LANGUAGES.values():
        if language.display_name == name:
            return language
    raise ValueError(f"Unsupported language: {name}")


def transcript_is_accepted(transcript: SpeechTranscript, *, min_words: int) -> tuple[bool, str]:
    text = transcript.text.strip()
    if not text:
        return False, "empty transcript"
    if len(text.split()) < min_words:
        return False, f"fewer than {min_words} words"
    return True, ""


def build_translation_service(args: argparse.Namespace) -> GemmaTranslationService:
    from app_utils.gemma import load_gemma

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
) -> SpeechRecognizer:
    from app_utils.speech import (
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


def run_text_mode(
    *,
    printer: CliPrinter,
    translator: GemmaTranslationService,
    language_state: LanguageState,
    stop_event: threading.Event,
):
    printer.show_header(text_mode=True)
    while not stop_event.is_set():
        try:
            line = input("→ ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.startswith("/"):
            cmd = line[1:].strip()
            if cmd == "q":
                break
            elif cmd in LANGUAGES:
                language = LANGUAGES[cmd]
                language_state.set_language(language)
                printer.status(f"[Language changed to: {language.display_name}]")
            else:
                printer.status(f"[Unknown command: {line!r}]  (use /1-/6 or /q)")
        else:
            printer.status(f"[You] {line}")
            language = language_state.current
            try:
                result = translator.translate(
                    line,
                    target_language=language.prompt_name,
                    on_partial=printer.translation_partial,
                )
                printer.translation_final(result)
            except KeyboardInterrupt:
                break
            except Exception as exc:
                printer.error(str(exc))
    stop_event.set()


def main():
    args = parse_args()
    configure_logging(args.logging)

    stop_event = threading.Event()
    language_state = LanguageState(initial_language(args.language))
    printer = CliPrinter(
        show_stats=not args.hide_stats,
        verbose_stats=args.verbose_stats,
    )
    keyboard = KeyboardLanguageController(
        state=language_state,
        printer=printer,
        stop_event=stop_event,
    )
    install_cli_shutdown_handlers(
        lambda: (stop_event.set(), keyboard.restore_terminal()),
        raise_on_signal=True,
    )

    recognizer = None

    try:
        if not args.no_npu_clock:
            ok, message = enable_npu_clock()
            printer.status(f"[NPU] {message}" if ok else f"[NPU] {message}")

        translator = build_translation_service(args)

        if args.text:
            run_text_mode(
                printer=printer,
                translator=translator,
                language_state=language_state,
                stop_event=stop_event,
            )
        else:
            audio_device = choose_audio_device(args.audio_device)
            recognizer = build_speech_recognizer(args, audio_device=audio_device)

            keyboard.start()
            printer.show_header()

            with recognizer:
                while not stop_event.is_set():
                    language = language_state.current
                    printer.ready(language)

                    transcript = recognizer.listen_once(stop_event=stop_event)
                    if transcript is None:
                        break

                    accepted, reason = transcript_is_accepted(
                        transcript,
                        min_words=args.min_words,
                    )
                    if not accepted:
                        printer.ignored(transcript, reason)
                        continue

                    printer.user(transcript)
                    language = language_state.current
                    try:
                        result = translator.translate(
                            transcript.text,
                            target_language=language.prompt_name,
                            on_partial=printer.translation_partial,
                        )
                        printer.translation_final(result)
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        printer.error(str(exc))
                    finally:
                        recognizer.drain()

    except KeyboardInterrupt:
        stop_event.set()
        printer.status("\nExiting.")
    finally:
        stop_event.set()
        keyboard.stop()
        if recognizer is not None:
            recognizer.source.stop()


if __name__ == "__main__":
    main()
