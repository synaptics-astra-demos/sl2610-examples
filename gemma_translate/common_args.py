from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
GEMMA_LLAMA_MODEL_PATH = (
    _THIS_DIR / ".." / "models" / "gemma-3-270m-it-Q8_0.gguf"
).resolve()
@dataclass(frozen=True)
class LanguageOption:
    display_name: str
    prompt_name: str


LANGUAGES = {
    "1": LanguageOption("Spanish", "Spanish"),
    "2": LanguageOption("French", "French"),
    "3": LanguageOption("German", "German"),
}


def add_common_translation_args(parser: argparse.ArgumentParser):
    """Add frontend-agnostic options shared by CLI and GUI translation runners."""

    add_translation_args(parser)
    add_inference_args(parser)
    add_speech_audio_args(parser)
    add_logging_runtime_args(parser)


def add_translation_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("translation options")
    group.add_argument(
        "--language",
        choices=[language.display_name for language in LANGUAGES.values()],
        default="Spanish",
        help="Initial target language.",
    )


def add_inference_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("inference options")
    group.add_argument(
        "--use-llama-gemma",
        action="store_true",
        help="Use llama.cpp (GGUF) backend instead of the default torq VMFB backend.",
    )
    group.add_argument(
        "--gemma-model",
        type=str,
        default=None,
        help=(
            "Path to the Gemma model file (.vmfb for torq, .gguf for llama). "
            "Defaults to HF download for torq or the GGUF download for llama."
        ),
    )
    group.add_argument(
        "--moonshine-model",
        type=str,
        default=None,
        help="Path to the Moonshine model directory. Defaults to HF download/discovery.",
    )
    group.add_argument(
        "--non-instruct-model",
        action="store_true",
        default=False,
        help="Use a non-instruct Gemma model.",
    )
    group.add_argument(
        "--no-npu-clock",
        action="store_true",
        help="Skip devmem NPU clock setup.",
    )


def add_speech_audio_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("speech/audio options")
    group.add_argument(
        "--audio-device",
        default=None,
        help="sounddevice input device index or name. If omitted, prompt interactively.",
    )
    group.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="Silero VAD speech threshold.",
    )
    group.add_argument(
        "--silence-ms",
        type=int,
        default=150,
        help="Silence duration that ends one utterance.",
    )
    group.add_argument(
        "--max-speech-secs",
        type=float,
        default=10.0,
        help="Maximum utterance length before forcing transcription.",
    )
    group.add_argument(
        "--min-segment-secs",
        type=float,
        default=0.20,
        help="Discard VAD segments shorter than this after resetting the recorder.",
    )
    group.add_argument(
        "--min-words",
        type=int,
        default=1,
        help="Minimum transcript word count to send to Gemma.",
    )


def add_logging_runtime_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("logging/runtime options")
    group.add_argument(
        "--logging",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging verbosity: %(choices)s (default: %(default)s)",
    )
    group.add_argument(
        "--show-native-logs",
        action="store_true",
        help="Show native C/C++ logs from PortAudio/ALSA and ONNX Runtime.",
    )


def resolve_gemma_model_path(args: argparse.Namespace) -> str | None:
    if args.gemma_model is not None:
        return args.gemma_model
    if args.use_llama_gemma:
        return str(GEMMA_LLAMA_MODEL_PATH)
    return None
