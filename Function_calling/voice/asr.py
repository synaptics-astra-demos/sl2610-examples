"""ASR (speech recognition) backends for the Coral demo.

ASR is a Protocol: any object with `transcribe(audio: np.ndarray, sample_rate: int) -> str`.

Two impls:
    - StubASR: rotates canned phrases (one per utterance). Useful for end-to-end
      pipeline verification without loading a real ASR model.
    - MoonshineASR: loads Moonshine on the Torq NPU.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


logger = logging.getLogger(__name__)


class VoiceUnavailable(RuntimeError):
    """Raised when a voice dependency (sounddevice / silero-vad-notorch / model) is missing."""


_MOONSHINE_SAMPLE_RATE: int = 16_000


@runtime_checkable
class ASR(Protocol):
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


_STUB_PHRASES: tuple[str, ...] = (
    "turn on the lights",
    "set the lights to red",
    "blink the lights three times",
    "play a beep",
    "set an alarm in one minute",
    "what's the cpu temperature",
    "turn off the lights and play success",
    "party mode",
)


class StubASR:
    """Returns rotating canned phrases. Audio is ignored.

    Use to validate the mic -> VAD -> ASR -> dispatcher path without a real
    ASR model. Each call to transcribe() advances the rotation.
    """

    def __init__(self, phrases: tuple[str, ...] = _STUB_PHRASES) -> None:
        self._phrases = phrases
        self._idx = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        text = self._phrases[self._idx % len(self._phrases)]
        self._idx += 1
        return text


class MoonshineASR:
    """Moonshine ASR on the Torq NPU.

    Resolution order for ``model_dir``:
      1. explicit constructor arg (e.g. from ``--moonshine-dir``)
      2. ``Function_calling/models/moonshine/`` (default; populated by
         ``scripts/setup.sh --voice``)

    Heavy deps (torq.runtime, onnxruntime, ml_dtypes, tokenizers, etc.)
    are imported lazily inside ``__init__`` so a host without them keeps
    the rest of the demo importable. The factory in ``voice/pipeline.py``
    catches the resulting VoiceUnavailable + FileNotFoundError so voice
    degrades to "disabled" rather than crashing the app.
    """

    def __init__(
        self,
        model_dir: str | os.PathLike | None = None,
    ) -> None:
        resolved = self._resolve_model_dir(model_dir)
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"Moonshine model dir not found: {resolved}. Pass "
                "--moonshine-dir, or stage the artifacts under "
                "Function_calling/models/moonshine/ (run "
                "`scripts/setup.sh --voice` to fetch from HuggingFace)."
            )

        try:
            from utils.speech import MoonshineTranscriber
        except ImportError as e:
            raise VoiceUnavailable(
                f"moonshine deps not available: {e}. The Moonshine voice "
                "path requires torq_runtime, onnxruntime, ml_dtypes, "
                "tokenizers, and sounddevice to be installed."
            ) from e

        logger.info("loading Moonshine via shared MoonshineTranscriber from %s", resolved)
        try:
            self._transcriber = MoonshineTranscriber(
                resolved,
                sample_rate=_MOONSHINE_SAMPLE_RATE,
                warmup=True,
            )
        except Exception as e:
            raise VoiceUnavailable(
                f"failed to initialize MoonshineTranscriber: {e}"
            ) from e
        self._sample_rate = _MOONSHINE_SAMPLE_RATE
        logger.info("Moonshine warm-up complete")

    @staticmethod
    def _resolve_model_dir(model_dir: str | os.PathLike | None) -> Path:
        if model_dir is not None:
            return Path(model_dir)
        # voice/asr.py → Function_calling/models/moonshine/
        return Path(__file__).resolve().parent.parent / "models" / "moonshine"

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != self._sample_rate:
            raise ValueError(
                f"Moonshine expects {self._sample_rate} Hz, got {sample_rate}"
            )
        text, _stats = self._transcriber.transcribe_audio(audio)
        return text

    @property
    def last_infer_time(self) -> float:
        return self._transcriber.runner.last_infer_time
