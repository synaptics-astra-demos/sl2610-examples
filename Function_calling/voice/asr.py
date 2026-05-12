"""ASR (speech recognition) backends for the Coral demo.

ASR is a Protocol: any object with `transcribe(audio: np.ndarray, sample_rate: int) -> str`.

Two impls:
    - StubASR: rotates canned phrases (one per utterance). Useful for end-to-end
      pipeline verification without loading a real ASR model.
    - MoonshineASR: loads Moonshine ONNX/VMFB on the Torq NPU via the vendored
      runtime under ``voice/_runtime/``. Mirrors the Transcriber in the
      standalone Moonshine example.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .mic import VoiceUnavailable


logger = logging.getLogger(__name__)


# Mirrors astra-2610-assistant-UI-app/src/app_voice_pyqt.py:417-421.
_MOONSHINE_INPUT_SECS: int = 5
_MOONSHINE_TOKENS_PER_SEC: int = 6
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
    """Moonshine ASR on the Torq NPU via the vendored runtime under
    ``voice/_runtime/``.

    Resolution order for ``model_dir``:
      1. explicit constructor arg (e.g. from ``--moonshine-dir``)
      2. ``Function_calling/models/moonshine/`` (default; populated by
         ``scripts/setup.sh --voice``)

    Heavy deps (onnxruntime, iree.runtime, ml_dtypes, tokenizers, plus
    requests/tqdm/huggingface_hub pulled in transitively) are imported
    lazily inside ``__init__`` so a host without them keeps the rest of
    the demo importable. The factory in ``voice/pipeline.py`` catches
    the resulting VoiceUnavailable + FileNotFoundError so voice degrades
    to "disabled" rather than crashing the app.
    """

    def __init__(
        self,
        model_dir: str | os.PathLike | None = None,
        model_size: str = "tiny",
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
            from tokenizers import Tokenizer  # type: ignore[import-not-found]

            from ._runtime.moonshine import load_moonshine
            from ._runtime.download import download_from_hf
        except ImportError as e:
            raise VoiceUnavailable(
                f"moonshine deps not available: {e}. The Moonshine voice "
                "path requires onnxruntime, ml_dtypes, tokenizers, "
                "huggingface_hub, requests, tqdm, and torq_runtime to be "
                "installed."
            ) from e

        tokenizer_path = resolved / "tokenizer.json"
        if tokenizer_path.is_file():
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(
                download_from_hf("UsefulSensors/moonshine-tiny", "tokenizer.json")
            )

        max_inp_len = _MOONSHINE_INPUT_SECS * _MOONSHINE_SAMPLE_RATE
        max_dec_len = _MOONSHINE_INPUT_SECS * _MOONSHINE_TOKENS_PER_SEC

        logger.info("loading Moonshine artifacts from %s", resolved)
        self._runner = load_moonshine(
            resolved, model_size, max_inp_len, max_dec_len,
        )
        self._tokenizer = tokenizer
        self._sample_rate = _MOONSHINE_SAMPLE_RATE

        # Warm-up pass on silence so the first user utterance hits a
        # primed graph (mirrors the Transcriber in the standalone
        # Moonshine example).
        warmup = np.zeros(self._sample_rate, dtype=np.float32)
        self._runner.run(warmup[np.newaxis, :])
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
        tokens = self._runner.run(audio[np.newaxis, :].astype(np.float32))
        text = self._tokenizer.decode_batch(tokens, skip_special_tokens=True)[0]
        return text.strip()

    @property
    def last_infer_time(self) -> float:
        return self._runner.last_infer_time
