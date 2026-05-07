"""VoicePipeline — thread-managed mic → VAD → ASR → text emit.

UI usage:
    pipe = make_voice_pipeline(
        on_text=lambda text: signal.emit(text),
        mode="moonshine",
        mic_device=None,
        moonshine_dir=None,
    )
    if pipe is None:           # voice unavailable; hide button
        ...
    pipe.start()               # opens mic, runs VAD loop
    pipe.stop()                # closes mic
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from .asr import ASR, MoonshineASR, StubASR
from .mic import MicStream, VoiceUnavailable
from .vad import VAD


logger = logging.getLogger(__name__)


VoiceMode = str  # "off" | "stub" | "moonshine"


class VoicePipeline:
    """Run mic + VAD + ASR in a background thread.

    on_text(text: str) is called from the worker thread for each finalised
    utterance. Callers wanting to touch Qt widgets should marshal via a
    Qt signal.
    """

    def __init__(
        self,
        asr: ASR,
        on_text: Callable[[str], None] | None = None,
        device: int | str | None = None,
    ) -> None:
        self._asr = asr
        self._on_text: Callable[[str], None] = on_text or (lambda _t: None)
        self._mic = MicStream(device=device)
        self._vad = VAD(
            sample_rate=self._mic.sample_rate,
            chunk_samples=self._mic.chunk_samples,
        )
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

    def set_callback(self, on_text: Callable[[str], None]) -> None:
        """Replace the on_text callback. Used to bind a Qt signal after
        the UI is constructed — the pipeline factory runs at startup,
        before the window (and its signals) exist."""
        self._on_text = on_text

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_evt.clear()
        self._mic.start()
        self._thread = threading.Thread(target=self._run, daemon=True, name="voice-pipeline")
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        self._mic.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop_evt.is_set():
            try:
                chunk, status = self._mic.queue.get(timeout=0.5)
            except Exception:
                continue
            if status:
                logger.debug("mic status: %s", status)
            utterance = self._vad.feed(chunk)
            if utterance is None:
                continue
            try:
                text = self._asr.transcribe(utterance, self._mic.sample_rate)
            except Exception:
                logger.exception("ASR failed")
                continue
            text = text.strip()
            if text:
                try:
                    self._on_text(text)
                except Exception:
                    logger.exception("on_text callback failed")


def _coerce_mic_device(raw: str | int | None) -> int | str | None:
    """Sounddevice accepts either a device index (int) or a substring (str).
    CLI args arrive as strings; coerce numeric strings to int so e.g.
    --mic 0 selects device index 0, while --mic hw:0,0 stays as a name."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    raw = raw.strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


def make_voice_pipeline(
    on_text: Callable[[str], None],
    mode: VoiceMode = "off",
    mic_device: str | int | None = None,
    moonshine_dir: str | Path | None = None,
) -> VoicePipeline | None:
    """Build a VoicePipeline; return None if voice is off or required deps
    are missing.

    Args:
        on_text: callback invoked from the pipeline thread for each
            finalised utterance.
        mode: "off" | "stub" | "moonshine".
        mic_device: optional sounddevice selector (index or name substring).
        moonshine_dir: optional path to staged Moonshine artifacts. Used
            only when mode="moonshine".
    """
    mode = (mode or "off").lower()
    if mode == "off":
        return None

    if mode == "stub":
        asr: ASR = StubASR()
    elif mode == "moonshine":
        try:
            asr = MoonshineASR(model_dir=moonshine_dir)
        except (VoiceUnavailable, NotImplementedError, FileNotFoundError) as e:
            logger.warning("moonshine ASR unavailable: %s", e)
            return None
    else:
        logger.warning("unknown voice mode %r; voice disabled", mode)
        return None

    device = _coerce_mic_device(mic_device)

    try:
        return VoicePipeline(asr=asr, on_text=on_text, device=device)
    except VoiceUnavailable as e:
        logger.warning("voice unavailable: %s", e)
        return None
