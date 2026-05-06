"""VoicePipeline — thread-managed mic → VAD → ASR → text emit.

UI usage:
    pipe = make_voice_pipeline(on_text=lambda text: signal.emit(text))
    if pipe is None:           # voice unavailable; hide button
        ...
    pipe.start()               # opens mic, runs VAD loop
    pipe.stop()                # closes mic
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable

from .asr import ASR, MoonshineASR, StubASR
from .mic import MicStream, VoiceUnavailable
from .vad import VAD


logger = logging.getLogger(__name__)


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


def make_voice_pipeline(
    on_text: Callable[[str], None],
) -> VoicePipeline | None:
    """Build a VoicePipeline per CORAL_VOICE; return None if voice is off
    or required deps are missing.

    CORAL_VOICE values: off | stub | moonshine
    CORAL_MIC: optional device index or substring (sounddevice device selector)
    """
    mode = os.environ.get("CORAL_VOICE", "off").lower()
    if mode == "off":
        return None

    if mode == "stub":
        asr: ASR = StubASR()
    elif mode == "moonshine":
        try:
            asr = MoonshineASR()
        except (VoiceUnavailable, NotImplementedError, FileNotFoundError) as e:
            logger.warning("moonshine ASR unavailable: %s", e)
            return None
    else:
        logger.warning("unknown CORAL_VOICE=%r; voice disabled", mode)
        return None

    raw_device = os.environ.get("CORAL_MIC")
    device: int | str | None
    if raw_device is None or raw_device == "":
        device = None
    else:
        try:
            device = int(raw_device)
        except ValueError:
            device = raw_device

    try:
        return VoicePipeline(asr=asr, on_text=on_text, device=device)
    except VoiceUnavailable as e:
        logger.warning("voice unavailable: %s", e)
        return None
