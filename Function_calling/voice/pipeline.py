"""VoicePipeline — thread-managed mic → VAD → ASR → text emit.

Delegates the audio capture, VAD segmentation, and Moonshine transcription
to the shared ``utils.speech`` module. This thin wrapper adds the
start/stop/callback API the Function_calling UI and CLI use.

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
import os
import threading
from pathlib import Path
from typing import Callable

from .asr import ASR, MoonshineASR, StubASR, VoiceUnavailable


logger = logging.getLogger(__name__)


VoiceMode = str  # "off" | "stub" | "moonshine"

SAMPLE_RATE = 16_000
CHUNK_SIZE = 512


class VoicePipeline:
    """Run mic + VAD + ASR in a background thread.

    For ``mode="moonshine"`` the full pipeline (audio source, speech
    segmenter, transcriber) is built from ``utils.speech``.

    For ``mode="stub"`` a lightweight fallback loops raw mic chunks
    through the shared VAD and hands completed utterances to the
    ``StubASR`` (which ignores audio and rotates canned phrases).

    on_text(text: str) is called from the worker thread for each finalised
    utterance. Callers wanting to touch Qt widgets should marshal via a
    Qt signal.
    """

    def __init__(
        self,
        asr: ASR,
        on_text: Callable[[str], None] | None = None,
        device: int | str | None = None,
        moonshine_dir: str | Path | None = None,
    ) -> None:
        self._asr = asr
        self._on_text: Callable[[str], None] = on_text or (lambda _t: None)
        self._device = device
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

        # Determine whether to use the full SpeechRecognizer path
        # (Moonshine) or the stub path (StubASR with raw mic + VAD).
        self._use_recognizer = isinstance(asr, MoonshineASR)

    def set_callback(self, on_text: Callable[[str], None]) -> None:
        """Replace the on_text callback. Used to bind a Qt signal after
        the UI is constructed — the pipeline factory runs at startup,
        before the window (and its signals) exist."""
        self._on_text = on_text

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="voice-pipeline")
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        if self._use_recognizer:
            self._run_recognizer()
        else:
            self._run_stub()

    def _run_recognizer(self) -> None:
        """Full Moonshine pipeline via utils.speech.SpeechRecognizer."""
        from utils.speech import (
            MoonshineTranscriber,
            SileroSpeechSegmenter,
            SoundDeviceAudioSource,
            SpeechRecognizer,
        )

        logger.info("voice pipeline thread starting (recognizer mode)")

        source = SoundDeviceAudioSource(
            device=self._device,
            sample_rate=SAMPLE_RATE,
            chunk_size=CHUNK_SIZE,
        )
        segmenter = SileroSpeechSegmenter(
            sample_rate=SAMPLE_RATE,
            chunk_size=CHUNK_SIZE,
        )
        recognizer = SpeechRecognizer(
            transcriber=self._asr._transcriber,
            source=source,
            segmenter=segmenter,
        )

        with recognizer:
            while not self._stop_evt.is_set():
                transcript = recognizer.listen_once(stop_event=self._stop_evt)
                if transcript is None:
                    continue
                text = transcript.text.strip()
                if not text:
                    continue
                logger.info(
                    "ASR transcribed (%s): %r",
                    transcript.stats.fmt() if transcript.stats else "?",
                    text,
                )
                try:
                    self._on_text(text)
                except Exception:
                    logger.exception("on_text callback failed")
        logger.info("voice pipeline thread exiting")

    def _run_stub(self) -> None:
        """Stub mode: real mic + VAD, fake ASR (canned phrases)."""
        import time

        from utils.speech import SileroSpeechSegmenter, SoundDeviceAudioSource

        logger.info("voice pipeline thread starting (stub mode)")

        source = SoundDeviceAudioSource(
            device=self._device,
            sample_rate=SAMPLE_RATE,
            chunk_size=CHUNK_SIZE,
        )
        segmenter = SileroSpeechSegmenter(
            sample_rate=SAMPLE_RATE,
            chunk_size=CHUNK_SIZE,
        )

        source.start()
        try:
            while not self._stop_evt.is_set():
                item = source.read_chunk(timeout_s=0.5)
                if item is None:
                    continue
                chunk, status = item
                if status:
                    logger.debug("mic status: %s", status)
                segment = segmenter.feed(chunk)
                if segment is None:
                    continue
                logger.info(
                    "VAD finalised utterance: %.2fs (%d samples)",
                    segment.duration_s,
                    len(segment.audio),
                )
                t0 = time.time()
                try:
                    text = self._asr.transcribe(segment.audio, SAMPLE_RATE)
                except Exception:
                    logger.exception("ASR failed")
                    continue
                asr_ms = (time.time() - t0) * 1000.0
                text = text.strip()
                if not text:
                    logger.info("ASR returned empty text in %.0f ms (silence?)", asr_ms)
                    continue
                logger.info("ASR transcribed in %.0f ms: %r", asr_ms, text)
                try:
                    self._on_text(text)
                except Exception:
                    logger.exception("on_text callback failed")
        finally:
            source.stop()
        logger.info("voice pipeline thread exiting")


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

    device = _coerce_mic_device(mic_device)

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

    try:
        return VoicePipeline(asr=asr, on_text=on_text, device=device)
    except VoiceUnavailable as e:
        logger.warning("voice unavailable: %s", e)
        return None
