# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Speech capture, segmentation, and transcription utilities.

This module intentionally has no terminal or GUI behavior. Frontends should use
``MoonshineTranscriber`` and ``SpeechRecognizer`` as services and decide how to
render status, transcripts, and errors themselves.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import os
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any

import numpy as np
from silero_vad_notorch import VADIterator, load_silero_vad
import sounddevice as sd
from sounddevice import InputStream
from tokenizers import Tokenizer

from utils.cli import suppress_native_stderr, suppress_native_stderr_at_exit
from utils.download import download_from_hf
from utils.stats import MoonshineInferenceStats

logger = logging.getLogger(__name__)


SAMPLING_RATE = 16_000
CHUNK_SIZE = 512
MAX_AUDIO_QUEUE_CHUNKS = 128


@dataclass(frozen=True)
class SpeechSegment:
    """Captured audio for one VAD-delimited utterance."""

    audio: np.ndarray
    sample_rate: int
    reason: str

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate


@dataclass(frozen=True)
class SpeechTranscript:
    """Speech-to-text result for one utterance."""

    text: str
    stats: MoonshineInferenceStats
    duration_s: float
    reason: str


class SoundDeviceAudioSource:
    """Small wrapper around sounddevice input streams.

    The PortAudio callback only copies chunks into a queue. Consumers can drain
    stale chunks before listening again, which is useful when the frontend is
    deliberately not accepting speech while another model is running.
    """

    def __init__(
        self,
        *,
        device: int | str | None,
        sample_rate: int = SAMPLING_RATE,
        chunk_size: int = CHUNK_SIZE,
        max_queue_chunks: int = MAX_AUDIO_QUEUE_CHUNKS,
        suppress_native_logs: bool = True,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.suppress_native_logs = suppress_native_logs
        self._queue: Queue[tuple[np.ndarray, Any]] = Queue(
            maxsize=max(1, int(max_queue_chunks))
        )
        self._stream: InputStream | None = None
        self._stderr_context = None
        if suppress_native_logs:
            suppress_native_stderr_at_exit()

    def start(self):
        if self._stream is not None:
            return
        os.environ["PA_ALSA_PLUGHW"] = "1"
        self._stderr_context = suppress_native_stderr(self.suppress_native_logs)
        self._stderr_context.__enter__()
        try:
            self._stream = InputStream(
                samplerate=self.sample_rate,
                channels=1,
                device=self.device,
                blocksize=self.chunk_size,
                dtype=np.float32,
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    logger.debug("Failed to close audio stream after start failure", exc_info=True)
                finally:
                    self._stream = None
            self._restore_native_stderr()
            raise

    def stop(self):
        if self._stream is None and self._stderr_context is None:
            return
        try:
            if self._stream is not None:
                try:
                    self._stream.stop()
                finally:
                    self._stream.close()
                    self._stream = None
            if self.suppress_native_logs:
                terminate = getattr(sd, "_terminate", None)
                if callable(terminate):
                    try:
                        terminate()
                    except Exception:
                        logger.debug("Failed to terminate PortAudio", exc_info=True)
        finally:
            self._restore_native_stderr()

    def _restore_native_stderr(self):
        if self._stderr_context is None:
            return
        self._stderr_context.__exit__(None, None, None)
        self._stderr_context = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    def _on_audio(self, data, frames, callback_time, status):
        del frames, callback_time
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except Empty:
                pass
        try:
            self._queue.put_nowait((data.copy().reshape(-1), status))
        except Full:
            pass

    def read_chunk(self, timeout_s: float = 0.1) -> tuple[np.ndarray, Any] | None:
        try:
            return self._queue.get(timeout=timeout_s)
        except Empty:
            return None

    def drain(self) -> int:
        n = 0
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return n
            n += 1


class SileroSpeechSegmenter:
    """Convert raw audio chunks into complete utterance segments."""

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLING_RATE,
        chunk_size: int = CHUNK_SIZE,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 300,
        lookback_chunks: int = 5,
        max_speech_secs: float = 10.0,
        min_segment_secs: float = 0.20,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.lookback_size = lookback_chunks * chunk_size
        self.max_speech_secs = max_speech_secs
        self.min_segment_secs = min_segment_secs
        self.vad_iterator = VADIterator(
            model=load_silero_vad(onnx=True),
            sampling_rate=sample_rate,
            threshold=threshold,
            min_silence_duration_ms=min_silence_duration_ms,
        )
        self._chunks: deque[np.ndarray] = deque()
        self._sample_count = 0
        self._recording = False

    def reset(self):
        self._chunks.clear()
        self._sample_count = 0
        self._recording = False
        self._reset_vad_iterator()

    def feed(self, chunk: np.ndarray) -> SpeechSegment | None:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        self._append_chunk(chunk)

        if not self._recording:
            self._trim_to_last(self.lookback_size)

        speech_event = self.vad_iterator(chunk)
        if speech_event:
            logger.debug("VAD event: %s", speech_event)
            if "start" in speech_event and not self._recording:
                self._recording = True

            if "end" in speech_event and self._recording:
                return self._finish_segment("vad_end")

        if self._recording and self._recorded_duration_s > self.max_speech_secs:
            return self._finish_segment("max_duration")

        return None

    @property
    def _recorded_duration_s(self) -> float:
        return self._sample_count / self.sample_rate

    def _finish_segment(self, reason: str) -> SpeechSegment | None:
        if not self._chunks:
            audio = np.empty(0, dtype=np.float32)
        elif len(self._chunks) == 1:
            audio = self._chunks[0]
        else:
            audio = np.concatenate(tuple(self._chunks))
        segment = SpeechSegment(
            audio=audio,
            sample_rate=self.sample_rate,
            reason=reason,
        )
        self.reset()

        if segment.duration_s < self.min_segment_secs:
            logger.debug(
                "Discarding %.3fs speech segment below %.3fs minimum",
                segment.duration_s,
                self.min_segment_secs,
            )
            return None
        return segment

    def _append_chunk(self, chunk: np.ndarray):
        if len(chunk) == 0:
            return
        self._chunks.append(chunk)
        self._sample_count += len(chunk)

    def _trim_to_last(self, max_samples: int):
        max_samples = max(0, int(max_samples))
        while self._sample_count > max_samples and self._chunks:
            excess = self._sample_count - max_samples
            first = self._chunks[0]
            if len(first) <= excess:
                self._chunks.popleft()
                self._sample_count -= len(first)
            else:
                self._chunks[0] = first[excess:]
                self._sample_count -= excess
                break

    def _reset_vad_iterator(self):
        reset_states = getattr(self.vad_iterator, "reset_states", None)
        if callable(reset_states):
            reset_states()

        # silero_vad_notorch exposes these fields but does not consistently
        # provide a public soft-reset API across versions.
        for attr, value in (
            ("triggered", False),
            ("temp_end", 0),
            ("current_sample", 0),
        ):
            if hasattr(self.vad_iterator, attr):
                setattr(self.vad_iterator, attr, value)


class MoonshineTranscriber:
    """Speech-to-text adapter around ``MoonshineRunner``."""

    def __init__(
        self,
        model_dir: str | os.PathLike | None,
        *,
        sample_rate: int = SAMPLING_RATE,
        warmup: bool = True,
        suppress_native_logs: bool = True,
    ):
        self.sample_rate = sample_rate
        logger.info(
            "Loading Moonshine model from '%s'...",
            model_dir if model_dir is not None else "default",
        )
        with suppress_native_stderr(suppress_native_logs):
            from utils.moonshine import load_moonshine

            self.runner = load_moonshine(model_dir)
        self.model_dir = self.runner.model_dir
        self.tokenizer = self._load_tokenizer()
        if warmup:
            with suppress_native_stderr(suppress_native_logs):
                self.transcribe_audio(np.zeros(sample_rate, dtype=np.float32))
        logger.info("Moonshine model loaded successfully")

    def transcribe(self, segment: SpeechSegment) -> SpeechTranscript:
        duration_s = segment.duration_s
        text, stats = self.transcribe_audio(segment.audio)
        return SpeechTranscript(
            text=text,
            stats=stats,
            duration_s=duration_s,
            reason=segment.reason,
        )

    def transcribe_audio(self, audio: np.ndarray) -> tuple[str, MoonshineInferenceStats]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        tokens = self.runner.run(audio[np.newaxis, :])
        text = self.tokenizer.decode_batch(tokens, skip_special_tokens=True)[0].strip()
        static_audio_len = self.runner.max_inp_len or len(audio)
        stats = MoonshineInferenceStats(
            total_time_ms=self.runner.last_infer_time,
            ttft_ms=self.runner.time_to_first_token,
            n_tokens=self.runner.generated_tokens,
            audio_duration_s=len(audio) / self.sample_rate,
            static_audio_duration_s=static_audio_len / self.runner.input_freq,
        )
        return text, stats

    def _load_tokenizer(self) -> Tokenizer:
        local_path = self.model_dir / "tokenizer.json"
        try:
            return Tokenizer.from_file(str(local_path))
        except (FileNotFoundError, OSError):
            tokenizer_file = download_from_hf("UsefulSensors/moonshine-tiny", "tokenizer.json")
            return Tokenizer.from_file(str(tokenizer_file))


class SpeechRecognizer:
    """High-level service that listens until one non-empty transcript is ready."""

    def __init__(
        self,
        *,
        transcriber: MoonshineTranscriber,
        source: SoundDeviceAudioSource,
        segmenter: SileroSpeechSegmenter,
    ):
        self.transcriber = transcriber
        self.source = source
        self.segmenter = segmenter

    def __enter__(self):
        self.source.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.source.stop()
        return False

    def listen_once(self, *, stop_event=None) -> SpeechTranscript | None:
        self.source.drain()
        self.segmenter.reset()

        while stop_event is None or not stop_event.is_set():
            item = self.source.read_chunk(timeout_s=0.1)
            if item is None:
                continue
            chunk, status = item
            if status:
                logger.debug("Audio input status: %s", status)

            segment = self.segmenter.feed(chunk)
            if segment is None:
                continue

            transcript = self.transcriber.transcribe(segment)
            if transcript.text.strip():
                return transcript

        return None

    def drain(self) -> int:
        return self.source.drain()


def query_input_devices() -> str:
    """Return sounddevice's formatted input/output device list."""

    return str(sd.query_devices())
