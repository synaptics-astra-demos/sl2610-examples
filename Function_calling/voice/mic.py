"""Mic capture via sounddevice.InputStream.

Wraps sounddevice into a Queue-backed chunk producer. Soft-imports
sounddevice — raises VoiceUnavailable with a clear message when missing,
so the UI can disable the mic button instead of crashing on import.
"""

from __future__ import annotations

from queue import Queue
from typing import Any


SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 512


class VoiceUnavailable(RuntimeError):
    """Raised when a voice dependency (sounddevice / silero-vad-notorch / model) is missing."""


def _require_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except ImportError as e:  # pragma: no cover - import guard
        raise VoiceUnavailable(
            "sounddevice is not installed. Install with `pip install sounddevice`. "
            "PortAudio is also required at runtime — extract `library/portaudio_libs.tgz` "
            "(or `sudo apt install libportaudio2` as a fallback)."
        ) from e
    return sd


class MicStream:
    """Open the input device and push float32 mono chunks onto self.queue.

    Each queue entry: (np.ndarray of shape (CHUNK_SAMPLES,), status).

    Lifecycle:
        m = MicStream()
        m.start()
        chunk, status = m.queue.get()  # blocking
        m.stop()
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_samples: int = CHUNK_SAMPLES,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.device = device
        self.queue: Queue = Queue()
        self._stream: Any | None = None

    def _callback(self, data, frames, time_info, status) -> None:  # noqa: ARG002
        self.queue.put((data.copy().flatten(), status))

    def start(self) -> None:
        if self._stream is not None:
            return
        sd = _require_sounddevice()
        import numpy as np

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            device=self.device,
            blocksize=self.chunk_samples,
            dtype=np.float32,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    @property
    def running(self) -> bool:
        return self._stream is not None
