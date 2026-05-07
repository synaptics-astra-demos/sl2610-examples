"""Mic capture via sounddevice.InputStream.

Wraps sounddevice into a Queue-backed chunk producer. Soft-imports
sounddevice — raises VoiceUnavailable with a clear message when missing,
so the UI can disable the mic button instead of crashing on import.
"""

from __future__ import annotations

import logging
import os
from queue import Queue
from typing import Any


logger = logging.getLogger(__name__)

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


# PortAudio's ALSA backend opens raw `hw:` by default, which fails when the
# device's native rate/channels don't match the request (e.g. a 48 kHz stereo
# USB mic asked for 16 kHz mono). Setting PA_ALSA_PLUGHW=1 makes PortAudio
# wrap the device with ALSA's `plughw:` which transparently resamples and
# up/down-mixes. The standalone Moonshine example uses the same trick.
os.environ.setdefault("PA_ALSA_PLUGHW", "1")


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

        # Resolve the actual device that sounddevice will open so users can
        # see in journalctl which mic the pipeline picked. None => "default".
        try:
            info = sd.query_devices(self.device, kind="input")
            chosen = (
                f"index={info.get('index', '?')} "
                f"name={info.get('name', '?')!r} "
                f"max_in={info.get('max_input_channels', '?')} "
                f"default_sr={info.get('default_samplerate', '?')}"
            )
        except Exception as e:  # noqa: BLE001
            chosen = f"<query_devices failed: {e}>"
        logger.info(
            "opening mic: requested device=%r → %s @ %d Hz, %d-sample chunks",
            self.device, chosen, self.sample_rate, self.chunk_samples,
        )

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            device=self.device,
            blocksize=self.chunk_samples,
            dtype=np.float32,
            callback=self._callback,
        )
        self._stream.start()
        logger.info("mic stream started")

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
            logger.info("mic stream stopped")
        finally:
            self._stream = None

    @property
    def running(self) -> bool:
        return self._stream is not None
