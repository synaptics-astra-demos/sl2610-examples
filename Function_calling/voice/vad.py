"""Silero VAD wrapper — detect utterance start/end on a mic chunk stream.

Soft-imports silero-vad-notorch (the torch-free fork — same package the
standalone speech_to_text/ example uses). Raises VoiceUnavailable if
missing so the pipeline factory can fall back to a "voice disabled" UI
state.
"""

from __future__ import annotations

from typing import Any

from .mic import VoiceUnavailable


class VAD:
    """Wrap silero-vad's VADIterator + a speech buffer.

    Call feed(chunk) per mic chunk. Returns a 1D float32 utterance ndarray
    when speech ends (or None otherwise). Internally manages the speech buffer,
    a pre-roll lookback window, and min/max speech-duration gates.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        threshold: float = 0.5,
        min_silence_ms: int = 400,
        min_speech_secs: float = 0.6,
        max_speech_secs: float = 10.0,
        lookback_chunks: int = 5,
        chunk_samples: int = 512,
    ) -> None:
        try:
            from silero_vad_notorch import VADIterator, load_silero_vad  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover - import guard
            raise VoiceUnavailable(
                "silero-vad-notorch is not installed. Install with "
                "`pip install silero-vad-notorch` (a torch-free fork; "
                "matches the package used by speech_to_text/)."
            ) from e

        import numpy as np

        self._np = np
        model = load_silero_vad(onnx=True)
        self._iterator = VADIterator(
            model=model,
            sampling_rate=sample_rate,
            threshold=threshold,
            min_silence_duration_ms=min_silence_ms,
        )
        self.sample_rate = sample_rate
        self.min_speech_secs = min_speech_secs
        self.max_speech_secs = max_speech_secs
        self._lookback_size = lookback_chunks * chunk_samples
        self._speech = np.empty(0, dtype=np.float32)
        self._recording = False
        self._speech_start_t: float = 0.0

    def feed(self, chunk: Any) -> Any | None:
        """Append a mic chunk; return finished utterance ndarray or None.

        Returns the buffered float32 array on speech end (silence detected
        OR max-duration timeout). Returns None while still listening.
        """
        import time

        np = self._np
        self._speech = np.concatenate((self._speech, chunk))
        if not self._recording:
            self._speech = self._speech[-self._lookback_size :]

        verdict = self._iterator(chunk)
        if verdict:
            if "start" in verdict and not self._recording:
                self._recording = True
                self._speech_start_t = time.time()
            if "end" in verdict and self._recording:
                if time.time() - self._speech_start_t >= self.min_speech_secs:
                    return self._flush()
                self._recording = False  # discard short blip
                self._speech = self._speech[-self._lookback_size :]
                return None

        if self._recording:
            duration = len(self._speech) / self.sample_rate
            if duration > self.max_speech_secs:
                return self._flush()
        return None

    def _flush(self) -> Any:
        np = self._np
        utterance = self._speech.copy()
        self._recording = False
        self._speech = np.empty(0, dtype=np.float32)
        self._iterator.triggered = False
        self._iterator.temp_end = 0
        self._iterator.current_sample = 0
        return utterance
