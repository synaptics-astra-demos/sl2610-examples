# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Voice activity detection backends for app.py.

Two interchangeable VAD backends — a zero-dependency self-calibrating energy
detector and Silero's neural VAD — share the same speech/silence endpointing
state machine (``_HangoverVAD``); only how each computes a per-chunk score
differs. See ``--vad-backend`` in app.py.
"""

import sys

import numpy as np


class _HangoverVAD:
    """
    Shared speech/silence endpointing: given a per-chunk score from a
    subclass, tracks speech_start/speech_end transitions with a fixed
    silence-duration hangover before ending an utterance. Both VAD backends
    below are driven by this identical state machine — only how the score is
    computed differs.
    """
    def __init__(self, threshold, silence_duration, sample_rate):
        self.threshold                = threshold
        self.silence_duration_samples = int(silence_duration * sample_rate)
        self.sample_rate              = sample_rate
        self.silence_counter          = 0
        self.is_speaking              = False
        self.last_score               = 0.0
        self.silence_remaining_sec    = 0.0

    def _score(self, audio_chunk) -> float:
        raise NotImplementedError

    def process_chunk(self, audio_chunk):
        score = self._score(audio_chunk)
        self.last_score = score

        is_speech = score > self.threshold
        if is_speech:
            self.silence_counter       = 0
            self.silence_remaining_sec = 0.0
            if not self.is_speaking:
                self.is_speaking = True
                return "speech_start"
            return "speech"
        else:
            if self.is_speaking:
                self.silence_counter += len(audio_chunk)
                remaining = max(0, self.silence_duration_samples - self.silence_counter)
                self.silence_remaining_sec = remaining / self.sample_rate
                if self.silence_counter >= self.silence_duration_samples:
                    self.is_speaking           = False
                    self.silence_counter        = 0
                    self.silence_remaining_sec  = 0.0
                    return "speech_end"
                return "speech"
            self.silence_remaining_sec = 0.0
            return "silence"


class EnergyVAD(_HangoverVAD):
    """
    Simple RMS energy-based voice activity detector for streaming.
    Self-calibrating: samples ambient noise during the first 12 chunks (~960 ms).
    """
    def __init__(self, threshold=0.015, silence_duration=2.5, sample_rate=16000,
                 report_calibration=False):
        super().__init__(threshold, silence_duration, sample_rate)
        self.base_threshold     = threshold
        self.report_calibration = report_calibration
        self.ambient_rms        = []
        self.calibrated         = False

    def _score(self, audio_chunk):
        return np.sqrt(np.mean(audio_chunk ** 2)) if len(audio_chunk) > 0 else 0.0

    def process_chunk(self, audio_chunk):
        if not self.calibrated:
            rms = self._score(audio_chunk)
            self.last_score = rms
            self.ambient_rms.append(rms)
            if len(self.ambient_rms) >= 12:
                mean_rms = np.mean(self.ambient_rms)
                std_rms  = np.std(self.ambient_rms)
                self.threshold = max(mean_rms + 4 * std_rms, self.base_threshold)
                if self.report_calibration:
                    print(
                        f"\n[VAD Calibration] Ambient Noise RMS: {mean_rms:.5f} "
                        f"(std: {std_rms:.5f}). Threshold set to: {self.threshold:.5f}",
                        file=sys.stderr,
                    )
                self.calibrated = True
            return "silence"
        return super().process_chunk(audio_chunk)


class SileroVAD(_HangoverVAD):
    """
    Neural VAD using the ``silero-vad-notorch`` package (same one used by
    speech_to_text/live_caption.py's SileroSpeechSegmenter), which ships the
    Silero ONNX graph as bundled package data — no network fetch needed. Its
    model only accepts fixed-size windows, so each incoming pipeline chunk is
    split into 512-sample (32 ms) windows, zero-padding a short final window,
    and the max per-window speech probability becomes the chunk's score — so
    one utterance onset anywhere in the chunk is enough to trigger
    speech_start.
    """
    _WINDOW = 512  # samples per native inference call at 16 kHz

    def __init__(self, model_path=None, threshold=0.5, silence_duration=2.5, sample_rate=16000,
                 report_calibration=False):
        if sample_rate != 16000:
            raise ValueError("SileroVAD only supports 16 kHz audio")
        super().__init__(threshold, silence_duration, sample_rate)
        try:
            from silero_vad_notorch import load_silero_vad
            from silero_vad_notorch.utils_vad import OnnxWrapper
        except ImportError:
            print("Error: silero-vad-notorch is required for --vad-backend silero. Install it with:",
                  file=sys.stderr)
            print("  pip install silero-vad-notorch", file=sys.stderr)
            sys.exit(1)
        # model_path overrides the bundled onnx graph with a custom one (e.g. a
        # different opset export); OnnxWrapper accepts an explicit path directly.
        self.model      = OnnxWrapper(str(model_path), force_onnx_cpu=True) if model_path else load_silero_vad(onnx=True)
        self.calibrated = True  # fixed probability threshold, no ambient-noise warm-up needed
        if report_calibration:
            print(f"\n[VAD] Silero neural VAD ready (silero-vad-notorch) — probability threshold: {threshold:.2f}",
                  file=sys.stderr)

    def _score(self, audio_chunk):
        if len(audio_chunk) == 0:
            return 0.0
        prob = 0.0
        for start in range(0, len(audio_chunk), self._WINDOW):
            window = audio_chunk[start:start + self._WINDOW]
            if len(window) < self._WINDOW:
                window = np.pad(window, (0, self._WINDOW - len(window)))
            # OnnxWrapper carries its own recurrent state across calls, so
            # windows must be fed in order (as they are here, chunk by chunk).
            out = self.model(window.astype(np.float32), self.sample_rate)
            prob = max(prob, float(out.item()))
        return prob
