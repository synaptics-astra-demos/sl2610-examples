# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Lightweight inference statistics containers."""

from __future__ import annotations


class InferenceStats:
    """Base stats snapshot from a single inference run."""

    __slots__ = ("total_time_ms", "ttft_ms", "n_tokens", "tps")

    def __init__(
        self,
        *,
        total_time_ms: float,
        ttft_ms: float,
        n_tokens: int,
    ):
        self.total_time_ms = total_time_ms
        self.ttft_ms = ttft_ms
        self.n_tokens = n_tokens
        decode_ms = total_time_ms - ttft_ms
        self.tps = n_tokens / (decode_ms / 1000) if decode_ms > 0 else 0.0

    def fmt(self, *, verbose: bool = False) -> str:
        del verbose
        return " | ".join([
            f"{self.n_tokens} tok",
            f"{self.tps:.1f} tok/s",
            f"{self.total_time_ms:.0f}ms",
        ])


class MoonshineInferenceStats(InferenceStats):
    """Stats for a Moonshine speech-to-text inference."""

    __slots__ = ("rtf", "audio_duration_s", "static_rtf", "static_audio_duration_s")

    def __init__(
        self,
        *,
        total_time_ms: float,
        ttft_ms: float,
        n_tokens: int,
        audio_duration_s: float | None = None,
        static_audio_duration_s: float | None = None,
    ):
        super().__init__(total_time_ms=total_time_ms, ttft_ms=ttft_ms, n_tokens=n_tokens)
        self.audio_duration_s = audio_duration_s
        self.rtf: float | None = (
            (total_time_ms / 1000) / audio_duration_s
            if audio_duration_s is not None and audio_duration_s > 0
            else None
        )
        self.static_audio_duration_s = static_audio_duration_s
        self.static_rtf: float | None = (
            (total_time_ms / 1000) / static_audio_duration_s
            if static_audio_duration_s is not None and static_audio_duration_s > 0
            else None
        )

    def fmt(self, *, verbose: bool = False) -> str:
        parts = []
        parts.append(f"{self.total_time_ms:.0f}ms")
        if self.rtf is not None:
            parts.append(f"RTF(audio)={self.rtf:.2f}")
        if verbose:
            parts = [
                f"{self.n_tokens} tok",
                f"{self.tps:.1f} tok/s",
                *parts,
            ]
            if self.static_rtf is not None:
                parts.append(f"RTF(static)={self.static_rtf:.2f}")
            parts.append(f"enc+first_dec_ms={self.ttft_ms:.0f}")
        return " | ".join(parts)


class Gemma3InferenceStats(InferenceStats):
    """Stats for a Gemma3 LLM inference."""

    __slots__ = ("n_input_tokens", "n_prefill_tokens", "prefill_tps")

    def __init__(
        self,
        *,
        total_time_ms: float,
        ttft_ms: float,
        n_tokens: int,
        n_input_tokens: int = 0,
        n_prefill_tokens: int = 0,
        prefill_tps: float | None = None,
    ):
        super().__init__(total_time_ms=total_time_ms, ttft_ms=ttft_ms, n_tokens=n_tokens)
        self.n_input_tokens = n_input_tokens
        self.n_prefill_tokens = n_prefill_tokens
        self.prefill_tps = (
            prefill_tps
            if prefill_tps is not None
            else n_prefill_tokens / (ttft_ms / 1000) if ttft_ms > 0 else 0.0
        )

    def fmt(self, *, verbose: bool = False) -> str:
        parts = [f"TTFT={self.ttft_ms:.0f}ms", f"{self.tps:.1f} tok/s"]
        if verbose:
            parts.insert(0, f"{self.n_tokens} tok")
            if self.prefill_tps:
                parts.append(f"prefill={self.prefill_tps:.1f} tok/s")
            parts.append(f"infer_ms={self.total_time_ms:.0f}")
        return " | ".join(parts)
