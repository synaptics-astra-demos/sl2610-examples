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

    def fmt(self) -> str:
        return " | ".join([
            f"{self.n_tokens} tok",
            f"{self.tps:.1f} tok/s",
            f"{self.total_time_ms:.0f}ms",
        ])


class MoonshineInferenceStats(InferenceStats):
    """Stats for a Moonshine speech-to-text inference."""

    __slots__ = ("rtf",)

    def __init__(
        self,
        *,
        total_time_ms: float,
        ttft_ms: float,
        n_tokens: int,
        audio_duration_s: float | None = None,
    ):
        super().__init__(total_time_ms=total_time_ms, ttft_ms=ttft_ms, n_tokens=n_tokens)
        self.rtf: float | None = (
            (total_time_ms / 1000) / audio_duration_s
            if audio_duration_s is not None and audio_duration_s > 0
            else None
        )

    def fmt(self) -> str:
        parts = [
            f"{self.n_tokens} tok",
            f"{self.tps:.1f} tok/s",
            f"{self.total_time_ms:.0f}ms",
        ]
        if self.rtf is not None:
            audio_s = self.total_time_ms / 1000 / self.rtf
            parts.append(f"RTF={self.rtf:.2f} ({audio_s:.1f}s audio)")
        return " | ".join(parts)


class Gemma3InferenceStats(InferenceStats):
    """Stats for a Gemma3 LLM inference."""

    __slots__ = ("n_input_tokens",)

    def __init__(
        self,
        *,
        total_time_ms: float,
        ttft_ms: float,
        n_tokens: int,
        n_input_tokens: int = 0,
    ):
        super().__init__(total_time_ms=total_time_ms, ttft_ms=ttft_ms, n_tokens=n_tokens)
        self.n_input_tokens = n_input_tokens

    def fmt(self) -> str:
        return " | ".join([
            f"{self.n_tokens} tok",
            f"{self.tps:.1f} tok/s",
            f"{self.total_time_ms:.0f}ms",
            f"TTFT={self.ttft_ms:.0f}ms",
        ])
