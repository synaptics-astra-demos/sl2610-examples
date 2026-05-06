"""Voice input pipeline — mic → VAD → ASR → text.

Public API:
    - VoicePipeline: thread-managed orchestrator (mic + VAD + ASR)
    - ASR (Protocol), StubASR, MoonshineASR (Phase B placeholder)
    - make_voice_pipeline(on_text): factory keyed off CORAL_VOICE

CORAL_VOICE values:
    off     - no voice (default; UI mic button hidden / disabled)
    stub    - real mic + VAD, stub ASR (returns rotating canned phrases)
    moonshine - real mic + VAD + Moonshine on Torq NPU (Phase B, NotImplementedError today)
"""

from .asr import ASR, StubASR, MoonshineASR
from .pipeline import VoicePipeline, VoiceUnavailable, make_voice_pipeline

__all__ = [
    "ASR",
    "StubASR",
    "MoonshineASR",
    "VoicePipeline",
    "VoiceUnavailable",
    "make_voice_pipeline",
]
