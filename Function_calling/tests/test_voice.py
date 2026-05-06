"""Tests for voice/ — only the parts that work without sounddevice / silero-vad / numpy.

The mic stream and VAD are exercised on-device; here we sanity-check the ASR
Protocol, the StubASR rotation, and the pipeline factory's fall-through paths.
"""

from __future__ import annotations

import os

import pytest

from voice import StubASR, make_voice_pipeline
from voice.asr import _STUB_PHRASES


def test_stub_asr_rotates_phrases_and_ignores_audio():
    asr = StubASR()
    seen = [asr.transcribe(audio=None, sample_rate=16_000) for _ in range(len(_STUB_PHRASES) + 2)]  # type: ignore[arg-type]

    assert seen[: len(_STUB_PHRASES)] == list(_STUB_PHRASES)
    assert seen[len(_STUB_PHRASES)] == _STUB_PHRASES[0]
    assert seen[len(_STUB_PHRASES) + 1] == _STUB_PHRASES[1]


def test_stub_asr_custom_phrase_set():
    phrases = ("alpha", "bravo")
    asr = StubASR(phrases=phrases)
    assert asr.transcribe(None, 16_000) == "alpha"  # type: ignore[arg-type]
    assert asr.transcribe(None, 16_000) == "bravo"  # type: ignore[arg-type]
    assert asr.transcribe(None, 16_000) == "alpha"  # type: ignore[arg-type]


def test_make_voice_pipeline_off_returns_none(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CORAL_VOICE", "off")
    pipe = make_voice_pipeline(on_text=lambda _t: None)
    assert pipe is None


def test_make_voice_pipeline_default_is_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CORAL_VOICE", raising=False)
    pipe = make_voice_pipeline(on_text=lambda _t: None)
    assert pipe is None


def test_make_voice_pipeline_unknown_mode_disables(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CORAL_VOICE", "espresso")
    pipe = make_voice_pipeline(on_text=lambda _t: None)
    assert pipe is None


def test_make_voice_pipeline_moonshine_disables_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Without staged Moonshine artifacts the factory must return None,
    not crash the caller. Phase B: MoonshineASR.__init__ raises
    FileNotFoundError / VoiceUnavailable when the model dir is missing or
    deps are unavailable. The factory swallows both and disables voice."""
    monkeypatch.setenv("CORAL_VOICE", "moonshine")
    monkeypatch.setenv("CORAL_MOONSHINE_DIR", str(tmp_path / "definitely-not-here"))
    pipe = make_voice_pipeline(on_text=lambda _t: None)
    assert pipe is None


def test_moonshine_asr_missing_dir_raises_filenotfound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """MoonshineASR(...) with a non-existent model_dir always raises
    FileNotFoundError — the dir check runs *before* the heavy-deps import,
    so the exact host configuration doesn't matter."""
    from voice.asr import MoonshineASR

    monkeypatch.delenv("CORAL_MOONSHINE_DIR", raising=False)
    missing = tmp_path / "no-such-dir"
    with pytest.raises(FileNotFoundError):
        MoonshineASR(model_dir=missing)
