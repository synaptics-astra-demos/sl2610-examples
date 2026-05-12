"""Tests for voice/ — only the parts that work without sounddevice / silero-vad-notorch / numpy.

Audio capture and VAD are handled by the shared utils.speech module;
here we sanity-check the ASR Protocol, the StubASR rotation, and the
pipeline factory's fall-through paths.
"""

from __future__ import annotations

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


def test_make_voice_pipeline_off_returns_none():
    pipe = make_voice_pipeline(on_text=lambda _t: None, mode="off")
    assert pipe is None


def test_make_voice_pipeline_default_is_off():
    pipe = make_voice_pipeline(on_text=lambda _t: None)
    assert pipe is None


def test_make_voice_pipeline_unknown_mode_disables():
    pipe = make_voice_pipeline(on_text=lambda _t: None, mode="espresso")
    assert pipe is None


def test_make_voice_pipeline_moonshine_disables_when_unavailable(tmp_path) -> None:
    """Without staged Moonshine artifacts the factory must return None,
    not crash the caller. MoonshineASR.__init__ raises FileNotFoundError /
    VoiceUnavailable when the model dir is missing or deps are unavailable;
    the factory swallows both and disables voice."""
    pipe = make_voice_pipeline(
        on_text=lambda _t: None,
        mode="moonshine",
        moonshine_dir=str(tmp_path / "definitely-not-here"),
    )
    assert pipe is None


def test_moonshine_asr_missing_dir_raises_filenotfound(tmp_path) -> None:
    """MoonshineASR(...) with a non-existent model_dir always raises
    FileNotFoundError — the dir check runs *before* the heavy-deps import,
    so the exact host configuration doesn't matter."""
    from voice.asr import MoonshineASR

    missing = tmp_path / "no-such-dir"
    with pytest.raises(FileNotFoundError):
        MoonshineASR(model_dir=missing)
