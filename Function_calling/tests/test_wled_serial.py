"""Unit tests for the WLED serial client (v8).

No real hardware required — we swap in a fake ``serial.Serial`` so the
payload format is captured as bytes and asserted directly.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

# Resolve imports against the parent Function_calling/ directory regardless
# of where pytest is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wled import (  # noqa: E402
    COLOR_NAMES,
    EFFECT_MAP,
    INTENSITY_MAP,
    NullWLEDClient,
    PALETTE_MAP,
    SPEED_MAP,
    WLEDSerialClient,
    _scale_brightness,
    resolve_color,
)


class FakeSerial:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.is_open = True
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False


@pytest.fixture
def patched_serial(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    fake_mod = types.ModuleType("serial")
    captured: list[bytes] = []

    class CapturingSerial(FakeSerial):
        def write(self, data: bytes) -> int:
            captured.append(data)
            return super().write(data)

    fake_mod.Serial = CapturingSerial  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "serial", fake_mod)
    return captured


def _parse_lines(raws: list[bytes]) -> list[dict[str, Any]]:
    return [json.loads(r.decode("ascii").rstrip()) for r in raws]


def _parse_line(raw: bytes) -> dict[str, Any]:
    return json.loads(raw.decode("ascii").rstrip())


# --------------------------------------------------------------- color resolver

def test_resolve_named_color_round_trips() -> None:
    assert resolve_color("red") == (255, 0, 0)
    assert resolve_color("WARM WHITE") == COLOR_NAMES["warm_white"]
    assert resolve_color("warm-white") == COLOR_NAMES["warm_white"]


def test_resolve_hex_color() -> None:
    assert resolve_color("#FF8800") == (255, 136, 0)
    assert resolve_color("#000000") == (0, 0, 0)


def test_resolve_new_color_names() -> None:
    """v8 expanded vocabulary — these were silently returning white in v7."""
    assert resolve_color("teal") == (0, 180, 180)
    assert resolve_color("amber") == (255, 150, 30)
    assert resolve_color("gold") == (255, 200, 40)
    assert resolve_color("violet") == (180, 80, 255)
    assert resolve_color("lime") == (180, 255, 30)
    assert resolve_color("mint") == (140, 255, 200)
    assert resolve_color("hot_pink") == (255, 80, 160)


def test_resolve_color_aliases() -> None:
    assert resolve_color("grey") == resolve_color("gray")


def test_resolve_color_fuzzy_match() -> None:
    """Typos should resolve to the nearest known color, not white."""
    assert resolve_color("purpel") == COLOR_NAMES["purple"]
    assert resolve_color("organge") == COLOR_NAMES["orange"]


def test_resolve_unknown_color_falls_back_to_white() -> None:
    assert resolve_color("zooblatxx") == COLOR_NAMES["white"]
    assert resolve_color("") == COLOR_NAMES["white"]


def test_scale_brightness_clamps_and_maps() -> None:
    assert _scale_brightness(0) == 0
    assert _scale_brightness(100) == 255
    assert _scale_brightness(50) == 128
    assert _scale_brightness(-5) == 0
    assert _scale_brightness(150) == 255


# --------------------------------------------------------------- on/off/solid

def test_on_sends_on_true(patched_serial: list[bytes]) -> None:
    c = WLEDSerialClient(port="/dev/ttyACM0")
    c.on()
    msg = _parse_line(patched_serial[0])
    assert msg == {"on": True, "bri": 255}


def test_off_sends_on_false(patched_serial: list[bytes]) -> None:
    c = WLEDSerialClient()
    c.off()
    assert _parse_line(patched_serial[0]) == {"on": False}


def test_set_solid_encodes_color_and_brightness(patched_serial: list[bytes]) -> None:
    c = WLEDSerialClient()
    c.set_solid("red", brightness_pct=50)
    msg = _parse_line(patched_serial[0])
    assert msg["on"] is True
    assert msg["bri"] == 128  # 50% -> 128
    assert msg["seg"][0]["col"] == [[255, 0, 0]]
    assert msg["seg"][0]["fx"] == 0  # solid


# --------------------------------------------------------------- set_effect

def test_set_effect_uses_effect_map(patched_serial: list[bytes]) -> None:
    c = WLEDSerialClient()
    c.set_effect("rainbow", speed="fast")
    seg = _parse_line(patched_serial[0])["seg"][0]
    assert seg["fx"] == EFFECT_MAP["rainbow"]["fx"]  # 9
    assert seg["sx"] == SPEED_MAP["fast"]


def test_set_effect_rainbow_omits_col(patched_serial: list[bytes]) -> None:
    """Rainbow is palette-spectrum; col would tint the result. Schema says ignored."""
    c = WLEDSerialClient()
    c.set_effect("rainbow", color="blue")
    seg = _parse_line(patched_serial[0])["seg"][0]
    assert "col" not in seg


def test_set_effect_chase_populates_secondary_color(patched_serial: list[bytes]) -> None:
    """Chase (fx 28) uses col[0] as the runners and col[1] as the background.
    We default col[1] to a dim version of col[0] so the trail is visible."""
    c = WLEDSerialClient()
    c.set_effect("chase", color="red")
    seg = _parse_line(patched_serial[0])["seg"][0]
    assert len(seg["col"]) == 2
    assert seg["col"][0] == [255, 0, 0]
    # col[1] is a dim shade of col[0] — non-zero on the red channel.
    assert seg["col"][1][0] > 0
    # And visibly dimmer than the primary.
    assert seg["col"][1][0] < seg["col"][0][0]


def test_set_effect_off_sends_on_false(patched_serial: list[bytes]) -> None:
    c = WLEDSerialClient()
    c.set_effect("off")
    assert _parse_line(patched_serial[0]) == {"on": False}


def test_set_effect_palette_writes_pal_field(patched_serial: list[bytes]) -> None:
    c = WLEDSerialClient()
    c.set_effect("aurora", palette="ocean")
    seg = _parse_line(patched_serial[0])["seg"][0]
    assert seg["pal"] == PALETTE_MAP["ocean"]  # 9


def test_set_effect_intensity_overrides_default_ix(patched_serial: list[bytes]) -> None:
    c = WLEDSerialClient()
    c.set_effect("fire", intensity="high")
    seg = _parse_line(patched_serial[0])["seg"][0]
    assert seg["ix"] == INTENSITY_MAP["high"]  # 220


def test_set_effect_intensity_low(patched_serial: list[bytes]) -> None:
    c = WLEDSerialClient()
    c.set_effect("sparkle", intensity="low")
    seg = _parse_line(patched_serial[0])["seg"][0]
    assert seg["ix"] == INTENSITY_MAP["low"]  # 64


def test_set_effect_unknown_effect_falls_back_to_solid(patched_serial: list[bytes]) -> None:
    c = WLEDSerialClient()
    c.set_effect("nonexistent-effect")
    seg = _parse_line(patched_serial[0])["seg"][0]
    assert seg["fx"] == EFFECT_MAP["solid"]["fx"]


def test_set_effect_invalid_palette_ignored(patched_serial: list[bytes]) -> None:
    """Unknown palette name should not crash; the field is just dropped."""
    c = WLEDSerialClient()
    c.set_effect("plasma", palette="not-a-real-palette")
    seg = _parse_line(patched_serial[0])["seg"][0]
    assert "pal" not in seg


def test_set_effect_with_new_color_name(patched_serial: list[bytes]) -> None:
    c = WLEDSerialClient()
    c.set_effect("solid", color="teal")
    seg = _parse_line(patched_serial[0])["seg"][0]
    assert seg["col"] == [[0, 180, 180]]


# --------------------------------------------------------------- blink

def test_blink_sends_discrete_frames(patched_serial: list[bytes]) -> None:
    """v8: blink runs N on/off cycles in software, NOT a continuous Strobe."""
    c = WLEDSerialClient()
    c.blink(count=3, color="blue", speed="fast")
    msgs = _parse_lines(patched_serial)
    # 3 cycles × (set_solid + off) + 1 final off in `finally`
    assert len(msgs) == 7
    # No frame should be the WLED Strobe effect (fx=23).
    for m in msgs:
        if "seg" in m:
            assert m["seg"][0]["fx"] != 23


def test_blink_alternates_solid_and_off(patched_serial: list[bytes]) -> None:
    c = WLEDSerialClient()
    c.blink(count=2, color="red", speed="fast")
    msgs = _parse_lines(patched_serial)
    # Cycles: solid, off, solid, off, off(final)
    assert msgs[0]["seg"][0]["col"] == [[255, 0, 0]]
    assert msgs[1] == {"on": False}
    assert msgs[2]["seg"][0]["col"] == [[255, 0, 0]]
    assert msgs[3] == {"on": False}


# --------------------------------------------------------------- null client

def test_null_client_is_noop() -> None:
    c = NullWLEDClient()
    c.on()
    c.off()
    c.set_solid("red", 50)
    c.set_effect("rainbow")
    c.set_effect("aurora", palette="ocean", intensity="high")
    c.blink(3, "blue")
    c.close()


# --------------------------------------------------------------- error boundary

def test_serial_write_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = types.ModuleType("serial")

    class ExplodingSerial:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise OSError("device disconnected")

    fake_mod.Serial = ExplodingSerial  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "serial", fake_mod)

    c = WLEDSerialClient()
    # Should log + swallow, not raise.
    c.on()
    c.off()
    c.set_effect("aurora")
