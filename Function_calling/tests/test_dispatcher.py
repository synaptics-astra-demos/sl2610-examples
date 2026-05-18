"""Unit tests for the v10 dispatcher.

Exercises the dispatcher's enum validation and routing to a stub
HardwareDevice that records every call. No real hardware required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compact_codec import ToolCall  # noqa: E402
from dispatcher import Dispatcher, DispatchResult  # noqa: E402


class StubHardware:
    """Minimal HardwareDevice surrogate that records every method call."""

    # Dispatcher reads this to construct a LightsController. None = HAT mode;
    # tests that exercise strip mode set this to a stub WLED client.
    _wled: Any = None

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def play_buzzer(self, pattern: str) -> None:
        self.calls.append(("play_buzzer", {"pattern": pattern}))

    def set_alarm(self, duration: str | None = None,
                  time_str: str | None = None,
                  label: str | None = None) -> dict[str, str]:
        out = {"label": label or "alarm_X", "trigger_at": "2026-05-12T00:00:00"}
        self.calls.append(("set_alarm", {
            "duration": duration, "time_str": time_str, "label": label,
        }))
        return out

    def cancel_alarm(self, label: str | None = None) -> dict[str, list[str]]:
        self.calls.append(("cancel_alarm", {"label": label}))
        return {"cancelled": [label] if label else []}

    def get_system_status(self, metric: str = "all") -> dict[str, Any]:
        self.calls.append(("get_system_status", {"metric": metric}))
        return {"cpu": 12.5, "memory": 40.0, "temperature": 45.0, "npu": "idle"}


@pytest.fixture
def hw_and_dispatcher() -> tuple[StubHardware, Dispatcher]:
    hw = StubHardware()
    return hw, Dispatcher(hw)  # type: ignore[arg-type]


# ----------------------------------------------- buzzer / status / respond pass-through

def test_buzzer_validates_pattern(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("play_buzzer", {"pattern": "explode"})])
    assert res.status == "fallback"


def test_respond_passes_message(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("respond",
                                     {"message": "Did you mean the neopixels?"})])
    assert res.status == "ok"
    assert "neopixels" in res.message


# --------------------------------------------------------------- removed tools

@pytest.mark.parametrize("removed", [
    # v7-era names dropped before v8
    "turn_on_lights", "turn_off_lights", "set_led_color", "blink_lights",
    "set_neopixel_pattern",
    # v9 LED surface collapsed into set_lights in v10
    "set_status_led", "blink_status_led", "set_neopixel_effect",
])
def test_removed_legacy_tools_rejected(removed: str,
                                       hw_and_dispatcher: tuple[StubHardware, Dispatcher]
                                       ) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall(removed, {})])
    assert res.status == "fallback"
    assert "didn't recognize" in res.message


# --------------------------------------------------------------- v10 set_lights

def test_set_lights_color_only(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_lights", {"color": "red"})])
    assert res.status == "ok"
    assert "color=red" in res.message


def test_set_lights_effect_only(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_lights", {"effect": "rainbow"})])
    assert res.status == "ok"


def test_set_lights_state_on(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_lights", {"state": "on"})])
    assert res.status == "ok"


def test_set_lights_state_off(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_lights", {"state": "off"})])
    assert res.status == "ok"


def test_set_lights_color_and_effect(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_lights",
                                     {"color": "blue", "effect": "pulse"})])
    assert res.status == "ok"
    d.cleanup()  # stop the looped pulse thread


def test_set_lights_empty_args_ok(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_lights", {})])
    # No args = solid all-on (HAT) or default solid (strip). Should not error.
    assert res.status == "ok"


def test_set_lights_invalid_color_fallback(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_lights", {"color": "ultraviolet"})])
    assert res.status == "fallback"
    assert "color" in res.message


def test_set_lights_invalid_effect_fallback(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_lights", {"effect": "warpdrive"})])
    assert res.status == "fallback"
    assert "effect" in res.message


def test_set_lights_invalid_state_fallback(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_lights", {"state": "maybe"})])
    assert res.status == "fallback"
    assert "state" in res.message


# --------------------------------------------------------------- multi-tool

def test_dispatch_all_runs_sequence_in_order(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    hw, d = hw_and_dispatcher
    results = d.dispatch_all([
        ToolCall("play_buzzer", {"pattern": "beep"}),
        ToolCall("get_system_status", {"metric": "memory"}),
    ])
    assert all(r.status == "ok" for r in results)
    assert [c[0] for c in hw.calls] == ["play_buzzer", "get_system_status"]
    d.cleanup()
