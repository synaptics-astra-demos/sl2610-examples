"""Unit tests for the v8 dispatcher.

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

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def set_status_led(self, led: str, state: str,
                       brightness: int = 100) -> None:
        self.calls.append(("set_status_led", {
            "led": led, "state": state, "brightness": brightness,
        }))

    def blink_status_led(self, led: str, count: int = 3,
                         speed: str = "normal") -> None:
        self.calls.append(("blink_status_led", {
            "led": led, "count": count, "speed": speed,
        }))

    def set_neopixel_effect(self, effect: str, color: str | None = None,
                            palette: str | None = None,
                            speed: str = "normal",
                            intensity: str | None = None) -> None:
        self.calls.append(("set_neopixel_effect", {
            "effect": effect, "color": color, "palette": palette,
            "speed": speed, "intensity": intensity,
        }))

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


# --------------------------------------------------------------- set_status_led

def test_set_status_led_ok(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    hw, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_status_led",
                                     {"led": "red", "state": "on"})])
    assert res.status == "ok"
    assert hw.calls == [("set_status_led",
                         {"led": "red", "state": "on", "brightness": 100})]


def test_set_status_led_all(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    hw, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_status_led",
                                     {"led": "all", "state": "off"})])
    assert res.status == "ok"
    assert hw.calls[0][1]["led"] == "all"


def test_set_status_led_invalid_led(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_status_led",
                                     {"led": "magenta", "state": "on"})])
    assert res.status == "error"
    assert "led" in res.message


def test_set_status_led_invalid_state(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_status_led",
                                     {"led": "red", "state": "flicker"})])
    assert res.status == "error"
    assert "state" in res.message


def test_set_status_led_invalid_brightness(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_status_led",
                                     {"led": "red", "state": "on", "brightness": 250})])
    assert res.status == "error"


# --------------------------------------------------------------- blink_status_led

def test_blink_status_led_defaults(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    hw, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("blink_status_led", {"led": "green"})])
    assert res.status == "ok"
    assert hw.calls == [("blink_status_led",
                         {"led": "green", "count": 3, "speed": "normal"})]


def test_blink_status_led_count_and_speed(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    hw, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("blink_status_led",
                                     {"led": "red", "count": 5, "speed": "fast"})])
    assert res.status == "ok"
    assert hw.calls[0][1] == {"led": "red", "count": 5, "speed": "fast"}


def test_blink_status_led_invalid_speed(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("blink_status_led",
                                     {"led": "red", "speed": "warp"})])
    assert res.status == "error"
    assert "speed" in res.message


def test_blink_status_led_zero_count_rejected(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("blink_status_led",
                                     {"led": "red", "count": 0})])
    assert res.status == "error"


# --------------------------------------------------------------- set_neopixel_effect

def test_set_neopixel_effect_minimal(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    hw, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_neopixel_effect",
                                     {"effect": "aurora"})])
    assert res.status == "ok"
    assert hw.calls[0][1]["effect"] == "aurora"


def test_set_neopixel_effect_full_args(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    hw, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_neopixel_effect", {
        "effect": "plasma", "color": "teal", "palette": "ocean",
        "speed": "slow", "intensity": "high",
    })])
    assert res.status == "ok"
    call = hw.calls[0][1]
    assert call["effect"] == "plasma"
    assert call["color"] == "teal"
    assert call["palette"] == "ocean"
    assert call["speed"] == "slow"
    assert call["intensity"] == "high"


def test_set_neopixel_effect_invalid_effect(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_neopixel_effect",
                                     {"effect": "tellmejoke"})])
    assert res.status == "error"
    assert "effect" in res.message


def test_set_neopixel_effect_invalid_palette(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_neopixel_effect",
                                     {"effect": "aurora", "palette": "vibrant"})])
    assert res.status == "error"
    assert "palette" in res.message


def test_set_neopixel_effect_invalid_intensity(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_neopixel_effect",
                                     {"effect": "fire", "intensity": "extreme"})])
    assert res.status == "error"
    assert "intensity" in res.message


def test_set_neopixel_effect_off(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    hw, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("set_neopixel_effect",
                                     {"effect": "off"})])
    assert res.status == "ok"
    assert hw.calls[0][1]["effect"] == "off"


# ----------------------------------------------- buzzer / status / respond pass-through

def test_buzzer_validates_pattern(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("play_buzzer", {"pattern": "explode"})])
    assert res.status == "error"


def test_respond_passes_message(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall("respond",
                                     {"message": "Did you mean the neopixels?"})])
    assert res.status == "ok"
    assert "neopixels" in res.message


# --------------------------------------------------------------- removed tools

@pytest.mark.parametrize("removed", [
    "turn_on_lights", "turn_off_lights", "set_led_color", "blink_lights",
    "set_neopixel_pattern",  # v7 name, now renamed
])
def test_removed_v7_tools_rejected(removed: str,
                                   hw_and_dispatcher: tuple[StubHardware, Dispatcher]
                                   ) -> None:
    _, d = hw_and_dispatcher
    [res] = d.dispatch_all([ToolCall(removed, {})])
    assert res.status == "error"
    assert "no handler" in res.message


# --------------------------------------------------------------- multi-tool

def test_dispatch_all_runs_sequence_in_order(hw_and_dispatcher: tuple[StubHardware, Dispatcher]) -> None:
    hw, d = hw_and_dispatcher
    results = d.dispatch_all([
        ToolCall("set_status_led", {"led": "red", "state": "on"}),
        ToolCall("set_neopixel_effect", {"effect": "aurora"}),
    ])
    assert all(r.status == "ok" for r in results)
    assert [c[0] for c in hw.calls] == ["set_status_led", "set_neopixel_effect"]
