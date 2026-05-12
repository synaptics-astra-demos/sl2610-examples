"""Tests for HardwareDevice._fire_alarm.

We can't run the real buzzer/LED writes off-device (and we don't want a
5-second pytest), so we monkeypatch ``play_buzzer`` + ``blink_lights`` to
recording mocks and verify the contract:

  - the async callback is invoked once with the full ``"ALARM FIRED: ..."``
    string before the blocking hardware sequence runs,
  - the loop dispatches the buzzer + blink sequence the expected number
    of cycles,
  - the alarm entry is removed from ``HardwareDevice._alarms`` afterwards,
  - the WLED ring is driven off at the end so the strobe doesn't outlive
    the fire sequence.
"""

from __future__ import annotations

import threading
from datetime import datetime
from unittest.mock import MagicMock

import pytest

import hardware
from hardware import HardwareDevice, _Alarm


def _make_alarm(label: str) -> _Alarm:
    # A real threading.Timer is fine — we never start it, but _Alarm is
    # frozen so we can't pass None for the timer field.
    return _Alarm(
        label=label,
        trigger_at=datetime.now(),
        timer=threading.Timer(60.0, lambda: None),
    )


@pytest.fixture
def quiet_hardware(monkeypatch: pytest.MonkeyPatch) -> HardwareDevice:
    """A HardwareDevice with the real GPIO/LED I/O stubbed out."""
    monkeypatch.setattr(hardware, "_all_status_leds_off", lambda: None)
    dev = HardwareDevice(wled=None)
    monkeypatch.setattr(dev, "play_buzzer", MagicMock())
    monkeypatch.setattr(dev, "blink_lights", MagicMock())
    return dev


def test_fire_alarm_invokes_callback_with_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hardware, "_all_status_leds_off", lambda: None)
    callback = MagicMock()
    dev = HardwareDevice(wled=None, on_async_event=callback)
    monkeypatch.setattr(dev, "play_buzzer", MagicMock())
    monkeypatch.setattr(dev, "blink_lights", MagicMock())
    dev._alarms["my-label"] = _make_alarm("my-label")

    dev._fire_alarm("my-label")

    callback.assert_called_once_with("ALARM FIRED: my-label")
    assert "my-label" not in dev._alarms


def test_fire_alarm_loops_buzzer_and_blink(quiet_hardware: HardwareDevice) -> None:
    quiet_hardware._fire_alarm("kitchen")

    cycles = HardwareDevice._ALARM_FIRE_CYCLES
    assert quiet_hardware.play_buzzer.call_count == cycles
    for call in quiet_hardware.play_buzzer.call_args_list:
        assert call.kwargs.get("pattern") == "alarm" or call.args == ("alarm",)

    assert quiet_hardware.blink_lights.call_count == cycles
    for call in quiet_hardware.blink_lights.call_args_list:
        assert call.kwargs == {"count": 5, "color": "red", "speed": "fast"}


def test_fire_alarm_turns_wled_off_after_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hardware, "_all_status_leds_off", lambda: None)
    wled = MagicMock()
    dev = HardwareDevice(wled=wled)
    monkeypatch.setattr(dev, "play_buzzer", MagicMock())
    monkeypatch.setattr(dev, "blink_lights", MagicMock())

    dev._fire_alarm("done")

    wled.off.assert_called_once_with()


def test_fire_alarm_with_no_callback_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(hardware, "_all_status_leds_off", lambda: None)
    dev = HardwareDevice(wled=None)
    monkeypatch.setattr(dev, "play_buzzer", MagicMock())
    monkeypatch.setattr(dev, "blink_lights", MagicMock())

    with caplog.at_level("WARNING", logger="functiongemma.hardware"):
        dev._fire_alarm("silent")

    assert any("ALARM FIRED: silent" in rec.message for rec in caplog.records)
