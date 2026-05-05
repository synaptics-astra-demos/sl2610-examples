"""Route parsed ``ToolCall`` objects to ``HardwareDevice`` methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from compact_codec import ToolCall
from hardware import HardwareDevice


# Closed enums declared in tools.json descriptions. Validated here because the
# compact codec is positional and accepts whatever literal the model emits, so
# a hallucinated string ("Tell me a joke" -> pattern="No joke here.") would
# otherwise silently fall through to the hardware layer.
_NEOPIXEL_PATTERNS = {"rainbow", "chase", "fade", "pulse", "sparkle", "solid"}
_BUZZER_PATTERNS = {"beep", "double_beep", "siren", "chirp", "alarm",
                    "success", "error"}
_SYSTEM_METRICS = {"cpu", "memory", "temperature", "npu", "all"}
_LED_TARGETS = {"all", "hat", "strip"}
_SPEEDS = {"slow", "normal", "fast"}


@dataclass(frozen=True)
class DispatchResult:
    tool: str
    status: str  # "ok" | "error"
    message: str
    detail: dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"{self.tool}: {self.message}" if self.status == "ok" \
            else f"{self.tool} ERROR: {self.message}"


def _summary(args: dict[str, Any], *keys: str) -> str:
    """Format a 'k=v k=v' summary of the args the model actually emitted."""
    if not keys:
        keys = tuple(args.keys())
    parts = [f"{k}={args[k]}" for k in keys if k in args]
    return " ".join(parts) if parts else "—"


def _reject(tool: str, arg: str, value: Any, valid: set[str]) -> DispatchResult:
    return DispatchResult(
        tool=tool, status="error",
        message=f"invalid {arg}={value!r} (expected one of: {', '.join(sorted(valid))})",
    )


class Dispatcher:
    """Translate ToolCall -> HardwareDevice method invocation."""

    def __init__(self, hardware: HardwareDevice) -> None:
        self._hw = hardware
        self._handlers: dict[str, Callable[[dict[str, Any]], DispatchResult]] = {
            "turn_on_lights": self._turn_on,
            "turn_off_lights": self._turn_off,
            "set_led_color": self._set_color,
            "blink_lights": self._blink,
            "set_neopixel_pattern": self._pattern,
            "play_buzzer": self._buzzer,
            "set_alarm": self._set_alarm,
            "cancel_alarm": self._cancel_alarm,
            "list_alarms": self._list_alarms,
            "get_system_status": self._status,
            "respond": self._respond,
        }

    def dispatch_all(self, calls: list[ToolCall]) -> list[DispatchResult]:
        return [self._dispatch_one(c) for c in calls]

    def _dispatch_one(self, call: ToolCall) -> DispatchResult:
        handler = self._handlers.get(call.name)
        if handler is None:
            return DispatchResult(tool=call.name, status="error",
                                  message=f"no handler for {call.name!r}")
        try:
            return handler(call.arguments)
        except Exception as exc:  # noqa: BLE001 — dispatcher boundary
            return DispatchResult(tool=call.name, status="error", message=str(exc))

    # -------------------------------------------------------------- handlers

    def _turn_on(self, args: dict[str, Any]) -> DispatchResult:
        self._hw.turn_on_lights()
        return DispatchResult("turn_on_lights", "ok", "lights on")

    def _turn_off(self, args: dict[str, Any]) -> DispatchResult:
        self._hw.turn_off_lights()
        return DispatchResult("turn_off_lights", "ok", "lights off")

    def _set_color(self, args: dict[str, Any]) -> DispatchResult:
        target = args.get("target", "all")
        if target not in _LED_TARGETS:
            return _reject("set_led_color", "target", target, _LED_TARGETS)
        self._hw.set_led_color(
            color=args["color"],
            target=target,
            brightness=int(args.get("brightness", 100)),
        )
        return DispatchResult("set_led_color", "ok",
                              _summary(args, "color", "target", "brightness"))

    def _blink(self, args: dict[str, Any]) -> DispatchResult:
        speed = args.get("speed", "normal")
        if speed not in _SPEEDS:
            return _reject("blink_lights", "speed", speed, _SPEEDS)
        self._hw.blink_lights(
            count=int(args.get("count", 3)),
            color=args.get("color", "white"),
            speed=speed,
        )
        return DispatchResult("blink_lights", "ok",
                              _summary(args, "count", "color", "speed"))

    def _pattern(self, args: dict[str, Any]) -> DispatchResult:
        pattern = args.get("pattern")
        if pattern not in _NEOPIXEL_PATTERNS:
            return _reject("set_neopixel_pattern", "pattern", pattern,
                           _NEOPIXEL_PATTERNS)
        speed = args.get("speed", "normal")
        if speed not in _SPEEDS:
            return _reject("set_neopixel_pattern", "speed", speed, _SPEEDS)
        self._hw.set_neopixel_pattern(
            pattern=pattern,
            color=args.get("color"),
            speed=speed,
        )
        return DispatchResult("set_neopixel_pattern", "ok",
                              _summary(args, "pattern", "color", "speed"))

    def _buzzer(self, args: dict[str, Any]) -> DispatchResult:
        pattern = args.get("pattern") or "beep"
        if pattern not in _BUZZER_PATTERNS:
            return _reject("play_buzzer", "pattern", pattern, _BUZZER_PATTERNS)
        self._hw.play_buzzer(pattern=pattern)
        return DispatchResult("play_buzzer", "ok", f"pattern={pattern}")

    def _set_alarm(self, args: dict[str, Any]) -> DispatchResult:
        result = self._hw.set_alarm(
            duration=args.get("duration"),
            time_str=args.get("time"),
            label=args.get("label"),
        )
        return DispatchResult("set_alarm", "ok",
                              f"scheduled {result['label']} @ {result['trigger_at']}",
                              detail=result)

    def _cancel_alarm(self, args: dict[str, Any]) -> DispatchResult:
        result = self._hw.cancel_alarm(label=args.get("label"))
        n = len(result["cancelled"])
        which = f"label={args['label']}" if "label" in args else "all"
        return DispatchResult("cancel_alarm", "ok",
                              f"cancelled {n} ({which})", detail=result)

    def _list_alarms(self, args: dict[str, Any]) -> DispatchResult:
        result = self._hw.list_alarms()
        return DispatchResult("list_alarms", "ok",
                              f"{len(result['alarms'])} alarm(s)",
                              detail=result)

    def _status(self, args: dict[str, Any]) -> DispatchResult:
        metric = args.get("metric", "all")
        if metric not in _SYSTEM_METRICS:
            return _reject("get_system_status", "metric", metric,
                           _SYSTEM_METRICS)
        result = self._hw.get_system_status(metric=metric)
        return DispatchResult("get_system_status", "ok", f"{result}", detail=result)

    def _respond(self, args: dict[str, Any]) -> DispatchResult:
        message = str(args.get("message", ""))
        return DispatchResult("respond", "ok", message)
