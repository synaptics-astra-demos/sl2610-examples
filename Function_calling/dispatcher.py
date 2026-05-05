"""Route parsed ``ToolCall`` objects to ``HardwareDevice`` methods."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from compact_codec import ToolCall
from hardware import HardwareDevice

log = logging.getLogger("functiongemma.dispatcher")


@dataclass(frozen=True)
class DispatchResult:
    tool: str
    status: str  # "ok" | "error"
    message: str
    detail: dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"{self.tool}: {self.message}" if self.status == "ok" \
            else f"{self.tool} ERROR: {self.message}"


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
            log.exception("Handler for %s failed", call.name)
            return DispatchResult(tool=call.name, status="error", message=str(exc))

    # -------------------------------------------------------------- handlers

    def _turn_on(self, args: dict[str, Any]) -> DispatchResult:
        self._hw.turn_on_lights()
        return DispatchResult("turn_on_lights", "ok", "lights on")

    def _turn_off(self, args: dict[str, Any]) -> DispatchResult:
        self._hw.turn_off_lights()
        return DispatchResult("turn_off_lights", "ok", "lights off")

    def _set_color(self, args: dict[str, Any]) -> DispatchResult:
        self._hw.set_led_color(
            color=args["color"],
            target=args.get("target", "all"),
            brightness=int(args.get("brightness", 100)),
        )
        return DispatchResult("set_led_color", "ok", f"color = {args['color']}")

    def _blink(self, args: dict[str, Any]) -> DispatchResult:
        self._hw.blink_lights(
            count=int(args.get("count", 3)),
            color=args.get("color", "white"),
            speed=args.get("speed", "normal"),
        )
        return DispatchResult("blink_lights", "ok", "blinking")

    def _pattern(self, args: dict[str, Any]) -> DispatchResult:
        self._hw.set_neopixel_pattern(
            pattern=args["pattern"],
            color=args.get("color"),
            speed=args.get("speed", "normal"),
        )
        return DispatchResult("set_neopixel_pattern", "ok", f"pattern = {args['pattern']}")

    def _buzzer(self, args: dict[str, Any]) -> DispatchResult:
        pattern = args.get("pattern") or "beep"
        self._hw.play_buzzer(pattern=pattern)
        return DispatchResult("play_buzzer", "ok", f"pattern = {pattern}")

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
        return DispatchResult("cancel_alarm", "ok",
                              f"cancelled {len(result['cancelled'])} alarm(s)",
                              detail=result)

    def _list_alarms(self, args: dict[str, Any]) -> DispatchResult:
        result = self._hw.list_alarms()
        return DispatchResult("list_alarms", "ok",
                              f"{len(result['alarms'])} alarm(s)",
                              detail=result)

    def _status(self, args: dict[str, Any]) -> DispatchResult:
        result = self._hw.get_system_status(metric=args.get("metric", "all"))
        return DispatchResult("get_system_status", "ok", f"{result}", detail=result)

    def _respond(self, args: dict[str, Any]) -> DispatchResult:
        message = str(args.get("message", ""))
        return DispatchResult("respond", "ok", message)
