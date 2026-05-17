"""Route parsed ``ToolCall`` objects to ``HardwareDevice`` methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from compact_codec import ToolCall
from hardware import HardwareDevice
from lights import KNOWN_COLORS, KNOWN_EFFECTS, KNOWN_STATES, LightsController


# Closed enums declared in tools.json descriptions. Validated here because the
# compact codec is positional and accepts whatever literal the model emits, so
# a hallucinated string ("Tell me a joke" -> effect="No joke here.") would
# otherwise silently fall through to the hardware layer.
_STATUS_LEDS = {"red", "green", "blue", "all"}
_STATES = {"on", "off"}
_EFFECTS = {
    "solid", "pulse", "fade", "chase", "rainbow", "sparkle", "off",
    "aurora", "plasma", "comet", "twinkle", "fireworks", "police",
    "heartbeat", "loading", "lightning", "glitter", "fire", "sunrise",
}
_PALETTES = {
    "auto", "ocean", "lava", "forest", "sunset", "party", "sherbet",
    "c9", "aurora", "beach", "fire", "sakura", "splash", "pastel",
}
_INTENSITIES = {"low", "medium", "high"}
_SPEEDS = {"slow", "normal", "fast"}
_BUZZER_PATTERNS = {"beep", "double_beep", "siren", "chirp", "alarm",
                    "success", "error"}
_SYSTEM_METRICS = {"cpu", "memory", "temperature", "npu", "all"}


@dataclass(frozen=True)
class DispatchResult:
    tool: str
    status: str  # "ok" | "error" | "fallback"
    message: str
    detail: dict[str, Any] | None = None

    def __str__(self) -> str:
        if self.status == "ok":
            return f"{self.tool}: {self.message}"
        if self.status == "fallback":
            return f"{self.tool} (fallback): {self.message}"
        return f"{self.tool} ERROR: {self.message}"


def _summary(args: dict[str, Any], *keys: str) -> str:
    """Format a 'k=v k=v' summary of the args the model actually emitted."""
    if not keys:
        keys = tuple(args.keys())
    parts = [f"{k}={args[k]}" for k in keys if k in args]
    return " ".join(parts) if parts else "—"


def _reject(tool: str, arg: str, value: Any, valid: set[str]) -> DispatchResult:
    """Friendly fallback for invalid arg values. Renders as a respond
    bubble in the UI instead of a red ERROR badge — per Google's
    feedback that error bubbles ruin demo flow. The original tool call
    and the bad value are preserved in turn_log via raw_text."""
    suggestions = ", ".join(sorted(valid))
    return DispatchResult(
        tool=tool, status="fallback",
        message=f"I can do {arg}: {suggestions}. Try one of those?",
    )


class Dispatcher:
    """Translate ToolCall -> HardwareDevice method invocation.

    Carries a ``LightsController`` for the unified v10 ``set_lights`` tool;
    the strip-vs-HAT routing happens inside the controller based on whether
    the ``HardwareDevice`` was constructed with a WLED client.
    """

    def __init__(self, hardware: HardwareDevice) -> None:
        self._hw = hardware
        self._lights = LightsController(wled=hardware._wled)
        self._handlers: dict[str, Callable[[dict[str, Any]], DispatchResult]] = {
            "set_status_led": self._set_status_led,
            "blink_status_led": self._blink_status_led,
            "set_neopixel_effect": self._effect,
            "set_lights": self._set_lights,
            "play_buzzer": self._buzzer,
            "set_alarm": self._set_alarm,
            "cancel_alarm": self._cancel_alarm,
            "get_system_status": self._status,
            "respond": self._respond,
        }

    def cleanup(self) -> None:
        """Stop any running HAT effect thread. Idempotent."""
        self._lights.cleanup()

    def dispatch_all(self, calls: list[ToolCall]) -> list[DispatchResult]:
        return [self._dispatch_one(c) for c in calls]

    def _dispatch_one(self, call: ToolCall) -> DispatchResult:
        # Rescue path for a known v10 failure mode: the 270M model sometimes
        # emits the correct respond MESSAGE but inside a wrong tool token
        # (e.g. <tool_1>(message="Hi there.") when it meant <tool_5>). Any
        # tool that wasn't `respond` but carries a `message=` arg gets
        # reinterpreted as a respond turn. Cheap, no retrain needed; loses
        # one diagnostic signal (model's original tool choice) which is
        # acceptable for demo polish.
        if call.name != "respond" and "message" in call.arguments:
            return DispatchResult(
                tool="respond", status="ok",
                message=str(call.arguments["message"]),
            )
        handler = self._handlers.get(call.name)
        if handler is None:
            return DispatchResult(
                tool=call.name, status="fallback",
                message="I didn't recognize that command — try \"turn on the lights\", \"blink red\", or \"play a beep\".",
            )
        try:
            return handler(call.arguments)
        except Exception as exc:  # noqa: BLE001 — dispatcher boundary
            return DispatchResult(
                tool=call.name, status="fallback",
                message=f"Something went wrong running that. ({exc})",
            )

    # -------------------------------------------------------------- handlers

    def _set_status_led(self, args: dict[str, Any]) -> DispatchResult:
        led = args.get("led")
        if led not in _STATUS_LEDS:
            return _reject("set_status_led", "led", led, _STATUS_LEDS)
        state = args.get("state")
        if state not in _STATES:
            return _reject("set_status_led", "state", state, _STATES)
        brightness = int(args.get("brightness", 100))
        if not 0 <= brightness <= 100:
            return DispatchResult(
                tool="set_status_led", status="fallback",
                message=f"Brightness should be 0-100. I got {brightness}.",
            )
        self._hw.set_status_led(led=led, state=state, brightness=brightness)
        return DispatchResult("set_status_led", "ok",
                              _summary(args, "led", "state", "brightness"))

    def _blink_status_led(self, args: dict[str, Any]) -> DispatchResult:
        led = args.get("led")
        if led not in _STATUS_LEDS:
            return _reject("blink_status_led", "led", led, _STATUS_LEDS)
        speed = args.get("speed", "normal")
        if speed not in _SPEEDS:
            return _reject("blink_status_led", "speed", speed, _SPEEDS)
        count = int(args.get("count", 3))
        if count < 1:
            return DispatchResult(
                tool="blink_status_led", status="fallback",
                message=f"Blink count should be at least 1. I got {count}.",
            )
        self._hw.blink_status_led(led=led, count=count, speed=speed)
        return DispatchResult("blink_status_led", "ok",
                              _summary(args, "led", "count", "speed"))

    def _effect(self, args: dict[str, Any]) -> DispatchResult:
        effect = args.get("effect")
        if effect not in _EFFECTS:
            return _reject("set_neopixel_effect", "effect", effect, _EFFECTS)
        speed = args.get("speed", "normal")
        if speed not in _SPEEDS:
            return _reject("set_neopixel_effect", "speed", speed, _SPEEDS)
        palette = args.get("palette")
        if palette is not None and palette not in _PALETTES:
            return _reject("set_neopixel_effect", "palette", palette, _PALETTES)
        intensity = args.get("intensity")
        if intensity is not None and intensity not in _INTENSITIES:
            return _reject("set_neopixel_effect", "intensity", intensity,
                           _INTENSITIES)
        self._hw.set_neopixel_effect(
            effect=effect,
            color=args.get("color"),
            palette=palette,
            speed=speed,
            intensity=intensity,
        )
        return DispatchResult("set_neopixel_effect", "ok",
                              _summary(args, "effect", "color", "palette",
                                       "speed", "intensity"))

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

    def _set_lights(self, args: dict[str, Any]) -> DispatchResult:
        """v10 unified handler. All three args are optional; the dispatcher
        validates each present arg against its known-good set and emits a
        friendly fallback (rendered as respond bubble) on any miss."""
        color = args.get("color")
        if color is not None and color.lower() not in KNOWN_COLORS:
            return _reject("set_lights", "color", color, set(KNOWN_COLORS))
        effect = args.get("effect")
        if effect is not None and effect.lower() not in KNOWN_EFFECTS:
            return _reject("set_lights", "effect", effect, set(KNOWN_EFFECTS))
        state = args.get("state")
        if state is not None and state.lower() not in KNOWN_STATES:
            return _reject("set_lights", "state", state, set(KNOWN_STATES))
        # All args clean (or absent). LightsController normalizes to
        # lowercase internally for HAT path; strip path accepts raw strings.
        self._lights.set_lights(
            color=color.lower() if isinstance(color, str) else None,
            effect=effect.lower() if isinstance(effect, str) else None,
            state=state.lower() if isinstance(state, str) else None,
        )
        return DispatchResult(
            "set_lights", "ok",
            _summary(args, "color", "effect", "state"),
        )
