"""Unified lights controller for v10 ``set_lights`` tool.

Routes ``(color, effect, state)`` intent to either the WLED strip (when
connected) or to HAT-mode approximations on the 3 status LEDs. Owns the
background thread for looped effects so each new ``set_lights`` call
preemptively cancels the previous animation.

Strip path: delegates to the existing ``WLEDSerialClient.set_effect``
which already knows the named-effect/palette space.

HAT path: implements ~19 effect approximations on the 3 discrete LEDs
(red / green / blue) using sysfs brightness writes. Approximations are
*charming, not literal* — the goal is for the take-home build to feel
responsive on minimal hardware.
"""

from __future__ import annotations

import logging
import random
import threading
from pathlib import Path
from typing import Callable

from wled import WLEDSerialClient

log = logging.getLogger("functiongemma.lights")


# -- sysfs primitives (mirrored from hardware.py to avoid a private import) --

LED_ROOT = Path("/sys/class/leds")
STATUS_LEDS = {
    "red":   LED_ROOT / "red:status"   / "brightness",
    "green": LED_ROOT / "green:status" / "brightness",
    "blue":  LED_ROOT / "blue:status"  / "brightness",
}


def _write_led(name: str, value: int) -> None:
    path = STATUS_LEDS.get(name)
    if path is None or not path.exists():
        return
    try:
        path.write_text(str(int(max(0, min(255, value)))))
    except OSError:
        log.exception("LED sysfs write failed: %s <- %d", path, value)


def _set_hat(*pairs: tuple[str, int]) -> None:
    """Drive specified LEDs to given values; all others to 0."""
    requested = dict(pairs)
    for name in ("red", "green", "blue"):
        _write_led(name, requested.get(name, 0))


# -- color name -> HAT LED combo --

_HAT_COLOR_MAP: dict[str, list[tuple[str, int]]] = {
    "red":     [("red", 255)],
    "green":   [("green", 255)],
    "blue":    [("blue", 255)],
    "yellow":  [("red", 255), ("green", 255)],
    "cyan":    [("green", 255), ("blue", 255)],
    "purple":  [("red", 255), ("blue", 255)],
    "magenta": [("red", 255), ("blue", 255)],
    "pink":    [("red", 255), ("blue", 76)],
    "orange":  [("red", 255), ("green", 76)],
    "white":   [("red", 255), ("green", 255), ("blue", 255)],
}


def _resolve_hat_color(color: str | None) -> list[tuple[str, int]]:
    if color is None:
        return [("red", 255), ("green", 255), ("blue", 255)]
    mapped = _HAT_COLOR_MAP.get(color.lower())
    if mapped is None:
        return [("red", 255), ("green", 255), ("blue", 255)]
    return mapped


def _scale_color(color: str | None, factor: float) -> list[tuple[str, int]]:
    leds = _resolve_hat_color(color)
    return [(n, int(v * factor)) for n, v in leds]


# -- HAT effect loops (each takes a stop event + optional color) --
# Each function runs on a background thread; check stop.wait(...) often.

def _loop_pulse(stop: threading.Event, color: str | None) -> None:
    while not stop.is_set():
        for v in range(0, 256, 16):
            _set_hat(*_scale_color(color, v / 255))
            if stop.wait(0.06):
                return
        for v in range(255, -1, -16):
            _set_hat(*_scale_color(color, v / 255))
            if stop.wait(0.06):
                return


def _loop_rainbow(stop: threading.Event, color: str | None) -> None:
    order = ("red", "green", "blue")
    i = 0
    while not stop.is_set():
        for j, n in enumerate(order):
            _write_led(n, 255 if j == i else 0)
        i = (i + 1) % 3
        if stop.wait(0.8):
            return


def _loop_fire(stop: threading.Event, color: str | None) -> None:
    while not stop.is_set():
        _set_hat(("red", random.randint(150, 255)))
        if stop.wait(random.uniform(0.05, 0.15)):
            return


def _loop_plasma(stop: threading.Event, color: str | None) -> None:
    while not stop.is_set():
        for v in range(0, 256, 16):
            _set_hat(("red", v), ("blue", 255 - v))
            if stop.wait(0.1):
                return
        for v in range(255, -1, -16):
            _set_hat(("red", v), ("blue", 255 - v))
            if stop.wait(0.1):
                return


def _loop_aurora(stop: threading.Event, color: str | None) -> None:
    while not stop.is_set():
        for v in range(13, 77, 4):
            _set_hat(("green", v))
            if stop.wait(0.15):
                return
        for v in range(76, 12, -4):
            _set_hat(("green", v))
            if stop.wait(0.15):
                return


def _loop_police(stop: threading.Event, color: str | None) -> None:
    while not stop.is_set():
        _set_hat(("red", 255))
        if stop.wait(0.125):
            return
        _set_hat(("blue", 255))
        if stop.wait(0.125):
            return


def _loop_fireworks(stop: threading.Event, color: str | None) -> None:
    while not stop.is_set():
        leds = random.choice([
            ("red",), ("green",), ("blue",),
            ("red", "green"), ("green", "blue"), ("red", "blue"),
            ("red", "green", "blue"),
        ])
        for v in (255, 200, 150, 100, 50, 0):
            _set_hat(*((n, v) for n in leds))
            if stop.wait(0.05):
                return
        if stop.wait(random.uniform(0.3, 0.8)):
            return


def _loop_sparkle(stop: threading.Event, color: str | None) -> None:
    while not stop.is_set():
        n = random.choice(("red", "green", "blue"))
        _set_hat((n, 255))
        if stop.wait(0.05):
            return
        _set_hat()
        if stop.wait(random.uniform(0.1, 0.4)):
            return


def _loop_twinkle(stop: threading.Event, color: str | None) -> None:
    while not stop.is_set():
        n = random.choice(("red", "green", "blue"))
        for v in range(0, 128, 8):
            _set_hat((n, v))
            if stop.wait(0.05):
                return
        for v in range(127, -1, -8):
            _set_hat((n, v))
            if stop.wait(0.05):
                return


def _loop_chase(stop: threading.Event, color: str | None) -> None:
    order = ("red", "green", "blue")
    i = 0
    while not stop.is_set():
        for j, n in enumerate(order):
            _write_led(n, 255 if j == i else 0)
        i = (i + 1) % 3
        if stop.wait(0.3):
            return


def _loop_comet(stop: threading.Event, color: str | None) -> None:
    order = ("red", "green", "blue")
    trail = {0: 255, 1: 80, 2: 30}
    i = 0
    while not stop.is_set():
        for j, n in enumerate(order):
            offset = (j - i) % 3
            _write_led(n, trail[offset])
        i = (i + 1) % 3
        if stop.wait(0.6):
            return


def _loop_heartbeat(stop: threading.Event, color: str | None) -> None:
    while not stop.is_set():
        _set_hat(("red", 255))
        if stop.wait(0.1):
            return
        _set_hat(("red", 50))
        if stop.wait(0.1):
            return
        _set_hat(("red", 255))
        if stop.wait(0.1):
            return
        _set_hat()
        if stop.wait(0.7):
            return


def _loop_glitter(stop: threading.Event, color: str | None) -> None:
    while not stop.is_set():
        n = random.choice(("red", "green", "blue"))
        _set_hat((n, 255))
        if stop.wait(0.04):
            return
        _set_hat()
        if stop.wait(0.04):
            return


def _loop_loading(stop: threading.Event, color: str | None) -> None:
    _loop_chase(stop, color)


def _loop_lightning(stop: threading.Event, color: str | None) -> None:
    while not stop.is_set():
        _set_hat(("red", 255), ("green", 255), ("blue", 255))
        if stop.wait(0.05):
            return
        for v in (180, 100, 40, 10, 0):
            _set_hat(("red", v), ("green", v), ("blue", v))
            if stop.wait(0.04):
                return
        if stop.wait(random.uniform(2.0, 6.0)):
            return


# -- One-shot HAT effects (run once, then exit) --

def _oneshot_blink(stop: threading.Event, color: str | None) -> None:
    leds = _resolve_hat_color(color or "white")
    for _ in range(3):
        if stop.is_set():
            return
        for name, val in leds:
            _write_led(name, val)
        if stop.wait(0.2):
            return
        _set_hat()
        if stop.wait(0.2):
            return


def _oneshot_fade(stop: threading.Event, color: str | None) -> None:
    for v in range(0, 256, 8):
        if stop.is_set():
            return
        _set_hat(*_scale_color(color or "white", v / 255))
        if stop.wait(0.04):
            return
    for v in range(255, -1, -8):
        if stop.is_set():
            return
        _set_hat(*_scale_color(color or "white", v / 255))
        if stop.wait(0.04):
            return


def _oneshot_sunrise(stop: threading.Event, color: str | None) -> None:
    # ~15s ramp: red dim → red bright → +green → +blue
    for v in range(0, 256, 4):
        if stop.is_set():
            return
        _set_hat(("red", v))
        if stop.wait(0.05):
            return
    for v in range(0, 256, 4):
        if stop.is_set():
            return
        _set_hat(("red", 255), ("green", v))
        if stop.wait(0.05):
            return
    for v in range(0, 256, 4):
        if stop.is_set():
            return
        _set_hat(("red", 255), ("green", 255), ("blue", v))
        if stop.wait(0.05):
            return


_HAT_LOOPS: dict[str, Callable[[threading.Event, str | None], None]] = {
    "pulse":     _loop_pulse,
    "rainbow":   _loop_rainbow,
    "fire":      _loop_fire,
    "plasma":    _loop_plasma,
    "aurora":    _loop_aurora,
    "police":    _loop_police,
    "fireworks": _loop_fireworks,
    "sparkle":   _loop_sparkle,
    "twinkle":   _loop_twinkle,
    "chase":     _loop_chase,
    "comet":     _loop_comet,
    "heartbeat": _loop_heartbeat,
    "glitter":   _loop_glitter,
    "loading":   _loop_loading,
    "lightning": _loop_lightning,
}

_HAT_ONESHOTS: dict[str, Callable[[threading.Event, str | None], None]] = {
    "blink":   _oneshot_blink,
    "fade":    _oneshot_fade,
    "sunrise": _oneshot_sunrise,
}


# Effects the WLED firmware knows natively. Anything outside this set the
# dispatcher will reject with a fallback suggestion.
KNOWN_EFFECTS: frozenset[str] = frozenset(
    list(_HAT_LOOPS.keys())
    + list(_HAT_ONESHOTS.keys())
    + ["solid", "off"]
)

KNOWN_COLORS: frozenset[str] = frozenset(_HAT_COLOR_MAP.keys())
KNOWN_STATES: frozenset[str] = frozenset({"on", "off"})


class LightsController:
    """Unified entry for the v10 set_lights tool.

    Routes intent to WLED (when present) or HAT (otherwise). Owns a single
    background thread for looped effects; a new ``set_lights`` cancels the
    previous animation before starting the next.
    """

    def __init__(self, wled: WLEDSerialClient | None) -> None:
        self._wled = wled
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        return "strip" if self._wled is not None else "hat"

    def set_lights(
        self,
        color: str | None = None,
        effect: str | None = None,
        state: str | None = None,
    ) -> None:
        log.info("set_lights mode=%s color=%s effect=%s state=%s",
                 self.mode, color, effect, state)
        with self._lock:
            self._cancel_thread()
            if state == "off" or effect == "off":
                self._all_off()
                return
            if state == "on" and effect is None:
                effect = "solid"
            if self._wled is not None:
                self._drive_strip(color=color, effect=effect)
                return
            self._drive_hat(color=color, effect=effect)

    def cleanup(self) -> None:
        with self._lock:
            self._cancel_thread()
            self._all_off()

    # -------------------------------------------------------- internals

    def _cancel_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=1.0)
        self._stop.clear()
        self._thread = None

    def _all_off(self) -> None:
        _set_hat()
        if self._wled is not None:
            try:
                self._wled.off()
            except Exception:  # noqa: BLE001
                log.exception("WLED off failed")

    def _start(self, fn: Callable[[threading.Event, str | None], None],
               color: str | None) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=fn, args=(self._stop, color), daemon=True,
            name=f"lights-{fn.__name__}",
        )
        self._thread.start()

    def _drive_strip(self, color: str | None, effect: str | None) -> None:
        if effect is None:
            effect = "solid"
        try:
            self._wled.set_effect(effect=effect, color=color)
        except Exception:  # noqa: BLE001
            log.exception("WLED set_effect failed (effect=%s, color=%s)",
                          effect, color)

    def _drive_hat(self, color: str | None, effect: str | None) -> None:
        if effect is None or effect == "solid":
            _set_hat(*_resolve_hat_color(color))
            return
        loop = _HAT_LOOPS.get(effect)
        if loop is not None:
            self._start(loop, color)
            return
        oneshot = _HAT_ONESHOTS.get(effect)
        if oneshot is not None:
            self._start(oneshot, color)
            return
        # Unknown effect — fall back to solid color so the user still sees
        # something. (Dispatcher should have rejected via fallback before
        # we get here, but this is a defensive belt-and-suspenders path.)
        log.warning("unknown HAT effect %r, falling back to solid", effect)
        _set_hat(*_resolve_hat_color(color))
