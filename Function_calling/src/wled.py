"""WLED JSON-over-USB-CDC serial client for the Adafruit Mini Sparkle Motion.

WLED accepts the same JSON payload over its USB serial port as it does over
HTTP (https://kno.wled.ge/interfaces/serial/). One JSON object per line,
115200 8-N-1 on /dev/ttyACM0 (default for Sparkle Motion CDC-ACM). Baud
doesn't actually matter on USB-CDC; that's just what WLED docs quote.

The dispatcher constructs a client only when ``--wled-port`` is passed, so
this whole module is a no-op when the user runs without the ring.
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger("functiongemma.wled")

DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT_S = 0.2

COLOR_NAMES: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "warm_white": (255, 200, 140),
    "cool_white": (220, 235, 255),
    "red": (255, 0, 0),
    "orange": (255, 120, 0),
    "yellow": (255, 220, 0),
    "green": (0, 255, 0),
    "cyan": (0, 255, 200),
    "blue": (0, 60, 255),
    "purple": (160, 0, 255),
    "pink": (255, 80, 180),
    "magenta": (255, 0, 200),
    "off": (0, 0, 0),
}

SPEED_MAP: dict[str, int] = {"slow": 60, "normal": 128, "fast": 220}

# Pattern -> WLED effect id + default intensity. Effect ids per
# https://kno.wled.ge/features/effects/
PATTERN_EFFECTS: dict[str, dict[str, int]] = {
    "solid":   {"fx":  0, "ix": 128},
    "pulse":   {"fx":  2, "ix": 128},  # Breathe
    "fade":    {"fx": 12, "ix": 128},
    "chase":   {"fx": 28, "ix": 180},
    "rainbow": {"fx":  9, "ix": 128},
    "sparkle": {"fx": 20, "ix": 200},
}


def resolve_color(color: str | None) -> tuple[int, int, int]:
    """Map a color name or '#RRGGBB' hex string to an (R, G, B) triple."""
    if not color:
        return COLOR_NAMES["white"]
    color = color.strip().lower().replace(" ", "_").replace("-", "_")
    if color in COLOR_NAMES:
        return COLOR_NAMES[color]
    if color.startswith("#") and len(color) == 7:
        try:
            return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
        except ValueError:
            pass
    log.warning("unknown color %r, defaulting to white", color)
    return COLOR_NAMES["white"]


def _scale_brightness(pct: int) -> int:
    pct = max(0, min(100, int(pct)))
    return int(round(pct * 255 / 100))


class WLEDSerialClient:
    """WLED client over USB-CDC serial.

    Each public method builds a WLED JSON payload and writes one
    newline-terminated line. Serial port is opened lazily so construction
    never raises on hardware-missing paths.
    """

    def __init__(self, port: str = DEFAULT_PORT, baud: int = DEFAULT_BAUD,
                 timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._port = port
        self._baud = baud
        self._timeout_s = timeout_s
        self._serial: Any = None

    def _ensure_open(self) -> Any:
        if self._serial is not None and getattr(self._serial, "is_open", False):
            return self._serial
        import serial  # lazy — pyserial only required when --wled-port is used

        self._serial = serial.Serial(
            port=self._port,
            baudrate=self._baud,
            timeout=self._timeout_s,
            write_timeout=self._timeout_s,
        )
        log.info("wled serial opened on %s @ %d", self._port, self._baud)
        return self._serial

    def _send(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        try:
            ser = self._ensure_open()
            ser.write(line.encode("ascii"))
            ser.flush()
        except Exception:  # noqa: BLE001 — serial is an external boundary
            log.exception("wled serial write failed: %s", line.rstrip())

    def on(self) -> None:
        self._send({"on": True, "bri": 255})

    def off(self) -> None:
        self._send({"on": False})

    def set_solid(self, color: str, brightness_pct: int = 100) -> None:
        r, g, b = resolve_color(color)
        self._send({
            "on": True,
            "bri": _scale_brightness(brightness_pct),
            "seg": [{"id": 0, "col": [[r, g, b]], "fx": 0}],
        })

    def set_pattern(self, pattern: str, color: str | None = None, speed: str = "normal") -> None:
        effect = PATTERN_EFFECTS.get(pattern.lower(), PATTERN_EFFECTS["solid"])
        sx = SPEED_MAP.get(speed.lower(), SPEED_MAP["normal"])
        r, g, b = resolve_color(color) if color else COLOR_NAMES["white"]
        self._send({
            "on": True,
            "bri": 255,
            "seg": [{
                "id": 0,
                "col": [[r, g, b]],
                "fx": effect["fx"],
                "sx": sx,
                "ix": effect["ix"],
            }],
        })

    def blink(self, count: int, color: str = "white", speed: str = "normal") -> None:
        sx = SPEED_MAP.get(speed.lower(), SPEED_MAP["normal"])
        r, g, b = resolve_color(color)
        self._send({
            "on": True,
            "bri": 255,
            "seg": [{
                "id": 0,
                "col": [[r, g, b]],
                "fx": 23,  # Strobe — handled in firmware, deterministic
                "sx": sx,
                "ix": max(32, min(255, 32 * max(1, int(count)))),
            }],
        })

    def close(self) -> None:
        if self._serial is not None and getattr(self._serial, "is_open", False):
            try:
                self._serial.close()
            except Exception:  # noqa: BLE001
                log.exception("wled serial close failed")
        self._serial = None
