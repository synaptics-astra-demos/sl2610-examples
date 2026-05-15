"""WLED JSON-over-serial client for the Adafruit Mini Sparkle Motion.

WLED accepts the same JSON payload over its serial port as it does over
HTTP (https://kno.wled.ge/interfaces/serial/). One JSON object per line,
115200 8-N-1 on /dev/ttyACM0.

The Mini Sparkle Motion exposes its UART through an on-board QinHeng CH343
USB-serial bridge (USB id 1a86:55d3), which enumerates as a standard CDC
ACM device — hence the /dev/ttyACM0 path on Linux. Baud rate matters on
this path (115200 is what WLED's serial config defaults to). It is NOT the
ESP32-S3 native USB-CDC interface that earlier hardware revisions used.

The dispatcher constructs a client only when ``--wled-port`` is passed, so
this whole module is a no-op when the user runs without the ring.
"""

from __future__ import annotations

import difflib
import json
import logging
import time
from typing import Any

log = logging.getLogger("functiongemma.wled")

DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUD = 115200
DEFAULT_TIMEOUT_S = 0.2

# Color names cover the v9 trained palette. Unknown names fall back to white
# via resolve_color (with a fuzzy-match attempt first).
COLOR_NAMES: dict[str, tuple[int, int, int]] = {
    "white":       (255, 255, 255),
    "warm_white":  (255, 200, 140),
    "cool_white":  (220, 235, 255),
    "soft_white":  (255, 240, 200),
    "red":         (255, 0, 0),
    "dark_red":    (140, 0, 0),
    "orange":      (255, 120, 0),
    "amber":       (255, 150, 30),
    "gold":        (255, 200, 40),
    "yellow":      (255, 220, 0),
    "lime":        (180, 255, 30),
    "green":       (0, 255, 0),
    "mint":        (140, 255, 200),
    "cyan":        (0, 255, 200),
    "turquoise":   (60, 220, 200),
    "sky_blue":    (120, 200, 255),
    "blue":        (0, 60, 255),
    "indigo":      (90, 30, 255),
    "violet":      (180, 80, 255),
    "purple":      (160, 0, 255),
    "magenta":     (255, 0, 200),
    "pink":        (255, 80, 180),
    "hot_pink":    (255, 80, 160),
    "peach":       (255, 180, 140),
    "teal":        (0, 180, 180),
    "gray":        (160, 160, 160),
    "grey":        (160, 160, 160),
    "black":       (0, 0, 0),
    "off":         (0, 0, 0),
}

SPEED_MAP: dict[str, int] = {"slow": 60, "normal": 128, "fast": 220}

INTENSITY_MAP: dict[str, int] = {"low": 64, "medium": 128, "high": 220}

# Effect name -> WLED effect id + default intensity. Effect IDs per
# https://kno.wled.ge/features/effects/
EFFECT_MAP: dict[str, dict[str, int]] = {
    "solid":     {"fx":   0, "ix": 128},
    "pulse":     {"fx":   2, "ix": 128},  # Breathe
    "fade":      {"fx":  12, "ix": 128},
    "chase":     {"fx":  28, "ix": 180},
    "rainbow":   {"fx":   9, "ix": 128},
    "sparkle":   {"fx":  20, "ix": 200},
    "aurora":    {"fx":  38, "ix": 128},  # Aurora Borealis
    "plasma":    {"fx":  97, "ix": 128},  # Plasma lamp
    "comet":     {"fx":  41, "ix": 160},  # Lighthouse — single trailing dot
    "twinkle":   {"fx":  80, "ix": 128},  # Twinklefox — slow fade
    "fireworks": {"fx":  42, "ix": 200},
    "police":    {"fx":  49, "ix": 255},
    "heartbeat": {"fx": 100, "ix": 128},
    "loading":   {"fx":  47, "ix": 180},
    "lightning": {"fx":  57, "ix": 200},
    "glitter":   {"fx":  87, "ix": 200},  # Rainbow + white sparkles
    "fire":      {"fx":  66, "ix": 180},  # Fire 2012
    "sunrise":   {"fx": 104, "ix": 128},
}

# Palette name -> WLED palette id. IDs per
# https://kno.wled.ge/features/palettes/
PALETTE_MAP: dict[str, int] = {
    "auto":    0,
    "ocean":   9,
    "lava":    8,
    "forest":  10,
    "sunset":  13,
    "party":   6,
    "sherbet": 27,
    "c9":      48,
    "aurora":  50,
    "beach":   22,
    "fire":    35,
    "sakura":  49,
    "splash":  19,
    "pastel":  20,
}

# Effects whose visible result depends on col[0] being the user's primary
# color. Chase additionally consumes col[1] as the trail.
_PRIMARY_COLOR_EFFECTS = {"solid", "pulse", "fade", "chase", "sparkle", "comet"}
_TRAIL_COLOR_EFFECTS = {"chase"}


def _normalize(color: str) -> str:
    return color.strip().lower().replace(" ", "_").replace("-", "_")


def resolve_color(color: str | None) -> tuple[int, int, int]:
    """Map a color name or '#RRGGBB' hex string to an (R, G, B) triple."""
    if not color:
        return COLOR_NAMES["white"]
    norm = _normalize(color)
    if norm in COLOR_NAMES:
        return COLOR_NAMES[norm]
    if norm.startswith("#") and len(norm) == 7:
        try:
            return (int(norm[1:3], 16), int(norm[3:5], 16), int(norm[5:7], 16))
        except ValueError:
            pass
    # Fuzzy match before giving up — catches typos like "purpel" -> "purple".
    # 0.8 cutoff is permissive enough for one-letter swaps without matching
    # truly unrelated names.
    near = difflib.get_close_matches(norm, COLOR_NAMES.keys(), n=1, cutoff=0.8)
    if near:
        log.info("color %r matched %r via fuzzy lookup", color, near[0])
        return COLOR_NAMES[near[0]]
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

    def set_effect(self, effect: str, color: str | None = None,
                   palette: str | None = None, speed: str = "normal",
                   intensity: str | None = None) -> None:
        effect_l = effect.lower()
        # "off" is a no-effect sentinel: just power the segment down.
        if effect_l == "off":
            self.off()
            return

        meta = EFFECT_MAP.get(effect_l, EFFECT_MAP["solid"])
        fx = meta["fx"]
        ix = INTENSITY_MAP.get((intensity or "").lower(), meta["ix"])
        sx = SPEED_MAP.get(speed.lower(), SPEED_MAP["normal"])

        seg: dict[str, Any] = {"id": 0, "fx": fx, "sx": sx, "ix": ix}

        if palette:
            pal_id = PALETTE_MAP.get(palette.lower())
            if pal_id is not None:
                seg["pal"] = pal_id
            else:
                log.warning("unknown palette %r, ignoring", palette)
        elif effect_l in _PRIMARY_COLOR_EFFECTS:
            # Force palette 0 (Default = use seg.col[0]) so a palette left
            # over from a prior effect (aurora, plasma, etc.) doesn't bleed
            # into the new solid/single-color output. Without this, "turn the
            # lights white" after a rainbow leaves the strip in palette mode.
            seg["pal"] = 0

        # Rainbow doesn't use a primary color (it spans the spectrum); omit
        # `col` entirely so any residual color from a prior call doesn't bleed
        # through. Other effects get the resolved primary; chase also gets a
        # dim version of the primary as a trail color (col[1]).
        if effect_l == "rainbow":
            pass
        elif effect_l in _PRIMARY_COLOR_EFFECTS:
            r, g, b = resolve_color(color) if color else COLOR_NAMES["white"]
            cols: list[list[int]] = [[r, g, b]]
            if effect_l in _TRAIL_COLOR_EFFECTS:
                cols.append([r // 6, g // 6, b // 6])
            seg["col"] = cols
        elif color:
            # Palette-driven effects (aurora, plasma, fire, etc.) tolerate
            # a primary color hint; pass it through but don't require it.
            r, g, b = resolve_color(color)
            seg["col"] = [[r, g, b]]

        self._send({"on": True, "bri": 255, "seg": [seg]})

    def blink(self, count: int, color: str = "white",
              speed: str = "normal") -> None:
        """Discrete blink: N solid frames + N off frames.

        WLED's continuous Strobe effect was used previously but never
        respected ``count`` — the ring would strobe forever while the HAT
        loop terminated cleanly. We drive the ring in software so HAT + ring
        blink together.
        """
        period = {"slow": 0.40, "normal": 0.20, "fast": 0.08}.get(speed, 0.20)
        try:
            for _ in range(max(1, int(count))):
                self.set_solid(color, brightness_pct=100)
                time.sleep(period)
                self.off()
                time.sleep(period)
        finally:
            self.off()

    def close(self) -> None:
        if self._serial is not None and getattr(self._serial, "is_open", False):
            try:
                self._serial.close()
            except Exception:  # noqa: BLE001
                log.exception("wled serial close failed")
        self._serial = None


class NullWLEDClient:
    """Stand-in client used off-device (no Sparkle Motion attached).

    Mirrors the public surface of ``WLEDSerialClient`` so the dispatcher
    can be wired up the same way on the dev laptop.
    """

    def on(self) -> None: ...
    def off(self) -> None: ...
    def set_solid(self, color: str, brightness_pct: int = 100) -> None: ...
    def set_effect(self, effect: str, color: str | None = None,
                   palette: str | None = None, speed: str = "normal",
                   intensity: str | None = None) -> None: ...
    def blink(self, count: int, color: str = "white",
              speed: str = "normal") -> None: ...
    def close(self) -> None: ...


def make_wled_client(port: str | None = None) -> WLEDSerialClient | NullWLEDClient:
    """Construct a real serial client if a port is configured, else null."""
    if not port:
        return NullWLEDClient()
    return WLEDSerialClient(port=port)
