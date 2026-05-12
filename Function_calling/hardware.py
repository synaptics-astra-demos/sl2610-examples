"""HAT hardware control: status LEDs + buzzer + camera + alarms.

Backed by the interfaces in Grinn's OOBE image report for the Coral Dev
Board (SL2619, AstraSOM-261x):

    Status LEDs : /sys/class/leds/{red,green,blue}:status/brightness  (0-255)
    Buzzer      : gpioset `gpiofind BUZZERn`  (binary, ACTIVE-LOW: 0=ON, 1=OFF)
    Camera      : gst-launch-1.0 v4l2src device=/dev/video0 (OV5647 NV12)
    Thermal     : /sys/class/thermal/thermal_zone0/temp  (millidegrees C)

Neopixel ring is delegated to a ``WLEDSerialClient`` if one was passed in.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import FrameType
from typing import Any, Callable

from wled import WLEDSerialClient, resolve_color

log = logging.getLogger("functiongemma.hardware")

LED_ROOT = Path("/sys/class/leds")
STATUS_LEDS = {
    "red":   LED_ROOT / "red:status"   / "brightness",
    "green": LED_ROOT / "green:status" / "brightness",
    "blue":  LED_ROOT / "blue:status"  / "brightness",
}
THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")

BUZZER_PATTERNS: dict[str, list[tuple[float, float]]] = {
    "beep":        [(0.15, 0.0)],
    "double_beep": [(0.10, 0.08), (0.10, 0.0)],
    "chirp":       [(0.05, 0.05), (0.05, 0.0)],
    "success":     [(0.08, 0.05), (0.08, 0.05), (0.15, 0.0)],
    "error":       [(0.25, 0.08), (0.25, 0.0)],
    "alarm":       [(0.20, 0.10)] * 4,
    "siren":       [(0.12, 0.06)] * 6,
}


@dataclass(frozen=True)
class _Alarm:
    label: str
    trigger_at: datetime
    timer: threading.Timer = field(compare=False)


def _write_sysfs(path: Path, value: str) -> None:
    try:
        path.write_text(value)
    except OSError:
        log.exception("sysfs write failed: %s <- %s", path, value)


def _run(cmd: list[str], timeout: float = 3.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False,
    )


_HAS_GPIOD = (
    shutil.which("gpiofind") is not None
    and shutil.which("gpioset") is not None
)

# BUZZERn is ACTIVE-LOW: gpioset value 0 = beeping, 1 = silent. The chip
# driver also LATCHES the last-driven value across `gpioset --mode=exit`
# calls (default mode), so once a value is set the line holds it
# until something else drives it. This means we only need to write the
# inverse polarity to play_buzzer's pulse semantics: state=True (pulse on)
# → 0, state=False (silence) → 1. And every pattern MUST end with state=False
# or the buzzer will be stuck on after exit.
_BUZZER_OFF = "1"
_BUZZER_ON = "0"


def _resolve_buzzer_line() -> tuple[str, str] | None:
    """Return (chip, line) for the BUZZERn GPIO, or None if unavailable."""
    if not _HAS_GPIOD:
        return None
    try:
        find = _run(["gpiofind", "BUZZERn"])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        log.warning("gpiofind not callable")
        return None
    if find.returncode != 0 or not find.stdout.strip():
        log.warning("gpiofind BUZZERn empty (rc=%d, stderr=%r)",
                    find.returncode, find.stderr)
        return None
    chip, _, line = find.stdout.strip().partition(" ")
    return chip, line


_BUZZER_LINE: tuple[str, str] | None = _resolve_buzzer_line()


def _drive_buzzer(value: str) -> None:
    """Drive BUZZERn to '0' or '1'. Releases the line on exit, but the
    Synaptics chip driver retains the latched value until next write."""
    if _BUZZER_LINE is None:
        return
    chip, line = _BUZZER_LINE
    try:
        _run(["gpioset", chip, f"{line}={value}"])
    except subprocess.TimeoutExpired:
        log.warning("gpioset BUZZERn=%s timed out", value)


def _gpio_buzzer(state: bool) -> None:
    """Drive BUZZERn for one half-pulse. ``state=True`` beeps, ``False`` silences."""
    _drive_buzzer(_BUZZER_ON if state else _BUZZER_OFF)


def _all_status_leds_off() -> None:
    for path in STATUS_LEDS.values():
        if path.exists():
            try:
                path.write_text("0")
            except OSError:
                pass


def _safe_shutdown_outputs() -> None:
    """Drive buzzer + status LEDs to silent/off. No instance state required —
    this is the bare minimum we always want, callable from a signal handler
    or atexit on a half-constructed/destroyed process."""
    _drive_buzzer(_BUZZER_OFF)
    _all_status_leds_off()


_ACTIVE_DEVICE: "weakref.ReferenceType[HardwareDevice] | None" = None
_HANDLERS_INSTALLED = False


def _safe_shutdown_full() -> None:
    """Outputs-off + per-instance cleanup (alarm timers, WLED close)."""
    if _ACTIVE_DEVICE is not None:
        dev = _ACTIVE_DEVICE()
        if dev is not None:
            try:
                dev._cleanup_instance()
            except Exception:  # noqa: BLE001 — best-effort during shutdown
                log.exception("HardwareDevice cleanup raised")
    _safe_shutdown_outputs()


def _signal_handler(signum: int, frame: FrameType | None) -> None:
    """Drive outputs to safe state, then re-raise the signal under its
    default disposition so normal termination proceeds (exit code, parent
    notification, core dump policy, etc.)."""
    _safe_shutdown_full()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _install_shutdown_handlers() -> None:
    """Idempotent: install atexit + signal handlers once per process.

    Covers normal exit, sys.exit, uncaught exceptions, SIGTERM (kill, init
    shutdown), and SIGHUP (terminal close, parent death). SIGINT is left
    alone — Python's default raises KeyboardInterrupt, which unwinds
    play_buzzer's try/finally and triggers atexit.

    NOT covered: SIGKILL, kernel OOM kill, segfault, power loss. Those
    require an OS-level safety net (e.g. a systemd unit with ExecStopPost
    that runs `gpioset gpiochip0 6=1`).
    """
    global _HANDLERS_INSTALLED
    if _HANDLERS_INSTALLED:
        return
    _HANDLERS_INSTALLED = True
    atexit.register(_safe_shutdown_full)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError, AttributeError):
            # ValueError: not in main thread. AttributeError: SIGHUP missing
            # on Windows. OSError: signal unsupported. All non-fatal — atexit
            # still gives us cleanup on normal exits.
            pass


def _read_cpu_temp_c() -> float | None:
    try:
        return int(THERMAL_ZONE.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def _parse_duration(s: str) -> timedelta:
    s = s.strip().lower()
    parts = s.split()
    if len(parts) < 2:
        raise ValueError(f"Can't parse duration: {s!r}")
    amount = float(parts[0])
    unit = parts[1].rstrip("s")
    factor = {"second": 1, "minute": 60, "hour": 3600}.get(unit)
    if factor is None:
        raise ValueError(f"Unknown unit: {unit!r}")
    return timedelta(seconds=amount * factor)


def _compute_trigger(duration: str | None, time_str: str | None) -> datetime:
    now = datetime.now()
    if duration:
        return now + _parse_duration(duration)
    if time_str:
        hh, mm = map(int, time_str.split(":"))
        trigger = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if trigger < now:
            trigger += timedelta(days=1)
        return trigger
    raise ValueError("Must provide duration or time")


class HardwareDevice:
    """All HAT-side hardware behind one façade. WLED ring is optional."""

    def __init__(
        self,
        wled: WLEDSerialClient | None = None,
        on_async_event: Callable[[str], None] | None = None,
    ) -> None:
        global _ACTIVE_DEVICE
        self._wled = wled
        self._on_async_event = on_async_event
        self._alarms: dict[str, _Alarm] = {}
        self._alarm_lock = threading.Lock()
        if not _HAS_GPIOD:
            log.warning("libgpiod (gpiofind/gpioset) not in PATH — buzzer is a no-op")
        elif _BUZZER_LINE is None:
            log.warning("BUZZERn line not found via gpiofind — buzzer is a no-op")
        # Drive everything to a safe state on construction in case a prior
        # process crashed mid-pulse and left a stuck output.
        _safe_shutdown_outputs()
        _ACTIVE_DEVICE = weakref.ref(self)
        _install_shutdown_handlers()
        log.info(
            "HardwareDevice ready (status_leds=%d, wled=%s, gpiod=%s, buzzer=%s)",
            sum(1 for p in STATUS_LEDS.values() if p.exists()),
            "yes" if wled else "no",
            "yes" if _HAS_GPIOD else "no",
            f"{_BUZZER_LINE[0]} {_BUZZER_LINE[1]}" if _BUZZER_LINE else "absent",
        )
        if wled is None:
            log.warning(
                "no WLED client — every neopixel-ring command "
                "(turn_on_lights, turn_off_lights, set_led_color target=strip, "
                "blink_lights, set_neopixel_pattern) will silently no-op on "
                "the ring. Pass --wled-port to drive it."
            )

    def cleanup(self) -> None:
        """Cancel timers, silence buzzer, kill LEDs, close WLED.

        Idempotent; safe to call multiple times. Wrap your demo's main loop
        in ``try / finally: hardware.cleanup()`` so a normal exit path also
        triggers it (signal handlers + atexit are the safety net for the
        abnormal paths)."""
        self._cleanup_instance()
        _safe_shutdown_outputs()

    def _cleanup_instance(self) -> None:
        with self._alarm_lock:
            for a in self._alarms.values():
                a.timer.cancel()
            self._alarms.clear()
        if self._wled is not None:
            try:
                self._wled.off()
                self._wled.close()
            except Exception:  # noqa: BLE001 — serial close is a boundary
                log.exception("WLED close failed")

    # ------------------------------------------------------------------ LEDs

    def _apply_status_leds(self, color: str, brightness_pct: int) -> None:
        r, g, b = resolve_color(color)
        scale = max(0, min(100, brightness_pct)) / 100.0
        threshold = 80
        channels = {
            "red":   int(r * scale) if r >= threshold else 0,
            "green": int(g * scale) if g >= threshold else 0,
            "blue":  int(b * scale) if b >= threshold else 0,
        }
        for name, path in STATUS_LEDS.items():
            if path.exists():
                _write_sysfs(path, str(channels[name]))

    def turn_on_lights(self) -> None:
        self._apply_status_leds("white", 100)
        if self._wled:
            # set_solid (not just .on()) — WLED retains the last fx/col across
            # master power toggles, so a bare `wled.on()` after a previous
            # set_neopixel_pattern would resume that pattern instead of
            # producing the white the user asked for.
            self._wled.set_solid("white", 100)
        log.info("lights ON (white)")

    def turn_off_lights(self) -> None:
        for path in STATUS_LEDS.values():
            if path.exists():
                _write_sysfs(path, "0")
        if self._wled:
            self._wled.off()
        log.info("lights OFF")

    def set_led_color(self, color: str, target: str = "all", brightness: int = 100) -> None:
        target = (target or "all").lower()
        if target in ("all", "hat"):
            self._apply_status_leds(color, brightness)
        if target in ("all", "strip") and self._wled:
            self._wled.set_solid(color, brightness)
        log.info("color=%s target=%s brightness=%d", color, target, brightness)

    def blink_lights(self, count: int = 3, color: str = "white", speed: str = "normal") -> None:
        if self._wled:
            self._wled.blink(count=count, color=color, speed=speed)
        period = {"slow": 0.40, "normal": 0.20, "fast": 0.08}.get(speed, 0.20)
        try:
            for _ in range(max(1, int(count))):
                self._apply_status_leds(color, 100)
                time.sleep(period)
                _all_status_leds_off()
                time.sleep(period)
        finally:
            _all_status_leds_off()

    def set_neopixel_pattern(self, pattern: str, color: str | None = None,
                             speed: str = "normal") -> None:
        if self._wled is None:
            log.info("set_neopixel_pattern %r ignored (no --wled-port)", pattern)
            return
        self._wled.set_pattern(pattern=pattern, color=color, speed=speed)
        log.info("pattern=%s color=%s speed=%s", pattern, color, speed)

    # ---------------------------------------------------------------- Buzzer

    def play_buzzer(self, pattern: str) -> None:
        seq = BUZZER_PATTERNS.get(pattern, BUZZER_PATTERNS["beep"])
        log.info("buzzer pattern=%s pulses=%d", pattern, len(seq))
        try:
            for on_s, off_s in seq:
                _gpio_buzzer(True)
                time.sleep(on_s)
                _gpio_buzzer(False)
                if off_s > 0:
                    time.sleep(off_s)
        finally:
            _gpio_buzzer(False)

    # ---------------------------------------------------------------- Alarms

    def set_alarm(self, duration: str | None = None, time_str: str | None = None,
                  label: str | None = None) -> dict[str, str]:
        trigger = _compute_trigger(duration, time_str)
        key = label or f"alarm_{int(time.time())}"
        delay = max(0.0, (trigger - datetime.now()).total_seconds())
        timer = threading.Timer(delay, self._fire_alarm, args=(key,))
        timer.daemon = True
        timer.start()
        with self._alarm_lock:
            self._alarms[key] = _Alarm(label=key, trigger_at=trigger, timer=timer)
        return {"label": key, "trigger_at": trigger.isoformat(timespec="seconds")}

    def cancel_alarm(self, label: str | None = None) -> dict[str, list[str]]:
        with self._alarm_lock:
            if label is None:
                cancelled = list(self._alarms.keys())
                for a in self._alarms.values():
                    a.timer.cancel()
                self._alarms.clear()
            elif label in self._alarms:
                self._alarms[label].timer.cancel()
                del self._alarms[label]
                cancelled = [label]
            else:
                cancelled = []
        return {"cancelled": cancelled}

    def _fire_alarm(self, label: str) -> None:
        if self._on_async_event is not None:
            self._on_async_event(f"ALARM FIRED: {label}")
        else:
            log.warning("ALARM FIRED: %s", label)
        self.play_buzzer(pattern="alarm")
        self.blink_lights(count=5, color="red", speed="fast")
        with self._alarm_lock:
            self._alarms.pop(label, None)

    # ---------------------------------------------------------- System status

    def get_system_status(self, metric: str = "all") -> dict[str, float | str]:
        try:
            import psutil
        except ImportError:
            psutil = None  # type: ignore[assignment]

        cpu = psutil.cpu_percent(interval=None) if psutil else 0.0
        mem = psutil.virtual_memory().percent if psutil else 0.0
        temp = _read_cpu_temp_c()
        status: dict[str, float | str] = {
            "cpu": float(cpu),
            "memory": float(mem),
            "temperature": temp if temp is not None else "unknown",
            "npu": "unknown",
        }
        if metric == "all":
            return status
        return {metric: status.get(metric, "unknown")}

    # --------------------------------------------------------------- Camera

    def capture_photo(self, save_as: str | None = None) -> str:
        name = save_as or f"photo_{int(time.time())}"
        out = Path("/tmp/functiongemma") / f"{name}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)

        if shutil.which("gst-launch-1.0") is None:
            log.warning("gst-launch-1.0 missing - skipping capture")
            return str(out)

        pipeline = [
            "gst-launch-1.0", "-q",
            "v4l2src", "device=/dev/video0", "num-buffers=1",
            "!", "video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1",
            "!", "videoconvert",
            "!", "jpegenc", "quality=90",
            "!", "filesink", f"location={out}",
        ]
        try:
            res = _run(pipeline, timeout=5.0)
            if res.returncode != 0:
                log.warning("gst-launch rc=%d stderr=%r", res.returncode, res.stderr)
        except subprocess.TimeoutExpired:
            log.warning("gst-launch timed out capturing photo")
        return str(out)
