"""HAT hardware control: status LEDs + buzzer + camera + alarms.

Backed by the interfaces in Grinn's OOBE image report for the Coral Dev
Board (SL2619, AstraSOM-261x):

    Status LEDs : /sys/class/leds/{red,green,blue}:status/brightness  (0-255)
    Buzzer      : gpioset `gpiofind BUZZERn`=1/0  (binary, no PWM)
    Camera      : gst-launch-1.0 v4l2src device=/dev/video0 (OV5647 NV12)
    Thermal     : /sys/class/thermal/thermal_zone0/temp  (millidegrees C)

Neopixel ring is delegated to a ``WLEDSerialClient`` if one was passed in.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

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


def _gpio_buzzer(state: bool) -> None:
    """Set the HAT buzzer via gpioset + gpiofind.

    ``gpiofind BUZZERn`` prints "<chip> <line>"; gpioset takes that + a value.
    """
    try:
        find = _run(["gpiofind", "BUZZERn"])
        if find.returncode != 0 or not find.stdout.strip():
            log.warning("gpiofind BUZZERn empty (rc=%d, stderr=%r)",
                        find.returncode, find.stderr)
            return
        chip, line = find.stdout.strip().split(maxsplit=1)
        _run(["gpioset", chip, f"{line}={1 if state else 0}"])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        log.exception("gpioset/gpiofind failed")


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

    def __init__(self, wled: WLEDSerialClient | None = None) -> None:
        self._wled = wled
        self._alarms: dict[str, _Alarm] = {}
        self._alarm_lock = threading.Lock()
        log.info(
            "HardwareDevice ready (status_leds=%d, wled=%s)",
            sum(1 for p in STATUS_LEDS.values() if p.exists()),
            "yes" if wled else "no",
        )

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
            self._wled.on()
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
        for _ in range(max(1, int(count))):
            self._apply_status_leds(color, 100)
            time.sleep(period)
            for path in STATUS_LEDS.values():
                if path.exists():
                    _write_sysfs(path, "0")
            time.sleep(period)

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
        for on_s, off_s in seq:
            _gpio_buzzer(True)
            time.sleep(on_s)
            _gpio_buzzer(False)
            if off_s > 0:
                time.sleep(off_s)

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

    def list_alarms(self) -> dict[str, list[dict[str, str]]]:
        with self._alarm_lock:
            items = [
                {"label": a.label,
                 "trigger_at": a.trigger_at.isoformat(timespec="seconds")}
                for a in self._alarms.values()
            ]
        return {"alarms": items}

    def _fire_alarm(self, label: str) -> None:
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
