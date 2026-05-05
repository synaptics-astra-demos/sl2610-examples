"""Background metrics sampler for the UI top pane (psutil-only).

Samples CPU / memory / temperature at a fixed interval and pushes snapshots
onto a queue. UI thread drains the queue on a Qt timer. NPU + power are
stubbed at None — sysfs paths are not yet stable on SL2619.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from queue import Queue


@dataclass(frozen=True)
class MetricsSnapshot:
    timestamp: float
    cpu_percent: float
    memory_percent: float
    temperature_c: float | None
    npu_percent: float | None
    power_w: float | None
    extras: dict[str, float] = field(default_factory=dict)


class PsutilProvider:
    """psutil-backed sampler. CPU + mem + temp where available."""

    def __init__(self) -> None:
        import psutil

        self._psutil = psutil
        self._psutil.cpu_percent(None)  # prime

    def sample(self) -> MetricsSnapshot:
        p = self._psutil
        temps = p.sensors_temperatures() if hasattr(p, "sensors_temperatures") else {}
        temp_c: float | None = None
        for _, entries in temps.items():
            if entries:
                temp_c = float(entries[0].current)
                break
        return MetricsSnapshot(
            timestamp=time.time(),
            cpu_percent=p.cpu_percent(None),
            memory_percent=p.virtual_memory().percent,
            temperature_c=temp_c,
            npu_percent=None,
            power_w=None,
        )


class MetricsPump:
    """Background thread pumping snapshots onto a queue."""

    def __init__(self, provider: PsutilProvider, interval_s: float = 1.0,
                 queue_size: int = 8) -> None:
        self.provider = provider
        self.interval_s = interval_s
        self.queue: Queue[MetricsSnapshot] = Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                snap = self.provider.sample()
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except Exception:
                        pass
                self.queue.put_nowait(snap)
            except Exception:
                pass
            time.sleep(self.interval_s)
