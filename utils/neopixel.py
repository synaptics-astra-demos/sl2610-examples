#!/usr/bin/env python3
import json
import logging
import os
import queue
import sys
import threading
import time

logger = logging.getLogger(__name__)

_Step = tuple[dict, float]  # (command, sleep_after_seconds)

PATTERNS: dict[str, list[_Step]] = {
    "ocean": [
        ({"on": True, "bri": 150, "seg": [{"fx": 43, "sx": 30, "ix": 255,
          "col": [[0, 100, 255], [0, 200, 255], [0, 255, 150]]}]}, 0),
    ],
    "listening": [
        ({"on": True, "bri": 200, "seg": [{"fx": 2, "sx": 100, "ix": 128,
          "col": [[0, 100, 255]]}]}, 0),
    ],
    "translating": [
        ({"on": True, "bri": 200, "seg": [{"fx": 10, "sx": 200, "ix": 128,
          "col": [[255, 255, 0], [0, 0, 0]]}]}, 0),
    ],
    "translation_complete": [
        ({"on": True, "bri": 200, "seg": [{"fx": 0, "col": [[0, 255, 0]]}]}, 2.0),
        ({"on": True, "bri": 50}, 0),
    ],
    "processing_image": [
        ({"on": True, "bri": 200, "seg": [{"fx": 10, "sx": 200, "ix": 64,
          "col": [[255, 255, 255]]}]}, 0),
    ],
    "generating_text": [
        ({"on": True, "bri": 200, "seg": [{"fx": 28, "sx": 150, "ix": 128,
          "col": [[255, 191, 0]]}]}, 0),
    ],
    "complete": [
        ({"on": True, "bri": 200, "seg": [{"fx": 3, "sx": 200, "col": [[0, 255, 0]]}]}, 1.5),
        ({"on": True, "bri": 50}, 0),
    ],
    "off": [
        ({"on": False}, 0),
    ],
}


class NeopixelAnimator:
    """Non-blocking NeoPixel LED animator.

    Call play(pattern) to start an animation. It returns immediately; commands
    and sleeps run in a single low-priority daemon thread. Submitting a new
    pattern discards any queued-but-not-yet-started work.
    """

    def __init__(self, tty: str = "/dev/ttyACM0"):
        self._tty = tty
        logger.debug("NeopixelAnimator: initializing with tty=%s", tty)
        self._queue: queue.Queue[list[_Step] | None] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="npxl-anim", daemon=True)
        self._thread.start()
        logger.debug("NeopixelAnimator: background thread started")
        self._init_tty()

    def play(self, pattern: str):
        steps = PATTERNS.get(pattern)
        if steps is None:
            raise ValueError(f"Unknown pattern {pattern!r}. Available: {list(PATTERNS)}")
        logger.debug("NeopixelAnimator: queuing pattern=%r (%d steps)", pattern, len(steps))
        # Drain pending unstarted animations so the new one isn't delayed
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(steps)

    def stop(self):
        logger.debug("NeopixelAnimator: stop requested")
        self._queue.put(None)
        self._thread.join(timeout=2.0)

    def _init_tty(self):
        cmd = f"stty -F {self._tty} 115200 raw -echo -hupcl"
        ret = os.system(cmd)
        if ret != 0:
            logger.warning("NeopixelAnimator: stty init failed (exit=%d) for %s", ret, self._tty)
        else:
            logger.debug("NeopixelAnimator: tty initialized OK (%s)", self._tty)

    def _send(self, cmd: dict):
        payload = json.dumps(cmd) + "\r\n"
        logger.debug("NeopixelAnimator: sending to %s: %s", self._tty, payload.strip())
        try:
            with open(self._tty, "w") as f:
                f.write(payload)
            logger.debug("NeopixelAnimator: send OK")
        except Exception:
            logger.exception("NeopixelAnimator: failed to send command to %s", self._tty)

    def _worker(self):
        logger.debug("NeopixelAnimator: worker thread running (nice+10)")
        try:
            os.nice(10)
        except Exception:
            logger.debug("NeopixelAnimator: os.nice() not available on this platform")
        while True:
            steps = self._queue.get()
            if steps is None:
                logger.debug("NeopixelAnimator: worker received stop sentinel, exiting")
                break
            logger.debug("NeopixelAnimator: executing %d-step animation", len(steps))
            for i, (cmd, delay) in enumerate(steps):
                logger.debug("NeopixelAnimator: step %d/%d delay=%.1fs", i + 1, len(steps), delay)
                self._send(cmd)
                if delay > 0:
                    time.sleep(delay)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: npxl.py <pattern>")
        print(f"Patterns: {', '.join(PATTERNS)}")
        sys.exit(1)

    animator = NeopixelAnimator()
    animator.play(sys.argv[1])
    animator.stop()
