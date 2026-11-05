# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""A log-formatted status line that updates in place instead of scrolling."""

from __future__ import annotations

import logging
import sys
import threading
import time

# Longest gap between redraws (TTY) or repeated lines (no TTY) while nothing changes.
TTY_REFRESH_SECONDS = 0.25
PIPE_HEARTBEAT_SECONDS = 10.0


class LiveStatus:
    """Show one frequently changing status in place, and keep other log lines intact.

    On a terminal the status is redrawn on a single line (at most every
    ``TTY_REFRESH_SECONDS``, or at once when its text changes). Any other log message
    first ends that line, so enrolment and removal messages print normally and the
    status resumes on a fresh line afterwards.

    When stderr is not a terminal (a pipe, a file, the smoke test) in-place updates would
    only produce one long garbled line, so a normal log line is written when the status
    text changes, plus a heartbeat every ``PIPE_HEARTBEAT_SECONDS``.
    """

    def __init__(self, logger: logging.Logger, level: int = logging.INFO):
        self._logger = logger
        self._level = level
        self._stream = sys.stderr
        self._tty = self._stream.isatty()
        self._lock = threading.Lock()
        self._line_open = False  # a status line is on screen with no trailing newline
        self._last_text: str | None = None
        self._last_draw = 0.0
        self._filter = _EndLineFilter(self)
        self._handlers: list[logging.Handler] = []

    def __enter__(self) -> "LiveStatus":
        if self._tty:
            # Only the console handler needs to know; file handlers never see the status line.
            for handler in logging.getLogger().handlers:
                if (isinstance(handler, logging.StreamHandler)
                        and not isinstance(handler, logging.FileHandler)
                        and getattr(handler, "stream", None) is self._stream):
                    handler.addFilter(self._filter)
                    self._handlers.append(handler)
        return self

    def __exit__(self, *exc) -> None:
        self.finish()
        for handler in self._handlers:
            handler.removeFilter(self._filter)
        self._handlers.clear()

    def update(self, status: str, text: str) -> None:
        """``status`` is the changing part (frame, timing, ...) and ``text`` what is compared
        for changes; the line shown is ``status`` followed by ``text``."""
        if not self._logger.isEnabledFor(self._level):
            return
        now = time.monotonic()
        changed = text != self._last_text
        if self._tty:
            if not changed and now - self._last_draw < TTY_REFRESH_SECONDS:
                return
            self._draw(f"{status} {text}", now)
        else:
            if not changed and now - self._last_draw < PIPE_HEARTBEAT_SECONDS:
                return
            self._logger.log(self._level, "%s %s", status, text)
            self._last_draw = now
        self._last_text = text

    def finish(self) -> None:
        """End the status line so whatever prints next starts on a new line."""
        self.end_line()

    def end_line(self) -> None:
        with self._lock:
            if self._line_open:
                self._stream.write("\n")
                self._stream.flush()
                self._line_open = False

    def _draw(self, message: str, now: float) -> None:
        prefix = f"[{logging.getLevelName(self._level):<8}] {self._logger.name}: "
        with self._lock:
            self._stream.write(f"\r{prefix}{message}\x1b[K")
            self._stream.flush()
            self._line_open = True
        self._last_draw = now


class _EndLineFilter(logging.Filter):
    """Starts any other log record on a new line when a status line is open."""

    def __init__(self, status: LiveStatus):
        super().__init__()
        self._status = status

    def filter(self, record: logging.LogRecord) -> bool:
        self._status.end_line()
        return True
