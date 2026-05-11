# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

import atexit
import logging
import select
import signal
import sys
import termios
import threading
import tty
from collections.abc import Callable, Iterable
from typing import TextIO

logger = logging.getLogger(__name__)


class TerminalMode:
    """Owns terminal cbreak mode and restores the original settings."""

    def __init__(self, stream: TextIO | None = None, log: logging.Logger | None = None):
        self._stream = stream or sys.stdin
        self._logger = log or logger
        self._lock = threading.Lock()
        self._fd = None
        self._settings = None

    def enter_cbreak(self) -> bool:
        if not self._stream.isatty():
            return False

        fd = self._stream.fileno()
        with self._lock:
            if self._settings is None:
                self._fd = fd
                self._settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        return True

    def read_key(self, timeout: float = 0.1) -> str | None:
        readable, _, _ = select.select([self._stream], [], [], timeout)
        if not readable:
            return None
        return self._stream.read(1)

    def restore(self):
        with self._lock:
            if self._fd is None or self._settings is None:
                return
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._settings)
            except termios.error:
                self._logger.debug("Failed to restore terminal settings", exc_info=True)
            finally:
                self._fd = None
                self._settings = None


def install_cli_shutdown_handlers(
    shutdown: Callable[[], None],
    *,
    exit_callbacks: Iterable[Callable[[], None]] = (),
):
    """Run ``shutdown`` for process exits, signals, and uncaught thread errors."""
    exit_callbacks = tuple(exit_callbacks)
    shutdown_lock = threading.Lock()
    shutdown_started = False

    def run_shutdown():
        nonlocal shutdown_started
        with shutdown_lock:
            if shutdown_started:
                return
            shutdown_started = True
        shutdown()
        for callback in exit_callbacks:
            callback()

    atexit.register(run_shutdown)

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def handle_signal(signum, frame):
        previous = previous_sigint if signum == signal.SIGINT else previous_sigterm
        run_shutdown()
        if callable(previous):
            previous(signum, frame)
        elif previous == signal.SIG_IGN:
            return
        elif signum == signal.SIGINT:
            raise KeyboardInterrupt
        else:
            raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    previous_threading_excepthook = threading.excepthook

    def handle_thread_exception(args):
        run_shutdown()
        previous_threading_excepthook(args)

    threading.excepthook = handle_thread_exception
