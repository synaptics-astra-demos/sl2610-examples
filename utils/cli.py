# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from __future__ import annotations

import atexit
from contextlib import contextmanager
import logging
import os
import select
import signal
import sys
import termios
import threading
import tty
from collections.abc import Callable, Iterable
from typing import TextIO

logger = logging.getLogger(__name__)

_native_atexit_lock = threading.Lock()
_native_atexit_suppression_registered = False


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
    raise_on_signal: bool = True,
):
    """Run ``shutdown`` for process exits, signals, and uncaught thread errors."""
    exit_callbacks = tuple(exit_callbacks)
    shutdown_lock = threading.Lock()
    shutdown_started = False

    def run_shutdown():
        nonlocal shutdown_started
        with shutdown_lock:
            if shutdown_started:
                return False
            shutdown_started = True
        shutdown()
        for callback in exit_callbacks:
            callback()
        return True

    atexit.register(run_shutdown)

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def handle_signal(signum, frame):
        previous = previous_sigint if signum == signal.SIGINT else previous_sigterm
        first_shutdown = run_shutdown()
        if first_shutdown and not raise_on_signal:
            return
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


@contextmanager
def suppress_native_stderr(enabled: bool = True):
    """Temporarily redirect process-level stderr to ``/dev/null``.

    Python logging levels do not affect native C/C++ libraries that write
    directly to file descriptor 2. Use this for short native-library
    startup/shutdown sections.
    """

    if not enabled:
        yield
        return

    try:
        sys.stderr.flush()
    except Exception:
        pass

    saved_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def suppress_native_stderr_at_exit():
    """Keep native atexit handlers quiet.

    Some native libraries emit shutdown diagnostics from their own atexit hooks
    after the Python application has already cleaned up. Registering this after
    those libraries are imported makes it run first because atexit handlers are
    last-in-first-out.
    """

    global _native_atexit_suppression_registered
    with _native_atexit_lock:
        if _native_atexit_suppression_registered:
            return
        _native_atexit_suppression_registered = True

    def redirect_stderr_to_devnull():
        try:
            sys.stderr.flush()
        except Exception:
            pass
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 2)
            os.close(devnull_fd)
        except Exception:
            pass

    atexit.register(redirect_stderr_to_devnull)
