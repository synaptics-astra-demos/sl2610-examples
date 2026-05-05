"""Command-line FunctionGemma physical-AI demo for the SL2619 Coral Dev Board.

Two modes:

    # One-shot
    python3 demo.py --prompt "Turn the lights red and beep twice"

    # Interactive REPL (model stays loaded; turn 2+ is sub-second after the
    # first turn pays the one-time tool-declaration prefill of ~45-50 s)
    python3 demo.py

By default the WLED Neopixel ring is OFF. Pass ``--wled-port /dev/ttyACM0``
to drive an Adafruit Mini Sparkle Motion + WS2812B ring over USB-CDC.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

from dispatcher import Dispatcher
from hardware import HardwareDevice
from llamacpp import FunctionGemmaModel
from wled import WLEDSerialClient


DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent
    / "models" / "functiongemma-physical-ai-v6-Q5_K_M.gguf"
)


_USE_ANSI = sys.stdout.isatty() and sys.stderr.isatty()
_DIM = "\033[2;37m" if _USE_ANSI else ""
_RESET = "\033[0m" if _USE_ANSI else ""


def _dim(text: str) -> str:
    return f"{_DIM}{text}{_RESET}"


class _Spinner:
    """Minimal stderr spinner: shows label with elapsed seconds + a rotating glyph.

    On exit, either prints "<label> done in N.Ns." or clears the line in
    place (clear_on_exit=True) so the spinner leaves no trace.
    """

    _GLYPHS = "|/-\\"

    def __init__(self, label: str, clear_on_exit: bool = False) -> None:
        self._label = label
        self._clear = clear_on_exit
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_Spinner":
        self._t0 = time.time()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        if self._clear:
            sys.stderr.write("\r" + " " * (len(self._label) + 20) + "\r")
        else:
            elapsed = time.time() - self._t0
            sys.stderr.write(f"\r{self._label} done in {elapsed:.1f}s.{' ' * 20}\n")
        sys.stderr.flush()

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set():
            elapsed = time.time() - self._t0
            sys.stderr.write(
                f"\r{_DIM}{self._label} {self._GLYPHS[i % 4]} {elapsed:5.1f}s{_RESET}",
            )
            sys.stderr.flush()
            i += 1
            self._stop.wait(0.1)


def run_turn(model: FunctionGemmaModel, dispatcher: Dispatcher, prompt: str) -> None:
    with _Spinner("thinking", clear_on_exit=True):
        result = model.generate(prompt)
    for call_result in dispatcher.dispatch_all(result.tool_calls):
        print(f"  {call_result}")
    n = len(result.tool_calls)
    plural = "s" if n != 1 else ""
    print(_dim(f"  ({n} tool call{plural} · {result.latency_ms:.0f} ms)"))


def main() -> None:
    p = argparse.ArgumentParser(
        description="FunctionGemma physical-AI demo on SL2619",
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                   help=f"Path to the GGUF model. Default: {DEFAULT_MODEL}")
    p.add_argument("--prompt",
                   help="One-shot prompt. Omit to start an interactive REPL.")
    p.add_argument(
        "--wled-port",
        help="USB-CDC port for the Mini Sparkle Motion (e.g. /dev/ttyACM0). "
             "Omit to run without the Neopixel ring.",
    )
    p.add_argument("--wled-baud", type=int, default=115200,
                   help="WLED serial baud rate (default 115200)")
    args = p.parse_args()

    with _Spinner(f"Loading model from {args.model.name}"):
        model = FunctionGemmaModel(str(args.model))

    wled = WLEDSerialClient(port=args.wled_port, baud=args.wled_baud) \
        if args.wled_port else None
    dispatcher = Dispatcher(HardwareDevice(wled=wled))

    if args.prompt:
        run_turn(model, dispatcher, args.prompt)
        return

    # Prime the prefix cache up front so turn 1 from the user is sub-second.
    # This pays the ~48 s cold prefill once, visibly, before we accept input.
    with _Spinner("Warming up (one-time ~50s prefill on the 2-core A55)"):
        model.generate("hello")

    print(_dim("Ready. Ctrl-D or empty line to exit."))
    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            break
        run_turn(model, dispatcher, prompt)


if __name__ == "__main__":
    main()
