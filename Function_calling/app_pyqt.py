"""PyQt5 launcher for the FunctionGemma physical-AI demo.

By default the WLED Neopixel ring is OFF. Pass ``--wled-port /dev/ttyACM0``
to drive an Adafruit Mini Sparkle Motion over USB-CDC.

Run (after exporting the wayland env vars listed in README.md):
    python3 app_pyqt.py --wled-port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

# Make the shared utils/ package importable from Function_calling/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from PyQt5.QtWidgets import QApplication

from chat_window import AlarmSignals, ChatWindow
from cpu_governor import ensure_performance_governor
from dispatcher import Dispatcher
from hardware import HardwareDevice
from llamacpp import FunctionGemmaModel
from theme import apply_theme
from voice import make_voice_pipeline
from wled import WLEDSerialClient


DEFAULT_MODEL = (
    Path(__file__).resolve().parent
    / "../models" / "coral-v10-Q5_K_M.gguf"
)


def main() -> int:
    ensure_performance_governor()
    p = argparse.ArgumentParser(
        description="FunctionGemma physical-AI PyQt demo on SL2619",
    )
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                   help=f"Path to the GGUF model. Default: {DEFAULT_MODEL}")
    p.add_argument(
        "--wled-port",
        help="USB-CDC port for the Mini Sparkle Motion. Omit to run without the ring.",
    )
    p.add_argument("--wled-baud", type=int, default=115200,
                   help="WLED serial baud rate (default 115200)")
    p.add_argument("--fullscreen", action="store_true",
                   help="Open the window full-screen (recommended on the 7\" panel)")
    p.add_argument(
        "--voice",
        choices=("off", "stub", "moonshine"),
        default="off",
        help="Voice input mode (default off). 'stub' uses canned phrases; "
             "'moonshine' uses the Torq-accelerated Moonshine ASR.",
    )
    p.add_argument(
        "--mic",
        help="Sounddevice selector (numeric index or name substring) for "
             "the input device. Defaults to the system default input.",
    )
    p.add_argument(
        "--moonshine-dir",
        type=Path,
        help="Directory holding Moonshine VMFB artifacts. Defaults to "
             "<repo>/models/Synaptics/moonshine-tiny-bf16-torq/ when "
             "--voice=moonshine.",
    )
    p.add_argument(
        "--screenshot-dir",
        type=Path,
        default=Path("/tmp"),
        help="Directory for Ctrl+P screenshots (default /tmp).",
    )
    args = p.parse_args()

    app = QApplication(sys.argv)
    apply_theme(app)

    print(f"Loading model from {args.model}")
    model = FunctionGemmaModel(str(args.model))

    # WLED auto-detect: if --wled-port is given, use it. Otherwise probe for
    # any /dev/ttyACM* and try the first one. This makes the strip "just
    # work" when plugged in without needing the user to re-run with a flag,
    # and gracefully falls back to HAT-only mode when no strip is present.
    wled_port = args.wled_port
    if not wled_port:
        candidates = sorted(glob.glob("/dev/ttyACM*"))
        if candidates:
            wled_port = candidates[0]
            logging.info("auto-detected WLED port: %s", wled_port)
    if wled_port:
        try:
            wled = WLEDSerialClient(port=wled_port, baud=args.wled_baud)
        except Exception:  # noqa: BLE001
            logging.exception(
                "WLED client failed to open %s — falling back to HAT-only",
                wled_port,
            )
            wled = None
    else:
        wled = None
        logging.info("no WLED device detected — running in HAT-only mode")
    alarm_signals = AlarmSignals()
    hardware = HardwareDevice(wled=wled, on_async_event=alarm_signals.fired.emit)
    dispatcher = Dispatcher(hardware)

    voice = None
    if args.voice != "off":
        voice = make_voice_pipeline(
            on_text=lambda _t: None,
            mode=args.voice,
            mic_device=args.mic,
            moonshine_dir=args.moonshine_dir,
        )

    win = ChatWindow(
        model=model,
        dispatcher=dispatcher,
        voice=voice,
        screenshot_dir=args.screenshot_dir,
        alarm_signals=alarm_signals,
    )
    if args.fullscreen:
        win.showFullScreen()
    else:
        win.show()
    try:
        return app.exec_()
    finally:
        if voice is not None:
            voice.stop()
        dispatcher.cleanup()
        hardware.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
