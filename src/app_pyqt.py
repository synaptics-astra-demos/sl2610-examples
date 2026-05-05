"""PyQt5 launcher for the FunctionGemma physical-AI demo.

Mirrors the entry pattern used by ``astra-2610-assistant-UI-app/src/app_voice_pyqt.py``.
By default the WLED Neopixel ring is OFF. Pass ``--wled-port /dev/ttyACM0``
to drive an Adafruit Mini Sparkle Motion over USB-CDC.

Run:
    source ../setup_wayland.sh
    python3 app_pyqt.py --wled-port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from chat_window import ChatWindow
from dispatcher import Dispatcher
from hardware import HardwareDevice
from llamacpp import FunctionGemmaModel
from theme import apply_theme
from wled import WLEDSerialClient


DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent
    / "models" / "functiongemma-physical-ai-Q4_K_M.gguf"
)


def main() -> int:
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
    args = p.parse_args()

    app = QApplication(sys.argv)
    apply_theme(app)

    print(f"Loading model from {args.model}")
    model = FunctionGemmaModel(str(args.model))

    wled = WLEDSerialClient(port=args.wled_port, baud=args.wled_baud) \
        if args.wled_port else None
    dispatcher = Dispatcher(HardwareDevice(wled=wled))

    win = ChatWindow(model=model, dispatcher=dispatcher)
    if args.fullscreen or os.environ.get("FUNCTIONGEMMA_FULLSCREEN", "").lower() in ("1", "true", "yes"):
        win.showFullScreen()
    else:
        win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
