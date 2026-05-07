"""Command-line FunctionGemma physical-AI demo for the SL2619 Coral Dev Board.

Two modes:

    # One-shot
    python3 demo.py --prompt "Turn the lights red and beep twice"

    # Interactive REPL (model stays loaded; turn 2+ is sub-second after the
    # first turn pays the one-time tool-declaration prefill of ~45-50 s)
    python3 demo.py

By default the WLED Neopixel ring is OFF. Pass ``--wled-port /dev/ttyACM0``
to drive an Adafruit Mini Sparkle Motion + WS2812B ring over USB-CDC.

The REPL supports slash commands (``/help``, ``/tools``, ``/stats``,
``/clear``, ``/exit``) and persists prompt history across sessions to
``~/.coral_demo_history``. Press Ctrl-D or type ``/exit`` to leave;
empty Enter just re-prompts.
"""

from __future__ import annotations

import argparse
import atexit
import readline
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from cpu_governor import ensure_performance_governor
from dispatcher import Dispatcher
from hardware import HardwareDevice
from llamacpp import FunctionGemmaModel
from voice import VoicePipeline, make_voice_pipeline
from wled import WLEDSerialClient


DEFAULT_MODEL = (
    Path(__file__).resolve().parent.parent
    / "models" / "functiongemma-physical-ai-v7-Q5_K_M.gguf"
)
HISTORY_FILE = Path.home() / ".coral_demo_history"
HISTORY_LIMIT = 1000
PROMPT = ">>> "


_USE_ANSI = sys.stdout.isatty() and sys.stderr.isatty()
_DIM = "\033[2;37m" if _USE_ANSI else ""
_RESET = "\033[0m" if _USE_ANSI else ""


def _dim(text: str) -> str:
    return f"{_DIM}{text}{_RESET}"


_HELP_TEXT = """\
Slash commands:
  /?, /help            Show this help
  /tools               List the tools the model can call
  /stats               Show session statistics
  /clear               Clear the screen (model stays loaded)
  /exit, /bye, /quit   Exit
Anything else is sent to the model as a prompt."""

_TOOLS_TEXT = """\
Available tools (v7, 10 functions):
  turn_on_lights                  Turn all LEDs on (default white)
  turn_off_lights                 Turn all LEDs off
  set_led_color <color>           Set RGB color (target / brightness optional)
  blink_lights [count] [color]    Discrete blink pattern
  set_neopixel_pattern <pattern>  Animated ring (rainbow, chase, fade, pulse, sparkle, solid)
  play_buzzer <pattern>           beep, double_beep, chirp, siren, alarm, success, error
  set_alarm <duration|time>       Schedule alarm (label optional)
  cancel_alarm [label]            Cancel one or all alarms
  get_system_status [metric]      CPU / memory / temperature / NPU
  respond <message>               Natural-language fallback when no tool fits"""


@dataclass
class SessionStats:
    turns: int = 0
    tool_calls: int = 0
    total_latency_ms: float = 0.0
    last_latency_ms: float = 0.0

    def render(self) -> str:
        if self.turns == 0:
            return "Session: 0 turns yet."
        avg = self.total_latency_ms / self.turns
        return (
            f"Session: {self.turns} turn(s), {self.tool_calls} tool call(s), "
            f"avg {avg:.0f} ms / turn, last {self.last_latency_ms:.0f} ms"
        )


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


def _setup_history() -> None:
    """Load prompt history from disk and arrange to save it on exit."""
    try:
        readline.read_history_file(HISTORY_FILE)
    except (FileNotFoundError, OSError):
        pass
    readline.set_history_length(HISTORY_LIMIT)
    atexit.register(_save_history)


def _save_history() -> None:
    try:
        readline.write_history_file(HISTORY_FILE)
    except OSError:
        pass


def _print_async(msg: str) -> None:
    """Print a background-thread message without trashing the readline prompt.

    Pushes the existing prompt up with a leading newline, writes the
    message, then re-emits a fresh ``>>> ``. In-progress typing on the
    original prompt line is lost (the user must retype) — that's a
    deliberate trade-off: trying to preserve the buffer via
    ``readline.get_line_buffer()`` returned stale text on the SL2619's
    dumb-terminal session and was worse than just losing it.
    """
    sys.stdout.write(f"\n{msg}\n{PROMPT}")
    sys.stdout.flush()


def run_turn(
    model: FunctionGemmaModel,
    dispatcher: Dispatcher,
    prompt: str,
    stats: SessionStats | None = None,
) -> None:
    with _Spinner("thinking", clear_on_exit=True):
        result = model.generate(prompt)
    for call_result in dispatcher.dispatch_all(result.tool_calls):
        print(f"  {call_result}")
    n = len(result.tool_calls)
    plural = "s" if n != 1 else ""
    print(_dim(f"  ({n} tool call{plural} · {result.latency_ms:.0f} ms)"))
    print()
    if stats is not None:
        stats.turns += 1
        stats.tool_calls += n
        stats.total_latency_ms += result.latency_ms
        stats.last_latency_ms = result.latency_ms


def _handle_slash(cmd: str, stats: SessionStats) -> str:
    """Return "exit" to break the loop, "ok" to re-prompt, "unknown" otherwise."""
    head = cmd.split(maxsplit=1)[0].lower()
    if head in ("/exit", "/bye", "/quit"):
        return "exit"
    if head in ("/?", "/help"):
        print(_HELP_TEXT)
        print()
        return "ok"
    if head == "/clear":
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        return "ok"
    if head == "/tools":
        print(_TOOLS_TEXT)
        print()
        return "ok"
    if head == "/stats":
        print(stats.render())
        print()
        return "ok"
    return "unknown"


def main() -> None:
    ensure_performance_governor()
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
    p.add_argument(
        "--voice",
        choices=("off", "stub", "moonshine"),
        default="off",
        help="Voice input mode (default off). 'stub' = real mic + canned "
             "phrases (validates pipeline plumbing). 'moonshine' = real "
             "Moonshine ASR on the Torq NPU. Spoken utterances are queued "
             "as if typed.",
    )
    p.add_argument(
        "--mic",
        help="Sounddevice selector for the input device. Accepts a numeric "
             "index (e.g. --mic 0) or a name substring (e.g. --mic hw:0,0). "
             "Defaults to the system default input.",
    )
    p.add_argument(
        "--moonshine-dir",
        type=Path,
        help="Directory holding Moonshine VMFB artifacts. Defaults to "
             "<repo>/models/moonshine/ when --voice=moonshine.",
    )
    args = p.parse_args()

    with _Spinner(f"Loading model from {args.model.name}"):
        model = FunctionGemmaModel(str(args.model))

    wled = WLEDSerialClient(port=args.wled_port, baud=args.wled_baud) \
        if args.wled_port else None
    hardware = HardwareDevice(wled=wled, on_async_event=_print_async)
    dispatcher = Dispatcher(hardware)

    voice: VoicePipeline | None = None
    if args.voice != "off":
        voice = make_voice_pipeline(
            on_text=lambda _t: None,
            mode=args.voice,
            mic_device=args.mic,
            moonshine_dir=args.moonshine_dir,
        )
        if voice is None:
            print(_dim(
                f"voice unavailable: --voice={args.voice} requires "
                "sounddevice + silero-vad-notorch (and libportaudio.so.2 — "
                "extract library/portaudio_libs.tgz). Continuing text-only."
            ))

    try:
        if args.prompt:
            run_turn(model, dispatcher, args.prompt)
            return

        _setup_history()
        stats = SessionStats()
        turn_lock = threading.Lock()

        # Prime the prefix cache up front so turn 1 from the user is
        # sub-second. This pays the ~48 s cold prefill once, visibly,
        # before we accept input. Holding turn_lock during warmup keeps
        # any --voice utterances spoken in this window from piling up:
        # the audio thread blocks at _on_voice_text rather than queuing
        # 50 s of buffered model.generate calls.
        with turn_lock, _Spinner(
            "Warming up (one-time ~50s prefill on the 2-core A55)"
        ):
            model.generate("hello")

        def _on_voice_text(text: str) -> None:
            text = text.strip()
            if not text:
                return
            with turn_lock:
                _print_async(f"(voice) {text}")
                run_turn(model, dispatcher, text, stats=stats)

        if voice is not None:
            voice.set_callback(_on_voice_text)
            voice.start()
            print(_dim(f"voice: listening (--voice={args.voice})."))

        print(_dim("Ready. /help for commands, Ctrl-D or /exit to leave."))
        while True:
            try:
                line = input(PROMPT).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line.startswith("/"):
                outcome = _handle_slash(line, stats)
                if outcome == "exit":
                    break
                if outcome == "unknown":
                    print(_dim(f"unknown command: {line.split()[0]} (try /help)"))
                    print()
                continue
            with turn_lock:
                run_turn(model, dispatcher, line, stats=stats)
    finally:
        if voice is not None:
            voice.stop()
        hardware.cleanup()


if __name__ == "__main__":
    main()
