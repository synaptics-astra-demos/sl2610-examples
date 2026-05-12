# FunctionGemma On-Device Function-Calling Guide

A self-contained, on-device voice + text physical-AI demo. A fine-tuned
FunctionGemma 270M LLM turns natural-language commands into tool calls
that drive real HAT hardware on the Coral Dev Board: status LEDs, a
piezo buzzer, the MIPI camera, and an optional Adafruit Mini Sparkle
Motion driving a WS2812B Neopixel ring over USB serial.

The model emits compact tool calls (`<tool_N>(args)<end>`) which decode
~5x faster on the 2-core A55 CPU than canonical JSON tool calls,
putting each turn comfortably under one second after the first.

Everything the demo needs — wheels, the portaudio library, the GGUF
model, the Moonshine VMFBs, the vendored ASR runtime — is fetched into
`Function_calling/` itself. No sibling directories required.

## Quick start

The board has the OOBE image already (`git`, `python3`, `gstreamer`,
`gpiod`, `weston`). On the board:

```bash
# 1. Clone
git clone https://github.com/synaptics-astra-demos/sl2610-examples.git
cd sl2610-examples/Function_calling

# 2. One-liner setup: venv + Python deps + GGUF model (+ optional voice)
bash scripts/setup.sh                  # core demo only
# or:
bash scripts/setup.sh --voice          # also: torq_runtime, portaudio, Moonshine

# 3. One-liner install-as-service (auto-starts the PyQt UI on boot)
sudo bash scripts/install-service.sh
```

That's it. The PyQt UI comes up on the 7" Wayland panel, the model
prefills in the background, and the demo accepts typed prompts (and
spoken ones if `--voice` was used).

The setup script is idempotent — re-run it anytime. To uninstall the
service: `sudo bash scripts/uninstall-service.sh`.

## Layout

```
Function_calling/
├── app_pyqt.py            # PyQt5 entrypoint (the UI demo)
├── demo.py                # CLI / REPL entrypoint
├── chat_window.py         # main UI window
├── command_log.py         # scrolling tool-call log widget
├── compact_codec.py       # <tool_N>(args)<end> ↔ ToolCall
├── cpu_governor.py        # forces "performance" governor on the A55
├── dispatcher.py          # ToolCall → HardwareDevice method
├── hardware.py            # status LEDs, buzzer, alarms, camera
├── llamacpp.py            # llama-cpp-python wrapper for the GGUF
├── metrics_panel.py       # top-pane sparklines
├── metrics_provider.py    # psutil + sysfs samplers
├── theme.py               # Qt palette / typography
├── tools.json             # 10-tool schema
├── wled.py                # Mini Sparkle Motion serial client
├── voice/
│   ├── asr.py             # StubASR + MoonshineASR
│   ├── mic.py             # sounddevice wrapper
│   ├── pipeline.py        # mic → VAD → ASR worker thread
│   ├── vad.py             # silero-vad-notorch wrapper
│   └── _runtime/          # vendored Moonshine + IREE/ORT runners
├── library/
│   └── portaudio_libs.tgz # libportaudio.so.2 (extracted into / by setup.sh --voice)
├── wheels/                # pre-built aarch64 wheels (llama-cpp-python, torq_runtime, …)
├── scripts/
│   ├── setup.sh           # first-time install
│   ├── install-service.sh # systemd autostart installer
│   └── uninstall-service.sh
├── tests/                 # pytest (unit tests for voice/)
├── requirements.txt
└── models/                # populated by setup.sh
    ├── functiongemma-physical-ai-v7-Q5_K_M.gguf      # core demo
    └── moonshine/                                    # only with --voice
        ├── encoder.onnx
        ├── decoder.vmfb
        ├── decoder_with_past.vmfb
        └── decoder_token_embeddings.npy
```

## Running the demo manually

The first model invocation prefills the tool-declaration prompt and
takes ~45-50 s on the 2-core A55 CPU. The interactive REPL pays this
cost once at start-up so every subsequent turn is sub-second; one-shot
mode pays it on every invocation. Default to the REPL.

For interactive runs of the **PyQt UI** from a terminal, export the
Wayland env vars first so Qt can find the OOBE image's weston
compositor (the systemd service handles this for you on autostart):

```bash
export XDG_RUNTIME_DIR=/var/run/user/0
export WAYLAND_DISPLAY=wayland-1
export QT_QPA_PLATFORM=wayland
export WESTON_DISABLE_GBM_MODIFIERS=true
```

Activate the venv, then run either entrypoint:

```bash
source .venv/bin/activate

# Interactive REPL
python3 demo.py

# One-shot prompt (still pays the ~50 s prefill on every invocation)
python3 demo.py --prompt "Turn the lights red and beep twice"

# PyQt UI
python3 app_pyqt.py

# PyQt UI full-screen on the 7" panel
python3 app_pyqt.py --fullscreen
```

`Ctrl+P` snapshots the PyQt window to `/tmp/`. `Esc` quits.

### Optional WLED Neopixel ring

Plug an Adafruit Mini Sparkle Motion (product 6314) running WLED
firmware + a WS2812B Neopixel ring (Adafruit 2539) into a USB-A port
and verify it enumerates as `/dev/ttyACM0`:

```bash
ls /dev/ttyACM*
```

Then pass `--wled-port`:

```bash
python3 demo.py --wled-port /dev/ttyACM0
python3 app_pyqt.py --wled-port /dev/ttyACM0 --fullscreen
```

## Voice input

The PyQt UI grows a **Mic** button when voice is enabled, and the REPL
accepts spoken utterances mixed with typed input. Both entrypoints
take the same flags:

| Flag | What it does |
|---|---|
| `--voice off` (default) | No voice. Mic button hidden; spoken input ignored. |
| `--voice stub` | Real mic + VAD; ASR returns rotating canned phrases. Use to validate the mic → VAD → tool-dispatch path without a real ASR model. |
| `--voice moonshine` | Real mic + VAD + Moonshine ASR on the Torq NPU. Reads VMFB artifacts from `--moonshine-dir` (default `Function_calling/models/moonshine/`, populated by `scripts/setup.sh --voice`). |
| `--mic <index\|substring>` | Sounddevice device selector. The HAT PDM mic enumerates as device 0 (`klamath-asoc, hw:0,0`); reach it with `--mic 0` or `--mic hw:0,0`. Defaults to the system default; a plugged-in USB mic typically takes that slot. |

```bash
# PyQt UI with stub voice (validates plumbing without a real ASR model)
python3 app_pyqt.py --voice stub

# REPL with stub voice
python3 demo.py --voice stub

# PyQt UI with real Moonshine ASR
python3 app_pyqt.py --voice moonshine

# Pin to the HAT PDM mic explicitly
python3 app_pyqt.py --voice moonshine --mic 0
```

`scripts/setup.sh --voice` installs the entire voice toolchain in one
shot: `sounddevice`, `silero-vad-notorch`, `onnxruntime`, `tokenizers`,
`huggingface_hub`, the `torq_runtime` wheel (with `--no-deps` per
upstream guidance), the four Moonshine VMFB artifacts (downloaded from
HuggingFace), and `libportaudio.so.2` (extracted from
`library/portaudio_libs.tgz`, which the OOBE image doesn't ship).

The Moonshine `tokenizer.json` is auto-fetched from
`UsefulSensors/moonshine-tiny` on first run; for fully offline use,
drop a copy alongside the VMFBs:

```bash
wget -O models/moonshine/tokenizer.json \
  https://huggingface.co/UsefulSensors/moonshine-tiny/resolve/main/tokenizer.json
```

**REPL caveat**: in `demo.py --voice` you can talk *or* type. Voice
transcription completes asynchronously, so if a spoken utterance
arrives while you are mid-keystroke at the `>>>` prompt, the readline
buffer is dropped and you have to retype. Acceptable for headless
smoke tests; the PyQt UI does not have this problem.

## Auto-start on boot (systemd)

After `scripts/setup.sh` has populated the venv and downloaded the
model, install the systemd service to launch the PyQt demo full-screen
at boot:

```bash
sudo bash scripts/install-service.sh
```

By default the unit runs `app_pyqt.py --fullscreen`. To pass extra
flags (e.g. WLED port, voice mode) at install time, append them — they
are baked into the generated `ExecStart=` line:

```bash
sudo bash scripts/install-service.sh --wled-port /dev/ttyACM0
sudo bash scripts/install-service.sh --voice stub --mic 0
```

Other flags:

- `--no-enable` — install the unit but don't enable it on boot
- `--no-start` — enable on boot but don't start it now

The unit:

- waits for `weston.service` (the OOBE image's Wayland compositor)
- exports the Wayland env vars used by weston
  (`XDG_RUNTIME_DIR`, `WAYLAND_DISPLAY`, `QT_QPA_PLATFORM`,
  `WESTON_DISABLE_GBM_MODIFIERS`)
- runs as `root` from `Function_calling/` with the venv's `python3`
- restarts on failure (5 s back-off, 120 s start timeout — the model
  takes ~50 s to cold-prefill)
- on stop drives `gpioset $(gpiofind BUZZERn)=1` so the buzzer is
  silenced even after SIGKILL

Day-to-day:

```bash
systemctl status functiongemma-demo
journalctl -u functiongemma-demo -f       # follow logs
systemctl restart functiongemma-demo      # after editing source

sudo bash scripts/uninstall-service.sh    # remove
```

## Expected output

```text
Loading model from functiongemma-physical-ai-v7-Q5_K_M.gguf done in 4.6s.
Warming up (one-time ~50s prefill on the 2-core A55) done in 48.3s.
Ready. /help for commands, Ctrl-D or /exit to leave.
>>> Turn the lights red and beep twice
  set_led_color: color=red
  play_buzzer: pattern=double_beep
  (2 tool calls · 612 ms)
>>>
```

## Tool schema (10 functions, v7)

| Tool | Args | Effect |
|---|---|---|
| `turn_on_lights` | - | All status LEDs + ring to default white |
| `turn_off_lights` | - | All lights off |
| `set_led_color` | color, target?, brightness? | RGB color set |
| `blink_lights` | count?, color?, speed? | Discrete blink |
| `set_neopixel_pattern` | pattern, color?, speed? | Animated ring effect (rainbow, chase, fade, pulse, sparkle, solid) |
| `play_buzzer` | pattern | Named pattern on the binary-GPIO buzzer (beep, double_beep, chirp, siren, alarm, success, error) |
| `set_alarm` | duration\|time, label? | Schedule alarm (buzzer + flashing) |
| `cancel_alarm` | label? | Cancel one or all alarms |
| `get_system_status` | metric? | CPU / memory / temperature / NPU |
| `respond` | message | Natural-language reply when no tool fits |

The full schema with descriptions lives in `tools.json`. v7 dropped
`list_alarms` — alarm-query prompts ("what alarms do I have?") route
via `respond()` instead.

## Hardware

- **Coral Dev Board (SL2619)** with the Grinn Coral HAT — RGB status
  LEDs at `/sys/class/leds/{red,green,blue}:status/brightness`, piezo
  buzzer on `BUZZERn` (binary GPIO).
- **Optional Adafruit Mini Sparkle Motion (6314)** running WLED
  firmware, enumerated as `/dev/ttyACM0` over USB-CDC. Drives a
  48-pixel WS2812B / SKC6812RV ring (Adafruit 2539). Pass
  `--wled-port /dev/ttyACM0` to enable.

### Buzzer wiring note

Despite the schematic name suggesting active-low, `BUZZERn` on the
Grinn Coral HAT is electrically wired such that the buzzer **silences
on the line being driven HIGH and beeps when LOW**. The kernel device
tree marks the line `active-high` (so `gpioset gpiochip0 6=1` drives
physical HIGH = silent; `=0` = beep). The chip driver also retains the
last-driven value across `gpioset --mode=exit`, so once a value is
written the line holds it.

`hardware.py` writes the inverted polarity (`0` to beep, `1` to
silence). If you port this code to a board with the polarity wired the
other way, flip the `_BUZZER_OFF` / `_BUZZER_ON` constants at the top
of `hardware.py`. Verify with `gpioinfo gpiochip0` (look for the line
named `"BUZZERn"`).

### Crash-safe cleanup

The demo guarantees the buzzer + status LEDs return to a silent/off
state on every exit path that the Python interpreter can observe:

- normal exit (`/exit`, EOF, end of `--prompt`)
- uncaught exceptions
- `KeyboardInterrupt` (Ctrl-C / SIGINT) mid-pattern
- `SIGTERM` (e.g. `kill <pid>`, init shutdown)
- `SIGHUP` (terminal close, parent process death)
- a fresh demo loading `hardware.py` after a crashed prior process

Layered as `try/finally` inside `play_buzzer` + `blink_lights`, a
`HardwareDevice.cleanup()` called from a `finally` block in `main()`,
signal handlers for `SIGTERM`/`SIGHUP`, and an `atexit` net.

`SIGKILL`, kernel OOM kill, segfault, and power loss bypass all
in-process cleanup. The systemd unit installed by
`scripts/install-service.sh` carries an `ExecStopPost=` that drives
`BUZZERn` back to silent on every service exit, including SIGKILL, so
the buzzer can never latch ON across an unclean restart.

## Model information

`huggingface.co/BrinqAI/functiongemma-270m-physical-ai` —
`functiongemma-physical-ai-v7-Q5_K_M.gguf`. Base model
`google/functiongemma-270m-it`, fine-tuned on 2000 train / 250 eval
examples covering all 10 tools and multi-tool routines (250 row eval
on the Q5_K_M GGUF: overall **86.8%**, single-tool **92.8%**,
multi-tool exact-match **75.0%**, parse failure **0.0%**). Compact
output format (`<tool_N>(args)<end>`) for ~5x faster decode on the
2-core A55 CPU.

The same HF repo also hosts the Moonshine VMFB artifacts under
`moonshine/` for the optional voice path.
