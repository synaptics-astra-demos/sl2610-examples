# FunctionGemma On-Device Function-Calling Guide

This guide describes how to run the FunctionGemma 270M physical-AI demo
directly on the target board. The model turns natural-language commands
into tool calls dispatched to real HAT hardware on the Coral Dev Board:
status LEDs, piezo buzzer, MIPI camera, and an optional Adafruit Mini
Sparkle Motion driving a WS2812B Neopixel ring over USB serial.

The model emits compact tool calls (`<tool_N>(args)<end>`) which decode
~5x faster on the 2-core A55 CPU than canonical JSON tool calls, putting
each turn comfortably under one second after the first.

## Setting up Astra Machina Board
For instructions on how to set up Astra Machina board, see the [Setting up the hardware](https://synaptics-astra.github.io/doc/v/latest/quickstart/hw_setup.html) guide.

## Prerequisites
Ensure your board has the following installed:

**Astra SDK "OOBE" Image**: Download and flash the SL2619 OOBE image from:
- [SL2619 OOBE Image](https://github.com/synaptics-astra/sdk/releases)
- The image includes important software components such as `git`, `python3`, `gstreamer`, and `gpiod`.

## 🔧 Installation

### Clone the Repository

Clone the repository using the following command:

```bash
git clone https://github.com/synaptics-astra-demos/sl2610-examples.git
```
Navigate to the Repository Directory:

```bash
cd sl2610-examples
```

### Setup Python Environment

To get started, set up your Python environment. This step ensures all required dependencies are installed and isolated within a virtual environment:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

Install dependencies

If online
```bash
pip install -r requirements.txt
```

If offline
```bash
pip install --no-index --find-links=./wheelhouse -r requirements.txt
```

If you plan to use voice input (see "Voice input (optional)" below), also install the PortAudio system library *before* `pip install` — `sounddevice` installs cleanly without it but fails at runtime with `OSError: PortAudio library not found`. The OOBE image ships it; on a stripped-down rootfs:

```bash
sudo apt install libportaudio2
```

> **Note**: the offline `wheelhouse/` does not yet include aarch64 wheels for the voice deps (`sounddevice`, `silero-vad`, `onnxruntime`, `tokenizers`, `ml_dtypes`). Until those are added, voice input requires the **online** install path — or a one-time `pip download --platform manylinux2014_aarch64 ...` against an aarch64 host to populate the wheelhouse.

### Download the Fine-Tuned Model

Download the v7 GGUF (248 MB, Q5_K_M, compact tool-call format) into the top-level `models/` directory:

```bash
mkdir -p models && cd models
wget https://huggingface.co/BrinqAI/functiongemma-270m-physical-ai/resolve/main/functiongemma-physical-ai-v7-Q5_K_M.gguf
cd ..
```

### (Optional) Download Moonshine ASR for voice input

> **TBD: VMFB tarball not yet uploaded.** The HF repo path below
> (`BrinqAI/moonshine-tiny-torq`) is a placeholder. Until those VMFBs
> are published you cannot run `CORAL_VOICE=moonshine` end-to-end —
> use `CORAL_VOICE=stub` instead, which exercises mic + VAD + dispatch
> without a real ASR model. This warning will be removed once the
> artifacts are live.

For real voice input (vs. the `stub` rotation) you'll need the Moonshine "tiny" artifacts — encoder, decoder, decoder-with-past as `.vmfb` files (Torq NPU compile target) plus the HuggingFace `tokenizer.json`. All four files must live in the same directory:

```bash
mkdir -p models/moonshine-tiny && cd models/moonshine-tiny
# encoder + decoder + decoder_with_past VMFBs (compile target: torq) — pending
wget https://huggingface.co/BrinqAI/moonshine-tiny-torq/resolve/main/encoder.vmfb
wget https://huggingface.co/BrinqAI/moonshine-tiny-torq/resolve/main/decoder.vmfb
wget https://huggingface.co/BrinqAI/moonshine-tiny-torq/resolve/main/decoder_with_past.vmfb
# tokenizer (live)
wget https://huggingface.co/UsefulSensors/moonshine-tiny/resolve/main/tokenizer.json
cd ../..
```

Override the directory with `CORAL_MOONSHINE_DIR=/path/to/dir`. The default resolves to `<repo>/models/moonshine-tiny/` (sibling of the GGUF directory).

### (Optional) Wire up the Neopixel Ring

Plug an Adafruit Mini Sparkle Motion (product 6314) running WLED firmware + a WS2812B Neopixel ring (Adafruit 2539) into one of the USB-A ports on the board, then verify it enumerates:

```bash
ls /dev/ttyACM*
```

## Running the Function-Calling Example

The first model invocation prefills the tool-declaration prompt and takes ~45-50 s on the 2-core A55 CPU. The interactive REPL pays this cost once at start-up so every subsequent turn is sub-second; one-shot mode pays it on every invocation. Default to the REPL.

Optionally set up the display environment (required for the PyQt UI variant):

```bash
export XDG_RUNTIME_DIR=/var/run/user/0
export WAYLAND_DISPLAY=wayland-1
export QT_QPA_PLATFORM=wayland
export WESTON_DISABLE_GBM_MODIFIERS=true
```

### Change to the Function Calling directory

```bash
cd Function_calling/
```

### Run the interactive CLI demo

```bash
python3 demo.py
```

The model warms up once (~50 s, shown by a spinner), then accepts prompts at the `>>>` prompt. Press Ctrl-D or enter an empty line to exit.

With the optional Mini Sparkle Motion ring attached:

```bash
python3 demo.py --wled-port /dev/ttyACM0
```

### Run a one-shot prompt

For scripting or smoke tests you can pass a single prompt and exit (note: this still pays the ~50 s prefill on every invocation):

```bash
python3 demo.py --prompt "Turn the lights red and beep twice"
```

### Run the PyQt UI

A PyQt5 UI variant is provided for the 7" Wayland panel: top half shows live system metrics with sparklines, bottom half shows a scrolling log of natural-language prompts and the tool calls they produced.

```bash
python3 app_pyqt.py
```

With the optional Mini Sparkle Motion ring attached:

```bash
python3 app_pyqt.py --wled-port /dev/ttyACM0
```

For full-screen on the 7" panel:

```bash
python3 app_pyqt.py --fullscreen
```

Press `Ctrl+P` for a screenshot to `/tmp/`. Press `Esc` to quit.

### Voice input (optional)

The PyQt UI grows a **Mic** button when voice is enabled, and the REPL gains a `--voice` flag that mixes spoken utterances with typed input.

Set `CORAL_VOICE` to pick the ASR backend:

| Value | What it does |
|---|---|
| `off` (default) | No voice. Mic button hidden; `--voice` warns and continues text-only. |
| `stub` | Real mic + VAD; ASR returns rotating canned phrases. Use to validate the mic → VAD → tool-dispatch path without a real ASR model. |
| `moonshine` | Real mic + VAD + Moonshine ASR on the Torq NPU (see Phase B). Requires the Moonshine VMFB artifacts in `models/moonshine-tiny/` (or override with `CORAL_MOONSHINE_DIR`). |

Optional: `CORAL_MIC=<index|substring>` selects a specific input device (sounddevice device selector). Defaults to the system default — usually a USB mic. The HAT PDM mic at `alsasrc hw:0,0` is reachable as `CORAL_MIC=hw:0,0` once the HAT is wired.

```bash
# PyQt UI with stub voice (validates plumbing without a real ASR model)
CORAL_VOICE=stub python3 app_pyqt.py

# REPL with stub voice (talk OR type — both feed run_turn)
CORAL_VOICE=stub python3 demo.py --voice

# PyQt UI with real Moonshine ASR (needs Moonshine artifacts staged)
CORAL_VOICE=moonshine python3 app_pyqt.py
```

System-level prerequisite for the mic stream: `libportaudio2` must be present (see install section above). The Python deps (`sounddevice`, `silero-vad`, `onnxruntime`, `tokenizers`) install from `requirements.txt`. `silero-vad` is loaded via `load_silero_vad(onnx=True)` so the Torch dependency path is bypassed.

**REPL caveat**: in `demo.py --voice` you can talk *or* type. Voice transcription completes asynchronously, so if a spoken utterance arrives while you are mid-keystroke at the `>>>` prompt, the readline buffer is dropped and you have to retype. Acceptable for headless smoke tests; the PyQt UI does not have this problem.

---

## Expected Output

You should see output similar to the following, confirming the model parsed the natural-language prompt into tool calls and dispatched them to the HAT hardware:

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

## Tool Schema (10 functions, v7)

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

The full schema with descriptions lives in `tools.json`. v7 dropped `list_alarms` —
alarm-query prompts ("what alarms do I have?") route via `respond()` instead.

## Hardware

- **Coral Dev Board (SL2619)** with the Grinn Coral HAT — RGB status LEDs at `/sys/class/leds/{red,green,blue}:status/brightness`, piezo buzzer on `BUZZERn` (binary GPIO).
- **Optional Adafruit Mini Sparkle Motion (6314)** running WLED firmware, enumerated as `/dev/ttyACM0` over USB-CDC. Drives a 36-pixel WS2812B ring (Adafruit 2539). Pass `--wled-port /dev/ttyACM0` to enable.

### Buzzer wiring note

Despite the schematic name suggesting active-low, `BUZZERn` on the Grinn Coral HAT is electrically wired such that the buzzer **silences on the line being driven HIGH and beeps when LOW**. The kernel device tree marks the line `active-high` (so `gpioset gpiochip0 6=1` drives physical HIGH = silent; `=0` = beep). The chip driver also retains the last-driven value across `gpioset --mode=exit`, so once a value is written the line holds it.

`hardware.py` writes the inverted polarity (`0` to beep, `1` to silence). If you port this code to a board with the polarity wired the other way, flip the `_BUZZER_OFF` / `_BUZZER_ON` constants at the top of `hardware.py`. Verify with `gpioinfo gpiochip0` (look for the line named `"BUZZERn"`).

### Crash-safe cleanup

The demo guarantees the buzzer + status LEDs return to a silent/off state on every exit path that the Python interpreter can observe:

- normal exit (`/exit`, EOF, end of `--prompt`)
- uncaught exceptions
- `KeyboardInterrupt` (Ctrl-C / SIGINT) mid-pattern
- `SIGTERM` (e.g. `kill <pid>`, init shutdown)
- `SIGHUP` (terminal close, parent process death)
- a fresh demo loading `hardware.py` after a crashed prior process

Layered as `try/finally` inside `play_buzzer` + `blink_lights`, a `HardwareDevice.cleanup()` called from a `finally` block in `main()`, signal handlers for `SIGTERM`/`SIGHUP`, and an `atexit` net.

`SIGKILL`, kernel OOM kill, segfault, and power loss bypass all in-process cleanup. If the demo is going to run as a long-lived service, drop a systemd unit with:

```ini
[Service]
ExecStopPost=/usr/bin/gpioset gpiochip0 6=1
```

(or run a oneshot `coral-buzzer-safe.service` at boot that just runs that command.)

## Model Information

`huggingface.co/BrinqAI/functiongemma-270m-physical-ai` —
`functiongemma-physical-ai-v7-Q5_K_M.gguf`. Base model `google/functiongemma-270m-it`,
fine-tuned on 2000 train / 250 eval examples covering all 10 tools and
multi-tool routines (250 row eval on the Q5_K_M GGUF: overall **86.8%**,
single-tool **92.8%**, multi-tool exact-match **75.0%**, parse failure **0.0%**).
Compact output format (`<tool_N>(args)<end>`) for ~5x faster decode on the
2-core A55 CPU.
