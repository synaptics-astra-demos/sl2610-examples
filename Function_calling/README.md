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

### One-shot setup script (recommended)

`Function_calling/scripts/setup.sh` is idempotent: it creates a venv, installs the requirements (offline-friendly via `wheelhouse/`), and downloads the v7 GGUF model. Run from the repo root:

```bash
bash Function_calling/scripts/setup.sh             # online install, no voice
bash Function_calling/scripts/setup.sh --offline   # use wheelhouse/, no voice
bash Function_calling/scripts/setup.sh --voice     # online + voice deps (sounddevice, silero-vad-notorch, torq_runtime, portaudio)
bash Function_calling/scripts/setup.sh --offline --voice
```

The `--voice` flag also extracts `library/portaudio_libs.tgz` into `/` (needs sudo or root) — the OOBE image doesn't ship `libportaudio.so.2`. The `torq_runtime` wheel installs separately with `--no-deps` per the standalone Moonshine example.

> **Note**: the offline `wheelhouse/` does not yet include aarch64 wheels for the voice-only deps (`sounddevice`, `silero-vad-notorch`, `onnxruntime`, `tokenizers`, `ml_dtypes`, `soundfile`). Until those are populated, `--voice` requires the **online** path (omit `--offline`).

### Manual setup (if you prefer)

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt                              # online
pip install --no-index --find-links=./wheelhouse -r requirements.txt   # offline

mkdir -p models && cd models
wget https://huggingface.co/BrinqAI/functiongemma-270m-physical-ai/resolve/main/functiongemma-physical-ai-v7-Q5_K_M.gguf
cd ..
```

### (Optional) Moonshine ASR artifacts for voice input

The Moonshine "tiny" VMFBs (`encoder.onnx`, `decoder.vmfb`, `decoder_with_past.vmfb`, `decoder_token_embeddings.npy`) ship in `models/moonshine/` alongside the standalone Moonshine example. The HuggingFace `tokenizer.json` is auto-fetched on first run by `MoonshineASR`; for fully offline use, drop a copy alongside the VMFBs:

```bash
wget -O models/moonshine/tokenizer.json \
  https://huggingface.co/UsefulSensors/moonshine-tiny/resolve/main/tokenizer.json
```

Override the artifact directory with `--moonshine-dir /path/to/dir`. The default is `<repo>/models/moonshine/`.

### (Optional) Wire up the Neopixel Ring

Plug an Adafruit Mini Sparkle Motion (product 6314) running WLED firmware + a WS2812B Neopixel ring (Adafruit 2539) into one of the USB-A ports on the board, then verify it enumerates:

```bash
ls /dev/ttyACM*
```

## Running the Function-Calling Example

The first model invocation prefills the tool-declaration prompt and takes ~45-50 s on the 2-core A55 CPU. The interactive REPL pays this cost once at start-up so every subsequent turn is sub-second; one-shot mode pays it on every invocation. Default to the REPL.

For interactive runs of the **PyQt UI** from a terminal, export the Wayland env vars first so Qt can find the OOBE image's weston compositor (the systemd service in [Auto-start on boot](#auto-start-on-boot-systemd) handles this for you):

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

The PyQt UI grows a **Mic** button when voice is enabled, and the REPL accepts spoken utterances mixed with typed input. Both entrypoints take the same flags:

| Flag | What it does |
|---|---|
| `--voice off` (default) | No voice. Mic button hidden; spoken input ignored. |
| `--voice stub` | Real mic + VAD; ASR returns rotating canned phrases. Use to validate the mic → VAD → tool-dispatch path without a real ASR model. |
| `--voice moonshine` | Real mic + VAD + Moonshine ASR on the Torq NPU. Reads VMFB artifacts from `--moonshine-dir` (default `<repo>/models/moonshine/`). |
| `--mic <index\|substring>` | Sounddevice device selector. The HAT PDM mic enumerates as device 0 (`klamath-asoc, hw:0,0`); reach it with `--mic 0` or `--mic hw:0,0`. Defaults to the system default; a plugged-in USB mic typically takes that slot. |

```bash
# PyQt UI with stub voice (validates plumbing without a real ASR model)
python3 app_pyqt.py --voice stub

# REPL with stub voice (talk OR type — both feed run_turn)
python3 demo.py --voice stub

# PyQt UI with real Moonshine ASR (needs Moonshine artifacts staged)
python3 app_pyqt.py --voice moonshine

# Pin to the HAT PDM mic explicitly
python3 app_pyqt.py --voice moonshine --mic 0
```

Voice prerequisites are installed in one shot by `bash Function_calling/scripts/setup.sh --voice` (see [Installation](#-installation)): the Python deps (`sounddevice`, `silero-vad-notorch`, plus the Moonshine stack `onnxruntime` / `tokenizers` / `ml_dtypes` / `soundfile`) from `requirements.txt` + `speech_to_text/requirements.txt`, the `torq_runtime` wheel with `--no-deps`, and `libportaudio.so.2` extracted from `library/portaudio_libs.tgz`. The VAD loads via `load_silero_vad(onnx=True)` and `silero-vad-notorch` is a torch-free fork, so no PyTorch is pulled in.

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

`SIGKILL`, kernel OOM kill, segfault, and power loss bypass all in-process cleanup. The systemd unit installed by `scripts/install-service.sh` (see below) carries an `ExecStopPost=` that drives `BUZZERn` back to silent on every service exit, including SIGKILL, so the buzzer can never latch ON across an unclean restart.

## Auto-start on boot (systemd)

After `scripts/setup.sh` has populated the venv and downloaded the model, install the systemd service to launch the PyQt demo full-screen at boot:

```bash
sudo bash Function_calling/scripts/install-service.sh
```

By default the unit runs `app_pyqt.py --fullscreen`. To pass extra flags (e.g. WLED port, voice mode) at install time, append them — they are baked into the generated `ExecStart=` line:

```bash
sudo bash Function_calling/scripts/install-service.sh --wled-port /dev/ttyACM0
sudo bash Function_calling/scripts/install-service.sh --voice stub --mic 0
```

Other flags:

- `--no-enable` — install the unit but don't enable it on boot
- `--no-start` — enable on boot but don't start it now

The unit:

- waits for `weston.service` (the OOBE image's Wayland compositor)
- exports the Wayland env vars used by weston (`XDG_RUNTIME_DIR`, `WAYLAND_DISPLAY`, `QT_QPA_PLATFORM`, `WESTON_DISABLE_GBM_MODIFIERS`)
- runs as `root` from `Function_calling/` with the venv's `python3`
- restarts on failure (5 s back-off, 120 s start timeout — the model takes ~50 s to cold-prefill)
- on stop drives `gpioset $(gpiofind BUZZERn)=1` so the buzzer is silenced even after SIGKILL

Day-to-day:

```bash
systemctl status functiongemma-demo
journalctl -u functiongemma-demo -f       # follow logs
systemctl restart functiongemma-demo      # after editing source

sudo bash Function_calling/scripts/uninstall-service.sh   # remove
```

## Model Information

`huggingface.co/BrinqAI/functiongemma-270m-physical-ai` —
`functiongemma-physical-ai-v7-Q5_K_M.gguf`. Base model `google/functiongemma-270m-it`,
fine-tuned on 2000 train / 250 eval examples covering all 10 tools and
multi-tool routines (250 row eval on the Q5_K_M GGUF: overall **86.8%**,
single-tool **92.8%**, multi-tool exact-match **75.0%**, parse failure **0.0%**).
Compact output format (`<tool_N>(args)<end>`) for ~5x faster decode on the
2-core A55 CPU.
