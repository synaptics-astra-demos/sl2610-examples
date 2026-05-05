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

### Download the Fine-Tuned Model

Download the v6 GGUF (260 MB, Q5_K_M, compact tool-call format) into the top-level `models/` directory:

```bash
mkdir -p models && cd models
wget https://huggingface.co/BrinqAI/functiongemma-270m-physical-ai/resolve/main/functiongemma-physical-ai-v6-Q5_K_M.gguf
cd ..
```

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

---

## Expected Output

You should see output similar to the following, confirming the model parsed the natural-language prompt into tool calls and dispatched them to the HAT hardware:

```text
Loading model from functiongemma-physical-ai-v6-Q5_K_M.gguf done in 4.6s.
Warming up (one-time ~50s prefill on the 2-core A55) done in 48.3s.
Ready. Ctrl-D or empty line to exit.
>>> Turn the lights red and beep twice
  set_led_color: color = red
  play_buzzer: pattern = double_beep
  (2 tool calls · 612 ms)
>>>
```

## Tool Schema (11 functions, v6)

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
| `list_alarms` | - | List active alarms |
| `get_system_status` | metric? | CPU / memory / temperature / NPU |
| `respond` | message | Natural-language reply when no tool fits |

The full schema with descriptions lives in `tools.json`.

## Hardware

- **Coral Dev Board (SL2619)** with the Grinn Coral HAT — RGB status LEDs at `/sys/class/leds/{red,green,blue}:status/brightness`, piezo buzzer on `BUZZERn` (binary GPIO).
- **Optional Adafruit Mini Sparkle Motion (6314)** running WLED firmware, enumerated as `/dev/ttyACM0` over USB-CDC. Drives a 36-pixel WS2812B ring (Adafruit 2539). Pass `--wled-port /dev/ttyACM0` to enable.

## Model Information

`huggingface.co/BrinqAI/functiongemma-270m-physical-ai` —
`functiongemma-physical-ai-v6-Q5_K_M.gguf`. Base model `google/functiongemma-270m-it`,
fine-tuned on 2000 train / 200 eval examples covering all 11 tools and
multi-tool routines (200 row eval: single-tool routing **95.5%**, multi-tool
exact-match 23.9%, parse failure 0.5%). Compact output format
(`<tool_N>(args)<end>`) for ~5x faster decode on the 2-core A55 CPU.
