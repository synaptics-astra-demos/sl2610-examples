# FunctionGemma Physical-AI Demo for SL2619

A fine-tuned FunctionGemma 270M model that turns natural-language commands
into tool calls dispatched to real HAT hardware on the Coral Dev Board:
status LEDs, piezo buzzer, MIPI camera, and an optional Adafruit Mini
Sparkle Motion driving a WS2812B Neopixel ring over USB serial.

The model emits compact tool calls (`<tool_N>(args)<end>`) which decode
~5x faster on the 2-core A55 CPU than canonical JSON tool calls, putting
the whole loop comfortably under one second per turn.

## Project Structure

```
functiongemma
├── README.md                   This file
├── README_pyQt.md              PyQt UI variant
├── requirements.txt            Pure-Python deps
├── setup_wayland.sh            Wayland env vars (source before running PyQt)
├── llama_cpp_python-...whl     Bundled aarch64 wheel for the SL2619
├── tools.json                  11-tool function schema
├── models/                     Place the GGUF here (see below)
└── src/
    ├── demo.py                 CLI: python3 demo.py --prompt "..."
    ├── app_pyqt.py             PyQt UI launcher
    ├── llamacpp.py             llama-cpp wrapper, builds prompt + parses output
    ├── compact_codec.py        Compact tool-call format decoder
    ├── hardware.py             Status LEDs + buzzer + camera + alarms
    ├── wled.py                 WLED JSON-over-USB-CDC serial client
    ├── dispatcher.py           Routes parsed tool calls to hardware
    ├── chat_window.py          PyQt main window
    ├── metrics_panel.py        Top-pane metrics tiles + sparklines
    ├── command_log.py          Bottom-pane scrolling tool log
    ├── metrics_provider.py     psutil-backed metrics sampler
    └── theme.py                Colors + typography
```

## Setting up the Astra Machina Board

For instructions on how to set up the Coral Dev Board, see the [Setting up
the hardware](https://synaptics-astra.github.io/doc/v/latest/quickstart/hw_setup.html)
guide.

## Prerequisites

Ensure your board has the following installed:

**Astra SDK "OOBE" Image**: Download and flash the SL2619 OOBE image from
the [Synaptics Astra SDK releases](https://github.com/synaptics-astra/sdk/releases).
The image includes important software components such as `git`, `python3`,
and `gstreamer`.

## Installation

Clone the repository:

```bash
git clone https://github.com/BrinqAI/functiongemma-physical-ai.git
cd functiongemma-physical-ai
```

Set up a Python virtual environment that inherits the system PyQt5 and
numpy already provided by the OOBE image:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

Install the bundled `llama-cpp-python` aarch64 wheel and the rest of the
dependencies:

```bash
pip install ./llama_cpp_python-0.3.16-cp312-cp312-linux_aarch64.whl
pip install -r requirements.txt
```

Download the fine-tuned GGUF model (260 MB, Q5_K_M, compact format, v6):

```bash
mkdir -p models && cd models
wget https://huggingface.co/BrinqAI/functiongemma-270m-physical-ai/resolve/main/functiongemma-physical-ai-v6-Q5_K_M.gguf
cd ..
```

(Optional) Plug an Adafruit Mini Sparkle Motion (product 6314) running
WLED firmware + a WS2812B Neopixel ring (Adafruit 2539) into one of the
USB-A ports on the board, then verify it enumerates:

```bash
ls /dev/ttyACM*
```

## Running the example

### CLI

From the `functiongemma/` directory:

```bash
cd src
python3 demo.py --prompt "Turn the lights red and beep twice"
```

With the optional Mini Sparkle Motion attached:

```bash
python3 demo.py --prompt "Play rainbow on the ring" --wled-port /dev/ttyACM0
```

### PyQt UI

See [`README_pyQt.md`](./README_pyQt.md).

## Expected output

```
[1/3] Loading model from /home/root/functiongemma-physical-ai/models/functiongemma-physical-ai-v6-Q5_K_M.gguf
[2/3] Running inference on prompt: 'Turn the lights red and beep twice'
  -> 2 tool call(s) in 612 ms: ['set_led_color', 'play_buzzer']
[3/3] Dispatching to hardware
  - set_led_color: color = red
  - play_buzzer: pattern = double_beep
```

## Tool schema (11 functions)

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

> Camera + vision tools (`capture_photo`, `describe_scene`) were dropped in v6 — out of scope for this demo. Camera-related prompts are routed to `respond` instead.

The full schema with descriptions lives in `tools.json`.

## Hardware

- **Coral Dev Board (SL2619)** with the Grinn Coral HAT — RGB status LEDs at
  `/sys/class/leds/{red,green,blue}:status/brightness`, piezo buzzer on
  `BUZZERn` (binary GPIO).
- **Optional Adafruit Mini Sparkle Motion (6314)** running WLED firmware,
  enumerated as `/dev/ttyACM0` over USB-CDC. Drives a 36-pixel WS2812B
  ring (Adafruit 2539). Pass `--wled-port /dev/ttyACM0` to enable.

## Model

`huggingface.co/BrinqAI/functiongemma-270m-physical-ai` —
`functiongemma-physical-ai-v6-Q5_K_M.gguf`. Base model `google/functiongemma-270m-it`,
fine-tuned on 2000 train / 200 eval examples covering all 11 tools and
multi-tool routines (200 row eval: single-tool routing **95.5%**, multi-tool
exact-match 23.9%, parse failure 0.5%). Compact output format
(`<tool_N>(args)<end>`) for ~5x faster decode on the 2-core A55 CPU.
