# Language Translation using Moonshine STT and Gemma3 LLM

This guide describes how to convert audible speech to text and use a large language models to translate phrases to different languages, all running on Astra SL2610-series processors. 

This example uses the following models:

- [Moonshine](https://github.com/moonshine-ai/moonshine), a modern speech-to-text (automatic speech recognition) model designed specifically for efficient, real-time, and low-latency operation. 

- [Gemma 3 270M](https://deepmind.google/models/gemma/gemma-3/), the most lightweight model in Google’s Gemma 3 family, designed specifically for extreme efficiency. 

- [Silero VAD (Voice Activity Detection)](https://github.com/snakers4/silero-vad), a lightweight, high-performance model designed to detect the presence of human speech in audio streams.

The User Interface is based on pyQt5, a set of Python bindings for Qt5.

## Project Structure
```
gemma_translate
├── app_translate.py
├── cli_translate.py
├── setup_demo.py
├── requirements.txt
├── README.md
└── fonts/
```

Additionally, code is used from `../utils/` (including `utils/moonshine/` and `utils/gemma/` subpackages). 

## 🔧 Hardware Setup

Attach a USB microphone to the board. 

Connect from a PC using ADB or SSH.

Optionally connect a display and USB keyboard/mouse and open a terminal directly. 

## Installation

Clone the repository and its torq-examples submodule using the following command:

```bash
git clone --recurse-submodules https://github.com/synaptics-astra-demos/sl2610-examples.git
```
Navigate to the Repository Directory:

```bash
cd sl2610-examples
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

### Setup Python Environment

To get started, set up your Python environment. This step ensures all required dependencies are installed and isolated within a virtual environment:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

#### Install general dependencies

If online
```bash
pip install -r requirements.txt
```

If offline
```bash
pip install --no-index --find-links=./wheelhouse -r requirements.txt
```


#### Install example-specific dependencies

[!WARNING] Please note that different examples require different versions of the Python Torq runtime. If using a shared virtual environment, always re-run installation of example-specific dependencies when switching between examples.


```bash
cd gemma_translate
```

Now install the additional dependencies for this specific example. 

```bash
pip install -r requirements.txt
```

If offline
```bash
pip install --no-index --find-links=../wheelhouse -r requirements.txt
```


### Download Models

Download the Moonshine and Gemma3 model files from HuggingFace:

[!WARNING] In mid-june 2026, Synaptics simplified the number of Moonshine model files. If you had previously downloaded the models, we recommend deleting them from `sl2610-examples/models/Synaptics/moonshine-tiny-bf16-torq` before running the next command. Future updates will incldue version control to eliminate the need going forward.

```bash
python setup_demo.py
```

**Optional:** If you plan to use the llama.cpp backend (`--use-llama-gemma`), download the GGUF model as well:

```bash
wget -P ../models https://huggingface.co/ggml-org/gemma-3-270m-it-GGUF/resolve/main/gemma-3-270m-it-Q8_0.gguf
```

Install the PortAudio system libraries for microphone input:

```bash
../configs/install_portaudio.sh
```

Connect a USB or PDM microphone

## Start

### Voice input (default)

No display — use the command line version with voice input:

```bash
python cli_translate.py
```

It will ask you to select your microphone:

```
List of Audio input devices:
  0 HyperX SoloCast: USB Audio (hw:0,0), ALSA (2 in, 0 out)
  1 sysdefault, ALSA (128 in, 0 out)
  2 spdif, ALSA (2 in, 0 out)
> 3 default, ALSA (128 in, 0 out)
Enter input device to listen on:
```

### Text input (no microphone required)

If you do not have a microphone, run in text mode:

```bash
python cli_translate.py --text
```

### Display mode

Set the following environment variables for using the display:

```bash
export XDG_RUNTIME_DIR=/var/run/user/0
export WAYLAND_DISPLAY=wayland-1
export QT_QPA_PLATFORM=wayland
export WESTON_DISABLE_GBM_MODIFIERS=true
```

For portrait mode on 800x480 display:
```bash
export ORIENTATION=portrait
export DISPLAY_HEIGHT=800
export DISPLAY_WIDTH=480
```

Start the app:

```bash
python app_translate.py
```

## Usage

### Voice mode

Press a number key at any time to switch the target language:

```
Press a listed number to change language:
  1: Spanish
  2: French
  3: German

Speak to translate. Press Ctrl+C to exit.
```

Speak phrases (in English) that are more than a few words but less than 5 seconds. The app will capture your speech and translate it to the selected language.

### Text mode

Type a phrase and press Enter to translate it:

```
Type a phrase and press Enter to translate.
Use /1-/6 to switch language, /q to quit:
  /1: Spanish
  /2: French
  /3: German

→ Good morning, how are you?
[You] Good morning, how are you?
[Translation] Buenos días, ¿cómo estás?
→ /2
[Language changed to: French]
→ Good morning, how are you?
[Translation] Bonjour, comment allez-vous ?
→ /q
```


# Citations

Useful Sensors. “Moonshine: On-Device Speech Recognition.” 2024.
https://github.com/usefulsensors/moonshine

Google DeepMind. “Gemma 3: Open Models by Google.” 2025.
https://ai.google.dev/gemma

Silero Team. “Silero VAD: Voice Activity Detection.” 2021.
https://github.com/snakers4/silero-vad
