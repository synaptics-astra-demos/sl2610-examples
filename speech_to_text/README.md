# Speech-To-Text using Moonshine

This guide describes how to convert audible speech to text and use a large language models to translate phrases to different languages, all running on Astra SL2610-series processors. 

This example uses the following models:

- [Moonshine](https://github.com/moonshine-ai/moonshine), a modern speech-to-text (automatic speech recognition) model designed specifically for efficient, real-time, and low-latency operation. 

- [Silero VAD (Voice Activity Detection)](https://github.com/snakers4/silero-vad), a lightweight, high-performance model designed to detect the presence of human speech in audio streams.

The User Interface is based on PyQt6, a set of Python bindings for Qt6.

## Project Structure
```
speech_to_text
├── setup_demo.py
├── live_caption.py
├── README.md
└── requirements.txt
```

Additionally, code is used from `../utils/` (including `utils/moonshine/` for the Moonshine inference and download backend). 

## 🔧 Hardware Setup

This example is compatible with the following hardware:
- Astra Machina SL2610 Dev Kit
- Synaptics Coralboard

Machina Dev Kit
- For setup instructions, see the [Setting up the hardware guide](https://synaptics-astra.github.io/doc/v/latest/quickstart/hw_setup.html)

Coralboard
- For setup instructions, see the [Synaptics Coralboard Site](https://developers.google.com/coral/products/SL2610-dev-board)

## Example Setup

Attach a USB microphone to the board. 

Connect from a PC using ADB or SSH.

Optionally connect a display and USB keyboard/mouse and open a terminal directly. 

## 🔧 Installation
 
### Setup the base environment

Clone the repository including submodules, run setup scripts, and install base Python dependencies according to the [Top Level Readme Installation Section](../README.md#installation)

### Install example-specific dependencies

```bash
cd speech_to_text

pip install -r requirements.txt
```

### Install the PortAudio system libraries for microphone input:

```bash
../setup/install_portaudio.sh
```


### Download Models

Download the Moonshine model files from HuggingFace:

[!WARNING] In mid-june 2026, Synaptics simplified the number of Moonshine model files. If you had previously downloaded the models, we recommend deleting them from `sl2610-examples/models/Synaptics/moonshine-tiny-bf16-torq` before running the next command. Future updates will incldue version control to eliminate the need going forward.

```bash
python setup_demo.py
```


## Start

```bash
python live_caption.py
```


## Speak into the microphone! 

Speak phrases (in English) that are more than a few words but less than 5 seconds. 

The app will capture your speech and show the text on the terminal.


# Citations

Useful Sensors. “Moonshine: On-Device Speech Recognition.” 2024.
https://github.com/usefulsensors/moonshine


Silero Team. “Silero VAD: Voice Activity Detection.” 2021.
https://github.com/snakers4/silero-vad
