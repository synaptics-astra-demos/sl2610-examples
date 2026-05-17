# Speech-To-Text using Moonshine

This guide describes how to convert audible speech to text and use a large language models to translate phrases to different languages, all running on Astra SL2610-series processors. 

This example uses the following models:

- [Moonshine](https://github.com/moonshine-ai/moonshine), a modern speech-to-text (automatic speech recognition) model designed specifically for efficient, real-time, and low-latency operation. 

- [Silero VAD (Voice Activity Detection)](https://github.com/snakers4/silero-vad), a lightweight, high-performance model designed to detect the presence of human speech in audio streams.

The User Interface is based on pyQt5, a set of Python bindings for Qt5.

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

## Installation

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

#### Install general dependencies

If online
```bash
pip install -r requirements.txt
```

Now install the additional dependencies for this specific example.

```bash
cd speech_to_text
pip install -r requirements.txt
```

If offline:
```bash
pip install --no-index --find-links=../wheelhouse -r requirements.txt
```

### Download Models

Download the Moonshine model files from HuggingFace:

```bash
python setup_demo.py
```

Install the PortAudio system libraries for microphone input:

```bash
../configs/install_configs.sh portaudio
```

Connect a USB or PDM microphone

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
