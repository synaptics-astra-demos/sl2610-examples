# Speech-To-Text using Moonshine

This guide describes how to convert audible speech to text and use a large language models to translate phrases to different languages, all running on Astra SL2610-series processors. 

This example uses the following models:

- [Moonshine](https://github.com/moonshine-ai/moonshine), a modern speech-to-text (automatic speech recognition) model designed specifically for efficient, real-time, and low-latency operation. 

[!WARNING]
This early-access version has the pre-compiled Moonshine models included. 
Before cloning, ensure that you have Git LFS installed - in order to handle the large files. 
In the future they will be downloaded from Huggingface. 


- [Silero VAD (Voice Activity Detection)](https://github.com/snakers4/silero-vad), a lightweight, high-performance model designed to detect the presence of human speech in audio streams.

The User Interface is based on pyQt5, a set of Python bindings for Qt5.

## Project Structure
```
speech_to_text
├── moonshine.py
├── portaudio_libs.tgz
├── README.md
└── requirements.txt
```

Additionaly, code it used from `../utils/` 

## 🔧 Hardware Setup

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

[Warning!] Due to a temporary dependency issue, the torq runtime must be installed after all other requirements are satisfied.
If the torq runtime is already installed, then uninstall it. 

```bash
pip uninstall torq_runtime
```

Now install the additional dependencies for this specific example. 

```bash
pip install -r requirements.txt
```

Now (re)install the torq runtime. 


```bash
pip install --no-deps ../wheelhouse/torq_runtime-1.5.0-cp312-cp312-manylinux_2_28_aarch64.whl
```

Extract the audio libraries

```bash
tar -xvzf portaudio_libs.tgz -C /
```

Connect a USB or PDM microphone

## Start


If you have a display, use the pyQt app version.

```bash
python moonshine.py
```

## Usage

It will ask you to select your microphone.

```bash
List of Audio input devices:
  0 HyperX SoloCast: USB Audio (hw:0,0), ALSA (2 in, 0 out)
  1 sysdefault, ALSA (128 in, 0 out)
  2 spdif, ALSA (2 in, 0 out)
> 3 default, ALSA (128 in, 0 out)
Enter input device to listen on:
```

The models will be loaded and the app will open. 

## Speak into the microphone! 

Speak phrases (in English) that are more than a few words but less than 5 seconds. 

The app will capture your speech and translate it to the selected language. 


# Citations

Useful Sensors. “Moonshine: On-Device Speech Recognition.” 2024.
https://github.com/usefulsensors/moonshine


Silero Team. “Silero VAD: Voice Activity Detection.” 2021.
https://github.com/snakers4/silero-vad
