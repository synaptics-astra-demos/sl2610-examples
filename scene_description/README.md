# Image Capture and Scene Description App

Visual scene description demonstrating image understanding on the **Synaptics Astra SL26xx series**.
Press the shutter button to capture a photo from the camera; **LiquidAI LFM2-VL-450M** then
describes it on the Torq NPU, with the vision encoder and decoder both running on-chip in bf16.

## Hardware Setup

This example is compatible with the following hardware:
- Astra Machina SL2610 Dev Kit
- Synaptics Coralboard

Machina Dev Kit
- For setup instructions, see the [Setting up the hardware guide](https://synaptics-astra.github.io/doc/v/latest/quickstart/hw_setup.html)

Coralboard
- For setup instructions, see the [Synaptics Coralboard Site](https://developer.synaptics.com/docs/developer-kits/coralboard/sl/get-started)

A **7" Waveshare Touchscreen panel** (or other display) is also required for the camera preview UI.

## Prerequisites
Ensure your board has the following installed:

**Astra SDK "OOBE" Image** (Default):
- [SL2619 OOBE Image](https://github.com/synaptics-astra/sdk/releases)
- The image includes important software components such as `git` and `python3`

## 🔧 Installation

### Setup the base environment

Clone the repository including submodules, run setup scripts, and install base Python dependencies according to the [Top Level Readme Installation Section](../README.md#installation)

### Install example-specific dependencies

```bash
cd scene_description

pip install -r requirements.txt
```

## 🖼️ Running the Scene Description Example

Optionally set up the display environment (required for visual output):

```bash
export XDG_RUNTIME_DIR=/var/run/user/0
export WAYLAND_DISPLAY=wayland-1
```

### Configure for Portrait Mode

For portrait mode on an 800x480 display, run the shared portrait setup script (see the [Top Level Readme Device Configuration section](../README.md#device-configuration)):

```bash
./setup/portrait_setup.sh
```

This configures Weston for portrait orientation and exports the following for apps:

```bash
export ORIENTATION=portrait
export DISPLAY_WIDTH=480
export DISPLAY_HEIGHT=800
```

If you are not using the shared script, set `ORIENTATION` directly in `run_gui.py`:

```python
ORIENTATION = "portrait"
```

### Change to the Scene Description directory

```bash
cd scene_description/
```

### Download Models

Download the LiquidAI LFM2-VL-450M model files from HuggingFace by running this setup script.

```bash
python setup_demo.py
```

Model files are stored to `models/Synaptics/LiquidAI-LFM2-VL-450M/`.

### Run the app

```bash
python3 run_gui.py
```

Press the shutter button in the UI to capture an image from the connected camera; the captured
photo and prompt are sent to LFM2-VL-450M on the NPU, and the description is shown once generated.

### Run without a display

`run_cli.py` takes one photo from the camera (`--camera-device auto`, the default) and prints the description in the terminal, using the same model and prompt as the GUI. The photo is saved to `scene_capture.jpg`.

```bash
python3 run_cli.py
```

To describe a picture file instead of using the camera, for example one of the samples:

```bash
python3 run_cli.py --image ../samples/dog_bike_car.jpg
```

It takes `--camera-device`, `--prompt`, `--models` and `--no-refresh` like the GUI.

### Cameras

Both USB (UVC) cameras and MIPI CSI cameras are supported. A USB camera is used as is, since it auto-exposes on its own. A MIPI CSI camera (for example the OV5647 on the Coralboard sensor hat) has no working auto-exposure and produces black frames at its default gain, so the app sets its exposure and white balance and tunes the gain from the image brightness when the camera opens (about one second). To use the CSI camera when a USB camera is also connected, pass `--camera-device /dev/video0`.

The camera image is not rotated by default. If it appears upside down, add `--rotate 180` (to `run_gui.py` or `run_cli.py`).

### Options

- `--camera-device`: camera to use, e.g. `/dev/video0`, or `auto` (default) to pick the first camera found (a USB camera is preferred over a MIPI CSI camera when both are connected)
- `--rotate`: rotate the camera feed clockwise by `0` (default), `90`, `180` or `270` degrees. If your camera is mounted upside down, use `--rotate 180`
- `--prompt`: prompt sent to the VLM for each captured photo, defaults to `"Describe this image in two sentences."`
- `--models`: directory holding the LFM2-VL-450M assets, defaults to `../models/Synaptics/LiquidAI-LFM2-VL-450M/`
- `--no-refresh`: skip the Hugging Face freshness check (offline/airgapped runs)

Example, asking a more specific question instead of the default caption prompt:

```bash
python3 run_gui.py --prompt "What is the main object in this photo, and what color is it?"
```

This example is built on **LFM2-VL-450M**, a redistribution of a model created by
**Liquid AI, Inc.**, licensed under the **LFM Open License v1.0**.
