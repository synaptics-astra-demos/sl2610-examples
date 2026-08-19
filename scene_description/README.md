# Image Capture and Scene Description App

Visual scene description demonstrating image understanding on the **Synaptics Astra SL26xx series**.

> ⚠️ **Warning: This app is not complete!**

## Hardware Setup

This example is compatible with the following hardware:
- Astra Machina SL2610 Dev Kit
- Synaptics Coralboard

Machina Dev Kit
- For setup instructions, see the [Setting up the hardware guide](https://synaptics-astra.github.io/doc/v/latest/quickstart/hw_setup.html)

Coralboard
- For setup instructions, see the [Synaptics Coralboard Site](https://developers.google.com/coral/products/SL2610-dev-board)

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

If you are not using the shared script, set `ORIENTATION` directly in `app_camera.py`:

```python
ORIENTATION = "portrait"
```

### Change to the Scene Description directory

```bash
cd scene_description/
```

### Run the app

```bash
python3 app_camera.py
```

Press the shutter button in the UI to capture an image from the connected camera and generate a scene description.
