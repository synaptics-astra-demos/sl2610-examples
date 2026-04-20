# MobileNetV2 On-Device Classification Guide

This guide describes how to run the MobileNetV2 image classification workflow directly on the target board.

## Setting up Astra Machina Board
For instructions on how to set up Astra Machina board, see the [Setting up the hardware](https://synaptics-astra.github.io/doc/v/latest/quickstart/hw_setup.html) guide.

## Prerequisites
Ensure your board has the following installed:

**Astra SDK "OOBE" Image**: Download and flash the SL2619 OOBE image from:
- [SL2619 OOBE Image](https://github.com/synaptics-astra/sdk/releases)
- The image includes important software components such as `git`, `python3`, and `gstreamer`.

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

## Runing the Image Classification Example

Login to the board and execute the script. The script handles preprocessing, inference, and post-processing (label mapping) automatically.

Optionally Set up display environment (Required for visual output).

```bash
export XDG_RUNTIME_DIR=/var/run/user/0
export WAYLAND_DISPLAY=wayland-1
```

### Go to the directory

```bash
cd image_classification/
```
### Run the image classification on an image file

**Note:** The Python runtime is compatible with newer compiler and runtime settings. Use a model that was compiled recently, including the provided model mbv2.vmfb.  

```bash
python3 classification.py \
  --model ../models/mbv2.vmfb \
  --image ../samples/cat.jpg \
  --labels labels.json \
  --device torq
```

---

## Expected Output

You should see output similar to the following, confirming the model ran successfully on the board hardware:

```text
[1/4] Preprocessing image...

[2/4] Running model on board...
Time: 7.248ms

[3/4] Processing output...
Warning: Output shape (1, 1000) doesn't match expected (1, 10000). Metadata might be needed.

[4/4] Classification Results:
  1. goldfish, Carassius auratus : 0.593750
  2. starfish, sea star : 0.019531
  3. triceratops : 0.015625
  4. rock beauty, Holocanthus tricolor : 0.011719
  5. cricket : 0.007812

Top Prediction: goldfish, Carassius auratus

```
