# MobileNetV2 On-Device Classification Guide

This guide describes how to run the MobileNetV2 image classification workflow directly on the target board.

## Hardware Setup

This example is compatible with the following hardware:
- Astra Machina SL2610 Dev Kit
- Synaptics Coralboard

Machina Dev Kit
- For setup instructions, see the [Setting up the hardware guide](https://synaptics-astra.github.io/doc/v/latest/quickstart/hw_setup.html)

Coralboard
- For setup instructions, see the [Synaptics Coralboard Site](https://developers.google.com/coral/products/SL2610-dev-board)

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
pip install -r requirements.txt
```

## Runing the Image Classification Example

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
