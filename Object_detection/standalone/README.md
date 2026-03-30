# YOLOv8 On-Device Object Detection Guide

This guide describes how to run the standalone YOLOv8n object detection on the **Synaptics Astra SL26xx series** using the Torq/Iree Python runtime. 

## Setting up Astra Machina Board
For instructions on how to set up Astra Machina board, see the [Setting up the hardware](https://synaptics-astra.github.io/doc/v/latest/quickstart/hw_setup.html) guide.

## Prerequisites
Ensure your board has the following installed:

**Astra SDK "OOBE" Image**: Download and flash the SL2619 OOBE image from:
- [SL2619 OOBE Image](https://github.com/synaptics-astra/sdk/releases)
- The image includes important software components such as `git` and `python3`

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

This will also install the python runtime included as a .whl file in the wheelhouse folder.


## 🖼️ Running Object Detection Example

The script applies YOLO-specific preprocessing (letterbox resizing), quantization, inference, and complex post-processing (dequantization, NMS, bounding box scaling).

Optionally Set up display environment (Required for visual output).

```bash
export XDG_RUNTIME_DIR=/var/run/user/0
export WAYLAND_DISPLAY=wayland-1
```

### Run the object detection job on an image file

**Note:** The Python runtime is compatible with newer compiler and runtime settings. Use a model that was compiled recently, including the provided model yolov8n_od.vmfb.  

```bash
cd Object_detection/standalone/
python3 object_detection.py \
  --model yolov8n_od.vmfb \
  --image dog_bike_car.jpg \
  --labels labels.json \
  --device torq
```

### Model information
The provided model is a quantized version of Yolo v8 Nano from Ultralytics with 320 x 320 input resolution and 80 output classes. The model has been compiled with the [Torq compiler](https://synaptics-torq.github.io/torq-compiler/v/latest/) for optimal performance on the Synaptics Torq and Coral NPU. 

Model Conversion
```
iree-import-tflite yolov8n_full_integer_quant_320_od.tflite -o yolov8n_full_integer_quant_320_od.tosa
```

Model Compilation
```
torq-compile -o yolov8n_full_integer_quant_320_od.vmfb yolov8n_full_integer_quant_320_od.tosa --torq-convert-dtypes --torq-disable-slicing --torq-enable-torq-hl-tiling --torq-enable-transpose-optimization --torq-convert-io-dtype --torq-hw=SL2610
```

## Testing YOLOv8s alternative

  Also provided is a compiled model for YOLOv8 Small. This is a better performing model at the expense of approximately 2x inference time.
  
  To test, switch the model to `yolov8s_od.vmfb` and update the output quantization parameters in `object_detection.py` to the following.


```python
    out_scale = 0.0051302798092365265
    out_zp = -108

```



## Expected Output

You should see output similar to the following, confirming the model successfully detected objects:

```text
[1/4] Preprocessing...

[2/4] Inference...
Time: 0.0359s

[3/4] Processing...

[4/4] Detections:
  dog             Conf: 0.8934  Box: [133 216 177 322]
  bicycle         Conf: 0.7886  Box: [138 150 425 267]
  car             Conf: 0.6292  Box: [465  76 260  93]
```
