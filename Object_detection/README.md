# YOLOv8 On-Device Object Detection Guide

This guide describes how to run the YOLOv8n object detection on the **Synaptics Astra SL26xx series** using the Torq/Iree Python runtime. 

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


## 🖼️ Running Object Detection Example

The script applies YOLO-specific preprocessing (letterbox resizing), quantization, inference, and complex post-processing (dequantization, NMS, bounding box scaling).

Optionally Set up display environment (Required for visual output).

```bash
export XDG_RUNTIME_DIR=/var/run/user/0
export WAYLAND_DISPLAY=wayland-1
```

### Change to the Object Detection directory
```bash
cd object_detection/
```

### Run the object detection on an image file


```bash
python3 object_detection.py \
  --model ../models/yolov8n_od.vmfb \
  --image ../samples/dog_bike_car.jpg \
  --labels labels.json \
  --device torq
```

### Run the object detection on USB camera input or video file

**Note:** A second Python script is provided for working with camera or video file input called `object_detection_video.py`.

This script supports both video file and USB camera input, live display for USB, and JSON results output.

#### Run with USB camera input

To check available cameras, run the command `v4l2-ctl --list-devices` 

For `--camera-device`, select a video device such as `/dev/video0`, or `auto`.

```bash
python3 object_detection_video.py \
  --model ../models/yolov8n_od.vmfb \
  --camera-device auto \
  --labels labels.json \
  --device torq
```

**Golden Command for Arducam on Portrait Display:**
This command optimizes the UI for portrait orientation (letterboxing, title, and stats) while ensuring proper hardware exposure control for Arducam modules.

```bash
python3 object_detection_video.py \
  --model ../models/yolov8n_od.vmfb \
  --camera-device /dev/video0 \
  --camera-control-device /dev/v4l-subdev2 \
  --labels labels.json \
  --device torq \
  --display \
  --exposure-auto 0
```

Optionally you can also set the following configurations:
- `--output`, Output video file
- `--json-results`, Output JSON file for detections
- `--camera-width`, USB camera width
- `--camera-height`, USB camera height
- `--camera-fps`, USB camera frame rate
- `--display`, Display annotated frames live
- `--display-sink`, GStreamer video sink for live display


#### Run with video file input (filesrc)

```bash
python3 object_detection_video.py \
  --model ../models/yolov8n_od.vmfb \
  --video <your_video>.mp4 \
  --labels labels.json \
  --device torq
```

#### Run with RTSP stream input

To stream from an RTSP source (e.g., IP camera, network stream):

```bash
python3 object_detection_video.py \
  --model ../models/yolov8n_od.vmfb \
  --rtsp-url rtsp://<camera_ip>:<port>/<stream_path> \
  --labels labels.json \
  --device torq
```

Example with a common IP camera:
```bash
python3 object_detection_video.py \
  --model ../models/yolov8n_od.vmfb \
  --rtsp-url rtsp://admin:123456@10.46.130.109:8554/stream0 \
  --labels labels.json \
  --device torq \
  --display
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
