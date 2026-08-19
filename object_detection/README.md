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
 
### Setup the base environment

Clone the repository including submodules, run setup scripts, and install base Python dependencies according to the [Top Level Readme Installation Section](../README.md#installation)

### Install example-specific dependencies

```bash
cd object_detection

pip install -r requirements.txt
```

## 🖼️ Running Object Detection Example

The script applies YOLO-specific preprocessing (letterbox resizing), quantization, inference, and complex post-processing (dequantization, NMS, bounding box scaling).

Optionally Set up display environment (Required for visual output).

```bash
export XDG_RUNTIME_DIR=/var/run/user/0
export WAYLAND_DISPLAY=wayland-1
```

For portrait mode on 800x480 display:
```bash
export ORIENTATION=portrait
export DISPLAY_HEIGHT=800
export DISPLAY_WIDTH=480
```

### Change to the Object Detection directory
```bash
cd object_detection/
```

### Download Models

Download the YoloV8n model files from HuggingFace by running this setup script.

```bash
python setup_demo.py
```

### Run the object detection on an image file

```bash
python3 object_detection.py --image ../samples/dog_bike_car.jpg
```


### Run the object detection on USB camera input or video file

**Note:** A second Python script is provided for working with camera or video file input called `object_detection_video.py`.

This script supports both video file and USB camera input, live display for USB, and JSON results output.

#### Run with USB camera input

To check available cameras, run the command `v4l2-ctl --list-devices` 

For `--camera-device`, select a video device such as `/dev/video0`, or `auto`.

```bash
python3 object_detection_video.py --camera-device auto
```

Optionally, you can pass the model, labels, and device. 

```bash
python3 object_detection_video.py \
  --model ../models/Synaptics/yolov8-od-nano-320-int8-torq/yolo_8n_2.0.0_npu.vmfb \
  --camera-device auto \
  --labels labels.json \
  --device torq
```


Optionally you can also set the following configurations:
- `--flip`, Flip the video input vertically (required depending on the camera orientation)
- `--model`, Path to model (default: ../models/Synaptics/yolov8-od-nano-320-int8-torq/yolo_8n_2.0.0_npu.vmfb)
- `--labels`, Path to labels (default: labels.json)
- `--device`, Device to run on (default: torq)
- `--output`, Output video file
- `--json-results`, Output JSON file for detections
- `--camera-width`, USB camera width
- `--camera-height`, USB camera height
- `--camera-fps`, USB camera frame rate
- `--display`, Display annotated frames live
- `--display-sink`, GStreamer video sink for live display


**Command for Coralboard with Sensor Hat Camera and Portrait Display:**
This command optimizes the UI for portrait orientation (letterboxing, title, and stats) while ensuring proper hardware exposure control for Arducam modules.

```bash
python3 object_detection_video.py \
  --camera-device /dev/video0 \
  --camera-control-device /dev/v4l-subdev2 \
  --display \
  --exposure-auto 0
```

#### Run with video file input (filesrc)

```bash
python3 object_detection_video.py --video <your_video>.mp4
```

#### Run with RTSP stream input

To stream from an RTSP source (e.g., IP camera, network stream):

```bash
python3 object_detection_video.py \
  --rtsp-url rtsp://<camera_ip>:<port>/<stream_path>
```

Example with a common IP camera:
```bash
python3 object_detection_video.py \
  --rtsp-url rtsp://admin:123456@10.46.130.109:8554/stream0 \
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

  This example also works with YOLOv8 Small. This is a better performing model at the expense of approximately 2x inference time.
  
  To test it out Yolov8 Small, follow these steps:
  - Manually download a compiled model file (e.g. yolov_8n_2.0.0_npu.vmfb) from [Synaptics/yolov8-od-small-320-int8-torq](https://huggingface.co/Synaptics/yolov8-od-small-320-int8-torq) to the `models` directory.
  
  - Update the output quantization parameters in `object_detection.py` to the following.

```python
    out_scale = 0.0051302798092365265
    out_zp = -108

```

  - Run the application, passing in the path to the model file.

```bash
  python3 object_detection_video.py \
  --model ../models/yolov_8n_2.0.0_npu.vmfb \
  --camera-device auto \
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
