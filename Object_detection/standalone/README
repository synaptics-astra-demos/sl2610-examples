# YOLOv8 On-Device Object Detection Guide

This guide describes how to run the standalone YOLOv8n object detection workflow directly on the target board.

## Prerequisites

### 1. Board Requirements
Ensure your board has the following installed:

**Use OOBE Image**: Download and flash the SL2619 OOBE image from:
- [SL2619 OOBE Image](http://iotmmswfileserver.synaptics.com:8000/sandal/LinuxSDK-Serpens/202601/20260120/202601201405/sl2619_oobe_scarthgap/Rel_Build/)
- **Python 3**
- **Python Libraries**: `numpy`, `pillow` (PIL)
- **Runtime**: `iree-run-module` binary (must be in system `$PATH`)

### 2. Required Files
You need to have the following files ready on your host machine:


| File                      | Description                                                    |
|---------------------------|----------------------------------------------------------------|
| `object_detection.py`     | The standalone Python YOLO inference script.                   |
| `requirements_board.txt`  | List of Python dependencies.                                   |
| `yolov8_od.vmfb`          | The compiled model binary (VMFB).                              |
| `labels.json`             | JSON file containing class labels (converted from YAML).       |
| `image.jpg`               | The input image to detect objects in (e.g., dog_bike_car.jpg). |


---

## Step 1: Transfer Files to Board

Use `scp` to copy all artifacts to a directory on the board (e.g., `/home/root/standalone`).

> **Note**: Replace `<BOARD_IP>` with your actual board IP address.

```bash
# 1. Create a directory on the board
ssh root@<BOARD_IP> "mkdir -p /home/root/standalone"

# 2. Copy the inference script and requirements
scp object_detection.py root@<BOARD_IP>:/home/root/standalone/
scp requirements_board.txt root@<BOARD_IP>:/home/root/standalone/

# 3. Copy the model and assets
scp yolov8_od.vmfb root@<BOARD_IP>:/home/root/standalone/
scp labels.json root@<BOARD_IP>:/home/root/standalone/
scp image.jpg root@<BOARD_IP>:/home/root/standalone/
```

## Step 2: Install Dependencies (On Board)

If the libraries are not already installed on your board image, you can install them using pip:

```bash
pip3 install -r requirements_board.txt
```

---

## Step 3: Run Inference

Login to the board and execute the script. The script applies YOLO-specific preprocessing (letterbox resizing), quantization, inference, and complex post-processing (dequantization, NMS, bounding box scaling).

```bash
# 1. SSH into the board
ssh root@<BOARD_IP>

# 2. Go to the directory
cd /home/root/standalone

# 3. Set up display environment (Required for visual output)
export XDG_RUNTIME_DIR=/var/run/user/0
export WAYLAND_DISPLAY=wayland-1

# 4. Run the object detection job
python3 object_detection.py \
  --model yolov8_od.vmfb \
  --image image.jpg \
  --labels labels.json \
  --device torq
```

---

## Expected Output

You should see output similar to the following, confirming the model successfully detected objects:

```text
[1/4] Preprocessing...

[2/4] Inference...
Command: iree-run-module --device=torq --module=yolov8_od.vmfb ...
EXEC @main
Time: 0.6516s

[3/4] Processing...

[4/4] Detections:
  dog             Conf: 0.9186  Box: [133 219 177 315]
  car             Conf: 0.5663  Box: [468  79 254  86]
  bicycle         Conf: 0.5663  Box: [151 137 412 280]
```

- **Conf**: Confidence score (0.0 to 1.0)
- **Box**: Bounding box coordinates `[x, y, width, height]` in original image pixels.
