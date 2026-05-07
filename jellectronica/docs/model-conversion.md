# Model Conversion Guide

How to convert a YOLOv8 jellyfish detection model for the Coral NPU.

## Overview

```
PyTorch (.pt) → ONNX (.onnx) → Torq Compiler → VMFB (.vmfb) → NPU Inference
```

The included `moon320.vmfb` is pre-compiled and ready to use. Follow this guide only if you need to retrain or update the model.

## Step 1: Export YOLOv8 to ONNX

On your development machine:

```bash
pip install ultralytics

python3 -c "
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='onnx', imgsz=320)
"
```

This produces a `best.onnx` file.

### Resolution Choice

| Resolution | Inference Speed | Accuracy |
|-----------|----------------|----------|
| **320×320** (recommended) | ~32ms / 31 FPS | Good for real-time music |
| 640×640 | ~120ms / 8 FPS | Higher detection fidelity |

## Step 2: Compile for NPU with Torq/IREE

The Torq compiler converts ONNX models to IREE FlatBuffer (`.vmfb`) format with INT8 quantization for the NPU.

Follow the [Torq Compiler Getting Started Guide](https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/getting_started.html) to set up your environment. 

Next, follow the steps in the [Step-by-Step Model deployment Examples](https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/step_by_step_examples.html#step-by-step-model-deployment-examples). Focus on the **Example: ONNX model** section. 


And finally, compile the model with these recommended settings.

```bash
torq-compile -o <Your Model>.vmfb <Your Model>.mlir \
    --torq-convert-dtypes \
    --torq-disable-slicing \
    --torq-enable-torq-hl-tiling \
    --torq-enable-transpose-optimization \
    --torq-convert-io-dtype \
    --torq-hw=SL2610 \
```

> **Note**: INT8 quantization requires ~100-500 representative sample images for calibration.

## Step 3: Deploy to Board

```bash
# Copy the compiled model
scp output/moon320.vmfb root@<board-ip>:/home/root/sl2610-examples/models/

# Or via ADB
adb push output/moon320.vmfb /home/root/sl2610-examples/models/
```

## Step 4: Test

```bash
# On the board
python3 -c "
import torq.runtime as tr
import numpy as np, time

runner = tr.VMFBInferenceRunner(
    'model/moon320.vmfb',
    device_uri='torq',
    function='main',
    load_model_to_mem=True,
)
inp = np.zeros((1, 320, 320, 3), dtype=np.int8)
runner.infer([inp])  # warmup

t0 = time.time()
for _ in range(10):
    runner.infer([inp])
ms = (time.time() - t0) / 10 * 1000
print(f'NPU inference: {ms:.1f}ms/frame ({1000/ms:.0f} FPS)')
"
```

## Quantization Parameters

The `detector.py` includes quantization parameters that must match the compiled model:

| Parameter | 320×320 Model | 640×640 Model |
|-----------|--------------|--------------|
| Input scale | 0.003921568859 | 0.003921568859 |
| Input zero-point | -128 | -128 |
| Output scale | 0.007293878589 | 0.003967983648 |
| Output zero-point | -105 | -128 |

If you retrain the model, update these values in `detector.py`.

## Source Model

The jellyfish detection model is from [seaphony-ml](https://github.com/patrickdmiller/seaphony-ml) by Patrick Miller. It's a custom-trained YOLOv8 nano model for moon jellyfish detection.
