# Model Conversion Guide

How to convert a YOLOv8 detection model for Synaptics Astra SL2610 with Torq / Coral NPU.

## Overview

```
PyTorch (.pt) → ONNX → Normalized ONNX → Full INT8 TFLite → TOSA MLIR → Torq Compiler → VMFB (.vmfb) → NPU Inference
```

The existing `moon320_int8.vmfb` model is pre-compiled and ready to use. Follow this guide only if you need to retrain or update the model.

> This is the validated conversion path (previously this guide used a direct
> `.pt → TFLite` export via `ultralytics`. That path is deprecated — the
> ONNX → `onnx2tf` → TOSA route below is what has actually been verified
> against the current Torq toolchain, byte-for-byte against the shipped
> `moon320_int8.vmfb`).

### Resolution Choice

| Resolution | Inference Speed | Accuracy |
|-----------|----------------|----------|
| **320×320** (recommended) | 30+ FPS | Good for real-time music |
| 640×640 | ~ 8 FPS | Higher detection fidelity |

## Prerequisites

Run the conversion on a Linux development machine (or WSL in Windows or macOS). The board is used for deployment/inference.

```bash
python3 -m venv .venv
source .venv/bin/activate

# YOLO export tooling
pip install -U ultralytics

# ONNX → TFLite conversion (version pinned — see note below)
pip install onnx2tf==2.6.8 onnxsim

# Torq compiler + runtime (torq-compile, torq-run-module) and the
# tosa-converter-for-tflite tool (installed via the TFLite extra on the
# Torq wheel) — follow the official installation guide:
# https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/getting_started.html
pip install "torq_compiler-<version>-<platform>.whl[tflite]"
```

Verify everything is on `PATH` before starting:

```bash
onnx2tf -V                        # expect 2.6.8
which onnxsim
which tosa-converter-for-tflite
which torq-compile
which torq-run-module
```

> **Version note**: The conversion was validated against `onnx2tf 2.6.8`
> specifically. Newer/older versions may change flag behavior or the
> default TFLite backend — if you use a different version, re-validate
> against a known-good `.vmfb` (see [Validating against an existing
> VMFB](#validating-against-an-existing-vmfb)).

## Step 1: Get a Dataset

Acquire a dataset of images with objects annotated in YoloV8 format, or create your own.

The dataset should be split into:
- training
- validation
- test

> **Note**: Full INT8 quantization requires representative calibration images. The validated workflow used 300 images; budget at least ~100.
- Randomly select images from the training set for calibration.

## Step 2: Use Ultralytics to fine-tune the YoloV8 model using the dataset

Follow an online guide to retrain the YoloV8 base model to detect your objects. You will get a `best.pt` file.

## Step 3: Export YOLOv8 to ONNX

On your development machine, use `ultralytics` to export the model to ONNX at `320x320` input resolution.

```bash
yolo export model=best.pt format=onnx imgsz=320
```

This produces a `best.onnx` file with output shape `[1, 5, 2100]` (4 box coordinates + 1 confidence, per anchor).

## Step 4: Normalize the ONNX Output

The raw ONNX box outputs are in pixel coordinates relative to the input resolution. Normalize the box channels by the input size before quantizing, so the exported model's coordinate scale is consistent regardless of resolution:

```
boxes      = output0[:, 0:4, :] / 320
confidence = output0[:, 4:5, :]
normalized_output = concat(boxes, confidence, axis=1)   # still [1, 5, 2100]
```

`320` here is the model's input resolution — it's a coordinate-normalization factor, not an INT8 quantization parameter. Adjust it if you export at a different resolution (e.g. divide by `640` for the 640×640 model).

This produces `best_normalized_output.onnx`.

## Step 5: Convert ONNX to Full INT8 TFLite

Full INT8 quantization needs a calibration set: representative images, RGB, letterboxed to `320x320`, NHWC, `float32`, normalized to `[0,1]`, saved as a single `.npy` of shape `(N, 320, 320, 3)`.

```python
import cv2
import numpy as np
from pathlib import Path

IMAGE_DIR = Path("calibration_images")
OUTPUT = "calibration_nhwc_float32.npy"
SIZE = 320

samples = []
paths = sorted(IMAGE_DIR.glob("*.jpg")) + sorted(IMAGE_DIR.glob("*.png"))

for path in paths:
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    scale = min(SIZE / w, SIZE / h)
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))

    padded = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    top, left = (SIZE - resized.shape[0]) // 2, (SIZE - resized.shape[1]) // 2
    padded[top:top + resized.shape[0], left:left + resized.shape[1]] = resized

    samples.append(padded.astype(np.float32) / 255.0)

calibration = np.stack(samples).astype(np.float32)
np.save(OUTPUT, calibration)
```

> Do not divide the saved `.npy` by 255 again downstream — it's already normalized to `[0,1]`.

Then convert with `onnx2tf`:

```bash
onnx2tf \
  -i best_normalized_output.onnx \
  -o onnx2tf_normalized_full_int8 \
  -oiqt \
  -iqd int8 \
  -oqd int8 \
  --tflite_backend tf_converter \
  -cind images calibration_nhwc_float32.npy \
  "[[[[0.0,0.0,0.0]]]]" \
  "[[[[1.0,1.0,1.0]]]]"
```

> **Critical**: `--tflite_backend tf_converter` is required. The default
> `flatbuffer_direct` backend produces a TFLite representation that fails
> during TFLite→TOSA conversion due to quantization-scale constraints.

Result: `onnx2tf_normalized_full_int8/best_normalized_output_full_integer_quant.tflite`, with interface `[1, 320, 320, 3] int8` in → `[1, 5, 2100] int8` out.

## Step 6: Convert TFLite to TOSA MLIR

```bash
tosa-converter-for-tflite \
  onnx2tf_normalized_full_int8/best_normalized_output_full_integer_quant.tflite \
  --text \
  -o normalized_full_integer.tosa.mlir
```

## Step 7: Compile for NPU with Torq/IREE

Follow the [Torq Compiler Getting Started Guide](https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/getting_started.html) to set up your environment, and the [Step-by-Step Model deployment Examples](https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/step_by_step_examples.html#step-by-step-model-deployment-examples) for background on the TOSA→VMFB step.

```bash
torq-compile \
  normalized_full_integer.tosa.mlir \
  -o best.vmfb \
  --torq-hw=SL2610
```

The compiler may print memory-planning diagnostics such as `Failed to swap in operand`; these can appear even on a successful build — check the process exit code (`EXIT=0`) and confirm the `.vmfb` was produced rather than treating the message itself as fatal.

> If you were previously passing `--torq-convert-dtypes`,
> `--torq-disable-slicing`, `--torq-enable-torq-hl-tiling`,
> `--torq-enable-transpose-optimization`, or `--torq-convert-io-dtype`,
> those flags applied to the older ONNX→MLIR path and are not part of the
> validated TOSA-MLIR flow above. Confirm current flag requirements
> against the [model conversion
> docs](https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/model_conversion.html)
> before dropping them for your Torq build.

## Step 8: Deploy to Board

```bash
# Copy the compiled model
scp best.vmfb root@<board-ip>:/home/root/sl2610-examples/models/

# Or via ADB
adb push best.vmfb /home/root/sl2610-examples/models/
```

## Step 9: Perform a Quick Test

On the board, in a Python script:

```python
import torq.runtime as tr
import numpy as np, time

runner = tr.VMFBInferenceRunner(
    'model/best.vmfb',
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
```

Alternatively, `torq-run-module` can be used on the dev machine (before deployment) to run a `.npy` input through the compiled `.vmfb` directly:

```bash
torq-run-module \
  --module=best.vmfb \
  --function=main \
  --input=@validation_input.npy \
  --output=@output.npy
```

## Step 10: Integrate into the Music generation demo app

Update `detector.py`:

- Change `model_path` (currently defaults to
  `../models/Synaptics/yolov8-od-nano-jellyfish-int8-torq/moon320_int8.vmfb`)
  to point at your new `.vmfb`.

- Update the quantization parameters in `QUANT_PARAMS`. These must match the compiled model.

If you retrain the model, update these values.

| Parameter | 320×320 Model | 640×640 Model |
|-----------|--------------|--------------|
| Input scale | 0.003921568859 | 0.003921568859 |
| Input zero-point | -128 | -128 |
| Output scale | 0.007293878589 | 0.003967983648 |
| Output zero-point | -105 | -128 |

## Troubleshooting

**`onnxsim` not found** — `onnx2tf` depends on it; `pip install onnxsim`.

**TFLite → TOSA reports a quantization-scale error** — confirm the TFLite model was generated with `--tflite_backend tf_converter`, not the default `flatbuffer_direct` backend.

**TFLite input layout looks different from ONNX** — expected: ONNX is `[1, 3, 320, 320]` (NCHW), TFLite is `[1, 320, 320, 3]` (NHWC). `onnx2tf` handles this transpose.

**Torq compiler prints `Failed to swap in operand`** — check the process exit code and whether the `.vmfb` was actually generated before treating this as an error.

## Validating against an existing VMFB

Optional, but recommended when replacing a known-good model: run the same quantized input through both the old and new `.vmfb` via `torq-run-module` and diff the outputs.

```bash
torq-run-module --module=<existing>.vmfb --function=main \
  --input=@validation_input.npy --output=@golden_output.npy

torq-run-module --module=<new>.vmfb --function=main \
  --input=@validation_input.npy --output=@new_output.npy

python -c 'import numpy as np; old=np.load("golden_output.npy"); new=np.load("new_output.npy"); d=np.abs(old.astype(np.int16)-new.astype(np.int16)); print("MAX ABS:", int(d.max())); print("EXACT MATCH:", bool(np.array_equal(old,new)))'
```

An exact match (`MAX ABS: 0`) confirms the new conversion path preserves the original model's behavior.

## Source Model

The jellyfish detection model is from [seaphony-ml](https://github.com/patrickdmiller/seaphony-ml) by Patrick Miller. It's a custom-trained YOLOv8 nano model for moon jellyfish detection.
