"""
Jellyfish Detector — Torq NPU backend (IREE/MLIR .vmfb models)

Backend: torq.runtime VMFBInferenceRunner on /dev/torq NPU.

Based on Patrick Miller's moon_lightweight.py from seaphony-ml.
"""

import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_utils.torq_examples.utils.inference import SimpleVMFBInferenceRunner

# ── Configuration ──────────────────────────────────────────────
INPUT_SIZE = 320            # Must match compiled model input size
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 15

# INT8 quantization parameters for the moon320.vmfb model
# (from seaphony-ml object_detection calibration)
IN_SCALE = 0.003921568859368563
IN_ZP = -128

# Per-resolution output dequantization parameters
QUANT_PARAMS = {
    320: (0.007293878588825464, -105),
    640: (0.003967983648180962, -128),
}


class Detector:
    """
    Jellyfish detector using the Torq NPU backend.

    Uses int8 quantized .vmfb models compiled with the Torq/IREE compiler
    for the Synaptics SL2610 NPU.
    """

    def __init__(self, model_path: str = "../models/Synaptics/yolov8-od-nano-jellyfish-int8-torq/moon320_int8.vmfb"):
        self.model_path = model_path
        self._runner = None      # SimpleVMFBInferenceRunner
        self._backend = None
        self._out_scale = None
        self._out_zp = None

    def load(self) -> None:
        """Load model onto the Torq NPU."""
        vmfb_path = self.model_path if self.model_path.endswith(".vmfb") else \
            os.path.splitext(self.model_path)[0] + ".vmfb"

        # Determine quantization params from filename
        if "320" in os.path.basename(self.model_path):
            self._out_scale, self._out_zp = QUANT_PARAMS[320]
        elif "640" in os.path.basename(self.model_path):
            self._out_scale, self._out_zp = QUANT_PARAMS[640]
        else:
            self._out_scale, self._out_zp = QUANT_PARAMS.get(INPUT_SIZE, QUANT_PARAMS[320])

        if not os.path.exists(vmfb_path):
            raise RuntimeError(
                f"Model not found: {vmfb_path}\n"
                f"Copy moon320.vmfb from seaphony-ml to deploy."
            )

        try:
            self._runner = SimpleVMFBInferenceRunner(
                vmfb_path,
                device_uri="torq",
                function="main",
                load_model_to_mem=True,
            )
            self._backend = "torq"
            print(f"[Detector] ✓ Torq NPU backend: {vmfb_path}")
        except Exception as e:
            raise RuntimeError(f"[Detector] Torq load failed: {e}")

    def detect(self, frame) -> tuple[list[dict], float]:
        """
        Run detection on a BGR frame (from OpenCV).
        Returns (detections, infer_ms) where detections is a list of dicts:
        {bbox: {x1,y1,x2,y2}, centroid: {x,y}, confidence: float}
        All coordinates normalized 0-1.
        """
        if self._backend is None:
            return [], 0.0

        h, w = frame.shape[:2]

        return self._detect_torq(frame, w, h)

    def _detect_torq(self, frame, orig_w: int, orig_h: int) -> list[dict]:
        """
        Torq NPU path — int8 quantized inference.
        Preprocessing: letterbox pad → normalize → quantize to int8.
        """
        # ── Preprocess: letterbox + int8 quantize ──
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Letterbox resize preserving aspect ratio
        r = min(INPUT_SIZE / orig_w, INPUT_SIZE / orig_h)
        new_w = int(round(orig_w * r))
        new_h = int(round(orig_h * r))
        resized = cv2.resize(img_rgb, (new_w, new_h))

        # Pad to INPUT_SIZE × INPUT_SIZE with neutral gray (114)
        pad = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        top = (INPUT_SIZE - new_h) // 2
        left = (INPUT_SIZE - new_w) // 2
        pad[top:top + new_h, left:left + new_w] = resized

        # Normalize [0,1] then quantize to int8
        blob = np.clip(
            np.array(pad, dtype=np.float32) / 255.0 / IN_SCALE + IN_ZP,
            -128, 127
        ).astype(np.int8)
        blob = np.expand_dims(blob, 0)  # (1, H, W, 3) — NHWC for Torq

        pad_info = (top / INPUT_SIZE, left / INPUT_SIZE)

        # ── Inference ──
        try:
            t0 = time.perf_counter()
            raw = self._runner.infer(blob)
            infer_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            print(f"[Detector] Torq inference error: {e}")
            return [], 0.0

        # ── Postprocess (int8 dequantize + NMS) ──
        return self._postprocess_torq(raw, (orig_h, orig_w), pad_info), infer_ms

    def _postprocess_torq(self, raw: np.ndarray, orig_shape: tuple,
                          pad_info: tuple) -> list[dict]:
        """
        YOLOv8 int8 output postprocessing.
        Dequantize → NMS → normalized coordinates.

        Based on seaphony-ml moon320.py postprocess().
        """
        # Dequantize int8 → float32
        out = ((raw.astype(np.float32) - self._out_zp) * self._out_scale)
        out = out.squeeze()  # Remove batch dim

        # Handle output shape: (5, N) → transpose to (N, 5)
        if out.ndim == 2:
            if out.shape[0] < out.shape[1]:
                out = out.T

        if out.ndim != 2 or out.shape[1] < 5:
            return []

        # Score filtering (multi-class or single-class)
        scores = np.max(out[:, 4:], axis=1)
        mask = scores >= CONF_THRESHOLD
        out = out[mask]
        scores = scores[mask]

        if len(out) == 0:
            return []

        # Boxes are in normalized coords with pad offset
        boxes = out[:, :4].copy()

        # Remove pad offset (in normalized space)
        boxes[:, 0] -= pad_info[1]   # x - left_pad_ratio
        boxes[:, 1] -= pad_info[0]   # y - top_pad_ratio

        # Scale to original pixel space
        max_dim = max(orig_shape)
        boxes[:, :4] *= max_dim

        # Convert cx,cy,w,h → x1,y1,x2,y2 in original pixel coords
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Normalize to 0-1 range
        oh, ow = orig_shape
        x1_n = np.clip(x1 / ow, 0.0, 1.0)
        y1_n = np.clip(y1 / oh, 0.0, 1.0)
        x2_n = np.clip(x2 / ow, 0.0, 1.0)
        y2_n = np.clip(y2 / oh, 0.0, 1.0)

        # NMS
        norm_boxes = np.stack([x1_n, y1_n, x2_n, y2_n], axis=1)
        keep = self._nms(norm_boxes, scores, IOU_THRESHOLD, MAX_DETECTIONS)

        detections = []
        for i in keep:
            detections.append({
                "bbox": {
                    "x1": float(norm_boxes[i, 0]),
                    "y1": float(norm_boxes[i, 1]),
                    "x2": float(norm_boxes[i, 2]),
                    "y2": float(norm_boxes[i, 3]),
                },
                "centroid": {
                    "x": float((norm_boxes[i, 0] + norm_boxes[i, 2]) / 2),
                    "y": float((norm_boxes[i, 1] + norm_boxes[i, 3]) / 2),
                },
                "confidence": float(scores[i]),
            })

        return detections

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray,
             iou_threshold: float, max_det: int) -> list[int]:
        """Greedy NMS — pure NumPy, no torch required."""
        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0 and len(keep) < max_det:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Compute IoU with remaining boxes
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h

            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_j = ((boxes[order[1:], 2] - boxes[order[1:], 0]) *
                      (boxes[order[1:], 3] - boxes[order[1:], 1]))
            union = area_i + area_j - inter
            iou = np.where(union > 0, inter / union, 0.0)

            order = order[1:][iou <= iou_threshold]

        return keep

    def dispose(self) -> None:
        self._runner = None
        self._backend = None
