# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Postprocessing for the three-output face detector."""

from __future__ import annotations

import numpy as np


def _dequantize(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale <= 0:
        raise ValueError("quantization scale must be positive")
    return (values.astype(np.float32) - zero_point) * scale


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must be between 0 and 1")
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size:
        index = int(order[0])
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        denominator = areas[index] + areas[order[1:]] - intersection
        overlap = np.divide(intersection, denominator, out=np.zeros_like(intersection), where=denominator > 0)
        order = order[np.where(overlap <= iou_threshold)[0] + 1]

    return keep


def decode_face_outputs(
    outputs,
    *,
    box1_scale: float,
    box1_zero_point: int,
    box2_scale: float,
    box2_zero_point: int,
    score_scale: float,
    score_zero_point: int,
    confidence_threshold: float,
    iou_threshold: float,
    image_width: int,
    image_height: int,
):
    if len(outputs) != 3:
        raise ValueError(f"expected three detector outputs, got {len(outputs)}")

    box1 = np.asarray(outputs[0]).reshape(-1, 2)
    scores = np.asarray(outputs[1]).reshape(-1)
    box2 = np.asarray(outputs[2]).reshape(-1, 2)
    if box1.shape != box2.shape or len(box1) != len(scores):
        raise ValueError("detector output tensors have incompatible shapes")

    box1 = _dequantize(box1, box1_scale, box1_zero_point)
    box2 = _dequantize(box2, box2_scale, box2_zero_point)
    scores = _dequantize(scores, score_scale, score_zero_point)
    boxes = np.column_stack((box1[:, 0], box1[:, 1], box2[:, 0], box2[:, 1]))

    valid = (
        (scores >= confidence_threshold)
        & (boxes[:, 2] > boxes[:, 0])
        & (boxes[:, 3] > boxes[:, 1])
    )
    boxes = boxes[valid]
    scores = scores[valid]
    if len(boxes) == 0:
        return []

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_height - 1)
    keep = _nms_xyxy(boxes, scores, iou_threshold)

    detections = []
    for index in keep:
        x1, y1, x2, y2 = boxes[index]
        detections.append(("face", float(scores[index]), np.array([x1, y1, x2 - x1, y2 - y1])))
    return detections
