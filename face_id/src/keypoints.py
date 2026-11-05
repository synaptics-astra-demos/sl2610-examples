# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Face keypoint preprocessing, decoding, and coordinate mapping."""

from __future__ import annotations

import cv2
import numpy as np


INPUT_SIZE = 56
KEYPOINT_COUNT = 68


def make_square_roi(box, frame_shape, expansion=1.3):
    """Return an even-aligned square ROI matching the embedded implementation."""
    x, y, width, height = [float(value) for value in box]
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    side = max(width, height) * expansion
    frame_height, frame_width = frame_shape[:2]

    x1 = max(int((center_x - side / 2.0) / 2.0) * 2, 0)
    y1 = max(int((center_y - side / 2.0) / 2.0) * 2, 0)
    x2 = min(int((center_x + side / 2.0) / 2.0) * 2, frame_width)
    y2 = min(int((center_y + side / 2.0) / 2.0) * 2, frame_height)
    return x1, y1, max(x2, x1 + 2), max(y2, y1 + 2)


def preprocess_keypoint_roi(frame, box, input_zero_point=-128):
    """Crop and resize a face ROI into the keypoint model input tensor."""
    x1, y1, x2, y2 = make_square_roi(box, frame.shape)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("Face ROI is empty")
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    quantized = np.clip(
        resized.astype(np.int16) - 128 + input_zero_point + 128,
        -128,
        127,
    ).astype(np.int8)
    return quantized[np.newaxis, :, :, :], (x1, y1, x2, y2)


def decode_keypoints(outputs, output_scale=1.0, output_zero_point=0):
    """Decode the 136-value landmark output into normalized x/y pairs."""
    landmarks = np.asarray(outputs[0]).reshape(-1)
    if landmarks.size < KEYPOINT_COUNT * 2:
        raise ValueError(f"Expected 136 landmark values, received {landmarks.size}")
    values = (landmarks.astype(np.float32) - output_zero_point) * output_scale
    return values[: KEYPOINT_COUNT * 2].reshape(KEYPOINT_COUNT, 2) / INPUT_SIZE


def map_keypoints_to_frame(keypoints, roi):
    """Map normalized ROI keypoints to camera-frame pixel coordinates."""
    x1, y1, x2, y2 = roi
    points = np.asarray(keypoints, dtype=np.float32).copy()
    points[:, 0] = x1 + points[:, 0] * (x2 - x1)
    points[:, 1] = y1 + points[:, 1] * (y2 - y1)
    return points