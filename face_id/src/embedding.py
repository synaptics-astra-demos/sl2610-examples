# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Face alignment and embedding model helpers."""

from __future__ import annotations

import cv2
import numpy as np


EMBEDDING_SIZE = 112
EMBEDDING_DIMENSION = 256
REFERENCE_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def extract_five_landmarks(keypoints):
    """Select the five averaged landmarks used by the embedded reference."""
    points = np.asarray(keypoints, dtype=np.float32)
    left_eye = points[[36, 37, 38, 39, 40, 41]].mean(axis=0)
    right_eye = points[[42, 43, 44, 45, 46, 47]].mean(axis=0)
    return np.array([left_eye, right_eye, points[30], points[48], points[54]], dtype=np.float32)


def align_face(frame, keypoints):
    """Warp a camera-frame face to the 112x112 recognition template."""
    landmarks = extract_five_landmarks(keypoints)
    transform, _ = cv2.estimateAffinePartial2D(
        landmarks, REFERENCE_LANDMARKS, method=cv2.LMEDS,
    )
    if transform is None:
        raise ValueError("Unable to estimate face alignment transform")
    aligned = cv2.warpAffine(
        frame, transform, (EMBEDDING_SIZE, EMBEDDING_SIZE),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
    )
    return aligned, landmarks


def preprocess_embedding(frame, keypoints):
    """Align and quantize an RGB face for the embedding model."""
    aligned, landmarks = align_face(frame, keypoints)
    rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
    quantized = np.clip(rgb.astype(np.int16) - 128, -128, 127).astype(np.int8)
    return quantized[np.newaxis, :, :, :], landmarks


def decode_embedding(output, scale=0.1855205, zero_point=-2):
    """Dequantize and L2-normalize the 256-dimensional embedding."""
    values = np.asarray(output).reshape(-1)
    if values.size < EMBEDDING_DIMENSION:
        raise ValueError(f"Expected 256 embedding values, received {values.size}")
    embedding = (values[:EMBEDDING_DIMENSION].astype(np.float32) - zero_point) * scale
    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError("Embedding has zero norm")
    return embedding / norm