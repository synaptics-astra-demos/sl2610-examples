# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Scene description: capture a photo, then describe it with LiquidAI LFM2-VL-450M on the NPU."""

import argparse
import gc
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap

from app_utils.camera import Camera, CameraError, find_camera_device
from app_utils.display import enable_wayland_display
from app_utils.npu import cleanup_npu_after_inference, setup_npu_for_inference

# The LFM2-VL-450M demo lives under app_utils/torq_examples/LiquidAI/LiquidAI-LFM2-VL-450M/,
# a directory whose name contains hyphens, so it isn't importable as a regular package.
# Add its src/ dir (for the runner) and its own dir (for the model-refresh helper) to
# sys.path directly instead, same as that demo's own src/infer.py does.
_LFM2VL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "app_utils", "torq_examples", "LiquidAI", "LiquidAI-LFM2-VL-450M",
)
sys.path.insert(0, os.path.join(_LFM2VL_DIR, "src"))
sys.path.insert(0, _LFM2VL_DIR)

from runner import LiquidVLStatic
from setup_demo import ensure_lfm2vl_models

ORIENTATION = "portrait" # or "landscape"

if ORIENTATION == "portrait":
    WINDOW_W = 480 - 6
    WINDOW_H = 800 - 33
else:
    WINDOW_W = 800
    WINDOW_H = 380

DEFAULT_MODELS_DIR = "../models/Synaptics/LiquidAI-LFM2-VL-450M/"
DEFAULT_PROMPT = "Describe this image in two sentences."
DECODER_FILE = "decoder_nolm.vmfb"
LM_HEAD_FILE = "lm_head.vmfb"
VISION_FILE = "vision_encoder_256.vmfb"
IMAGE_DECODER_PREFIX = "decoder_image_2part_"
CAPTURE_IMAGE_PATH = "scene_capture.jpg"
VISION_INPUT_SIZE = 256  # vision_encoder_256.vmfb's one-shot image-prefill path is a static
                          # 256-patch (16x16) shape, so the source image must be exactly
                          # 256x256 pixels going in (see LiquidAI-LFM2-VL-450M/README.md).
PREVIEW_FPS = 15
CAPTURE_HOLD_SECONDS = 10


class Signals(QObject):
    update_image = pyqtSignal(object)  # BGR numpy frame
    update_description = pyqtSignal(str)
    set_button_enabled = pyqtSignal(bool)


def _center_crop_square(frame):
    """Crop ``frame`` to a centered square, using the shorter side as edge length."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0:y0 + side, x0:x0 + side]


def build_vl(models_dir):
    """Load the LFM2-VL-450M vision encoder + decoder for one-shot image description."""
    models_dir = Path(models_dir)
    required = (DECODER_FILE, LM_HEAD_FILE, VISION_FILE)
    for filename in required:
        path = models_dir / filename
        if not path.exists():
            sys.exit(f"LFM2-VL-450M asset not found: {path}")
    return LiquidVLStatic(
        str(models_dir / DECODER_FILE),
        str(models_dir / VISION_FILE),
        lmhead_path=str(models_dir / LM_HEAD_FILE),
        image_decoder_prefix=str(models_dir / IMAGE_DECODER_PREFIX),
    )


class CameraApp(QWidget):
    def __init__(self, camera, models_dir, prompt):
        super().__init__()
        self.camera = camera
        self.models_dir = models_dir
        self.prompt = prompt
        self.signals = Signals()
        self._running = True
        self._frozen = False
        self._last_frame = None
        self._init_ui()
        self.signals.update_image.connect(self._show_image)
        self.signals.update_description.connect(self.description_label.setText)
        self.signals.set_button_enabled.connect(self.capture_btn.setEnabled)

        self._preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self._preview_thread.start()

    def _init_ui(self):
        self.setWindowTitle("Scene Description — Astra SL2610")
        self.resize(WINDOW_W, WINDOW_H)
        self.setStyleSheet("background-color: #000000;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Image fills the screen edge-to-edge
        self.image_label = QLabel("Starting camera preview...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setStyleSheet("background-color: #000000; color: #444444; font-size: 15px;")
        layout.addWidget(self.image_label)

        # Bottom bar — dark strip like the Android camera controls area
        bottom_bar = QWidget()
        bottom_bar.setFixedHeight(110)
        bottom_bar.setStyleSheet("background-color: #111111;")
        bottom_layout = QHBoxLayout(bottom_bar)
        bottom_layout.setContentsMargins(20, 0, 20, 0)
        bottom_layout.setSpacing(12)

        # Description text — left side, no box, subtle colour
        self.description_label = QLabel("")
        self.description_label.setWordWrap(True)
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.description_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.description_label.setStyleSheet(
            "color: #AAAAAA; font-size: 15px; font-family: 'Segoe UI', sans-serif;"
            "background: transparent;"
        )
        bottom_layout.addWidget(self.description_label)

        # Circular shutter button — Android-style white ring + white fill
        self.capture_btn = QPushButton()
        self.capture_btn.setFixedSize(80, 80)
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background-color: white;
                border-radius: 40px;
                border: 4px solid #888888;
            }
            QPushButton:disabled {
                background-color: #444444;
                border: 4px solid #333333;
            }
            QPushButton:pressed {
                background-color: #CCCCCC;
                border: 4px solid #666666;
            }
        """)
        self.capture_btn.clicked.connect(self._on_capture)
        bottom_layout.addWidget(self.capture_btn, alignment=Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(bottom_bar)

    def _on_capture(self):
        if self._frozen or self._last_frame is None:
            return
        self.capture_btn.setEnabled(False)
        self._frozen = True
        self.signals.update_description.emit("Thinking...")
        threading.Thread(target=self._describe_then_resume, args=(self._last_frame,), daemon=True).start()

    def _describe_then_resume(self, frame):
        try:
            square = _center_crop_square(frame)
            resized = cv2.resize(square, (VISION_INPUT_SIZE, VISION_INPUT_SIZE), interpolation=cv2.INTER_AREA)
            cv2.imwrite(CAPTURE_IMAGE_PATH, resized)
            # The NPU vision encoder and decoder can't be co-resident (the encoder
            # is freed once the decoder loads), so the vmfb-vision path is one-shot
            # per LiquidVLStatic instance — rebuild it fresh for every capture.
            vl = build_vl(self.models_dir)
            try:
                description = vl.run(CAPTURE_IMAGE_PATH, self.prompt)
            finally:
                del vl
                gc.collect()
            self.signals.update_description.emit(description.strip())
        except Exception as exc:
            self.signals.update_description.emit(f"Error: {exc}")
        time.sleep(CAPTURE_HOLD_SECONDS)
        self._frozen = False
        self.signals.update_description.emit("")
        self.signals.set_button_enabled.emit(True)

    def _preview_loop(self):
        frame_interval = 1.0 / PREVIEW_FPS
        try:
            # exposure_auto=0: OpenCV's CAP_PROP_AUTO_EXPOSURE abstraction is
            # unreliable for OV5647, so set it directly via v4l2-ctl on open.
            with Camera(self.camera, controls={"exposure_auto": 0}) as cam:
                for frame in cam.stream():
                    if not self._running:
                        break
                    # Skip updating the display while a captured frame is being held.
                    if not self._frozen:
                        self._last_frame = frame
                        self.signals.update_image.emit(frame)
                    time.sleep(frame_interval)
        except CameraError as exc:
            self.signals.update_description.emit(f"Error: {exc}")

    def closeEvent(self, event):
        self._running = False
        super().closeEvent(event)

    def _show_image(self, bgr_frame):
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(pix)

def parse_args():
    parser = argparse.ArgumentParser(description="Capture a photo and describe it with LiquidAI LFM2-VL-450M.")
    parser.add_argument("--models", default=DEFAULT_MODELS_DIR, help="Directory holding the LFM2-VL-450M assets (default: %(default)s)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent to the VLM for each captured photo (default: %(default)s)")
    parser.add_argument("--no-refresh", action="store_true", help="Skip the model freshness check (offline runs)")
    return parser.parse_args()


def main():
    args = parse_args()

    camera = find_camera_device()
    if camera is None:
        print("ERROR: No camera found")
        sys.exit(1)
    print(f"Camera: {camera}")

    ensure_lfm2vl_models(args.models, refresh=not args.no_refresh)

    setup_npu_for_inference()

    enable_wayland_display()
    os.environ["WESTON_DISABLE_GBM_MODIFIERS"] = "true"
    os.environ["QT_QPA_PLATFORM"] = "wayland"

    app = QApplication(sys.argv)
    window = CameraApp(camera, args.models, args.prompt)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_npu_after_inference()
