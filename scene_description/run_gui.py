# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Scene description: capture a photo, then describe it with LiquidAI LFM2-VL-450M on the NPU."""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app_utils.demo_utils import DependencyChecker, IncompatibleVersionError, MissingRequirementsError

try:
    DependencyChecker(os.path.join(os.path.dirname(__file__), "requirements.txt")).run()
except (MissingRequirementsError, IncompatibleVersionError) as e:
    print(f"{e}\nRun: python setup_demo.py", file=sys.stderr)
    sys.exit(1)

import cv2

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QImage, QPixmap

from app_utils.camera import Camera, CameraError, resolve_camera_device, rotate_frame
from app_utils.display import enable_wayland_display
from app_utils.npu import cleanup_npu_after_inference, setup_npu_for_inference

# The LFM2-VL-450M demo lives under app_utils/torq_examples/LiquidAI/LiquidAI-LFM2-VL-450M/,
# a directory whose name contains hyphens, so it isn't importable as a regular package.
# Add its src/ dir (for the LFM2VL engine) and its own dir (for the model-refresh helper) to
# sys.path directly instead, same as that demo's own src/infer.py does.
_LFM2VL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "app_utils", "torq_examples", "LiquidAI", "LiquidAI-LFM2-VL-450M",
)
sys.path.insert(0, os.path.join(_LFM2VL_DIR, "src"))
sys.path.insert(0, _LFM2VL_DIR)

from lfm2vl import LFM2VL
from setup_demo import ensure_lfm2vl_int8_models

ORIENTATION = "portrait" # or "landscape"

if ORIENTATION == "portrait":
    WINDOW_W = 480 - 6
    WINDOW_H = 800 - 33
else:
    WINDOW_W = 800
    WINDOW_H = 380

DEFAULT_MODELS_DIR = "../models/Synaptics/LiquidAI-LFM2-VL-450M/"
DEFAULT_PROMPT = "Describe this image in two sentences."
CAPTURE_IMAGE_PATH = "scene_capture.jpg"
VISION_INPUT_SIZE = 256  # the vision encoder is a static 256-patch (16x16) shape, so the
                          # image is center-cropped and resized to 256x256 before encoding
                          # (see LiquidAI-LFM2-VL-450M/README.md).
PREVIEW_FPS = 15
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 640, 480, 30  # an explicit size is required: left
                                                        # unconstrained, the ISP camera negotiates
                                                        # 4K and fails to open
CAPTURE_HOLD_SECONDS = 10


class Signals(QObject):
    update_image = pyqtSignal(object)  # BGR numpy frame
    update_description = pyqtSignal(str)
    set_button_enabled = pyqtSignal(bool)
    update_stage = pyqtSignal(str)      # pipeline stage text for the status line
    inference_finished = pyqtSignal()   # stop the dot animation, hide the status line


def _center_crop_square(frame):
    """Crop ``frame`` to a centered square, using the shorter side as edge length."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0:y0 + side, x0:x0 + side]


def load_vl(models_dir):
    """Load the LFM2-VL-450M (W8, all-NPU) model set once; the handle is reused for every capture."""
    return LFM2VL(str(models_dir))


class CameraApp(QWidget):
    def __init__(self, camera, models_dir, prompt, rotate=0):
        super().__init__()
        self.camera = camera
        self.rotate = rotate
        self.models_dir = models_dir
        self.prompt = prompt
        self.signals = Signals()
        self._running = True
        self._vl = None  # loaded lazily on the first capture, then reused
        self._vl_lock = threading.Lock()  # the engine is single-flow: one inference at a time
        self._frozen = False
        self._last_frame = None
        self._init_ui()
        self.signals.update_image.connect(self._show_image)
        self.signals.update_description.connect(self.description_label.setText)
        self.signals.set_button_enabled.connect(self.capture_btn.setEnabled)
        self.signals.update_stage.connect(self._set_stage)
        self.signals.inference_finished.connect(self._on_inference_finished)

        # Cycling "..." animation for the status line — driven on the GUI thread;
        # the worker thread only ever moves the stage text via signals.
        self._stage_text = ""
        self._dots_phase = 0
        self._dots_timer = QTimer(self)
        self._dots_timer.setInterval(350)
        self._dots_timer.timeout.connect(self._tick_dots)

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

        # Left side — a status line (pipeline stage + animated dots) sits above the
        # description text and is only visible while a capture is being processed.
        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(4)

        self.status_label = QLabel("")
        self.status_label.setVisible(False)
        self.status_label.setStyleSheet(
            "color: #5A9BD4; font-size: 13px; font-family: 'Segoe UI', sans-serif;"
            "background: transparent;"
        )
        text_col.addWidget(self.status_label)

        # Description text — no box, subtle colour
        self.description_label = QLabel("")
        self.description_label.setWordWrap(True)
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.description_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.description_label.setStyleSheet(
            "color: #AAAAAA; font-size: 15px; font-family: 'Segoe UI', sans-serif;"
            "background: transparent;"
        )
        text_col.addWidget(self.description_label)

        bottom_layout.addLayout(text_col, stretch=1)

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
        self.description_label.setText("")
        self._stage_text = "Preparing photo"
        self._dots_phase = 0
        self.status_label.setText(self._stage_text)
        self.status_label.setVisible(True)
        self._dots_timer.start()
        threading.Thread(target=self._describe_then_resume, args=(self._last_frame,), daemon=True).start()

    def _describe_then_resume(self, frame):
        try:
            square = _center_crop_square(frame)
            resized = cv2.resize(square, (VISION_INPUT_SIZE, VISION_INPUT_SIZE), interpolation=cv2.INTER_AREA)
            cv2.imwrite(CAPTURE_IMAGE_PATH, resized)
            with self._vl_lock:
                if self._vl is None:
                    self.signals.update_stage.emit("Loading model")
                    self._vl = load_vl(self.models_dir)
                self.signals.update_stage.emit("Looking at the image")
                self._vl.encode_image(resized)
                self.signals.update_stage.emit("Writing description")
                description = ""
                for chunk in self._vl.infer_stream(self.prompt):
                    description += chunk
                    self.signals.update_description.emit(description.strip())
            self.signals.update_description.emit(description.strip() or "(no description)")
        except Exception as exc:
            self.signals.update_description.emit(f"Error: {exc}")
        finally:
            self.signals.inference_finished.emit()
        time.sleep(CAPTURE_HOLD_SECONDS)
        self._frozen = False
        self.signals.update_description.emit("")
        self.signals.set_button_enabled.emit(True)

    def _tick_dots(self):
        self._dots_phase = (self._dots_phase + 1) % 4
        dots = "●" * self._dots_phase + "·" * (3 - self._dots_phase)
        self.status_label.setText(f"{self._stage_text}  {dots}")

    def _set_stage(self, text):
        self._stage_text = text
        self.status_label.setText(text)

    def _on_inference_finished(self):
        self._dots_timer.stop()
        self.status_label.setVisible(False)
        self.status_label.setText("")

    def _preview_loop(self):
        frame_interval = 1.0 / PREVIEW_FPS
        try:
            with Camera(self.camera, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS) as cam:
                for frame in cam.stream():
                    if not self._running:
                        break
                    # Skip updating the display while a captured frame is being held.
                    if not self._frozen:
                        frame = rotate_frame(frame, self.rotate)
                        self._last_frame = frame
                        self.signals.update_image.emit(frame)
                    time.sleep(frame_interval)
        except CameraError as exc:
            self.signals.update_description.emit(f"Error: {exc}")

    def closeEvent(self, event):
        self._running = False
        # Release the NPU contexts unless an inference is mid-flight (process exit frees them anyway).
        if self._vl_lock.acquire(blocking=False):
            try:
                if self._vl is not None:
                    self._vl.close()
                    self._vl = None
            finally:
                self._vl_lock.release()
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
    parser.add_argument("--camera-device", default="auto", help="Camera device, e.g. /dev/video0, or 'auto' to pick the first camera found (default: %(default)s)")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="Rotate the camera feed (degrees clockwise, default: %(default)s); use 180 if the camera is mounted upside down")
    parser.add_argument("--no-refresh", action="store_true", help="Skip the model freshness check (offline runs)")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        camera = resolve_camera_device(args.camera_device)
    except CameraError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print(f"Camera: {camera}")

    ensure_lfm2vl_int8_models(args.models, refresh=not args.no_refresh)
    if not (Path(args.models) / "config.json").is_file():
        print(f"ERROR: LFM2-VL-450M model set not found in {args.models}. Run: python setup_demo.py")
        sys.exit(1)

    setup_npu_for_inference()

    enable_wayland_display()
    os.environ["WESTON_DISABLE_GBM_MODIFIERS"] = "true"
    os.environ["QT_QPA_PLATFORM"] = "wayland"

    app = QApplication(sys.argv)
    window = CameraApp(camera, args.models, args.prompt, args.rotate)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_npu_after_inference()
