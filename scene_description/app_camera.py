import os
import sys
import threading

import cv2

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap

ORIENTATION = "portrait" # or "landscape"

if ORIENTATION == "portrait":
    WINDOW_W = 480 - 6
    WINDOW_H = 800 - 33
else:   
    WINDOW_W = 800
    WINDOW_H = 380

DUMMY_DESCRIPTION = "A person sitting at a desk with a laptop and a coffee cup nearby."


def find_camera():
    import subprocess
    try:
        result = subprocess.run(["v4l2-ctl", "--list-devices"],
                                capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            in_usb = False
            for line in result.stdout.splitlines():
                if "usb-" in line.lower():
                    in_usb = True
                    continue
                if in_usb and "/dev/video" in line:
                    dev = line.strip()
                    if os.path.exists(dev):
                        return dev
                if not line.strip():
                    in_usb = False
    except Exception:
        pass
    for i in range(10):
        dev = f"/dev/video{i}"
        if os.path.exists(dev):
            return dev
    return None


class Signals(QObject):
    update_image = pyqtSignal(object)  # BGR numpy frame
    update_description = pyqtSignal(str)
    set_button_enabled = pyqtSignal(bool)


class CameraApp(QWidget):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.signals = Signals()
        self._init_ui()
        self.signals.update_image.connect(self._show_image)
        self.signals.update_description.connect(self.description_label.setText)
        self.signals.set_button_enabled.connect(self.capture_btn.setEnabled)

    def _init_ui(self):
        self.setWindowTitle("Scene Description — Astra SL2610")
        self.resize(WINDOW_W, WINDOW_H)
        self.setStyleSheet("background-color: #000000;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Image fills the screen edge-to-edge
        self.image_label = QLabel("Press the shutter button to capture")
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
        self.capture_btn.setEnabled(False)
        self.signals.update_description.emit("Capturing...")
        threading.Thread(target=self._capture, daemon=True).start()

    def _capture(self):
        try:
            cam_index = int(self.camera)
        except (ValueError, TypeError):
            cam_index = self.camera

        import subprocess

        # Set auto exposure directly via v4l2-ctl before opening the device.
        # OpenCV's CAP_PROP_AUTO_EXPOSURE abstraction is unreliable for OV5647.
        device_path = self.camera if isinstance(self.camera, str) else f"/dev/video{self.camera}"
        subprocess.run(
            ["v4l2-ctl", "-d", device_path, "--set-ctrl=auto_exposure=0"],
            capture_output=True,
        )

        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.signals.update_description.emit(f"Error: cannot open {self.camera}")
            self.signals.set_button_enabled.emit(True)
            return

        for _ in range(20):  # give AEC time to converge
            cap.read()
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            self.signals.update_description.emit("Error: failed to capture frame")
            self.signals.set_button_enabled.emit(True)
            return

        self.signals.update_image.emit(frame)
        self.signals.update_description.emit(DUMMY_DESCRIPTION)
        self.signals.set_button_enabled.emit(True)

    def _show_image(self, bgr_frame):
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(pix)

#  NPU Clock 
def enable_npu_clock():
    """Enable NPU clock via devmem (required before Torq inference)."""
    try:
        subprocess.run(["devmem", "0xf7e104b0", "32", "0x216"],
                       capture_output=True, timeout=5)
        print("[NPU] Clock enabled")
    except Exception as e:
        print(f"[NPU] Clock enable failed: {e}")

def main():
    camera = find_camera()
    if camera is None:
        print("ERROR: No camera found")
        sys.exit(1)
    print(f"Camera: {camera}")

    # Set NPU clock
    enable_npu_clock()

    os.environ["XDG_RUNTIME_DIR"] = "/var/run/user/0"
    os.environ["WESTON_DISABLE_GBM_MODIFIERS"] = "true"
    os.environ["WAYLAND_DISPLAY"] = "wayland-1"
    os.environ["QT_QPA_PLATFORM"] = "wayland"

    app = QApplication(sys.argv)
    window = CameraApp(camera)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
