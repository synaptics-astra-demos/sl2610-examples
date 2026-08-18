# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Camera discovery, configuration, photo capture, and frame streaming.

Standalone from the ``torq_examples`` submodule (no dependency on it) so any
app in this repo can use camera capture without pulling in the Torq runtime.
Requires OpenCV (``cv2``) and, for device discovery/config, the ``v4l2-ctl``
CLI tool.

Typical use::

    from app_utils.camera import Camera

    with Camera("auto", width=640, height=480, fps=30) as cam:
        frame = cam.capture_photo()          # single warmed-up frame
        for frame in cam.stream():           # continuous latest-frame stream
            ...
"""

from __future__ import annotations

import os
import subprocess
import threading
import time

import cv2

V4L2_CONTROL_ALIASES = {
    "brightness": ["brightness"],
    "contrast": ["contrast"],
    "saturation": ["saturation"],
    "sharpness": ["sharpness"],
    "gain": ["gain", "analogue_gain"],
    "exposure_auto": ["exposure_auto", "auto_exposure"],
    "exposure_absolute": ["exposure_absolute", "exposure"],
    "exposure_auto_priority": ["exposure_auto_priority"],
}


class CameraError(RuntimeError):
    """Raised for camera discovery, open, or capture failures."""


def find_camera_device():
    """Return the first USB camera's ``/dev/videoN`` path, or None."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if result.returncode == 0:
            in_usb_device = False
            for line in result.stdout.splitlines():
                if "usb-" in line.lower():
                    in_usb_device = True
                    continue
                if in_usb_device and "/dev/video" in line:
                    device = line.strip()
                    if os.path.exists(device):
                        return device
                if not line.strip():
                    in_usb_device = False
    except Exception:
        pass

    for index in range(10):
        device = f"/dev/video{index}"
        if os.path.exists(device):
            return device
    return None


def resolve_camera_device(camera_device):
    """Resolve ``"auto"`` to a discovered device; validate an explicit path."""
    if camera_device == "auto":
        resolved = find_camera_device()
        if resolved is None:
            raise CameraError("No USB camera device found")
        return resolved

    if not os.path.exists(camera_device):
        raise CameraError(f"Camera device not found: {camera_device}")
    return camera_device


def set_camera_controls(device, controls):
    """Set V4L2 controls (brightness, gain, exposure, ...) on ``device``.

    ``controls`` maps logical names (see ``V4L2_CONTROL_ALIASES``) to values;
    None values are skipped. Drivers vary in what they name a control (UVC
    uses ``exposure_absolute``, CSI drivers often use ``exposure``), so each
    logical name tries its known aliases in order until one succeeds.
    """
    for logical_name, value in (controls or {}).items():
        if value is None:
            continue

        success = False
        errors = []
        for v4l2_name in V4L2_CONTROL_ALIASES.get(logical_name, [logical_name]):
            command = ["v4l2-ctl", "-d", device, "-c", f"{v4l2_name}={value}"]
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    print(f"[Camera] Success: {' '.join(command)}")
                    success = True
                    break
                errors.append(f"{v4l2_name}: {result.stderr.strip()}")
            except Exception as exc:
                errors.append(f"{v4l2_name}: {exc}")

        if not success:
            print(f"[Camera] Warning: failed to set {logical_name}={value} on {device}: {' | '.join(errors)}")


class FrameGrabber:
    """Background thread that keeps only the most recently read frame.

    Wraps any already-open ``cv2.VideoCapture`` (camera device or RTSP
    stream) so slow consumers - e.g. NPU inference - never process a
    backlog of stale buffered frames. ``Camera.stream()`` uses this
    internally; use it directly when you already have your own
    ``VideoCapture`` (for example, an RTSP source opened outside this
    module).
    """

    def __init__(self, cap):
        self._cap = cap
        self._frame = None
        self._lock = threading.Lock()
        self._stopped = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stopped:
            ret, frame = self._cap.read()
            if not ret:
                self._stopped = True
                break
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=1.0)


class Camera:
    """A camera device: discovery/open, V4L2 config, photo capture, streaming.

    Args:
        device: ``/dev/videoN`` path, an integer index, or ``"auto"`` to pick
            the first detected USB camera.
        width, height, fps: requested capture settings (best-effort; not all
            drivers honor all three).
        controls: dict of V4L2 controls to apply on open, see
            ``set_camera_controls``.
        control_device: V4L2 device to apply ``controls`` to, if it differs
            from ``device`` (e.g. a separate ``/dev/v4l-subdevN``).
        warmup_frames: frames to discard in ``capture_photo()`` before
            grabbing the "real" one, letting auto-exposure/white-balance
            settle after opening.
    """

    def __init__(self, device="auto", width=None, height=None, fps=None,
                 controls=None, control_device=None, warmup_frames=20):
        self.requested_device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.controls = controls
        self.control_device = control_device
        self.warmup_frames = warmup_frames

        self.device = None
        self._cap = None
        self._grabber = None

    def open(self):
        self.device = resolve_camera_device(self.requested_device)

        if self.controls:
            set_camera_controls(self.control_device or self.device, self.controls)

        try:
            cam_index = int(self.device)
        except (ValueError, TypeError):
            cam_index = self.device

        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not cap.isOpened():
            raise CameraError(f"Cannot open camera device: {self.device}")

        self._cap = cap
        return self

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc_info):
        self.close()

    @property
    def frame_size(self):
        """Actual (width, height) the driver is delivering."""
        if self._cap is None:
            raise CameraError("Camera not open")
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def capture_photo(self):
        """Grab a single BGR frame, discarding warmup frames so AEC settles."""
        if self._cap is None:
            raise CameraError("Camera not open")
        for _ in range(self.warmup_frames):
            self._cap.read()
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise CameraError("Failed to capture frame")
        return frame

    def stream(self):
        """Yield the latest available BGR frame continuously.

        Backed by a background grabber thread, so this never blocks waiting
        on frames a slow consumer has fallen behind on. Runs until the
        caller stops iterating (e.g. ``break``) or the camera disconnects.
        """
        if self._cap is None:
            raise CameraError("Camera not open")
        if self._grabber is None:
            self._grabber = FrameGrabber(self._cap)
        while not self._grabber._stopped:
            ret, frame = self._grabber.read()
            if not ret:
                time.sleep(0.005)
                continue
            yield frame

    def read(self):
        """Direct, non-streaming passthrough to the underlying VideoCapture."""
        if self._cap is None:
            raise CameraError("Camera not open")
        return self._cap.read()

    def close(self):
        if self._grabber is not None:
            self._grabber.stop()
            self._grabber = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
