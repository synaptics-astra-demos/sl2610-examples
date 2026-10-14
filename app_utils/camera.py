# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Camera discovery, configuration, photo capture, and frame streaming.

Standalone from the ``torq_examples`` submodule (no dependency on it) so any
app in this repo can use camera capture without pulling in the Torq runtime.
Requires OpenCV (``cv2``) built with GStreamer support (used for capture -
see ``Camera._gstreamer_pipeline()`` for why, not cv2's own V4L2 backend)
and, for device discovery/config, the ``v4l2-ctl`` CLI tool.

Typical use::

    from app_utils.camera import Camera

    with Camera("auto", width=640, height=480, fps=30) as cam:
        frame = cam.capture_photo()          # single warmed-up frame
        for frame in cam.stream():           # continuous latest-frame stream
            ...
"""

from __future__ import annotations

import glob
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
    "white_balance_auto": ["white_balance_automatic", "white_balance_temperature_auto"],
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


def is_usb_camera(device):
    """True if ``device`` is listed under a USB device by ``v4l2-ctl --list-devices``.

    USB (UVC) cameras auto-expose on their own. Anything else (a MIPI CSI
    sensor behind an ISP) is treated as a platform camera, see
    ``find_sensor_subdev``. When the listing is unavailable, assume USB so
    nothing gets reconfigured.
    """
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except Exception:
        return True
    if result.returncode != 0:
        return True
    heading = ""
    for line in result.stdout.splitlines():
        if line.strip() and not line.startswith((" ", "\t")):
            heading = line
        elif line.strip() == device:
            return "usb-" in heading.lower()
    return True


def find_sensor_subdev():
    """The V4L2 subdevice of a CSI image sensor (the one with ``analogue_gain``), or None.

    On a MIPI CSI camera the exposure/gain controls are on the sensor's
    ``/dev/v4l-subdevN`` node, not on the ``/dev/videoN`` capture node.
    """
    for path in sorted(glob.glob("/dev/v4l-subdev*")):
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", path, "--list-ctrls"],
                capture_output=True, text=True, timeout=2, check=False,
            )
        except Exception:
            continue
        if "analogue_gain" in result.stdout:
            return path
    return None


def rotate_frame(frame, degrees):
    """Rotate a BGR frame by 0, 90, 180 or 270 degrees clockwise."""
    code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(degrees)
    return frame if code is None else cv2.rotate(frame, code)


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
        device: ``/dev/videoN`` path, or ``"auto"`` to pick the first
            detected USB camera (falling back to the first ``/dev/videoN``
            found at all, e.g. for a CSI/platform camera with no "usb-"
            entry in ``v4l2-ctl --list-devices``).
        width, height, fps: requested capture settings (best-effort; not all
            drivers honor all three).
        controls: dict of V4L2 controls to apply on open, see
            ``set_camera_controls``.
        control_device: V4L2 device to apply ``controls`` to, if it differs
            from ``device`` (e.g. a separate ``/dev/v4l-subdevN``).
        warmup_frames: frames to discard in ``capture_photo()`` before
            grabbing the "real" one, letting auto-exposure/white-balance
            settle after opening.
        auto_exposure: when True (default) and no ``controls`` were given,
            configure exposure for the camera type on ``open()``. USB (UVC)
            cameras already auto-expose, so nothing is touched. A MIPI CSI
            sensor has no working auto-exposure (frames are black at its
            default gain), so its sensor subdevice is set to manual exposure
            plus auto white balance and the gain is tuned from the frame
            brightness, see ``tune_csi_exposure``.
    """

    def __init__(self, device="auto", width=None, height=None, fps=None,
                 controls=None, control_device=None, warmup_frames=20,
                 auto_exposure=True):
        self.requested_device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.controls = controls
        self.control_device = control_device
        self.warmup_frames = warmup_frames
        self.auto_exposure = auto_exposure

        self.sensor_subdev = None  # set on open() for a CSI camera
        self.device = None
        self._cap = None
        self._grabber = None

    def _gstreamer_pipeline(self):
        """A v4l2src pipeline string for `self.device`, for cv2.CAP_GSTREAMER.

        Not cv2's own V4L2 backend (cv2.CAP_V4L2): CSI/ISP camera pipelines
        (as opposed to USB/UVC cameras, which are almost always
        single-plane) commonly expose only the V4L2 "multiplanar" capture
        API, which cv2.CAP_V4L2 doesn't reliably drive - on an OpenCV build
        with no V4L2 support compiled in at all (`cv2.getBuildInformation()`
        has no "V4L" line - true for at least one target this runs on), it
        silently opens the device via some fallback backend that never
        actually negotiates a format, so every read() blocks forever.
        GStreamer's v4l2src correctly handles both single- and multi-plane
        V4L2 devices (USB and CSI alike), and this repo already hard-depends
        on GStreamer for display (app_utils/display.py), so it's not a new
        dependency.

        Only width/height (not pixel format) are put in the source pad's
        requested caps - letting v4l2src auto-negotiate the actual pixel
        format at a given resolution is far more reliable than forcing one;
        some drivers reject a forced format outright even when that exact
        format is what they'd have picked anyway. videoconvert normalizes
        whatever comes out to BGR (matching a plain cv2.VideoCapture frame)
        before appsink.
        """
        caps = []
        if self.width and self.height:
            caps.append(f"width={self.width},height={self.height}")
        if self.fps:
            caps.append(f"framerate={int(self.fps)}/1")
        src_caps = f"video/x-raw,{','.join(caps)}" if caps else "video/x-raw"

        return (
            f"v4l2src device={self.device} ! {src_caps} ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1"
        )

    def open(self):
        self.device = resolve_camera_device(self.requested_device)

        if self.controls:
            set_camera_controls(self.control_device or self.device, self.controls)

        cap = cv2.VideoCapture(self._gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise CameraError(f"Cannot open camera device: {self.device}")

        self._cap = cap
        if self.auto_exposure and not self.controls and not is_usb_camera(self.device):
            self.sensor_subdev = self.control_device or find_sensor_subdev()
            if self.sensor_subdev:
                self.tune_csi_exposure()
        return self

    def tune_csi_exposure(self, target=90, tolerance=20, max_iterations=12):
        """Set a CSI sensor to manual exposure and tune its gain until frames reach a usable brightness.

        The sensor's own auto-exposure does not work (frames stay black), so:
        auto white balance on, manual exposure at its maximum, then adjust
        ``analogue_gain`` from the mean frame brightness (0-255) until it is
        within ``tolerance`` of ``target``. Best-effort: a failure leaves the
        last gain in place. Returns the final mean brightness.
        """
        dev = self.sensor_subdev

        def set_ctrl(setting):
            subprocess.run(["v4l2-ctl", "-d", dev, "-c", setting], capture_output=True, check=False)

        set_ctrl("white_balance_automatic=1")
        set_ctrl("auto_exposure=1")   # menu value 1 = manual mode
        set_ctrl("gain_automatic=0")
        set_ctrl("exposure=500")      # the sensor's maximum

        gain, gain_min, gain_max = 128, 16, 1023
        mean = 0.0
        for _ in range(max_iterations):
            set_ctrl(f"analogue_gain={int(gain)}")
            frame = None
            # A gain change takes about 4 frames to reach the stream (more right after the
            # stream starts); frames before that still show the old brightness, so read
            # well past it, or an old in-range frame is mistaken for convergence.
            for _ in range(10):
                ok, grabbed = self._cap.read()
                if ok and grabbed is not None:
                    frame = grabbed
            if frame is None:
                break
            mean = float(frame.mean())
            if abs(mean - target) <= tolerance or (mean < target and gain >= gain_max) \
                    or (mean > target and gain <= gain_min):
                break
            gain = min(gain_max, max(gain_min, gain * min(4.0, max(0.5, target / max(mean, 1.0)))))
        return mean

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
