# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Show a saved JPEG on the board's Wayland display via GStreamer."""

from __future__ import annotations

import os
import shutil
import subprocess
import time

from PIL import Image


def enable_wayland_display() -> None:
    """Set the env vars GStreamer's waylandsink needs. Call once before showing anything."""
    os.environ["XDG_RUNTIME_DIR"] = "/var/run/user/0"
    os.environ["WAYLAND_DISPLAY"] = "wayland-1"


def display_resolution() -> tuple[int, int]:
    """(width, height) from the ORIENTATION/DISPLAY_WIDTH/DISPLAY_HEIGHT env vars."""
    orientation = os.environ.get("ORIENTATION", "landscape")
    width = int(os.environ.get("DISPLAY_WIDTH", 800 if orientation == "landscape" else 480))
    height = int(os.environ.get("DISPLAY_HEIGHT", 480 if orientation == "landscape" else 800))
    return width, height


def fit_to_display(img: Image.Image, width: int | None = None, height: int | None = None,
                    fill: tuple[int, int, int] = (0, 0, 0)):
    """Letterbox ``img`` onto a ``width``x``height`` canvas (default: ``display_resolution()``).

    Preserves aspect ratio, centering the resized image on a ``fill``-colored
    canvas - this holds in portrait mode too, where the display is narrower
    than most source images. Returns ``(canvas, scale, dx, dy)``; ``scale``/
    ``dx``/``dy`` let callers remap coordinates (e.g. detection boxes) from
    the original image onto the canvas.
    """
    box_w, box_h = (width, height) if width and height else display_resolution()
    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    scale = min(box_w / w, box_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR) if (new_w, new_h) != (w, h) else img

    canvas = Image.new("RGB", (box_w, box_h), fill)
    dx, dy = (box_w - new_w) // 2, (box_h - new_h) // 2
    canvas.paste(resized, (dx, dy))
    return canvas, scale, dx, dy


def show_image(image_path, seconds: float = 5, width: int | None = None, height: int | None = None) -> None:
    """Show a JPEG on the Wayland display for ``seconds`` via gst-launch-1.0, then close it.

    Assumes ``image_path`` is already sized to fit the display (see
    ``fit_to_display``); it is not itself aspect-ratio aware.
    """
    if not shutil.which("gst-launch-1.0"):
        print("gst-launch-1.0 not found; cannot display image.")
        return

    disp_w, disp_h = (width, height) if width and height else display_resolution()
    print(f"Displaying {image_path} at {disp_w}x{disp_h} for {seconds}s...")

    cmd = [
        "gst-launch-1.0",
        "filesrc", f"location={image_path}", "!",
        "jpegdec", "!",
        "videoconvert", "!",
        "imagefreeze", "!",
        "videoscale", "!",
        f"video/x-raw,width={disp_w},height={disp_h}", "!",
        "waylandsink",
    ]
    try:
        proc = subprocess.Popen(cmd)
    except Exception as exc:
        print(f"Failed to launch gst-launch-1.0: {exc}")
        return

    try:
        time.sleep(seconds)
    except KeyboardInterrupt:
        pass
    proc.terminate()
    proc.wait()
    print("Display closed.")


class LiveDisplay:
    """Push a live stream of BGR frames to the Wayland display via GStreamer ``appsrc``.

    Unlike ``show_image()`` (which relaunches ``gst-launch-1.0`` per still), this
    keeps one pipeline open so frames stream continuously - use it for a camera
    preview. Pushing the same frame once and pausing (rather than pushing again)
    holds it on screen, since the sink just keeps displaying the last buffer it
    received. Requires PyGObject (``gi.repository.Gst``).
    """

    def __init__(self, width: int, height: int, fps: int = 15, sink_name: str = "waylandsink"):
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        self._Gst = Gst
        self._fps = fps
        self._frame_index = 0

        pipeline_str = (
            "appsrc name=src format=time is-live=true block=true ! "
            f"video/x-raw,format=BGRA,width={width},height={height},framerate={fps}/1 ! "
            f"{sink_name} sync=false"
        )
        self._pipeline = Gst.parse_launch(pipeline_str)
        self._appsrc = self._pipeline.get_by_name("src")
        self._pipeline.set_state(Gst.State.PLAYING)

    def push(self, bgr_frame) -> None:
        """Push one BGR frame, already sized to the pipeline's width/height."""
        import cv2

        Gst = self._Gst
        data = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2BGRA).tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        if self._fps > 0:
            duration = Gst.SECOND // self._fps
            buf.pts = self._frame_index * duration
            buf.duration = duration
        self._frame_index += 1

        ret = self._appsrc.emit("push-buffer", buf)
        if ret != Gst.FlowReturn.OK:
            print(f"Warning: failed to display frame: {ret}")

    def close(self) -> None:
        Gst = self._Gst
        try:
            self._appsrc.emit("end-of-stream")
        except Exception:
            pass

        bus = self._pipeline.get_bus()
        if bus is not None:
            bus.timed_pop_filtered(Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        self._pipeline.set_state(Gst.State.READY)
        self._pipeline.get_state(Gst.SECOND)
        self._pipeline.set_state(Gst.State.NULL)
        self._pipeline.get_state(Gst.SECOND)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
