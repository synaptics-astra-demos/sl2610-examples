#!/usr/bin/env python3
"""
Standalone Board Object Detection Script (YOLOv8)

Usage:
    python3 run_cli.py --image image.jpg [--labels labels.json]
    python3 run_cli.py --video input.mp4
    python3 run_cli.py --camera-device auto --display
    python3 run_cli.py --rtsp-url rtsp://<camera_ip>:<port>/<stream_path>
"""
import argparse
from collections import deque
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_utils.demo_utils import DependencyChecker, IncompatibleVersionError, MissingRequirementsError

try:
    DependencyChecker(
        os.path.join(os.path.dirname(__file__), "requirements.txt"),
        version_map={"torq.runtime": ">=2.0.0a1"},
    ).run()
except (MissingRequirementsError, IncompatibleVersionError) as e:
    print(f"{e}\nRun: python setup_demo.py", file=sys.stderr)
    sys.exit(1)

from PIL import Image

import results
import vision
from app_utils.camera import FrameGrabber, resolve_camera_device, rotate_frame, set_camera_controls
from app_utils.display import LiveDisplay, enable_wayland_display, fit_to_display, show_image
from app_utils.npu import cleanup_npu_after_inference, setup_npu_for_inference
from app_utils.torq_examples.utils.inference import SimpleVMFBInferenceRunner

MAX_DETECTIONS_TO_KEEP = 60


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 object detection on an image, video file, USB camera, or RTSP stream."
    )
    parser.add_argument("--model", default="../models/Synaptics/yolov8-od-nano-320-int8-torq/yolo_od.vmfb")
    parser.add_argument("--labels", default="labels.json")
    parser.add_argument("--device", default="torq")
    parser.add_argument("--display", action="store_true", help="Display annotated output")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", help="Path to a single image file")
    source_group.add_argument("--video", help="Path to a video file")
    source_group.add_argument("--camera-device", help="USB camera device, e.g. /dev/video0, or 'auto'")
    source_group.add_argument("--rtsp-url", help="RTSP stream URL")

    image_group = parser.add_argument_group("Image mode options (--image)")
    image_group.add_argument(
        "--save-image", action="store_true",
        help="Save the annotated output image (implied by --display)",
    )

    video_group = parser.add_argument_group("Video/camera/RTSP mode options")
    video_group.add_argument("--output", default=None, help="Output video file (optional)")
    video_group.add_argument("--json-results", default="detection_results.json", help="Output JSON file for detections")
    video_group.add_argument("--camera-width", type=int, default=640, help="USB camera width")
    video_group.add_argument("--camera-height", type=int, default=480, help="USB camera height")
    video_group.add_argument("--camera-fps", type=int, default=30, help="USB camera frame rate")
    video_group.add_argument("--display-sink", default="waylandsink", help="GStreamer video sink for live display")
    video_group.add_argument(
        "--rotate", type=int, choices=[0, 90, 180, 270], default=0,
        help="Rotate camera feed (degrees clockwise, default: %(default)s)",
    )
    video_group.add_argument("--flip", action="store_true", help="Flip video frames vertically")

    cam_group = parser.add_argument_group("Camera Config")
    cam_group.add_argument("--camera-control-device", help="V4L2 device for controls (e.g. /dev/v4l-subdev2)")
    cam_group.add_argument("--brightness", type=int, help="V4L2 brightness")
    cam_group.add_argument("--contrast", type=int, help="V4L2 contrast")
    cam_group.add_argument("--saturation", type=int, help="V4L2 saturation")
    cam_group.add_argument("--sharpness", type=int, help="V4L2 sharpness")
    cam_group.add_argument("--gain", type=int, help="V4L2 gain")
    cam_group.add_argument("--exposure-auto", type=int, help="V4L2 auto exposure")
    cam_group.add_argument("--exposure-absolute", type=int, help="V4L2 absolute exposure time")

    return parser.parse_args()


def run_image_mode(args, runner, labels):
    print("\n[1/4] Preprocessing...")
    input_data, pad_info, orig_shape = vision.preprocess_image(args.image)

    print("\n[2/4] Inference...")
    try:
        raw_out = runner.infer(input_data)
    except Exception as e:
        print(f"Inference failed: {e}")
        sys.exit(1)
    print(f"Time: {runner.infer_time_ms:.3f}ms")

    print("\n[3/4] Processing...")
    if raw_out.shape != (1, 84, 2100):
        print(f"Warning: Output shape {raw_out.shape} doesn't match expected (1, 84, 2100). Metadata might be needed.")
    outputs = vision.dequantize_out(raw_out)
    detections = vision.postprocess(outputs, orig_shape, pad_info, labels)

    results.log_image_detections(detections)

    if (args.save_image or args.display) and detections:
        print("\n[5/5] Saving result image...")
        try:
            img, scale, dx, dy = fit_to_display(Image.open(args.image))
            print(f"Resized image to {img.width}x{img.height} (letterboxed).")
            vision.annotate_image(img, detections, scale, dx, dy)
            out_path = results.save_result_image(img, "output_yolo.jpg")
            if args.display:
                show_image(out_path, width=img.width, height=img.height)
        except Exception as e:
            print(f"Failed to save result image: {e}")


def _open_capture(args):
    """Open the configured video source. Returns (cap, source_desc)."""
    import cv2

    if args.rtsp_url:
        return cv2.VideoCapture(args.rtsp_url), args.rtsp_url
    if args.video:
        return cv2.VideoCapture(args.video), args.video

    # camera_device may be a path ("/dev/video0") or an integer index
    try:
        cam_index = int(args.camera_device)
    except (ValueError, TypeError):
        cam_index = args.camera_device
    # Force V4L2 backend to avoid spurious GStreamer warnings
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if args.camera_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    if args.camera_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    if args.camera_fps:
        cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    return cap, args.camera_device


def run_video_mode(args, runner, labels):
    import cv2

    cap, source_desc = _open_capture(args)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source_desc}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or args.camera_fps or 15
    out_fps = int(src_fps) if src_fps > 0 else 15
    display_fps = out_fps if out_fps > 0 else 15

    orientation = os.environ.get("ORIENTATION", "landscape")
    disp_w, disp_h = (480, 800) if orientation == "portrait" else (800, 480)

    live = LiveDisplay(disp_w, disp_h, fps=display_fps, sink_name=args.display_sink) if args.display else None

    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(args.output, fourcc, out_fps, (width, height))

    all_detections = deque(maxlen=MAX_DETECTIONS_TO_KEEP)
    last_labels = []
    json_writer = results.RotatingJsonArrayWriter(args.json_results, MAX_DETECTIONS_TO_KEEP)

    # Background grabber for camera/RTSP sources so slow inference never reads
    # stale buffered frames. Video files are read directly via cap.read().
    grabber = FrameGrabber(cap) if (args.camera_device or args.rtsp_url) else None

    print(f"Processing {source_desc} with Torq Python runtime... Press Ctrl+C to stop.")
    frame_count = 0
    fps = 0.0
    fps_time = time.time()

    try:
        while True:
            if grabber is not None:
                while True:
                    ret, bgr_frame = grabber.read()
                    if ret:
                        break
                    time.sleep(0.005)
            else:
                ret, bgr_frame = cap.read()
            if not ret or bgr_frame is None:
                break

            bgr_frame = rotate_frame(bgr_frame, args.rotate)
            if args.flip:
                bgr_frame = cv2.flip(bgr_frame, 0)

            input_data, pad_info, orig_shape = vision.preprocess_frame(bgr_frame)
            raw_out = runner.infer(input_data)
            infer_time = runner.infer_time_ms

            outputs = vision.dequantize_out(raw_out)
            detections = vision.postprocess(outputs, orig_shape, pad_info, labels)

            last_labels = results.log_video_frame(frame_count, infer_time, detections, last_labels)

            annotated = vision.draw_detections(bgr_frame, detections)

            frame_result = {
                "frame": frame_count,
                "detections": [results.detection_to_record(*d) for d in detections],
            }
            all_detections.append(frame_result)
            json_writer.append(frame_result)

            if live is not None:
                # letterbox_frame keeps the aspect ratio correct on portrait displays.
                display_frame, video_rect = vision.letterbox_frame(annotated, (disp_w, disp_h))
                stats = {"fps": fps, "npu": infer_time, "count": len(detections)}
                vision.draw_ui(display_frame, "Object Detection", stats, video_rect)
                live.push(display_frame)

            if out_writer is not None:
                out_writer.write(annotated)

            frame_count += 1
            if frame_count % 10 == 0:
                now = time.time()
                fps = 10.0 / (now - fps_time)
                fps_time = now

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if grabber is not None:
            grabber.stop()
        cap.release()
        if out_writer is not None:
            out_writer.release()
        if live is not None:
            live.close()
        json_writer.close()
        results.log_video_summary(frame_count, args.output, args.json_results, json_writer, all_detections)


def main():
    args = parse_args()

    setup_npu_for_inference()
    if args.display:
        enable_wayland_display()

    try:
        runner = SimpleVMFBInferenceRunner(
            args.model, device_uri=args.device, function="main", load_model_to_mem=True,
            device_io=True,  # the default dmabuf allocator needs device-backed I/O
        )
        labels = results.load_labels(args.labels)

        if args.image:
            run_image_mode(args, runner, labels)
            return

        if args.camera_device and args.camera_device != "auto":
            ctrl_device = args.camera_control_device or args.camera_device
            set_camera_controls(ctrl_device, {
                "brightness": args.brightness,
                "contrast": args.contrast,
                "saturation": args.saturation,
                "sharpness": args.sharpness,
                "gain": args.gain,
                "exposure_auto": args.exposure_auto,
                "exposure_absolute": args.exposure_absolute,
            })
        if args.camera_device:
            try:
                args.camera_device = resolve_camera_device(args.camera_device)
            except RuntimeError as exc:
                print(f"ERROR: {exc}")
                sys.exit(1)

        run_video_mode(args, runner, labels)
    finally:
        cleanup_npu_after_inference()


if __name__ == "__main__":
    main()
