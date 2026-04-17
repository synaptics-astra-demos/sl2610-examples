#!/usr/bin/env python3
"""
Standalone Board Object Detection Script (YOLOv8)
Usage: python3 object_detection_video.py --model model.vmfb --video input.mp4 [--labels labels.json]
"""
import argparse
from collections import deque
import json
import numpy as np
import os
import subprocess
import sys
import threading
import time
import torq.runtime as torq_rt

MAX_DETECTIONS_TO_KEEP = 60

# ==========================================
# Helpers (Ported from helpers/yolo.py)
# ==========================================

def preprocess_frame_cv(bgr_frame, target_size=(320, 320)):
    """
    OpenCV/NumPy version of preprocess_frame — accepts BGR numpy array directly.
    Avoids PIL and the BGR->RGB conversion, saving ~15-20ms per frame.
    Returns: quantized input (1, 320, 320, 3), pad_info, orig_shape
    """
    import cv2
    new_w, new_h = target_size
    h, w = bgr_frame.shape[:2]

    # Scale ratio (new / old)
    r = min(new_w / w, new_h / h)
    new_unpad = (int(round(w * r)), int(round(h * r)))

    # Resize
    resized = cv2.resize(bgr_frame, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Pad with grey (114, 114, 114)
    padded = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
    top  = int(round((new_h - new_unpad[1]) / 2 - 0.1))
    left = int(round((new_w - new_unpad[0]) / 2 - 0.1))
    padded[top:top + new_unpad[1], left:left + new_unpad[0]] = resized

    # BGR -> RGB, normalize, quantize
    rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
    in_scale = 0.003921568859368563
    in_zp    = -128
    input_data = np.clip(rgb / in_scale + in_zp, -128, 127).astype(np.int8)
    input_data = np.expand_dims(input_data, axis=0)  # (1, 320, 320, 3)

    pad_info = (top / new_h, left / new_w)
    return input_data, pad_info, (h, w)


def dequantize_out(y, out_scale, out_zp, int8=True):
    if int8:
        return (y.astype(np.float32) - out_zp) * out_scale
    return y

def nms_numpy(boxes, scores, iou_threshold):
    """
    Pure Numpy NMS
    boxes: (N, 4) in format [x1, y1, w, h] (top-left x, top-left y, width, height)
    """
    if len(boxes) == 0:
        return []

    # Convert to x1, y1, x2, y2 for NMS calculation
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def postprocess(outputs, orig_shape, pad_info, labels=None):
    # outputs: (1, 84, 2100) or similar
    # orig_shape: (h, w)
    # pad_info: (pad_h_ratio, pad_w_ratio)
    
    # Squeeze batch
    outputs = np.squeeze(outputs) # (84, 2100) usually for YOLOv8
    
    # Transpose to (2100, 84) -> (Num_Proposals, 4_coords + Classes)
    outputs = outputs.transpose() 
    
    # Extract boxes and scores
    if outputs.shape[1] < 5:
        print(f"Error: Output shape {outputs.shape} too small")
        return []
        
    boxes = outputs[:, :4]
    scores_data = outputs[:, 4:]
    
    # Get max score and class ID for each proposal
    class_ids = np.argmax(scores_data, axis=1)
    scores = np.max(scores_data, axis=1)
    
    # Filter by confidence
    CONF_THRESH = 0.25
    mask = scores > CONF_THRESH
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return []

    # Prepare for NMS
    
    # Correct Logic for Normalized Output [0,1]
    
    # 1. Adjust for Padding (in normalized space)
    # pad_info = (pad_top_ratio, pad_left_ratio)
    boxes[:, 0] -= pad_info[1] # x - left_pad
    boxes[:, 1] -= pad_info[0] # y - top_pad
    
    # 2. Scale to Original Image Pixels
    
    max_dim = max(orig_shape)
    boxes[:, :4] *= max_dim
    
    # 3. Convert Center-WH to TopLeft-WH
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    
    # NMS
    IOU_THRESH = 0.45
    indices = nms_numpy(boxes, scores, IOU_THRESH)
    
    results = []
    for i in indices[:10]: # Top 10
        cls_id = class_ids[i]
        label = labels.get(str(cls_id), f"Class {cls_id}") if labels else f"Class {cls_id}"
        results.append((label, scores[i], boxes[i]))
        
    return results

def run_inference_torq(runner, input_data):
    outputs = runner.infer([input_data])  # <-- wrap in list
    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 1:
            return outputs[0]
        raise RuntimeError(
            f"Expected a single output tensor from the model, but got {len(outputs)} "
            "outputs. This script currently supports only single-output models. "
            "Please update the code to select the desired output tensor."
        )
    # If the runtime already returns a single tensor (e.g., a NumPy array), return it as-is.
    return outputs

def find_working_camera():
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            lines = result.stdout.splitlines()
            in_usb_device = False
            for line in lines:
                if "usb-" in line.lower():
                    in_usb_device = True
                    continue
                if in_usb_device and "/dev/video" in line:
                    device = line.strip()
                    if os.path.exists(device):
                        return device
                if line.strip() == "":
                    in_usb_device = False
    except Exception:
        pass

    for index in range(10):
        device = f"/dev/video{index}"
        if os.path.exists(device):
            return device
    return None


def resolve_camera_device(camera_device):
    if camera_device == "auto":
        resolved = find_working_camera()
        if resolved is None:
            raise RuntimeError("No USB camera device found")
        return resolved

    if not os.path.exists(camera_device):
        raise RuntimeError(f"Camera device not found: {camera_device}")
    return camera_device


def create_display_pipeline(Gst, width, height, fps, sink_name):
    pipeline_str = (
        "appsrc name=display_src format=time is-live=true block=true ! "
        f"video/x-raw,format=BGRA,width={width},height={height},framerate={fps}/1 ! "
        "synavideoconvertscale ! "
        f"{sink_name} sync=false"
    )
    pipeline = Gst.parse_launch(pipeline_str)
    appsrc = pipeline.get_by_name("display_src")
    return pipeline, appsrc


def push_display_frame(Gst, appsrc, frame_rgb, frame_index, fps):
    data = frame_rgb.tobytes()
    gst_buffer = Gst.Buffer.new_allocate(None, len(data), None)
    gst_buffer.fill(0, data)
    if fps > 0:
        frame_duration = Gst.SECOND // fps
        gst_buffer.pts = frame_index * frame_duration
        gst_buffer.duration = frame_duration
    return appsrc.emit("push-buffer", gst_buffer)


class RotatingJsonArrayWriter:
    def __init__(self, path, max_entries):
        self.path = path
        self.max_entries = max_entries
        self.rotated_path = self._build_rotated_path(path)
        self.file = None
        self.first_entry = True
        self.current_entries = 0
        self._open_new_file()

    @staticmethod
    def _build_rotated_path(path):
        base, ext = os.path.splitext(path)
        return f"{base}.1{ext or '.json'}"

    def _open_new_file(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.file = open(self.path, "w", encoding="utf-8")
        self.file.write("[\n")
        self.file.flush()
        self.first_entry = True
        self.current_entries = 0

    def _close_current_file(self):
        if self.file is None:
            return
        if not self.first_entry:
            self.file.write("\n")
        self.file.write("]\n")
        self.file.flush()
        self.file.close()
        self.file = None

    def _rotate(self):
        self._close_current_file()
        if os.path.exists(self.rotated_path):
            os.remove(self.rotated_path)
        if os.path.exists(self.path):
            os.replace(self.path, self.rotated_path)
        self._open_new_file()

    def append(self, record):
        if self.current_entries >= self.max_entries:
            self._rotate()

        prefix = "" if self.first_entry else ",\n"
        self.file.write(prefix)
        self.file.write(json.dumps(record, separators=(",", ":")))
        self.file.flush()
        self.first_entry = False
        self.current_entries += 1

    def close(self):
        self._close_current_file()

class FrameGrabber:
    """
    Daemon thread that continuously reads from a VideoCapture and keeps only
    the most recent frame.  The inference loop calls grabber.read() which
    returns immediately with the latest frame, so slow inference never
    accumulates a backlog of stale frames.
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
        """Return (ok, frame_copy) — frame_copy is None if no frame yet."""
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=1.0)


def shutdown_display_pipeline(Gst, pipeline, appsrc):
    if pipeline is None:
        return

    if appsrc is not None:
        try:
            appsrc.emit("end-of-stream")
        except Exception:
            pass

    bus = pipeline.get_bus()
    if bus is not None:
        bus.timed_pop_filtered(
            Gst.SECOND,
            Gst.MessageType.EOS | Gst.MessageType.ERROR,
        )

    pipeline.set_state(Gst.State.READY)
    pipeline.get_state(Gst.SECOND)
    pipeline.set_state(Gst.State.NULL)
    pipeline.get_state(Gst.SECOND)


def run_with_opencv(args, runner, labels):
    """OpenCV capture + preprocess main loop with optional GStreamer display."""
    import cv2

    Gst = None
    display_pipeline = None
    display_appsrc = None
    display_fps = 0
    if args.display:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst as _Gst
        _Gst.init(None)
        Gst = _Gst

    if args.display and Gst is None:
        raise RuntimeError("Failed to initialize GStreamer display")

    # -- open source ----------------------------------------------------------
    if args.rtsp_url:
        cap = cv2.VideoCapture(args.rtsp_url)
        source_desc = args.rtsp_url
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        source_desc = args.video
    else:
        # camera_device may be a path ("/dev/video0") or an integer index
        dev = args.camera_device
        try:
            cam_index = int(dev)
        except (ValueError, TypeError):
            cam_index = dev
        # Force V4L2 backend to avoid spurious GStreamer warnings
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        if args.camera_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        if args.camera_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        if args.camera_fps:
            cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
        source_desc = dev

    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source_desc}")
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or args.camera_fps or 15
    out_fps = int(src_fps) if src_fps > 0 else 15
    display_fps = out_fps if out_fps > 0 else 15

    # -- optional output writer ------------------------------------------------
    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(args.output, fourcc, out_fps, (width, height))

    all_detections = deque(maxlen=MAX_DETECTIONS_TO_KEEP)
    last_detection_labels = []
    json_writer = RotatingJsonArrayWriter(args.json_results, MAX_DETECTIONS_TO_KEEP)

    # Start background grabber for camera sources so slow inference never
    # reads stale buffered frames. For video files use cap.read() directly
    grabber = None
    if args.camera_device or args.rtsp_url:
        grabber = FrameGrabber(cap)

    print(f"Processing {source_desc} with Torq Python runtime... Press Ctrl+C to stop.")
    frame_count = 0

    try:
        while True:
            # -- pull frame ---------------------------------------------------
            if grabber is not None:
                # Wait until the grabber has at least one frame
                while True:
                    ret, bgr_frame = grabber.read()
                    if ret:
                        break
                    time.sleep(0.005)
            else:
                ret, bgr_frame = cap.read()
            if not ret or bgr_frame is None:
                break

            # -- preprocess ---------------------------------------------------
            input_data, pad_info, orig_shape = preprocess_frame_cv(bgr_frame)

            # -- inference ----------------------------------------------------
            raw_out = run_inference_torq(runner, input_data)

            # -- postprocess --------------------------------------------------
            out_scale = 0.004194467328488827
            out_zp = -128
            outputs = dequantize_out(raw_out, out_scale, out_zp, int8=True)
            detections = postprocess(outputs, orig_shape, pad_info, labels)
            
            detection_labels = [d[0] for d in detections]

            objects_changed = False
            if set(detection_labels) != set(last_detection_labels):
                objects_changed = True

            if objects_changed:
                # jump to new line
                print(f"\n", end="", flush=True)
            else:        
                # clear existing line
                print("\r" + " " * 50 + "\r", end="", flush=True)
            # print frame count
            print(f"{frame_count} ", end="", flush=True)
            # print objects
            for label, conf, box in detections:
                print(f" {label} {conf:.2f}", end="", flush=True)
            last_detection_labels = detection_labels

            # -- draw (OpenCV) ------------------------------------------------
            annotated = bgr_frame.copy()
            frame_detections = []
            for label, conf, box in detections:
                x1, y1, w_box, h_box = [float(v) for v in box]
                x2, y2 = x1 + w_box, y1 + h_box
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                text = f"{label} {conf:.2f}"
                text_y = int(y1) - 8 if int(y1) - 8 > 10 else int(y1) + 18
                cv2.putText(annotated, text, (int(x1), text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                frame_detections.append({
                    "label": label,
                    "confidence": float(conf),
                    "bounding_box": {
                        "origin": {"x": int(round(x1)), "y": int(round(y1))},
                        "size":   {"x": int(round(w_box)), "y": int(round(h_box))}
                    }
                })

            # -- JSON write ---------------------------------------------------
            frame_result = {"frame": frame_count, "detections": frame_detections}
            all_detections.append(frame_result)
            json_writer.append(frame_result)

            # -- display / write ----------------------------------------------
            if args.display:
                assert Gst is not None
                if display_pipeline is None:
                    display_pipeline, display_appsrc = create_display_pipeline(
                        Gst,
                        width,
                        height,
                        display_fps,
                        args.display_sink,
                    )
                    display_pipeline.set_state(Gst.State.PLAYING)

                rendered_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2BGRA)
                ret = push_display_frame(
                    Gst,
                    display_appsrc,
                    rendered_frame,
                    frame_count,
                    display_fps,
                )
                if ret != Gst.FlowReturn.OK:
                    print(f"Warning: failed to display frame: {ret}")

            if out_writer is not None:
                out_writer.write(annotated)

            frame_count += 1

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if grabber is not None:
            grabber.stop()
        cap.release()
        if out_writer is not None:
            out_writer.release()
        if args.display:
            assert Gst is not None
            shutdown_display_pipeline(Gst, display_pipeline, display_appsrc)
        json_writer.close()
        print(f"Done. Processed {frame_count} frames. Output: {args.output if args.output else 'not saved'}")
        print(
            f"Detection results saved to: {args.json_results} "
            f"(previous file: {json_writer.rotated_path if os.path.exists(json_writer.rotated_path) else 'none'})"
        )
        print(f"Kept the last {len(all_detections)} detections in memory.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--rtsp-url", help="RTSP stream URL")
    source_group.add_argument("--video", help="Path to video file")
    source_group.add_argument(
        "--camera-device",
        help="USB camera device, for example /dev/video0, or 'auto'",
    )
    parser.add_argument("--labels")
    parser.add_argument("--device", default="torq")
    parser.add_argument("--output", default=None, help="Output video file (optional)")
    parser.add_argument("--json-results", default="detection_results.json", help="Output JSON file for detections")
    parser.add_argument("--camera-width", type=int, default=640, help="USB camera width")
    parser.add_argument("--camera-height", type=int, default=480, help="USB camera height")
    parser.add_argument("--camera-fps", type=int, default=30, help="USB camera frame rate")
    parser.add_argument("--display", action="store_true", help="Display annotated frames live")
    parser.add_argument("--display-sink", default="waylandsink", help="GStreamer video sink for live display")
    args = parser.parse_args()

    # 0. Load the model with Torq Runtime
    runner = torq_rt.VMFBInferenceRunner(
        args.model,
        device_uri=args.device,
        function="main",
        load_model_to_mem=True
    )
    if args.camera_device:
        try:
            args.camera_device = resolve_camera_device(args.camera_device)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

    labels = {}
    if args.labels:
        with open(args.labels) as f:
            data = json.load(f)
            if "names" in data:
                labels = {str(k): v for k, v in data["names"].items()}
            else:
                labels = data

    run_with_opencv(args, runner, labels)

if __name__ == "__main__":
    main()

