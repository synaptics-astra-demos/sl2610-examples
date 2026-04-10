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
from PIL import Image, ImageDraw, ImageFont
import torq.runtime as torq_rt

MAX_DETECTIONS_TO_KEEP = 60

# ==========================================
# Helpers (Ported from helpers/yolo.py)
# ==========================================

def preprocess_frame(frame_rgb, target_size=(320, 320)):
    """
    Preprocess frame (numpy array HWC RGB) for inference
    Returns: quantized input (1, 320, 320, 3), pad_info, orig_shape
    """
    img = Image.fromarray(frame_rgb)
    w, h = img.size
    new_w, new_h = target_size
    
    # Scale ratio (new / old)
    r = min(new_w / w, new_h / h)
    
    # Compute padding
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = (new_w - new_unpad[0]) / 2, (new_h - new_unpad[1]) / 2

    # Resize
    img_resized = img.resize(new_unpad, Image.BILINEAR)
    
    # Pad
    # Create a new image with grey background (114, 114, 114)
    padded_img = Image.new("RGB", target_size, (114, 114, 114))
    
    # Paste resized image at center
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    padded_img.paste(img_resized, (left, top))
    
    # Preprocessing for Model
    # Normalize to [0, 1] and add batch dimension
    input_data = np.array(padded_img, dtype=np.float32)
    input_data /= 255.0
    
    # === QUANTIZE INPUT (Float32 -> Int8) ===

    # Must match the models quantization parameters
    in_scale = 0.003921568859368563
    in_zp = -128
    
    # Quantize: (float / scale) + zp
    input_data = (input_data / in_scale + in_zp)
    input_data = np.clip(input_data, -128, 127) # Ensure range
    input_data = input_data.astype(np.int8) 
    # ========================================

    input_data = np.expand_dims(input_data, axis=0) # (1, 320, 320, 3)
    
    pad_info = (top / new_h, left / new_w) # (dh_ratio, dw_ratio)
    
    return input_data, pad_info, (h, w) # Return original (h, w)

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


def build_input_pipeline(args):
    if args.video:
        return (
            f"filesrc location={args.video} ! qtdemux name=demux demux.video_0 !  h264parse ! avdec_h264 ! synavideoconvertscale !"
            "video/x-raw,format=RGB, width=640, height=480 ! appsink name=sink emit-signals=true max-buffers=1"
        )

    return (
        f"v4l2src device={args.camera_device} ! "
        f"video/x-raw,width={args.camera_width},height={args.camera_height},framerate={args.camera_fps}/1 ! "
        "synavideoconvertscale ! video/x-raw,format=RGB, width=640,height=480 ! "
        "appsink name=sink emit-signals=true max-buffers=1 drop=true"
    )


def create_display_pipeline(Gst, width, height, fps, sink_name):
    pipeline_str = (
        "appsrc name=display_src format=time is-live=true block=true ! "
        f"video/x-raw,format=RGB,width={width},height={height},framerate={fps}/1 ! "
        "videoconvert ! "
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

def main():
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    Gst.init(None)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
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

    pipeline_str = build_input_pipeline(args)
    pipeline = Gst.parse_launch(pipeline_str)
    appsink = pipeline.get_by_name('sink')
    display_pipeline = None
    display_appsrc = None

    cv2 = None
    fourcc = None
    out_writer = None
    if args.output:
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        except ImportError:
            cv2 = None

    all_detections = deque(maxlen=MAX_DETECTIONS_TO_KEEP)
    last_detections = []
    json_writer = RotatingJsonArrayWriter(args.json_results, MAX_DETECTIONS_TO_KEEP)

    source_desc = args.video if args.video else args.camera_device
    print(f"Processing {source_desc} with Torq Python runtime... Press Ctrl+C to stop.")
    try:
        pipeline.set_state(Gst.State.PLAYING)
        frame_count = 0
        try:
            while True:
                sample = appsink.emit('pull-sample')
                if sample is None:
                    break
                buf = sample.get_buffer()
                caps = sample.get_caps()
                structure = caps.get_structure(0)
                width = structure.get_value('width')
                height = structure.get_value('height')
                success, map_info = buf.map(Gst.MapFlags.READ)
                if not success:
                    break
                frame_data = np.frombuffer(map_info.data, dtype=np.uint8)
                frame_rgb = frame_data.reshape((height, width, 3)).copy()
                buf.unmap(map_info)
                # Preprocess and run inference
                input_data, pad_info, orig_shape = preprocess_frame(frame_rgb)
                raw_out = run_inference_torq(runner, input_data)
                out_scale = 0.004194467328488827
                out_zp = -128
                outputs = dequantize_out(raw_out, out_scale, out_zp, int8=True)
                detections = postprocess(outputs, orig_shape, pad_info, labels)
                
                for label, conf, _ in detections:
                    # compare the object labels with last detections
                    if label not in [d[0] for d in last_detections]:
                        # new object: commit it on its own line
                        print(f"\n{frame_count} {label} {conf:.2f}", flush=True)
                    else:
                        # existing object: overwrite the current line
                        print("\r" + " " * 40 + "\r" + f"{frame_count} {label} {conf:.2f}", end="", flush=True)
                    
                last_detections = detections


                # Draw detections
                img = Image.fromarray(frame_rgb)
                draw = ImageDraw.Draw(img)
                if frame_count == 0:
                    try:
                        label_font = ImageFont.truetype("/usr/share/fonts/ttf/LiberationSans-Regular.ttf", 25)
                    except OSError:
                        label_font = ImageFont.load_default()
                frame_detections = []
                for label, conf, box in detections:
                    x1, y1, w_box, h_box = [float(x) for x in box]
                    x2 = x1 + w_box
                    y2 = y1 + h_box
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    text = f"{label}"
                    text_pos = [x1, y1 - 25]
                    if text_pos[1] < 0: text_pos[1] = y1 + 5
                    draw.text((text_pos[0], text_pos[1]), text, fill="red", font=label_font)
                    frame_detections.append({
                        "label": label,
                        "confidence": float(conf),
                        "bounding_box": {
                            "origin": {"x": int(round(x1)), "y": int(round(y1))},
                            "size": {"x": int(round(w_box)), "y": int(round(h_box))}
                        }
                    })
                frame_result = {
                    "frame": frame_count,
                    "detections": frame_detections
                }
                all_detections.append(frame_result)
                json_writer.append(frame_result)
                rendered_frame = np.array(img)
                if args.display:
                    if display_pipeline is None:
                        display_fps = args.camera_fps if args.camera_device else 15
                        display_pipeline, display_appsrc = create_display_pipeline(
                            Gst,
                            width,
                            height,
                            display_fps,
                            args.display_sink,
                        )
                        display_pipeline.set_state(Gst.State.PLAYING)
                    ret = push_display_frame(
                        Gst,
                        display_appsrc,
                        rendered_frame,
                        frame_count,
                        args.camera_fps if args.camera_device else 15,
                    )
                    if ret != Gst.FlowReturn.OK:
                        print(f"Warning: failed to display frame: {ret}")
                if args.output and cv2:
                    if out_writer is None:
                        out_writer = cv2.VideoWriter(args.output, fourcc, 15, (width, height))
                    out_writer.write(cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
                frame_count += 1
        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            pipeline.set_state(Gst.State.NULL)
            if display_pipeline is not None:
                if display_appsrc is not None:
                    display_appsrc.emit("end-of-stream")
                display_pipeline.set_state(Gst.State.NULL)
            if out_writer:
                out_writer.release()
            json_writer.close()
            print(f"Done. Processed {frame_count} frames. Output: {args.output if args.output else 'not saved'}")
            print(
                f"Detection results saved to: {args.json_results} "
                f"(previous file: {json_writer.rotated_path if os.path.exists(json_writer.rotated_path) else 'none'})"
            )
            print(f"Kept the last {len(all_detections)} detections in memory.")
    except Exception as e:
        json_writer.close()
        print(f"Torq Python runtime inference failed while processing {source_desc}: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
