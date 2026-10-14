# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Image/video preprocessing, detection post-processing, and frame/image annotation for YOLOv8.

Shared by run_cli.py's image mode (PIL) and video/camera/RTSP mode (OpenCV).
"""
import os

import numpy as np
from PIL import ImageDraw, ImageFont

# Quantization parameters — must match the compiled model. See the README's
# "Testing YOLOv8s alternative" section for how to override OUT_SCALE/OUT_ZP.
IN_SCALE = 0.003921568859368563
IN_ZP = -128
OUT_SCALE = 0.004194467328488827
OUT_ZP = -128

CONF_THRESH = 0.25
IOU_THRESH = 0.45


def preprocess_image(image_path, target_size=(320, 320)):
    """Letterbox-resize and quantize a still image. Returns (input, pad_info, orig_shape)."""
    from PIL import Image

    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        raise SystemExit(1)

    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    new_w, new_h = target_size
    r = min(new_w / w, new_h / h)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = (new_w - new_unpad[0]) / 2, (new_h - new_unpad[1]) / 2

    img_resized = img.resize(new_unpad, Image.BILINEAR)
    padded_img = Image.new("RGB", target_size, (114, 114, 114))
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    padded_img.paste(img_resized, (left, top))

    input_data = np.array(padded_img, dtype=np.float32) / 255.0
    input_data = np.clip(input_data / IN_SCALE + IN_ZP, -128, 127).astype(np.int8)
    input_data = np.expand_dims(input_data, axis=0)  # (1, 320, 320, 3)

    pad_info = (top / new_h, left / new_w)
    return input_data, pad_info, (h, w)


def preprocess_frame(bgr_frame, target_size=(320, 320)):
    """OpenCV/NumPy equivalent of preprocess_image() for a BGR frame straight off a capture device."""
    import cv2

    new_w, new_h = target_size
    h, w = bgr_frame.shape[:2]
    r = min(new_w / w, new_h / h)
    new_unpad = (int(round(w * r)), int(round(h * r)))

    resized = cv2.resize(bgr_frame, new_unpad, interpolation=cv2.INTER_LINEAR)
    padded = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
    top = int(round((new_h - new_unpad[1]) / 2 - 0.1))
    left = int(round((new_w - new_unpad[0]) / 2 - 0.1))
    padded[top:top + new_unpad[1], left:left + new_unpad[0]] = resized

    rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
    input_data = np.clip(rgb / IN_SCALE + IN_ZP, -128, 127).astype(np.int8)
    input_data = np.expand_dims(input_data, axis=0)  # (1, 320, 320, 3)

    pad_info = (top / new_h, left / new_w)
    return input_data, pad_info, (h, w)


def dequantize_out(y, out_scale=OUT_SCALE, out_zp=OUT_ZP):
    return (y.astype(np.float32) - out_zp) * out_scale


def nms(boxes, scores, iou_threshold=IOU_THRESH):
    """Pure-NumPy non-max suppression. boxes: (N, 4) as [x1, y1, w, h]."""
    if len(boxes) == 0:
        return []

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


def postprocess(outputs, orig_shape, pad_info, labels=None, conf_thresh=CONF_THRESH, iou_thresh=IOU_THRESH):
    """Decode raw model output into up to 10 (label, confidence, [x, y, w, h]) detections."""
    outputs = np.squeeze(outputs)  # (84, 2100) usually for YOLOv8
    outputs = outputs.transpose()  # (Num_Proposals, 4_coords + Classes)

    if outputs.shape[1] < 5:
        print(f"Error: Output shape {outputs.shape} too small")
        return []

    boxes = outputs[:, :4]
    scores_data = outputs[:, 4:]

    class_ids = np.argmax(scores_data, axis=1)
    scores = np.max(scores_data, axis=1)

    mask = scores > conf_thresh
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    # Undo letterbox padding (normalized space), then scale to original image pixels.
    boxes[:, 0] -= pad_info[1]  # x - left_pad
    boxes[:, 1] -= pad_info[0]  # y - top_pad
    max_dim = max(orig_shape)
    boxes[:, :4] *= max_dim

    # Center-WH -> TopLeft-WH
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2

    indices = nms(boxes, scores, iou_thresh)

    results = []
    for i in indices[:10]:  # Top 10
        cls_id = class_ids[i]
        label = labels.get(str(cls_id), f"Class {cls_id}") if labels else f"Class {cls_id}"
        results.append((label, scores[i], boxes[i]))

    return results


def _load_annotation_font(size=35):
    try:
        font_path = "/usr/share/fonts/ttf/LiberationSans-Regular.ttf"
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)
        for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            if os.path.exists(p):
                return ImageFont.truetype(p, size + 5)
    except Exception:
        pass
    return ImageFont.load_default()


def annotate_image(img, detections, scale=1.0, dx=0, dy=0):
    """Draw detection boxes/labels in place on a PIL image (already scaled/letterboxed via scale/dx/dy)."""
    draw = ImageDraw.Draw(img)
    font = _load_annotation_font()

    for label, conf, box in detections:
        x1 = box[0] * scale + dx
        y1 = box[1] * scale + dy
        x2 = x1 + box[2] * scale
        y2 = y1 + box[3] * scale

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        text = f"{label}"
        text_pos = [x1, y1 - 45]
        if text_pos[1] < 0:
            text_pos[1] = y1 + 5  # Move inside if clipped at top

        try:
            left, top, right, bottom = draw.textbbox((text_pos[0], text_pos[1]), text, font=font)
            draw.rectangle((left - 5, top - 5, right + 5, bottom + 5), fill="red")
        except AttributeError:
            draw.rectangle((text_pos[0], text_pos[1], text_pos[0] + len(text) * 20, text_pos[1] + 40), fill="red")

        draw.text((text_pos[0], text_pos[1]), text, fill="white", font=font)

    return img


def letterbox_frame(frame, target_size):
    """Maintain aspect ratio and pad a BGR frame with black to target_size. Returns (canvas, (top, left, w, h))."""
    import cv2

    h, w = frame.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas, (top, left, new_w, new_h)


def draw_ui(canvas, title, stats, video_rect):
    """Draw a title and FPS/NPU/detection-count stats into the letterbox bars around the video."""
    import cv2

    target_h, target_w = canvas.shape[:2]
    top, left, vw, vh = video_rect
    font = cv2.FONT_HERSHEY_SIMPLEX

    if top > 40:
        (tw, th), _ = cv2.getTextSize(title, font, 1.1, 2)
        tx = (target_w - tw) // 2
        ty = (top // 2) + (th // 2)
        cv2.putText(canvas, title, (tx + 1, ty + 1), font, 1.1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, title, (tx, ty), font, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    bot_y_start = top + vh
    if target_h - bot_y_start > 60:
        y_cursor = bot_y_start + 40
        cv2.putText(canvas, f"FPS: {stats['fps']:.1f}", (30, y_cursor),
                    font, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"NPU: {stats['npu']:.1f} ms", (30, y_cursor + 35),
                    font, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
        count_text = f"DETECTIONS: {stats['count']}"
        (cw, ch), _ = cv2.getTextSize(count_text, font, 0.7, 2)
        cv2.putText(canvas, count_text, (target_w - cw - 30, y_cursor + 15),
                    font, 0.7, (0, 255, 100), 2, cv2.LINE_AA)


def draw_detections(frame, detections):
    """Draw detection boxes/labels on a copy of a BGR frame. Returns the annotated copy."""
    import cv2

    annotated = frame.copy()
    for label, conf, box in detections:
        x1, y1, w_box, h_box = [float(v) for v in box]
        x2, y2 = x1 + w_box, y1 + h_box
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        text = f"{label} {conf:.2f}"
        text_y = int(y1) - 8 if int(y1) - 8 > 10 else int(y1) + 18
        cv2.putText(annotated, text, (int(x1), text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return annotated
