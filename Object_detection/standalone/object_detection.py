#!/usr/bin/env python3
"""
Standalone Board Object Detection Script (YOLOv8)
Usage: python3 board_job_yolo.py --model model.vmfb --image image.jpg [--labels labels.json]
"""
import argparse
import json
import numpy as np
import os
import subprocess
import time
import sys
import shutil
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# Helpers (Ported from helpers/yolo.py)
# ==========================================

def preprocess_image(image_path, target_size=(320, 320)):
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        sys.exit(1)
        
    # Convert to RGB if needed
    if img.mode != "RGB":
        img = img.convert("RGB")

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

def run_inference(model_path, input_npy_path, output_bin_path, device="torq"):
    executable = "iree-run-module"
    cmd = [
        executable,
        f"--device={device}",
        f"--module={model_path}",
        "--function=main",
        f"--input=@{input_npy_path}",
        f"--output=@{output_bin_path}"
    ]
    print(f"Command: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--device", default="torq")
    args = parser.parse_args()
    
    # 1. Preprocess
    print("\n[1/4] Preprocessing...")
    input_data, pad_info, orig_shape = preprocess_image(args.image)
    input_file = "input_board.npy"
    np.save(input_file, input_data)
    
    # 2. Inference
    print("\n[2/4] Inference...")
    output_file = "output_board.bin"
    if os.path.exists(output_file): os.remove(output_file)
    
    start = time.time()
    try:
        run_inference(args.model, input_file, output_file, args.device)
    except Exception as e:
        print(f"Inference failed: {e}")
        sys.exit(1)
    
    print(f"Time: {time.time() - start:.4f}s")
    
    # 3. Load Output

    
    print("\n[3/4] Processing...")
    raw_out = np.fromfile(output_file, dtype=np.int8) # Assuming int8 model
    
    # Reshape

    expected_elems = 1 * 84 * 2100
    if raw_out.size == expected_elems:
        raw_out = raw_out.reshape((1, 84, 2100))
    else:
        print(f"Warning: Output size {raw_out.size} doesn't match expected (1, 84, 2100). Metadata might be needed.")

 
    out_scale = 0.004194467328488827
    out_zp = -128
    
    outputs = dequantize_out(raw_out, out_scale, out_zp, int8=True)
    
    # 4. Postprocess
    labels = {}
    if args.labels:
        with open(args.labels) as f:
            data = json.load(f)
            # Handle {names: {0: "person", ...}} format from YAML->JSON check
            if "names" in data:
                labels = {str(k): v for k, v in data["names"].items()}
            else:
                labels = data
                
    results = postprocess(outputs, orig_shape, pad_info, labels)
    
    print("\n[4/4] Detections:")
    if not results:
        print("No objects detected.")
    
    for label, conf, box in results:
        print(f"  {label:<15} Conf: {conf:.4f}  Box: {box.astype(int)}")

    # 5. Save Annotated Image
    if results:
        print("\n[5/5] Saving result image...")
        try:
            img = Image.open(args.image)
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # --- 1. Resize to 1280x720 (Letterbox) ---
            target_w, target_h = 1280, 720
            w, h = img.size
            
            # Calculate scale to fit
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize source
            img_resized = img.resize((new_w, new_h), Image.BILINEAR)
            
            # Create standard canvas (black)
            canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            
            # Paste centered
            dx = (target_w - new_w) // 2
            dy = (target_h - new_h) // 2
            canvas.paste(img_resized, (dx, dy))
            
            img = canvas # Use the 720p canvas
            print(f"Resized image to {target_w}x{target_h} (letterboxed).")
            # -----------------------------------------

            draw = ImageDraw.Draw(img)
            
            # --- 2. Load Font ---
            font = None
            try:
                font_path = "/usr/share/fonts/ttf/LiberationSans-Regular.ttf"
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, 35)
                else: 
                     # Fallback check
                    fallback_paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
                    for p in fallback_paths:
                        if os.path.exists(p):
                            font = ImageFont.truetype(p, 40)
                            break
            except Exception:
                pass
            
            if font is None:
                font = ImageFont.load_default()
            # --------------------

            for label, conf, box in results:
                # box is [x, y, w, h] in ORIGINAL coordinates
                
                # --- 3. Transform Box Coordinates ---
                # Apply scaling and shift
                x1 = box[0] * scale + dx
                y1 = box[1] * scale + dy
                w_box = box[2] * scale
                h_box = box[3] * scale
                
                x2 = x1 + w_box
                y2 = y1 + h_box
                # ------------------------------------
                
                # Draw Box (Red, thicker line for resolution)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                
                # Draw Label (Label ONLY, no score)
                text = f"{label}" # Just label
                
                # Draw Label Background
                # Draw text slightly above box, or inside if too high
                text_pos = [x1, y1 - 45]
                if text_pos[1] < 0: text_pos[1] = y1 + 5 # Move inside if clip top
                
                try:
                    left, top, right, bottom = draw.textbbox((text_pos[0], text_pos[1]), text, font=font)
                    draw.rectangle((left-5, top-5, right+5, bottom+5), fill="red")
                except AttributeError:
                    draw.rectangle((text_pos[0], text_pos[1], text_pos[0] + len(text)*20, text_pos[1]+40), fill="red")
                
                draw.text((text_pos[0], text_pos[1]), text, fill="white", font=font)
            
            out_img = "output_yolo.jpg"
            img.save(out_img)
            print(f"Result image saved to: {out_img}")

            # 6. Attempt Display
            print("Attempting to display image...")
            
            # Option A: GStreamer (Wayland/Embedded)
            # Pipeline: filesrc -> jpegdec -> videoconvert -> imagefreeze -> waylandsink
            if shutil.which("gst-launch-1.0"):
                try:
                    print("Found gst-launch-1.0. Displaying with waylandsink for 5 seconds...")
                    
                    cmd = [
                        "gst-launch-1.0",
                        "filesrc", f"location={out_img}", "!",
                        "jpegdec", "!",
                        "videoconvert", "!",
                        "imagefreeze", "!",
                        "waylandsink", "fullscreen=true"
                    ]
                    
                    # Run in background
                    proc = subprocess.Popen(cmd)
                    
                    # Wait 5 seconds or for interrupt
                    try:
                        time.sleep(5)
                    except KeyboardInterrupt:
                        pass # Handle early exit
                        
                    # Clean up
                    proc.terminate()
                    proc.wait()
                    print("Display closed.")
                    
                except Exception as e:
                    print(f"GStreamer failed: {e}")
                    

        except Exception as e:
            print(f"Failed to save result image: {e}")

if __name__ == "__main__":
    main()
