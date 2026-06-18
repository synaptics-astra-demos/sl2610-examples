#!/usr/bin/env python3
"""
Standalone Board Inference Script
Usage: python3 classification.py --model model.vmfb --image image.jpg [--labels labels.json]
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.inference import SimpleVMFBInferenceRunner

# ==========================================
# Ported from helpers/mobilenet.py
# ==========================================
def preprocess_image(image_path, input_dtype=np.int8):
    # Load image
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        sys.exit(1)

    orig_image = image.copy()

    width, height = image.size
    if width > height:
        left = (width - height) // 2
        right = left + height
        top = 0
        bottom = height
    else:
        top = (height - width) // 2
        bottom = top + width
        left = 0
        right = width
    
    image = image.crop((left, top, right, bottom))
    
    # Resize original image 
    image = orig_image.resize((224, 224))

    # convert image to numpy array with correct quantization
    if input_dtype == np.int8:
        input_data = np.array(image, dtype=np.uint8)
        input_data = input_data.astype(np.float32)
        input_data = input_data - 127
        input_data = input_data.astype(np.int8)
        input_data = np.expand_dims(input_data, axis=0)
    elif input_dtype == np.uint8:
        input_data = np.array(image, dtype=np.uint8)
        input_data = np.expand_dims(input_data, axis=0)
    else:
        raise Exception(f"Unsupported input dtype {input_dtype}")

    return input_data

def dequantize_output(output_data, output_bias, output_scale):
    return (output_data.astype(np.float32) - output_bias) / output_scale

# ==========================================
# Main Logic
# ==========================================
def run_inference_torq(runner, input_data):
    return runner.infer(input_data)

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
    parser = argparse.ArgumentParser(description="Run complete inference workflow on board")
    parser.add_argument("--model", required=True, help="Path to .vmfb model file")
    parser.add_argument("--image", required=True, help="Path to input image file")
    parser.add_argument("--labels", help="Path to labels json file")
    parser.add_argument("--device", default="torq", help="Device to target (default: torq)")
    args = parser.parse_args()

    os.environ["XDG_RUNTIME_DIR"] = "/var/run/user/0"
    os.environ["WAYLAND_DISPLAY"] = "wayland-1"

    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.image):
        print(f"Image file not found: {args.image}")
        sys.exit(1)

    #Set NPU clock
    enable_npu_clock()

    # 0. Load the model with Torq Runtime
    runner = SimpleVMFBInferenceRunner(
        args.model,
        device_uri=args.device,
        function="main",
        load_model_to_mem=True
    )

    # 1. Preprocess
    print("\n[1/4] Preprocessing image...")
    # MobileNetV2 usually expects int8 input
    try:
        input_data = preprocess_image(args.image, np.int8)
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        sys.exit(1)

    # 2. Run Inference
    print("\n[2/4] Running model on board...")

    try:
        raw_out = run_inference_torq(runner, input_data)
    except Exception as e:
        print(f"Inference failed: {e}")
        sys.exit(1)
    print(f"Time: {runner.infer_time_ms:.3f}ms")

    # 3. Postprocess
    print("\n[3/4] Processing output...")
    
    if raw_out.shape != (1, 10000):
        print(f"Warning: Output shape {raw_out.shape} doesn't match expected (1, 10000). Metadata might be needed.")
    
    # Scale/Bias from test_run_classification_e2e.py
    OUTPUT_SCALE = 256.0
    OUTPUT_BIAS = -128
    
    probs = dequantize_output(raw_out, OUTPUT_BIAS, OUTPUT_SCALE)
    
    # 4. Results
    print("\n[4/4] Classification Results:")
    
    # Load labels if provided
    labels = {}
    if args.labels:
        try:
            with open(args.labels, 'r') as f:
                data = json.load(f)
                # Handle simplified_mobilenetv2_info format
                if isinstance(data, dict) and 'labels' in data:
                    # labels might be a dict "0": "label" or list
                    l = data['labels']
                    if isinstance(l, list):
                        labels = {i: v for i, v in enumerate(l)}
                    elif isinstance(l, dict):
                         labels = {int(k): v for k, v in l.items()}
                else:
                    print("Warning: Unknown labels JSON format")
        except Exception as e:
            print(f"Warning: Failed to load labels: {e}")

    # Top 5
    probs = probs.flatten() # Ensure it's 1D
    predicted_idx = np.argmax(probs)
    sorted_indices = np.argsort(probs)[::-1] # Descending order
    
    for i in range(min(5, len(probs))):
        idx = sorted_indices[i]
        score = probs[idx]
        label_str = labels.get(int(idx), f"Class {idx}")
        print(f"  {i+1}. {label_str} : {score:.6f}")

    print(f"\nTop Prediction: {labels.get(int(predicted_idx), f'Class {predicted_idx}')}")

    # 5. Save Annotated Image
    print("\n[5/5] Saving result image...")
    try:
        # Re-open original image to draw on
        img = Image.open(args.image)
        # Convert to RGB if needed (e.g. if PNG with alpha)
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        # Resize to match display resolution based on environment variables
        orientation = os.environ.get("ORIENTATION", "landscape")
        target_w = int(os.environ.get("DISPLAY_WIDTH", 800 if orientation == "landscape" else 480))
        target_h = int(os.environ.get("DISPLAY_HEIGHT", 480 if orientation == "landscape" else 800))
        # Create black background
        new_img = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        
        # Calculate aspect-safe resize
        w, h = img.size
        ratio = min(target_w / w, target_h / h)
        new_size = (int(w * ratio), int(h * ratio))
        
        if new_size != (w, h):
            img_resized = img.resize(new_size, Image.BILINEAR)
        else:
            img_resized = img
            
        # Paste centered
        paste_pos = ((target_w - new_size[0]) // 2, (target_h - new_size[1]) // 2)
        new_img.paste(img_resized, paste_pos)
        
        img = new_img # Replace working image with 720p version
        print(f"Resized image to {target_w}x{target_h} (letterboxed) for display.")

        draw = ImageDraw.Draw(img)
        
        # Load a larger font if possible
        font = None
        try:
            # Common paths on Linux/Yocto
            font_paths = [
                "/usr/share/fonts/ttf/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf" 
            ]
            for p in font_paths:
                if os.path.exists(p):
                    font = ImageFont.truetype(p, 45) # Increased size
                    print(f"Loaded font: {p}")
                    break
        except Exception as e:
            print(f"Could not load custom font: {e}")
            
        if font is None:
            # Fallback (will be small)
            font = ImageFont.load_default()

        # Create clear label text (Label only, no score)
        top_label = labels.get(int(predicted_idx), f"Class {predicted_idx}")
        # Cleanup label (remove ", ..." extra names if present)
        top_label = top_label.split(",")[0] 
        text = f"{top_label}"
        
        # Draw a black rectangle background for the text
        text_pos = (50, 50)
        
        try:
            # Modern Pillow
            left, top, right, bottom = draw.textbbox(text_pos, text, font=font)
            draw.rectangle((left-10, top-10, right+10, bottom+10), fill="black", outline="white")
        except AttributeError:
            # Older Pillow fallback (approximate size)
            # Rough estimate: 20px height per char (if we found a font), else default is tiny
            # This fallback is tricky with unknown font size, so we'll just make a big box if we have a font
            char_w = 25 if font != ImageFont.load_default() else 7
            char_h = 45 if font != ImageFont.load_default() else 15
            w_est = len(text) * char_w
            draw.rectangle((text_pos[0]-10, text_pos[1]-10, text_pos[0]+w_est, text_pos[1]+char_h), fill="black", outline="white")

        draw.text(text_pos, text, fill="white", font=font)
        
        output_image_path = "output_classification.jpg"
        img.save(output_image_path)
        print(f"Result image saved to: {output_image_path}")

        # 6. Attempt Display
        print("Attempting to display image...")
        
        # Option A: GStreamer (Wayland/Embedded)
        if shutil.which("gst-launch-1.0"):
            try:
                print("Found gst-launch-1.0. Displaying with waylandsink for 5 seconds...")
                
                cmd = [
                    "gst-launch-1.0",
                    "filesrc", f"location={output_image_path}", "!",
                    "jpegdec", "!",
                    "videoconvert", "!",
                    "imagefreeze", "!",
                    "videoscale", "!",
                    f"video/x-raw,width={target_w},height={target_h}", "!",
                    "waylandsink"
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
