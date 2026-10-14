# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Scene description CLI: describe a photo with LiquidAI LFM2-VL-450M on the NPU.

Same model and prompt as ``run_gui.py``, but it takes one photo from the camera
(or reads an image file with --image) and prints the description to the terminal.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app_utils.demo_utils import DependencyChecker, IncompatibleVersionError, MissingRequirementsError

try:
    DependencyChecker(os.path.join(os.path.dirname(__file__), "requirements.txt")).run()
except (MissingRequirementsError, IncompatibleVersionError) as e:
    print(f"{e}\nRun: python setup_demo.py", file=sys.stderr)
    sys.exit(1)

import cv2

from app_utils.camera import Camera, CameraError, resolve_camera_device, rotate_frame
from app_utils.npu import cleanup_npu_after_inference

# The LFM2-VL-450M demo lives under app_utils/torq_examples/LiquidAI/LiquidAI-LFM2-VL-450M/,
# a directory whose name contains hyphens, so it isn't importable as a regular package.
# Add its src/ dir (for the LFM2VL engine) and its own dir (for the model-refresh helper) to
# sys.path directly instead, same as that demo's own src/infer.py does.
_LFM2VL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..",
    "app_utils", "torq_examples", "LiquidAI", "LiquidAI-LFM2-VL-450M",
)
sys.path.insert(0, os.path.join(_LFM2VL_DIR, "src"))
sys.path.insert(0, _LFM2VL_DIR)

from lfm2vl import LFM2VL
from setup_demo import ensure_lfm2vl_int8_models

DEFAULT_MODELS_DIR = "../models/Synaptics/LiquidAI-LFM2-VL-450M/"
CAPTURE_IMAGE_PATH = "scene_capture.jpg"
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS = 640, 480, 30  # an explicit size is required: left
                                                        # unconstrained, the ISP camera negotiates
                                                        # 4K and fails to open
DEFAULT_PROMPT = "Describe this image in two sentences."


def parse_args():
    parser = argparse.ArgumentParser(description="Describe a picture with LiquidAI LFM2-VL-450M.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera-device", help="Take a photo from this camera, e.g. /dev/video0, or 'auto' to pick the first camera found (default: auto)")
    source.add_argument("--image", help="Describe this image file instead of taking a photo, e.g. ../samples/dog_bike_car.jpg")
    parser.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="Rotate the camera photo (degrees clockwise, default: %(default)s); use 180 if the camera is mounted upside down")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt sent to the VLM (default: %(default)s)")
    parser.add_argument("--models", default=DEFAULT_MODELS_DIR, help="Directory holding the LFM2-VL-450M assets (default: %(default)s)")
    parser.add_argument("--no-refresh", action="store_true", help="Skip the model freshness check (offline runs)")
    return parser.parse_args()


def get_frame(args):
    """Return (BGR frame, description of its source) from --image or the camera."""
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: could not read image: {args.image}")
            sys.exit(1)
        return frame, args.image

    try:
        device = resolve_camera_device(args.camera_device or "auto")
        print(f"Camera: {device}")
        with Camera(device, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS) as cam:
            frame = rotate_frame(cam.capture_photo(), args.rotate)
    except CameraError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    cv2.imwrite(CAPTURE_IMAGE_PATH, frame)
    return frame, f"{device} (saved to {CAPTURE_IMAGE_PATH})"


def main():
    args = parse_args()

    frame, source = get_frame(args)

    ensure_lfm2vl_int8_models(args.models, refresh=not args.no_refresh)
    if not (Path(args.models) / "config.json").is_file():
        print(f"ERROR: LFM2-VL-450M model set not found in {args.models}. Run: python setup_demo.py")
        sys.exit(1)

    print(f"Image:  {source}")
    print(f"Prompt: {args.prompt}")

    print("\n[1/3] Loading model...")
    t0 = time.perf_counter()
    with LFM2VL(args.models) as vl:
        print(f"Time: {(time.perf_counter() - t0) * 1000:.0f}ms")

        print("\n[2/3] Encoding image...")
        vl.encode_image(frame)
        print(f"Time: vision {vl.vision_time:.0f}ms, image prefill {vl.img_prefill_time:.0f}ms, "
              f"prefix {vl.prefix_time:.0f}ms")

        print("\n[3/3] Description:")
        description = ""
        for chunk in vl.infer_stream(args.prompt):
            description += chunk
            sys.stdout.write(chunk)
            sys.stdout.flush()
        print()

        s = vl.stats
        print(f"\n(TTFT {s['ttft_ms']:.0f}ms | prompt {s['prompt_tokens']} tok, "
              f"gen {s['n_tokens']} tok @ {s['tok_per_s']:.1f} tok/s)")

    if not description.strip():
        print("ERROR: the model returned an empty description")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_npu_after_inference()
