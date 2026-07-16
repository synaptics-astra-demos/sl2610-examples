# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download Yolo v8 object detection model files from HuggingFace.

Usage::

    python setup_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging

from app_utils.demo_utils import run_demo_setup_cli
from app_utils.paths import MODELS_DIR
from app_utils.torq_examples.object_detection.setup_demo import download_object_detection

logger = logging.getLogger("object_detection.setup")


def setup_object_detection(
    object_detection_models: list[str] | None = None,
):

    def _download_models():
        download_object_detection(object_detection_models, base_dir=MODELS_DIR)

    requirements_txt = Path(__file__).parent / "requirements.txt"
    run_demo_setup_cli(
        _download_models, requirements_txt, logger,
        version_map={"torq.runtime": ">=2.0.0a1"},
        demo_name="object_detection",
    )


if __name__ == "__main__":
    import argparse
    from app_utils.log import add_logging_args, configure_logging
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Download Yolo v8 model files.",
    )
    parser.add_argument(
        "--object-detection-models",
        nargs="*",
        default=None,
        help="Object detection model names or HF repo IDs.",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging, args.log_file)
    setup_object_detection(args.object_detection_models)
