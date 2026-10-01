# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Set up the SL2610 Face ID Qt application models."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app_utils.demo_utils import run_demo_setup_cli
from app_utils.log import add_logging_args, configure_logging
from app_utils.paths import MODELS_DIR
from app_utils.torq_examples.Face_ID.setup_demo import download_face_id

logger = logging.getLogger("face_id.setup")


def setup_face_id() -> None:
    """Download Face ID VMFBs into the SL2610 model directory."""
    run_demo_setup_cli(
        lambda: download_face_id(base_dir=MODELS_DIR),
        Path(__file__).parent / "requirements.txt",
        logger,
        version_map={"torq.runtime": ">=2.0.0a1"},
        demo_name="face_id",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Face ID VMFB files.")
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging, args.log_file)
    setup_face_id()
