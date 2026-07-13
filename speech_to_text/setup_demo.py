# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download Moonshine model files from HuggingFace.

Usage::

    python setup_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging

from app_utils.demo_utils import run_demo_setup_cli
from app_utils.paths import MODELS_DIR
from app_utils.torq_examples.moonshine.setup_demo import download_moonshine

logger = logging.getLogger("speech_to_text.setup")


def setup_speech_to_text(
    moonshine_models: list[str] | None = None,
):

    def _download_models():
        download_moonshine(moonshine_models, base_dir=MODELS_DIR)

    requirements_txt = Path(__file__).parent / "requirements.txt"
    run_demo_setup_cli(
        _download_models, requirements_txt, logger,
        version_map={"torq.runtime": ">=2.0.0a1"},
        demo_name="speech_to_text",
    )


if __name__ == "__main__":
    import argparse
    from app_utils.log import add_logging_args, configure_logging
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Download Moonshine model files.",
    )
    parser.add_argument(
        "--moonshine-models",
        nargs="*",
        default=None,
        help="Moonshine model names or HF repo IDs.",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging, args.log_file)
    setup_speech_to_text(args.moonshine_models)
