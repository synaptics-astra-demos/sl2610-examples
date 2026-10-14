# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download Moonshine streaming model files from HuggingFace.

Usage::

    python setup_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging

from app_utils.demo_utils import run_demo_setup_cli
from app_utils.paths import MODELS_DIR
from app_utils.torq_examples.moonshine_streaming.setup_demo import download_moonshine_streaming

logger = logging.getLogger("speech_to_text_streaming.setup")

# The streaming model has not been published for the current torq-examples
# release (v2.2.1); v2.1.0 is the newest tag on Hugging Face. Pin to it so setup
# works out of the box. An explicit ``name:version`` in --moonshine-models, e.g.
# ``streaming-tiny-en:v2.2.1``, overrides this. Remove the pin once a tag that
# matches torq_examples/VERSION is published.
MOONSHINE_STREAMING_MODEL_VERSION = "v2.1.0"


def setup_speech_to_text(
    moonshine_models: list[str] | None = None,
):

    def _download_models():
        download_moonshine_streaming(
            moonshine_models,
            base_dir=MODELS_DIR,
            model_version=MOONSHINE_STREAMING_MODEL_VERSION,
        )

    requirements_txt = Path(__file__).parent / "requirements.txt"
    run_demo_setup_cli(
        _download_models, requirements_txt, logger,
        version_map={"torq.runtime": ">=2.0.0a1"},
        demo_name="speech_to_text_streaming",
    )


if __name__ == "__main__":
    import argparse
    from app_utils.log import add_logging_args, configure_logging
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Download Moonshine Streaming model files.",
    )
    parser.add_argument(
        "--moonshine-models",
        nargs="*",
        default=None,
        help="Moonshine Streaming model names or HF repo IDs.",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging, args.log_file)
    setup_speech_to_text(args.moonshine_models)
