# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download Moonshine and Gemma3 model files from HuggingFace.

Usage::

    python download_models.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging

from utils.moonshine import download_moonshine
from utils.gemma import download_gemma3
from utils.demo_utils import run_demo_setup_cli

logger = logging.getLogger("gemma_translate.setup")


def setup_gemma_translate(
    moonshine_models: list[str] | None = None,
    gemma3_models: list[str] | None = None
):

    def _download_models():
        download_moonshine(moonshine_models)
        download_gemma3(gemma3_models)

    requirements_txt = Path(__file__).parent / "requirements.txt"
    run_demo_setup_cli(
        _download_models, requirements_txt, logger,
        version_map={"torq.runtime": ">=2.0.0a1"},
        demo_name="gemma_translate",
    )


if __name__ == "__main__":
    import argparse
    from utils.log import add_logging_args, configure_logging
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Download Moonshine and Gemma3 model files.",
    )
    parser.add_argument(
        "--moonshine-models",
        nargs="*",
        default=None,
        help="Moonshine model names or HF repo IDs.",
    )
    parser.add_argument(
        "--gemma3-models",
        nargs="*",
        default=None,
        help="Gemma3 model names or HF repo IDs.",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging, args.log_file)
    setup_gemma_translate(args.moonshine_models, args.gemma3_models)
