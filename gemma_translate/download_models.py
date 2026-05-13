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

logger = logging.getLogger("gemma_translate.setup")


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
    requirements_txt = Path(__file__).parent / "requirements.txt"

    try:
        download_moonshine(args.moonshine_models)
        download_gemma3(args.gemma3_models)
    except Exception as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
