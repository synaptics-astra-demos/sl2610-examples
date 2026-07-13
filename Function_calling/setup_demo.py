# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download FunctionGemma and Moonshine model files.

Usage::

    python setup_demo.py
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app_utils.demo_utils import run_demo_setup_cli
from app_utils.paths import MODELS_DIR
from app_utils.torq_examples.moonshine.setup_demo import download_moonshine
from app_utils.torq_examples.utils.download import download_from_url

logger = logging.getLogger("function_calling.setup")

FUNCTIONGEMMA_REPO_ID = "BrinqAI/functiongemma-270m-physical-ai"
FUNCTIONGEMMA_FILENAME = "functiongemma-physical-ai-v10-Q5_K_M.gguf"
FUNCTIONGEMMA_URL = (
    f"https://huggingface.co/{FUNCTIONGEMMA_REPO_ID}/resolve/main/"
    f"{FUNCTIONGEMMA_FILENAME}"
)


def download_functiongemma(
    *,
    base_dir: str | os.PathLike | None = None,
) -> Path:
    """Download the FunctionGemma GGUF used by the demo."""

    models_dir = Path(base_dir) if base_dir is not None else MODELS_DIR
    model_path = models_dir / FUNCTIONGEMMA_FILENAME
    if model_path.exists():
        logger.info("Using local FunctionGemma model from '%s'", model_path)
        return model_path

    logger.info(
        "Downloading FunctionGemma model from %s to '%s'",
        FUNCTIONGEMMA_REPO_ID,
        model_path,
    )
    return download_from_url(FUNCTIONGEMMA_URL, model_path)


def setup_function_calling(
    moonshine_models: list[str] | None = None,
    *,
    skip_moonshine: bool = False,
):

    def _download_models():
        download_functiongemma()
        if not skip_moonshine:
            download_moonshine(moonshine_models, base_dir=MODELS_DIR)

    requirements_txt = Path(__file__).parent / "requirements.txt"
    run_demo_setup_cli(
        _download_models,
        requirements_txt,
        logger,
        version_map={"torq.runtime": ">=2.0.0a1"},
        demo_name="Function_calling",
    )


if __name__ == "__main__":
    import argparse

    from app_utils.log import add_logging_args, configure_logging

    parser = argparse.ArgumentParser(
        description="Download FunctionGemma and Moonshine model files.",
    )
    parser.add_argument(
        "--moonshine-models",
        nargs="*",
        default=None,
        help="Moonshine model names or HF repo IDs.",
    )
    parser.add_argument(
        "--skip-moonshine",
        action="store_true",
        help="Only download the FunctionGemma GGUF; skip Moonshine ASR files.",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging, args.log_file)
    setup_function_calling(
        args.moonshine_models,
        skip_moonshine=args.skip_moonshine,
    )
