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
from typing import Final

from huggingface_hub import HfApi

from utils.download import download_from_hf

logger = logging.getLogger(__name__)


MOONSHINE_HF_REPO_MAP: Final[dict[str, str]] = {
    "tiny-en": "Synaptics/moonshine-tiny-bf16-torq",
}
GEMMA3_HF_REPO_MAP: Final[dict[str, str]] = {
    "instruct": "Synaptics/gemma-3-270m-it-torq",
}
_GEMMA3_MODEL_FILENAMES: Final[list[str]] = [
    "model.vmfb",
    "model.vmfb.trim",
]


def _download_preprocessor(repo_id: str) -> None:
    """Download the optional Moonshine preprocessor model."""
    hf_api = HfApi()
    has_vmfb = hf_api.file_exists(repo_id=repo_id, filename="preprocessor.vmfb")
    has_onnx = hf_api.file_exists(repo_id=repo_id, filename="preprocessor.onnx")

    if has_vmfb:
        if has_onnx:
            logger.warning(
                "Both preprocessor files exist in %s; using preprocessor.vmfb.",
                repo_id,
            )
        download_from_hf(repo_id, "preprocessor.vmfb")
        return

    if has_onnx:
        download_from_hf(repo_id, "preprocessor.onnx")
        return

    logger.info("No preprocessor model found in %s; continuing without it.", repo_id)


def _download_gemma3_model(repo_id: str) -> None:
    """Download model.vmfb and/or model.vmfb.trim as they exist."""
    hf_api = HfApi()
    downloaded_any = False
    for filename in _GEMMA3_MODEL_FILENAMES:
        if hf_api.file_exists(repo_id=repo_id, filename=filename):
            download_from_hf(repo_id, filename)
            logger.info("Downloaded %s from %s", filename, repo_id)
            downloaded_any = True

    if not downloaded_any:
        raise FileNotFoundError(
            f"Neither model.vmfb nor model.vmfb.trim found in {repo_id}"
        )


def download_moonshine(models: list[str] | None = None) -> None:
    """Download Moonshine model files from HuggingFace.

    Args:
        models: List of model names (keys in ``MOONSHINE_HF_REPO_MAP``)
            or raw HF repo IDs.  Defaults to ``["tiny-en"]``.
    """
    if models is None:
        models = ["tiny-en"]
    logger.info("Downloading Moonshine models: [%s]", ", ".join(models))
    repos = [MOONSHINE_HF_REPO_MAP.get(m, m) for m in models]

    for repo_id in repos:
        _download_preprocessor(repo_id)
        download_from_hf(repo_id, "encoder.vmfb")
        download_from_hf(repo_id, "decoder.vmfb")
        download_from_hf(repo_id, "decoder_with_past.vmfb")
        download_from_hf(repo_id, "decoder_token_embeddings.npy")
        download_from_hf(repo_id, "tokenizer.json")
        logger.info("Downloaded Moonshine model files from %s", repo_id)

    logger.info("Moonshine model download complete.")


def download_gemma3(models: list[str] | None = None) -> None:
    """Download Gemma3 model files from HuggingFace.

    Args:
        models: List of model names (keys in ``GEMMA3_HF_REPO_MAP``)
            or raw HF repo IDs.  Defaults to ``["instruct"]``.
    """
    if models is None:
        models = ["instruct"]
    logger.info("Downloading Gemma3 models: [%s]", ", ".join(models))
    repos = [GEMMA3_HF_REPO_MAP.get(m, m) for m in models]

    for repo_id in repos:
        _download_gemma3_model(repo_id)
        download_from_hf(repo_id, "token_embeddings.npy")
        download_from_hf(repo_id, "config.json")
        download_from_hf(repo_id, "tokenizer.json")
        # Optional: trimmed vocab LUT
        try:
            hf_api = HfApi()
            if hf_api.file_exists(repo_id=repo_id, filename="token_id_lut.npy"):
                download_from_hf(repo_id, "token_id_lut.npy")
        except Exception:
            pass
        logger.info("Downloaded Gemma3 model files from %s", repo_id)

    logger.info("Gemma3 model download complete.")


if __name__ == "__main__":
    import argparse
    from utils.log import add_logging_args, configure_logging

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
    configure_logging(args.logging)

    try:
        download_moonshine(args.moonshine_models)
        download_gemma3(args.gemma3_models)
    except Exception as e:
        logger.error("%s", e)
        if e.__cause__:
            logger.error("Caused by: %s", e.__cause__)
        sys.exit(1)
