# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download Moonshine model files from HuggingFace."""

import logging
from typing import Final

from huggingface_hub import HfApi

from ..download import download_from_hf

logger = logging.getLogger(__name__)


MOONSHINE_HF_REPO_MAP: Final[dict[str, str]] = {
    "tiny-en": "Synaptics/moonshine-tiny-bf16-torq",
}


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
