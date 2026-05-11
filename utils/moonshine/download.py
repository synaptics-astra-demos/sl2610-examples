# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download Moonshine model files from HuggingFace."""

import logging
from pathlib import Path
from typing import Final

from huggingface_hub import HfApi

from ..download import default_models_dir, download_from_hf

logger = logging.getLogger(__name__)


MOONSHINE_HF_REPO_MAP: Final[dict[str, str]] = {
    "tiny-en": "Synaptics/moonshine-tiny-bf16-torq",
}


def _download_preprocessor(repo_id: str, base_dir: Path) -> None:
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
        download_from_hf(repo_id, "preprocessor.vmfb", base_dir=base_dir)
        return

    if has_onnx:
        download_from_hf(repo_id, "preprocessor.onnx", base_dir=base_dir)
        return

    logger.info("No preprocessor model found in %s; continuing without it.", repo_id)


def download_moonshine(models: list[str] | None = None) -> dict[str, Path]:
    """Download Moonshine model files from HuggingFace.

    Args:
        models: List of model names (keys in ``MOONSHINE_HF_REPO_MAP``)
            or raw HF repo IDs.  Defaults to ``["tiny-en"]``.

    Returns:
        A dict mapping each model name to its local directory path.
    """
    if models is None:
        models = ["tiny-en"]
    logger.info("Downloading Moonshine models: [%s]", ", ".join(models))
    repos = [MOONSHINE_HF_REPO_MAP.get(m, m) for m in models]
    base_dir = default_models_dir()

    result: dict[str, Path] = {}
    for name, repo_id in zip(models, repos):
        _download_preprocessor(repo_id, base_dir)
        download_from_hf(repo_id, "encoder.vmfb", base_dir=base_dir)
        download_from_hf(repo_id, "decoder.vmfb", base_dir=base_dir)
        download_from_hf(repo_id, "decoder_with_past.vmfb", base_dir=base_dir)
        download_from_hf(repo_id, "decoder_token_embeddings.npy", base_dir=base_dir)
        download_from_hf(repo_id, "tokenizer.json", base_dir=base_dir)
        result[name] = base_dir / repo_id
        logger.info("Downloaded Moonshine model files from %s to '%s'", repo_id, str(result[name]))

    logger.info("Moonshine model download complete.")
    return result
