# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download Gemma3 model files from HuggingFace."""

import logging
from typing import Final

from huggingface_hub import HfApi

from ..download import download_from_hf

logger = logging.getLogger(__name__)


GEMMA3_HF_REPO_MAP: Final[dict[str, str]] = {
    "instruct": "Synaptics/gemma-3-270m-it-torq",
}
_GEMMA3_MODEL_FILENAMES: Final[list[str]] = [
    "model.vmfb",
    "model.vmfb.trim",
]


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
