# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download Moonshine model files from HuggingFace."""

import logging
from pathlib import Path
from typing import Final

from ..download import default_models_dir, download_from_hf

logger = logging.getLogger(__name__)


MOONSHINE_HF_REPO_MAP: Final[dict[str, str]] = {
    "tiny-en": "Synaptics/moonshine-tiny-bf16-torq",
}

_MOONSHINE_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "encoder.vmfb",
    "decoder.vmfb",
    "decoder_with_past.vmfb",
    "decoder_token_embeddings.npy",
    "tokenizer.json",
)


def moonshine_repo_id(model: str) -> str:
    return MOONSHINE_HF_REPO_MAP.get(model, model)


def local_moonshine_model_dir(
    model: str = "tiny-en",
    *,
    base_dir: str | Path | None = None,
) -> Path | None:
    """Return the managed local model directory if required files exist."""

    if base_dir is None:
        base_dir = default_models_dir()
    model_dir = Path(base_dir) / moonshine_repo_id(model)
    if all((model_dir / filename).exists() for filename in _MOONSHINE_REQUIRED_FILES):
        return model_dir
    return None


def _download_preprocessor(repo_id: str, base_dir: Path) -> None:
    """Download the optional Moonshine preprocessor model."""
    local_dir = base_dir / repo_id
    if (local_dir / "preprocessor.vmfb").exists() or (local_dir / "preprocessor.onnx").exists():
        return

    from huggingface_hub import HfApi

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
    logger.info("Resolving Moonshine models: [%s]", ", ".join(models))
    repos = [MOONSHINE_HF_REPO_MAP.get(m, m) for m in models]
    base_dir = default_models_dir()

    result: dict[str, Path] = {}
    for name, repo_id in zip(models, repos):
        local_dir = local_moonshine_model_dir(name, base_dir=base_dir)
        if local_dir is not None:
            result[name] = local_dir
            logger.info("Using local Moonshine model files from '%s'", str(local_dir))
            continue

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
