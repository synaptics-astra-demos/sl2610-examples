# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download Moonshine model files from HuggingFace."""

import logging
from pathlib import Path
from typing import Final

from ..download import default_models_dir, download_from_hf, verify_manifest, write_manifest

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
    """Return the managed local model directory if manifest is valid."""

    if base_dir is None:
        base_dir = default_models_dir()
    model_dir = Path(base_dir) / moonshine_repo_id(model)
    if verify_manifest(model_dir):
        return model_dir
    return None


def _download_preprocessor(repo_id: str, base_dir: Path) -> str | None:
    """Download the optional Moonshine preprocessor model.

    Returns:
        The filename downloaded (e.g. ``"preprocessor.vmfb"``), or None
        if the repo has no preprocessor (flow 1: embedded in encoder).
    """
    local_dir = base_dir / repo_id
    if (local_dir / "preprocessor.vmfb").exists():
        return "preprocessor.vmfb"
    if (local_dir / "preprocessor.onnx").exists():
        return "preprocessor.onnx"

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
        return "preprocessor.vmfb"

    if has_onnx:
        download_from_hf(repo_id, "preprocessor.onnx", base_dir=base_dir)
        return "preprocessor.onnx"

    logger.info("No preprocessor model found in %s; continuing without it.", repo_id)
    return None


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

        downloaded_files: list[str] = []

        preprocessor = _download_preprocessor(repo_id, base_dir)
        if preprocessor:
            downloaded_files.append(preprocessor)

        for filename in _MOONSHINE_REQUIRED_FILES:
            download_from_hf(repo_id, filename, base_dir=base_dir)
            downloaded_files.append(filename)

        model_dir = base_dir / repo_id
        write_manifest(model_dir, repo_id, downloaded_files)
        result[name] = model_dir
        logger.info("Downloaded Moonshine model files from %s to '%s'", repo_id, str(model_dir))

    logger.info("Moonshine model download complete.")
    return result
