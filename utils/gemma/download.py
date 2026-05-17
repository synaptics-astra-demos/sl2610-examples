# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download Gemma3 model files from HuggingFace."""

import logging
from pathlib import Path
from typing import Final

from ..download import default_models_dir, download_from_hf, verify_manifest, write_manifest

logger = logging.getLogger(__name__)


GEMMA3_HF_REPO_MAP: Final[dict[str, str]] = {
    "instruct": "Synaptics/gemma-3-270m-it-torq",
}
_GEMMA3_MODEL_FILENAMES: Final[list[str]] = [
    "model.vmfb",
    "model.vmfb.trim",
]
_GEMMA3_TRIM_LUT_FILENAME: Final[str] = "token_id_lut.npy"
_GEMMA3_REQUIRED_FILES: Final[tuple[str, ...]] = (
    "token_embeddings.npy",
    "config.json",
    "tokenizer.json",
)


def gemma3_repo_id(model: str) -> str:
    return GEMMA3_HF_REPO_MAP.get(model, model)


def local_gemma3_model_dir(
    model: str = "instruct",
    *,
    base_dir: str | Path | None = None,
) -> Path | None:
    """Return the managed local model directory if manifest is valid."""

    if base_dir is None:
        base_dir = default_models_dir()
    model_dir = Path(base_dir) / gemma3_repo_id(model)
    if verify_manifest(model_dir):
        return model_dir
    return None


def local_gemma3_model_path(
    model: str = "instruct",
    *,
    base_dir: str | Path | None = None,
) -> Path | None:
    """Return the default local trimmed VMFB path for a managed Gemma3 model."""

    if base_dir is None:
        base_dir = default_models_dir()
    model_dir = Path(base_dir) / gemma3_repo_id(model)
    model_path = model_dir / "model.vmfb.trim"
    return model_path if model_path.exists() else None


def _download_gemma3_model(repo_id: str, base_dir: Path) -> list[str]:
    """Download model.vmfb and/or model.vmfb.trim as they exist.

    Returns:
        List of model filenames that are present locally after this call.
    """
    local_dir = base_dir / repo_id
    existing = [f for f in _GEMMA3_MODEL_FILENAMES if (local_dir / f).exists()]
    if existing:
        return existing

    from huggingface_hub import HfApi

    hf_api = HfApi()
    downloaded: list[str] = []
    for filename in _GEMMA3_MODEL_FILENAMES:
        if hf_api.file_exists(repo_id=repo_id, filename=filename):
            download_from_hf(repo_id, filename, base_dir=base_dir)
            logger.info("Downloaded %s from %s", filename, repo_id)
            downloaded.append(filename)

    if not downloaded:
        raise FileNotFoundError(
            f"Neither model.vmfb nor model.vmfb.trim found in {repo_id}"
        )
    return downloaded


def download_gemma3(models: list[str] | None = None) -> dict[str, Path]:
    """Download Gemma3 model files from HuggingFace.

    Args:
        models: List of model names (keys in ``GEMMA3_HF_REPO_MAP``)
            or raw HF repo IDs.  Defaults to ``["instruct"]``.

    Returns:
        A dict mapping each model name to its local directory path.
    """
    if models is None:
        models = ["instruct"]
    logger.info("Resolving Gemma3 models: [%s]", ", ".join(models))
    repos = [gemma3_repo_id(m) for m in models]
    base_dir = default_models_dir()

    result: dict[str, Path] = {}
    for name, repo_id in zip(models, repos):
        local_dir = local_gemma3_model_dir(name, base_dir=base_dir)
        if local_dir is not None:
            result[name] = local_dir
            logger.info("Using local Gemma3 model files from '%s'", str(local_dir))
            continue

        downloaded_files: list[str] = []

        model_filenames = _download_gemma3_model(repo_id, base_dir)
        downloaded_files.extend(model_filenames)

        for filename in _GEMMA3_REQUIRED_FILES:
            download_from_hf(repo_id, filename, base_dir=base_dir)
            downloaded_files.append(filename)

        # Optional: trimmed vocab LUT
        try:
            from huggingface_hub import HfApi

            hf_api = HfApi()
            if hf_api.file_exists(repo_id=repo_id, filename=_GEMMA3_TRIM_LUT_FILENAME):
                download_from_hf(repo_id, _GEMMA3_TRIM_LUT_FILENAME, base_dir=base_dir)
                downloaded_files.append(_GEMMA3_TRIM_LUT_FILENAME)
        except Exception:
            pass

        model_dir = base_dir / repo_id
        write_manifest(model_dir, repo_id, downloaded_files)
        result[name] = model_dir
        logger.info("Downloaded Gemma3 model files from %s to '%s'", repo_id, str(model_dir))

    logger.info("Gemma3 model download complete.")
    return result
