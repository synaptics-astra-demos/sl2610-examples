# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Synaptics Incorporated.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from utils.download import DownloadError, default_models_dir, verify_manifest
from third_party.torq_examples.gemma3 import setup_demo as _torq_gemma_setup

logger = logging.getLogger(__name__)

GEMMA3_HF_REPO_MAP: Final[dict[str, str]] = dict(_torq_gemma_setup._HF_REPO_MAP)


def gemma3_repo_id(model: str) -> str:
    return GEMMA3_HF_REPO_MAP.get(model, model)


def local_gemma3_model_dir(
    model: str = "instruct",
    *,
    base_dir: str | Path | None = None,
) -> Path | None:
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
    if base_dir is None:
        base_dir = default_models_dir()
    model_dir = Path(base_dir) / gemma3_repo_id(model)
    model_path = model_dir / "model.vmfb.trim"
    return model_path if model_path.exists() else None


def download_gemma3(models: list[str] | None = None) -> dict[str, Path]:
    if models is None:
        models = ["instruct"]

    logger.info("Resolving Gemma3 models: [%s]", ", ".join(models))
    base_dir = default_models_dir()
    result: dict[str, Path] = {}

    for name in models:
        repo_id = gemma3_repo_id(name)
        model_dir = base_dir / repo_id
        try:
            _torq_gemma_setup._refresh_gemma3(repo_id, model_dir, base_dir)
        except Exception as exc:
            raise DownloadError(f"Unable to download Gemma3 files from {repo_id}") from exc
        result[name] = model_dir
        logger.info("Gemma3 model files ready at '%s'", model_dir)

    return result
