# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Synaptics Incorporated.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from utils.download import DownloadError, default_models_dir, verify_manifest
from third_party.torq_examples.moonshine import setup_demo as _torq_moonshine_setup

logger = logging.getLogger(__name__)

MOONSHINE_HF_REPO_MAP: Final[dict[str, str]] = dict(_torq_moonshine_setup._HF_REPO_MAP)


def moonshine_repo_id(model: str) -> str:
    return MOONSHINE_HF_REPO_MAP.get(model, model)


def local_moonshine_model_dir(
    model: str = "tiny-en",
    *,
    base_dir: str | Path | None = None,
) -> Path | None:
    if base_dir is None:
        base_dir = default_models_dir()
    model_dir = Path(base_dir) / moonshine_repo_id(model)
    if verify_manifest(model_dir):
        return model_dir
    return None


def download_moonshine(models: list[str] | None = None) -> dict[str, Path]:
    if models is None:
        models = ["tiny-en"]

    logger.info("Resolving Moonshine models: [%s]", ", ".join(models))
    base_dir = default_models_dir()
    result: dict[str, Path] = {}

    for name in models:
        repo_id = moonshine_repo_id(name)
        model_dir = base_dir / repo_id
        try:
            _torq_moonshine_setup._refresh_moonshine(repo_id, model_dir, base_dir)
        except Exception as exc:
            raise DownloadError(f"Unable to download Moonshine files from {repo_id}") from exc
        result[name] = model_dir
        logger.info("Moonshine model files ready at '%s'", model_dir)

    return result
