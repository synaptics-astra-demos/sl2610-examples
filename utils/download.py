# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Synaptics Incorporated.

from __future__ import annotations

import os
from pathlib import Path

from utils.torq_examples.utils import download as _torq_download

DownloadError = _torq_download.DownloadError
ModelStatus = _torq_download.ModelStatus
base_dir_for = _torq_download.base_dir_for
check_model_status = _torq_download.check_model_status
clear_model_dir = _torq_download.clear_model_dir
download_from_url = _torq_download.download_from_url
ensure_model = _torq_download.ensure_model
get_hf_revision = _torq_download.get_hf_revision
read_manifest = _torq_download.read_manifest
verify_manifest = _torq_download.verify_manifest
write_manifest = _torq_download.write_manifest

__all__ = [
    "DownloadError",
    "ModelStatus",
    "base_dir_for",
    "check_model_status",
    "clear_model_dir",
    "default_models_dir",
    "download_from_hf",
    "download_from_url",
    "ensure_model",
    "get_hf_revision",
    "read_manifest",
    "verify_manifest",
    "write_manifest",
]


def default_models_dir() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return Path(os.getenv("MODELS", str(repo_root / "models")))


def download_from_hf(
    repo_id: str,
    filename: str | os.PathLike,
    base_dir: str | os.PathLike | None = None,
) -> Path:
    if base_dir is None:
        base_dir = default_models_dir()
    return _torq_download.download_from_hf(repo_id, filename, base_dir=base_dir)
