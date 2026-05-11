# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import os
import requests
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

__all__ = [
    "default_models_dir",
    "download_from_url",
    "download_from_hf",
]

logger = logging.getLogger(__name__)


def download_from_url(url: str, filename: str | os.PathLike, chunk_size: int = 8192):
    filename = Path(filename)
    if filename.exists():
        logger.debug("File found locally at: %s", filename)
        return filename

    filename.parent.mkdir(exist_ok=True, parents=True)

    logger.debug("Attempting download from %s...", url)
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total = int(response.headers.get('content-length', 0))
    progress = tqdm(total=total, unit='B', unit_scale=True, desc=str(filename))

    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                progress.update(len(chunk))
    progress.close()
    logger.debug("Download completed.")

    return filename


def default_models_dir() -> Path:
    _repo_root = Path(__file__).resolve().parent.parent
    return Path(os.getenv("MODELS", str(_repo_root / "models")))


def download_from_hf(
    repo_id: str,
    filename: str | os.PathLike,
    base_dir: str | os.PathLike | None = None,
) -> Path:
    if base_dir is None:
        base_dir = default_models_dir()
    base_dir = Path(base_dir)
    local_file = base_dir / repo_id / filename
    local_file.parent.mkdir(parents=True, exist_ok=True)

    if local_file.exists():
        logger.debug("File found locally at: %s", local_file)
        return local_file

    logger.debug("Attempting to download %s from %s...", filename, repo_id)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(base_dir / repo_id),
    )
    logger.debug("Download from HuggingFace completed.")
    return local_file
