# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import os
import requests
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

__all__ = [
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


def download_from_hf(repo_id: str, filename: str | os.PathLike):
    _repo_root = Path(__file__).resolve().parent.parent
    base_dir = os.getenv("MODELS", str(_repo_root / "models"))
    local_file = os.path.join(base_dir, repo_id, filename)
    os.makedirs(os.path.dirname(local_file), exist_ok=True)

    if os.path.exists(local_file):
        logger.debug("File found locally at: %s", local_file)
        return local_file

    logger.debug("Attempting to download %s from %s...", filename, repo_id)
    downloaded_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=os.path.join(base_dir, repo_id),
    )
    logger.debug("Download from HuggingFace completed.")
    return downloaded_file
