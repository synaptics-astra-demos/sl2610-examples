# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

__all__ = [
    "default_models_dir",
    "download_from_url",
    "download_from_hf",
    "write_manifest",
    "verify_manifest",
    "read_manifest",
]

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME: Final[str] = ".manifest.json"


def download_from_url(url: str, filename: str | os.PathLike, chunk_size: int = 8192):
    filename = Path(filename)
    if filename.exists():
        logger.debug("File found locally at: %s", filename)
        return filename

    filename.parent.mkdir(exist_ok=True, parents=True)

    import requests
    from tqdm import tqdm

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

    from huggingface_hub import hf_hub_download

    logger.debug("Attempting to download %s from %s...", filename, repo_id)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(base_dir / repo_id),
    )
    logger.debug("Download from HuggingFace completed.")
    return local_file


def write_manifest(model_dir: Path, repo_id: str, files: list[str]) -> Path:
    """Write a download manifest after a successful setup.

    Args:
        model_dir: Local directory containing the downloaded model files.
        repo_id: HuggingFace repo ID the files were fetched from.
        files: List of filenames that were resolved/downloaded.

    Returns:
        Path to the written manifest file.
    """
    manifest = {
        "repo_id": repo_id,
        "files": sorted(files),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = model_dir / _MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.debug("Wrote manifest to %s", manifest_path)
    return manifest_path


def verify_manifest(model_dir: Path) -> bool:
    """Check that a manifest exists and all listed files are present.

    Returns:
        True if the manifest exists and every file it lists is on disk.
    """
    manifest_path = model_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Corrupt manifest at %s", manifest_path)
        return False

    files = manifest.get("files", [])
    if not files:
        return False
    return all((model_dir / f).exists() for f in files)


def read_manifest(model_dir: Path) -> dict | None:
    """Read and return the manifest dict, or None if missing/corrupt."""
    manifest_path = model_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
