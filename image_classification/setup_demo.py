# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download MobileNetV2 model files from HuggingFace.

    Use available torq_examples download utilities. 

Usage::

    python setup_demo.py
"""

import sys
import os
import logging
from pathlib import Path
from typing import Final

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app_utils.demo_utils import run_demo_setup_cli
from app_utils.paths import MODELS_DIR

from app_utils.torq_examples.utils.download import (
    DownloadError,
    ModelStatus,
    ensure_model,
    download_from_hf,
    get_hf_revision,
    resolve_repo_id, 
    verify_manifest,
)

logger = logging.getLogger("image_classification.setup")

_MOBILENETV2_HF_REPO_MAP: Final[dict[str, str]] = {
    "v2": "Synaptics/mobilenet_v2-int8-torq",
}
_MODEL_FILENAME: Final[str] = "MobileNetv2_int8.vmfb"
_LABELS_FILENAME: Final[str] = "labels.json"
_SAMPLES_PREFIX: Final[str] = "samples/"

_MODELS_DIR: Final[str] = "../models"

def _hf_file_exists(repo_id: str, filename: str) -> bool:
    from huggingface_hub import HfApi

    return HfApi().file_exists(repo_id=repo_id, filename=filename)

def _list_sample_files(repo_id: str) -> list[str]:
    from huggingface_hub import HfApi

    return [
        path for path in HfApi().list_repo_files(repo_id=repo_id)
        if path.startswith(_SAMPLES_PREFIX) and not path.endswith("/")
    ]

def _has_required_files(model_dir: Path) -> bool:
    return (model_dir / _MODEL_FILENAME).exists() and (model_dir / _LABELS_FILENAME).exists()


def _download_model_assets(repo_id: str, base_dir: Path) -> list[str]:
    """Download model assets; return the manifest file list."""
    manifest_files = []

    for filename in (_MODEL_FILENAME, _LABELS_FILENAME):
        if not _hf_file_exists(repo_id, filename):
            raise FileNotFoundError(f"Required file '{filename}' not found in {repo_id}")
        download_from_hf(repo_id, filename, base_dir=base_dir)
        manifest_files.append(filename)

    for sample_file in _list_sample_files(repo_id):
        download_from_hf(repo_id, sample_file, base_dir=base_dir)
        manifest_files.append(sample_file)

    return manifest_files

def _refresh_model(repo_id: str, model_dir: Path, base_dir: Path) -> ModelStatus:
    files_present = verify_manifest(model_dir) and _has_required_files(model_dir)
    revision = get_hf_revision(repo_id)
    return ensure_model(
        model_dir,
        repo_id,
        files_present=files_present,
        revision=revision,
        download=lambda: _download_model_assets(repo_id, base_dir),
    )

def download_models(
    models: list[str] | None = None,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Download/refresh the given MobileNetV2 models; return ``{name: model_dir}``."""

    if models is None:
        models = ["v2"]
    if base_dir is None:
        base_dir = _MODELS_DIR
    base_dir = Path(base_dir)

    logger.info("Resolving MobileNetV2 models: [%s]", ", ".join(models))
    result: dict[str, Path] = {}
    for name in models:
        repo_id = resolve_repo_id(name, _MOBILENETV2_HF_REPO_MAP)
        model_dir = base_dir / repo_id
        try:
            _refresh_model(repo_id, model_dir, base_dir)
        except Exception as exc:
            raise DownloadError(f"Unable to download MobileNetV2 files from {repo_id}") from exc
        result[name] = model_dir
        logger.info("MobileNetV2 model files ready at '%s'", model_dir)
    return result


def setup_image_classification(
    image_classification_models: list[str] | None = None,
):

    def _download_models():
        download_models(image_classification_models, base_dir=MODELS_DIR)

    requirements_txt = Path(__file__).parent / "requirements.txt"
    run_demo_setup_cli(
        _download_models, requirements_txt, logger,
        version_map={"torq.runtime": ">=2.0.0a1"},
        demo_name="image_classification",
    )


if __name__ == "__main__":
    import argparse
    from app_utils.log import add_logging_args, configure_logging

    parser = argparse.ArgumentParser(
        description="Download MobileNetV2 model files.",
    )
    parser.add_argument(
        "--image-classification-models",
        nargs="*",
        default=None,
        help="Image classification model names or HF repo IDs.",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging, args.log_file)
    setup_image_classification(args.image_classification_models)
