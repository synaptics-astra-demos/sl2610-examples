# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download Yolo V8 nano Moon Jellyfish Detection model files from HuggingFace.

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
    verify_manifest,
)

logger = logging.getLogger("yolov8_nano_moon_jellyfish_detection.setup")

_YOLO_V8_NANO_MOON_JELLYFISH_DETECTION_HF_REPO_ID: Final[str] = "Synaptics/yolov8-od-nano-jellyfish-int8-torq"
_MODEL_FILENAME: Final[str] = "moon320_int8.vmfb"

_MODELS_DIR: Final[str] = "../models"

def _hf_file_exists(repo_id: str, filename: str) -> bool:
    from huggingface_hub import HfApi

    return HfApi().file_exists(repo_id=repo_id, filename=filename)

def _has_required_files(model_dir: Path) -> bool:
    return (model_dir / _MODEL_FILENAME).exists()


def _download_model_assets(repo_id: str, base_dir: Path) -> list[str]:
    """Download model assets; return the manifest file list."""
    if not _hf_file_exists(repo_id, _MODEL_FILENAME):
        raise FileNotFoundError(f"Required file '{_MODEL_FILENAME}' not found in {repo_id}")
    download_from_hf(repo_id, _MODEL_FILENAME, base_dir=base_dir)
    return [_MODEL_FILENAME]

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
    """Download/refresh the given Yolo V8 nano Moon Jellyfish Detection models; return ``{name: model_dir}``."""

    if models is None:
        models = [_YOLO_V8_NANO_MOON_JELLYFISH_DETECTION_HF_REPO_ID]
    if base_dir is None:
        base_dir = _MODELS_DIR
    base_dir = Path(base_dir)

    logger.info("Resolving Yolo V8 nano Moon Jellyfish Detection models: [%s]", ", ".join(models))
    result: dict[str, Path] = {}
    for name in models:
        repo_id = name
        model_dir = base_dir / repo_id
        try:
            _refresh_model(repo_id, model_dir, base_dir)
        except Exception as exc:
            raise DownloadError(f"Unable to download Yolo V8 nano Moon Jellyfish Detection files from {repo_id}") from exc
        result[name] = model_dir
        logger.info("Yolo V8 nano Moon Jellyfish Detection model files ready at '%s'", model_dir)
    return result


def setup_jellyfish_detection(
    jellyfish_detection_models: list[str] | None = None,
):

    def _download_models():
        download_models(jellyfish_detection_models, base_dir=MODELS_DIR)

    requirements_txt = Path(__file__).parent / "requirements.txt"
    run_demo_setup_cli(
        _download_models, requirements_txt, logger,
        version_map={"torq.runtime": ">=2.0.0a1"},
        demo_name="yolo_v8_nano_moon_jellyfish_detection",
    )


if __name__ == "__main__":
    import argparse
    from app_utils.log import add_logging_args, configure_logging

    parser = argparse.ArgumentParser(
        description="Download Yolo V8 nano Moon Jellyfish Detection model files.",
    )
    parser.add_argument(
        "--jellyfish-detection-model",
        nargs="*",
        default=None,
        help="Yolo V8 nano Moon Jellyfish Detection model names or HF repo IDs.",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging, args.log_file)
    setup_jellyfish_detection(args.jellyfish_detection_model)
