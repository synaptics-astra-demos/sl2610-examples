# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Download LiquidAI LFM2-VL-450M model files from HuggingFace.

Usage::

    python setup_demo.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import importlib.util
import logging
from pathlib import Path

from app_utils.demo_utils import run_demo_setup_cli
from app_utils.paths import MODELS_DIR

logger = logging.getLogger("scene_description.setup")

_LFM2VL_DEMO_DIR = (
    Path(__file__).resolve().parent.parent
    / "app_utils" / "torq_examples" / "LiquidAI" / "LiquidAI-LFM2-VL-450M"
)


def _load_lfm2vl_setup_demo():
    """Import the LiquidAI-LFM2-VL-450M demo's setup_demo.py by file path.

    Its directory name contains dots and hyphens, so it isn't importable as a
    regular package.
    """
    spec = importlib.util.spec_from_file_location(
        "lfm2vl_setup_demo", _LFM2VL_DEMO_DIR / "setup_demo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_scene_description():

    def _download_models():
        # LiquidAI-LFM2-VL-450M's setup_demo.py (vendored under app_utils/torq_examples/,
        # not ours to edit) has no base_dir param; it always downloads under the directory
        # named by the MODELS env var, so set that to land models in sl2610-examples/models
        # instead of its own default (app_utils/torq_examples/models).
        os.environ.setdefault("MODELS", str(MODELS_DIR))
        _load_lfm2vl_setup_demo().setup_liquidvl(["default"])

    requirements_txt = Path(__file__).parent / "requirements.txt"
    run_demo_setup_cli(
        _download_models, requirements_txt, logger,
        demo_name="scene_description",
    )


if __name__ == "__main__":
    import argparse
    from app_utils.log import add_logging_args, configure_logging

    parser = argparse.ArgumentParser(
        description="Download LiquidAI LFM2-VL-450M model files.",
    )
    add_logging_args(parser)
    args = parser.parse_args()
    configure_logging(args.logging, args.log_file)
    setup_scene_description()
