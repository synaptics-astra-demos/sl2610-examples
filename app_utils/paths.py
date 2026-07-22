# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Synaptics Incorporated.

"""Filesystem locations for sl2610-examples.

``MODELS_DIR`` is the sl2610 models directory. It is passed explicitly as
``base_dir`` to the (parameterized) torq-examples download helpers so models land
here rather than inside the ``torq_examples`` submodule.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS", str(_REPO_ROOT / "models")))

__all__ = ["MODELS_DIR"]
