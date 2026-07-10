# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

# The bundled torq_examples submodule (app_utils/torq_examples/) is a standalone
# project vendored via git submodule. Its own code imports its own helpers with
# bare names, e.g. ``from utils.download import ...``, expecting its own
# directory to be a sys.path root. Put that directory on sys.path so those bare
# imports resolve to *its* ``utils`` package.
#
# This package is named ``app_utils`` (not ``utils``) specifically so there is
# no name collision with torq_examples's own ``utils`` package - our own code
# always resolves "app_utils" to this package, torq_examples's internal code
# always resolves "utils" to its own bundled copy. See app_utils/README.md for
# the one gotcha this doesn't cover: when our code reaches into torq_examples
# via ``app_utils.torq_examples.utils.x`` (dotted) for the same file
# torq_examples's own internal code loads via bare ``utils.x``, Python treats
# those as two separate module objects.
import sys as _sys
from pathlib import Path as _Path

_torq_examples_dir = str(_Path(__file__).parent / "torq_examples")
if _torq_examples_dir not in _sys.path:
    _sys.path.insert(0, _torq_examples_dir)

del _sys, _Path, _torq_examples_dir
