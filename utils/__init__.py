# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

# The bundled torq_examples modules import their download helpers with a bare
# ``from utils.download import ...``. Run inside sl2610-examples, ``utils`` is this
# package, so point ``utils.download`` at torq's real module (the same object
# imported elsewhere via ``utils.torq_examples.utils.download``) rather than a
# shim file. This keeps a single ``DownloadError``/``ModelStatus`` class.
import sys as _sys

from .torq_examples.utils import download as _download

_sys.modules[__name__ + ".download"] = _download

del _sys, _download
