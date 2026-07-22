"""Pytest configuration for Function_calling tests.

Ensures the repo root is on sys.path so ``app_utils.*`` is importable,
and adds Function_calling/ itself so ``voice``, ``llamacpp``, etc.
resolve as top-level packages.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)

for p in (_THIS_DIR, _REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
