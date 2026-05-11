"""Inference helpers for gemma_translate.

This module re-exports the optimized Moonshine runner from ``library.moonshine``
and the Gemma backends from ``library.gemma``. 
Provides a compatibility layer between the legacy API and the current backends.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Final

from library.moonshine import MoonshineRunner  # noqa: F401
from library.gemma import (  # noqa: F401
    GemmaBackend,
    GemmaTorq,
    GemmaLlama,
    load_gemma,
)


def load_moonshine(
    model_dir: str | os.PathLike,
    **kwargs,
) -> MoonshineRunner:
    """Convenience wrapper for instantiating a MoonshineRunner from a model directory."""
    return MoonshineRunner(model_dir, **kwargs)


def format_answer(
    answer: str,
    infer_time: float,
    stats: list[str] | None = None,
    agent_name: str = "Agent"
) -> str:
    GREEN: Final[str] = "\033[32m"
    RESET: Final[str] = "\033[0m"
    result: str = GREEN + f"{agent_name}: {answer}" + RESET + f" ({infer_time * 1000:.3f} ms"
    stats = stats or []
    for stat in stats:
        result += ", " + str(stat)
    return result + ")"
