"""Inference helpers for speech_to_text.

Re-exports MoonshineRunner from ``utils.moonshine`` and provides
a backward-compatible ``load_moonshine`` wrapper and ``format_answer`` utility.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Final

from utils.moonshine import MoonshineRunner


def load_moonshine(
    model_dir: str | os.PathLike,
    **kwargs,
) -> MoonshineRunner:
    """Instantiate a MoonshineRunner from a model directory."""
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
