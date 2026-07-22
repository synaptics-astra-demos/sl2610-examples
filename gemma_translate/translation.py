# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Text translation service utilities.

The classes here wrap model backends and prompt construction. Frontends should
only pass text, target language, and optional partial-output callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Callable

from app_utils.stats import Gemma3InferenceStats

if TYPE_CHECKING:
    from app_utils.gemma import GemmaBackend

logger = logging.getLogger(__name__)


PartialCallback = Callable[[str], None]


@dataclass(frozen=True)
class TranslationResult:
    text: str
    stats: Gemma3InferenceStats
    target_language: str


class GemmaTranslationService:
    """Prompt-building translation adapter around a ``GemmaBackend``."""

    def __init__(self, backend: GemmaBackend):
        self.backend = backend
        logger.info("Translation service ready (backend=%s)", type(backend).__name__)

    def translate(
        self,
        text: str,
        *,
        target_language: str,
        on_partial: PartialCallback | None = None,
    ) -> TranslationResult:
        prompt = self._build_prompt(text, target_language)
        final_text = ""

        for partial in self.backend.stream_response(prompt):
            final_text = str(partial)
            if on_partial is not None:
                on_partial(final_text)

        stats = Gemma3InferenceStats(
            total_time_ms=self.backend.last_infer_time_ms,
            ttft_ms=self.backend.time_to_first_token_ms,
            n_tokens=self.backend.last_n_output_tokens,
            n_input_tokens=self.backend.last_n_input_tokens,
            n_prefill_tokens=getattr(self.backend, "last_n_prefill_tokens", 0),
            prefill_tps=getattr(self.backend, "last_prefill_tps", None),
        )
        return TranslationResult(
            text=final_text.strip(),
            stats=stats,
            target_language=target_language,
        )

    @staticmethod
    def _build_prompt(text: str, target_language: str) -> str:
        return (
            f"Translate the text in quotes to {target_language}. "
            f"Output only the translated text.\n\"{text}\"\n"
        )
