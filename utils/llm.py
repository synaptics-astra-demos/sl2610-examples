# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Synaptics Incorporated.

from __future__ import annotations

from platform.torq_examples.utils import llm as _torq_llm

DecoderOnlyLLMRunner = _torq_llm.DecoderOnlyLLMRunner
InferenceInterrupted = _torq_llm.InferenceInterrupted
discover_lm_head_path = _torq_llm.discover_lm_head_path
resolve_lm_head_path = _torq_llm.resolve_lm_head_path
resolve_token_id_lut = _torq_llm.resolve_token_id_lut

__all__ = [
    "DecoderOnlyLLMRunner",
    "InferenceInterrupted",
    "discover_lm_head_path",
    "resolve_lm_head_path",
    "resolve_token_id_lut",
]
