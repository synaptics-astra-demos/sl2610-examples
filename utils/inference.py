# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Synaptics Incorporated.

from __future__ import annotations

from third_party.torq_examples.utils import inference as _torq_inference

BaseManagedCacheRunner = _torq_inference.BaseManagedCacheRunner
ManagedEncDecCacheRunner = _torq_inference.ManagedEncDecCacheRunner
ManagedSelfAttnCacheRunner = _torq_inference.ManagedSelfAttnCacheRunner
SimpleVMFBInferenceRunner = _torq_inference.SimpleVMFBInferenceRunner
SplitLMHeadRunner = _torq_inference.SplitLMHeadRunner

__all__ = [
    "BaseManagedCacheRunner",
    "ManagedEncDecCacheRunner",
    "ManagedSelfAttnCacheRunner",
    "SimpleVMFBInferenceRunner",
    "SplitLMHeadRunner",
]
