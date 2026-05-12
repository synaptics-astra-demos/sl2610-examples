# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from .download import (
    GEMMA3_HF_REPO_MAP,
    download_gemma3,
    gemma3_repo_id,
    local_gemma3_model_dir,
    local_gemma3_model_path,
)

__all__ = [
    "GemmaBackend",
    "GemmaTorq",
    "GemmaLlama",
    "load_gemma",
    "download_gemma3",
    "gemma3_repo_id",
    "local_gemma3_model_dir",
    "local_gemma3_model_path",
    "GEMMA3_HF_REPO_MAP",
]


def __getattr__(name):
    if name in {"GemmaBackend", "GemmaTorq", "GemmaLlama", "load_gemma"}:
        from .runner import GemmaBackend, GemmaTorq, GemmaLlama, load_gemma

        return {
            "GemmaBackend": GemmaBackend,
            "GemmaTorq": GemmaTorq,
            "GemmaLlama": GemmaLlama,
            "load_gemma": load_gemma,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
