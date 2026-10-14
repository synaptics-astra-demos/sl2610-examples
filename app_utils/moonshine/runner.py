# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Synaptics Incorporated.

from __future__ import annotations

import os

from app_utils.torq_examples.moonshine.src.runner import MoonshineRunner
from app_utils.torq_examples.utils.runtime import build_runtime_flags


def load_moonshine(
    model_path: str | os.PathLike | None = None,
    *,
    model_name: str = "tiny-en",
    input_freq: int = 16000,
    n_threads: int | None = None,
    runtime_flags: list[str] | None = None,
    device_io: bool | None = None,
    tda: str = "dmabuf",
) -> MoonshineRunner:
    # Extra runtime_flags come last, so a caller-supplied
    # --torq_device_allocator still wins. device_io defaults to on with dmabuf
    # (host inputs are not valid with dmabuf) and off otherwise.
    if device_io is None:
        device_io = tda == "dmabuf"
    if model_path is None:
        from app_utils.paths import MODELS_DIR
        from app_utils.torq_examples.moonshine.setup_demo import (
            MOONSHINE_HF_REPO_MAP,
            download_moonshine,
        )
        from app_utils.torq_examples.utils.download import local_model_dir

        model_path = local_model_dir(model_name, MOONSHINE_HF_REPO_MAP, base_dir=MODELS_DIR)
        if model_path is None:
            dirs = download_moonshine([model_name], base_dir=MODELS_DIR)
            model_path = dirs[model_name]

    return MoonshineRunner(
        model_path,
        input_freq=input_freq,
        n_threads=n_threads,
        runtime_flags=build_runtime_flags(tda, runtime_flags),
        device_io=device_io,
    )


__all__ = [
    "MoonshineRunner",
    "load_moonshine",
]
