# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Synaptics Incorporated.

from __future__ import annotations

import os

_TorqMoonshineRunner = None


def _runner_class():
    global _TorqMoonshineRunner
    if _TorqMoonshineRunner is None:
        from platform.torq_examples.moonshine.src.runner import (
            MoonshineRunner as TorqMoonshineRunner,
        )

        _TorqMoonshineRunner = TorqMoonshineRunner
    return _TorqMoonshineRunner


def MoonshineRunner(*args, **kwargs):
    return _runner_class()(*args, **kwargs)


def load_moonshine(
    model_path: str | os.PathLike | None = None,
    *,
    model_name: str = "tiny-en",
    input_freq: int = 16000,
    n_threads: int | None = None,
    runtime_flags: list[str] | None = None,
    device_io: bool = False,
) -> MoonshineRunner:
    if model_path is None:
        from .download import download_moonshine, local_moonshine_model_dir

        model_path = local_moonshine_model_dir(model_name)
        if model_path is None:
            dirs = download_moonshine([model_name])
            model_path = dirs[model_name]

    return _runner_class()(
        model_path,
        input_freq=input_freq,
        n_threads=n_threads,
        runtime_flags=runtime_flags,
        device_io=device_io,
    )


__all__ = [
    "MoonshineRunner",
    "load_moonshine",
]
