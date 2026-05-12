# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

from .download import (
    MOONSHINE_HF_REPO_MAP,
    download_moonshine,
    local_moonshine_model_dir,
    moonshine_repo_id,
)

__all__ = [
    "MoonshineRunner",
    "load_moonshine",
    "download_moonshine",
    "local_moonshine_model_dir",
    "moonshine_repo_id",
    "MOONSHINE_HF_REPO_MAP",
]


def __getattr__(name):
    if name in {"MoonshineRunner", "load_moonshine"}:
        from .runner import MoonshineRunner, load_moonshine

        return {
            "MoonshineRunner": MoonshineRunner,
            "load_moonshine": load_moonshine,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
