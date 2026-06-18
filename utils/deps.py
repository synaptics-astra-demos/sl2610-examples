# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Synaptics Incorporated.

from __future__ import annotations

from third_party.torq_examples.utils import deps as _torq_deps

MissingRequirementsError = _torq_deps.MissingRequirementsError
check_requirements = _torq_deps.check_requirements

__all__ = [
    "MissingRequirementsError",
    "check_requirements",
]
