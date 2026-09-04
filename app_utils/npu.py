# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Board-level NPU setup, re-exported from torq_examples so every app shares one import path.

The Torq runtime is what actually runs on the NPU, so its own clock/frequency setup is the
source of truth here rather than a local reimplementation — this module is just the stable
``app_utils`` entry point apps use instead of reaching into ``app_utils.torq_examples`` directly.
"""

from app_utils.torq_examples.utils.npu import configure_npu_userspace_frequency, enable_npu_clock
from app_utils.torq_examples.utils.runtime import (build_runtime_flags, cleanup_npu_after_inference,
                                                    setup_npu_for_inference)

__all__ = [
    "build_runtime_flags", "cleanup_npu_after_inference", "configure_npu_userspace_frequency",
    "enable_npu_clock", "setup_npu_for_inference",
]
