"""Pin all CPU cores to the ``performance`` scaling governor.

The 2-core Cortex-A55 in the SL2619 defaults to ``schedutil`` in the OOBE
image. On a cold prefill that adds roughly 3x to the first-token latency
of the FunctionGemma 270M model, since the cores stay parked at the lower
P-state through the prompt-evaluation phase. This module flips every CPU
to ``performance`` at demo startup. The change is reverted on the next
reboot or by writing a different value back to ``scaling_governor``.

Best-effort: silently no-op if sysfs is not writable (non-root, dev
laptop, missing files).
"""

from __future__ import annotations

from pathlib import Path

_CPUFREQ_ROOT = Path("/sys/devices/system/cpu")


def ensure_performance_governor() -> None:
    """Write "performance" to every CPU's ``scaling_governor`` if needed.

    Prints one line listing which CPUs changed, so the action is visible.
    Silent if every CPU was already on ``performance`` or if the sysfs
    isn't writable.
    """
    if not _CPUFREQ_ROOT.exists():
        return
    changed: list[str] = []
    try:
        for cpu in sorted(_CPUFREQ_ROOT.glob("cpu[0-9]*")):
            gov = cpu / "cpufreq" / "scaling_governor"
            if not gov.exists():
                continue
            if gov.read_text().strip() != "performance":
                gov.write_text("performance")
                changed.append(cpu.name)
    except (PermissionError, OSError):
        return
    if changed:
        print(f"[setup] CPU governor: pinned {', '.join(changed)} to performance")
