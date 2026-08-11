# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Plot the raw arrays dumped by ``infer.py --profile`` (see ``WorkerProfiler``).

Run after a profiled session:

    python src/infer.py -m ../models/... --wav sample.wav --profile
    python src/plot_profile.py

Reads ``profile_results/*.npy`` (default: ``speech_to_text_stream/profile_results``,
i.e. sibling to ``src/``) and writes PNGs to ``profile_results/plots/``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is not installed. Please run:", file=sys.stderr)
    print("  pip install matplotlib", file=sys.stderr)
    sys.exit(1)

DEFAULT_PROFILE_DIR = Path(__file__).resolve().parent.parent / "profile_results"


def _load(profile_dir: Path, name: str, required: bool = True):
    path = profile_dir / f"{name}.npy"
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing {path} — run infer.py with --profile first.")
        return None
    return np.load(path)


def plot_chunk_times(chunk_ms, had_decode, budget_ms, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    idx = np.arange(len(chunk_ms))
    ax.scatter(idx[~had_decode], chunk_ms[~had_decode], s=8, alpha=0.6,
               color="tab:blue", label="cheap (no decode)")
    ax.scatter(idx[had_decode], chunk_ms[had_decode], s=8, alpha=0.6,
               color="tab:red", label="with decode")
    if budget_ms is not None:
        ax.axhline(budget_ms, color="black", linestyle="--", linewidth=1,
                   label=f"real-time budget ({budget_ms:.1f} ms)")
    ax.set_xlabel("chunk index")
    ax.set_ylabel("chunk time (ms)")
    ax.set_title("Worker chunk time per audio chunk")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "chunk_times.png", dpi=150)
    plt.close(fig)


def plot_latency_hist(data, title, xlabel, filename, out_dir, color):
    if data is None or len(data) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=30, color=color, alpha=0.85)
    for pct, style in ((50, "-"), (95, "--"), (99, ":")):
        v = np.percentile(data, pct)
        ax.axvline(v, color="black", linestyle=style, linewidth=1, label=f"p{pct}={v:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def plot_decode_steps(steps, out_dir):
    if steps is None or len(steps) == 0:
        return
    lo, hi = int(steps.min()), int(steps.max())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(steps, bins=range(lo, hi + 2), align="left", color="tab:orange", alpha=0.85)
    ax.set_xlabel("decoder forward passes per decode call")
    ax.set_ylabel("count")
    ax.set_title("Decoder steps per decode call")
    fig.tight_layout()
    fig.savefig(out_dir / "decode_steps.png", dpi=150)
    plt.close(fig)


def plot_queue_depth(queue_depth, out_dir):
    if queue_depth is None or queue_depth.ndim != 2 or len(queue_depth) == 0:
        return
    t, q = queue_depth[:, 0], queue_depth[:, 1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, q, color="tab:purple", linewidth=1)
    if len(t) >= 2 and t.max() > t.min():
        slope, intercept = np.polyfit(t, q, 1)
        ax.plot(t, slope * t + intercept, color="black", linestyle="--", linewidth=1,
               label=f"trend: {slope:+.2f} items/s")
        ax.legend()
    ax.set_xlabel("time (s)")
    ax.set_ylabel("queue depth (chunks)")
    ax.set_title("Audio queue depth over time (sustained growth ⇒ falling behind)")
    fig.tight_layout()
    fig.savefig(out_dir / "queue_depth.png", dpi=150)
    plt.close(fig)


def plot_realtime_factor(chunk_ms, budget_ms, out_dir):
    """Cumulative work time / cumulative audio time over the run (running RTF)."""
    if budget_ms is None or len(chunk_ms) == 0:
        return
    audio_s = np.arange(1, len(chunk_ms) + 1) * (budget_ms / 1000.0)
    work_s  = np.cumsum(chunk_ms) / 1000.0
    rtf     = work_s / audio_s
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(audio_s, rtf, color="tab:green", label="running RTF (work/audio)")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="real-time (1.0x)")
    ax.set_xlabel("audio time (s)")
    ax.set_ylabel("work / audio")
    ax.set_title("Real-time factor over the run (>1.0x ⇒ cannot keep up)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "realtime_factor.png", dpi=150)
    plt.close(fig)


def main(args: argparse.Namespace):
    profile_dir = Path(args.profile_dir)
    if not profile_dir.is_dir():
        print(f"Error: profile dir {profile_dir} not found. "
              f"Run infer.py with --profile first.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else profile_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_ms     = _load(profile_dir, "worker_chunk_ms")
    had_decode   = _load(profile_dir, "worker_chunk_had_decode").astype(bool)
    encode_ms    = _load(profile_dir, "worker_encode_ms", required=False)
    decode_ms    = _load(profile_dir, "worker_decode_ms", required=False)
    decode_steps = _load(profile_dir, "worker_decode_steps", required=False)
    queue_depth  = _load(profile_dir, "worker_queue_depth", required=False)
    budget_arr   = _load(profile_dir, "chunk_budget_ms", required=False)
    budget_ms    = float(budget_arr) if budget_arr is not None else None

    plot_chunk_times(chunk_ms, had_decode, budget_ms, out_dir)
    plot_latency_hist(encode_ms, "Encoder time per chunk", "encode time (ms)",
                      "encode_times.png", out_dir, color="tab:blue")
    plot_latency_hist(decode_ms, "Decode call latency", "decode time (ms)",
                      "decode_times.png", out_dir, color="tab:red")
    plot_decode_steps(decode_steps, out_dir)
    plot_queue_depth(queue_depth, out_dir)
    plot_realtime_factor(chunk_ms, budget_ms, out_dir)

    print(f"Wrote plots to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot infer.py --profile dumps.")
    parser.add_argument("--profile-dir", type=str, default=str(DEFAULT_PROFILE_DIR),
                        help=f"Directory holding the --profile .npy dumps (default: {DEFAULT_PROFILE_DIR})")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Directory to write plots to (default: <profile-dir>/plots)")
    main(parser.parse_args())
