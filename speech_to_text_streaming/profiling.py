# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Per-chunk worker timing profiler for app.py's --profile flag.

See ``plot_profile.py`` to turn the raw arrays this dumps into plots.
"""

import sys
import time
from pathlib import Path

import numpy as np


class WorkerProfiler:
    def __init__(self, chunk_budget_ms: float):
        self.chunk_budget_ms = chunk_budget_ms
        self.chunk_ms          = []
        self.chunk_had_decode  = []
        self.encode_ms         = []
        self.decode_ms         = []
        self.decode_steps      = []
        self.queue_depth       = []
        self.missed            = 0
        self.t_start           = time.perf_counter()

    def record_chunk(self, ms: float, had_decode: bool):
        self.chunk_ms.append(ms)
        self.chunk_had_decode.append(had_decode)
        if ms > self.chunk_budget_ms:
            self.missed += 1

    @staticmethod
    def _stats(samples):
        if not len(samples):
            return dict(n=0, mean=0.0, p50=0.0, p95=0.0, p99=0.0, max=0.0)
        a = np.asarray(samples, dtype=np.float64)
        return dict(n=len(a), mean=float(a.mean()),
                    p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)),
                    p99=float(np.percentile(a, 99)), max=float(a.max()))

    def summary(self, out_dir=None):
        import numpy as _np
        cm  = _np.asarray(self.chunk_ms, dtype=_np.float64)
        had = _np.asarray(self.chunk_had_decode, dtype=bool)
        cheap = cm[~had] if had.any() else cm
        heavy = cm[had]

        def row(label, s):
            return (f"    {label:18s} n={s['n']:>5d}  p50={s['p50']:7.2f}  "
                     f"p95={s['p95']:7.2f}  p99={s['p99']:7.2f}  max={s['max']:7.2f} ms")

        n = len(cm)
        miss_pct = 100.0 * self.missed / n if n else 0.0
        col = "\033[32m" if miss_pct < 1 else "\033[33m" if miss_pct < 5 else "\033[31m"
        print("\n" + "=" * 64, file=sys.stderr)
        print("  Worker profile — real-time keep-up (2-Split)", file=sys.stderr)
        print("=" * 64, file=sys.stderr)
        print(f"  chunk budget: {self.chunk_budget_ms:.1f} ms/chunk   "
              f"chunks processed: {n}", file=sys.stderr)
        print(f"  missed real-time: {col}{self.missed} ({miss_pct:.1f}%)\033[0m",
              file=sys.stderr)
        print(row("chunk (all)",      self._stats(cm)),    file=sys.stderr)
        print(row("chunk (cheap)",    self._stats(cheap)), file=sys.stderr)
        print(row("chunk (w/ decode)",self._stats(heavy)), file=sys.stderr)
        print(row("encode/chunk",     self._stats(self.encode_ms)), file=sys.stderr)
        print(row("decode/call",      self._stats(self.decode_ms)), file=sys.stderr)
        print(row("decode steps/call", self._stats(self.decode_steps)), file=sys.stderr)

        # Amortize decoder forward passes over the audio cadence.
        # 1 chunk = feature_stride (4) frames; decode runs only on trigger chunks.
        steps     = _np.asarray(self.decode_steps, dtype=_np.float64)
        n_decodes = len(steps)
        total_steps = float(steps.sum()) if n_decodes else 0.0
        frames_per_chunk = 4  # feature_stride
        steps_per_chunk  = total_steps / n if n else 0.0
        steps_per_frame  = total_steps / (n * frames_per_chunk) if n else 0.0
        print(f"    decoder forward passes: {int(total_steps)} over {n_decodes} decode calls",
              file=sys.stderr)
        print(f"    amortized: {steps_per_chunk:.2f} steps/chunk   "
              f"{steps_per_frame:.2f} steps/frame", file=sys.stderr)
        if self.queue_depth:
            qd = _np.asarray([q for _, q in self.queue_depth], dtype=_np.float64)
            print(f"    queue depth        max={int(qd.max())}  "
                  f"mean={qd.mean():.2f}  (sustained growth ⇒ falling behind)",
                  file=sys.stderr)

        # ── Real-time keep-up: work-time / audio-time (≥1.0× ⇒ cannot keep up) ──
        # Each processed chunk == chunk_budget_ms of audio (e.g. 80 ms).
        audio_ms   = n * self.chunk_budget_ms
        audio_s    = audio_ms / 1000.0
        enc_total  = float(_np.sum(self.encode_ms)) if self.encode_ms else 0.0
        dec_total  = float(_np.sum(self.decode_ms)) if self.decode_ms else 0.0
        work_total = float(cm.sum())

        def _rtf_row(label, work_ms):
            rtf = work_ms / audio_ms if audio_ms else 0.0
            c = "\033[32m" if rtf < 0.8 else "\033[33m" if rtf < 1.0 else "\033[31m"
            return (f"    {label:8s} {c}{rtf:5.2f}x real-time\033[0m"
                    f"  ({work_ms/1000:6.1f}s work / {audio_s:5.1f}s audio)")

        print("  ── keep-up (work/audio; total ≥ 1.0x ⇒ cannot keep up) ──", file=sys.stderr)
        print(_rtf_row("total",   work_total), file=sys.stderr)
        print(_rtf_row("encoder", enc_total),  file=sys.stderr)
        print(_rtf_row("decoder", dec_total),  file=sys.stderr)

        if total_steps and audio_s:
            print(f"    decoder: {dec_total / total_steps:5.1f} ms/token   "
                  f"{total_steps / audio_s:5.1f} steps/s  "
                  f"(speech is ~4-6.5 tok/s; more ⇒ re-decode waste)", file=sys.stderr)

        # Queue-depth slope: a sustained positive trend is the definitive
        # "falling behind" signal (max/mean alone can hide it).
        if self.queue_depth and len(self.queue_depth) >= 2:
            ts = _np.asarray([t for t, _ in self.queue_depth], dtype=_np.float64)
            qz = _np.asarray([q for _, q in self.queue_depth], dtype=_np.float64)
            if ts.max() > ts.min():
                slope = float(_np.polyfit(ts, qz, 1)[0])  # queue items / second
                c = "\033[32m" if slope < 0.5 else "\033[33m" if slope < 2 else "\033[31m"
                print(f"    queue growth: {c}{slope:+.2f} items/s\033[0m"
                      f"  (>0 sustained ⇒ backlog growing)", file=sys.stderr)

        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            _np.save(out_dir / "worker_chunk_ms.npy", cm)
            _np.save(out_dir / "worker_chunk_had_decode.npy", had)
            _np.save(out_dir / "worker_encode_ms.npy", _np.asarray(self.encode_ms))
            _np.save(out_dir / "worker_decode_ms.npy", _np.asarray(self.decode_ms))
            _np.save(out_dir / "worker_decode_steps.npy", _np.asarray(self.decode_steps))
            _np.save(out_dir / "worker_queue_depth.npy",
                     _np.asarray(self.queue_depth, dtype=_np.float64))
            _np.save(out_dir / "chunk_budget_ms.npy", _np.array(self.chunk_budget_ms))
            print(f"  dumped raw arrays to {out_dir}/", file=sys.stderr)
        print("=" * 64, file=sys.stderr)
