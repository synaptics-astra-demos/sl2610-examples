#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Synaptics Incorporated.

"""Smoke-test every app's CLI on the board with its default arguments.

Run from anywhere, with the project venv active, on the target board:

    python tests/smoke_cli_apps.py              # run everything
    python tests/smoke_cli_apps.py --list       # show the cases
    python tests/smoke_cli_apps.py --only image_classification gemma_translate
    python tests/smoke_cli_apps.py --skip jellectronica --strict

Before the apps are tested, the top-level ``requirements.txt`` (the dependencies the
examples share) is installed once with pip. Before each app, its own ``requirements.txt``
(only what the other apps do not need) is installed and its ``setup_demo.py`` is run to
download or refresh the models, each once per app directory. Skip the installs with
--skip-install and the downloads with --skip-setup for offline runs.
Each case then runs the app's ``run_cli.py`` from the app's own directory (so
the relative default model paths resolve) and checks:

* one-shot apps: exit code 0, expected output present, no error signatures;
* long-running apps (live mic, web server): the app reaches its "ready" state,
  stays alive for a few seconds, and prints no error signatures.

Results:

* PASS  - the checks held.
* FAIL  - the app ran and something is wrong (crash, bad exit, missing output).
* SETUP - the app could not run here (setup_demo.py failed, or a model or
          Python package is missing). This is not a regression; pass --strict
          to treat it as a failure.

Only the Python standard library is used. Full per-case logs are written to
--log-dir. Exit code is 1 if any case FAILs (or SETUPs with --strict).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "samples"

# Output that means the app broke, wherever it appears.
ERROR_PATTERNS = [
    r"Traceback \(most recent call last\)",
    r"Inference failed",
    r"failed to writeXram",
    r"xram_write",
    r"RuntimeError",
]

# Output that means the app could not run here at all (not a regression).
SETUP_PATTERNS = [
    r"ModuleNotFoundError",
    r"Missing packages?:",
    r"Run: python setup_demo\.py",
    r"Model file not found",
    r"No such file or directory: .*(vmfb|gguf)",
]

# Refuse to run the install/download steps below this much free disk (GB): a full
# disk during a model download has hung the board.
MIN_FREE_GB = 1.5

# Noise to hide when printing a failure tail.
NOISE = re.compile(r"^\s+0x[0-9a-f]+ .*<\?\?\?>")


@dataclass
class Case:
    name: str
    cwd: str  # directory under the repo root
    args: list[str]
    kind: str = "oneshot"  # "oneshot" | "service"
    stdin: str | None = None
    # Paths (relative to cwd) that must exist, else the case is SETUP.
    requires: list[str] = field(default_factory=list)
    # Regexes that must all match the output (oneshot) / appear (service).
    expect: list[str] = field(default_factory=list)
    # Regexes that must NOT match, in addition to ERROR_PATTERNS.
    forbid: list[str] = field(default_factory=list)
    timeout: int = 300  # seconds; for services, the time allowed to become ready
    # service only: output regex, or URL, that signals ready; then stay up this long.
    ready_regex: str | None = None
    ready_url: str | None = None
    settle: int = 5
    note: str = ""

    @property
    def script(self) -> str:
        return "run_cli.py"

    @property
    def has_requirements(self) -> bool:
        """True if the app's requirements.txt lists at least one requirement (comments don't count)."""
        path = REPO / self.cwd / "requirements.txt"
        if not path.is_file():
            return False
        return any(line.strip() and not line.strip().startswith("#") for line in path.read_text().splitlines())

    @property
    def has_setup(self) -> bool:
        return (REPO / self.cwd / "setup_demo.py").is_file()


# Stand-in for the repository root, used to run the top-level pip install.
ROOT_CASE = Case(name="top-level", cwd=".", args=[])

CASES = [
    Case(
        name="image_classification",
        cwd="image_classification",
        args=["--image", str(SAMPLES / "cat.jpg")],
        requires=["../models/Synaptics/mobilenet_v2-int8-torq/MobileNetv2_int8.vmfb"],
        expect=[r"Top Prediction:", r"(?i)cat|tabby|lynx|tiger"],
        timeout=180,
    ),
    Case(
        name="object_detection",
        cwd="object_detection",
        args=["--image", str(SAMPLES / "bus.jpg")],
        requires=["../models/Synaptics/yolov8-od-nano-320-int8-torq/yolo_od.vmfb"],
        expect=[r"\[4/4\] Detections:"],
        forbid=[r"No objects detected"],
        timeout=180,
    ),
    Case(
        name="object_detection_video",
        cwd="object_detection",
        args=["--video", str(SAMPLES / "jellyfish_short.mp4")],  # first 5 s (152 frames) of jellyfish.mp4
        requires=["../models/Synaptics/yolov8-od-nano-320-int8-torq/yolo_od.vmfb"],
        timeout=120,
    ),
    Case(
        name="scene_description",
        cwd="scene_description",
        args=["--image", str(SAMPLES / "dog_bike_car.jpg")],
        requires=["../models/Synaptics/LiquidAI-LFM2-VL-450M/config.json"],
        expect=[r"\[3/3\] Description:\n\S", r"tok/s"],
        timeout=300,
        note="Describes a sample picture; with no --image the CLI takes a photo from the camera.",
    ),
    Case(
        name="speech_to_text_streaming",
        cwd="speech_to_text_streaming",
        args=["--wav", str(SAMPLES / "sample.wav")],
        requires=["../models/Synaptics/moonshine-streaming-tiny-torq"],
        expect=[r"Transcribing"],
        timeout=300,
    ),
    Case(
        name="gemma_translate",
        cwd="gemma_translate",
        args=["--text"],
        stdin="Hello, how are you today?\n/q\n",
        requires=["../models/Synaptics/gemma-3-270m-it-torq"],
        expect=[r"\[Translation\]", r"TTFT="],
        timeout=300,
        note="--text feeds a typed phrase instead of the mic, so only Gemma is exercised.",
    ),
    Case(
        name="Function_calling",
        cwd="Function_calling",
        args=["--prompt", "What is the system temperature?"],
        requires=["../models/functiongemma-physical-ai-v10-Q5_K_M.gguf"],
        expect=[r"\d+ tool calls?"],
        timeout=300,
        note="--prompt runs one turn and exits instead of the interactive REPL.",
    ),
    Case(
        name="speech_to_text",
        cwd="speech_to_text",
        args=["--audio-device", "default"],
        kind="service",
        requires=["../models/Synaptics/moonshine-tiny-bf16-torq"],
        ready_regex=r"Moonshine model loaded successfully",
        timeout=180,
        note="Live-mic app: checks it loads and keeps running. --audio-device avoids the interactive prompt.",
    ),
    Case(
        name="jellectronica",
        cwd="jellectronica",
        args=[],
        kind="service",
        requires=["../models/Synaptics/yolov8-od-nano-jellyfish-int8-torq/moon320_int8.vmfb"],
        ready_url="http://127.0.0.1:5002/",
        timeout=180,
        note="Web server: checks it serves HTTP on the default port.",
    ),
]


@dataclass
class Result:
    case: Case
    status: str  # PASS | FAIL | SETUP
    reason: str
    seconds: float
    log: Path
    tail: str = ""


class Capture:
    """Collect a process's combined output on a background thread."""

    def __init__(self, proc: subprocess.Popen, log_file):
        self.proc = proc
        self._chunks: list[str] = []
        self._log = log_file
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        for line in iter(self.proc.stdout.readline, ""):
            with self._lock:
                self._chunks.append(line)
            self._log.write(line)
            self._log.flush()

    def text(self) -> str:
        with self._lock:
            return "".join(self._chunks)

    def join(self, timeout: float = 5) -> None:
        self._thread.join(timeout)


def kill_group(proc: subprocess.Popen, sig: int = signal.SIGINT, wait: float = 10) -> None:
    """Signal the whole process group, escalating to SIGKILL."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
        proc.wait(timeout=wait)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def start(case: Case, log_file) -> tuple[subprocess.Popen, Capture]:
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        [sys.executable, case.script, *case.args],
        cwd=REPO / case.cwd,
        env=env,
        stdin=subprocess.PIPE if case.stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,  # own process group so we can stop children too
    )
    if case.stdin is not None:
        proc.stdin.write(case.stdin)
        proc.stdin.close()
    return proc, Capture(proc, log_file)


def find(patterns: list[str], text: str) -> str | None:
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(0)
    return None


def url_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return 200 <= r.status < 400
    except Exception:
        return False


HEARTBEAT_SECONDS = 20


def last_line(text: str, width: int = 90) -> str:
    """Most recent non-empty output line (progress bars rewrite with \\r), for the heartbeat."""
    for chunk in reversed(text.replace("\r", "\n").splitlines()):
        chunk = chunk.strip()
        if chunk and not NOISE.match(chunk):
            return chunk[:width]
    return ""


def tail_of(text: str, n: int = 14) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip() and not NOISE.match(ln)]
    return "\n".join(lines[-n:])


def run_step(cmd: list[str], case: Case, log_path: Path, timeout: int) -> tuple[bool, str, str]:
    """Run a prep command in the app's directory. Returns (ok, reason, output)."""
    label = " ".join(cmd[1:3] if cmd[1] == "-m" else cmd[1:2])
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO / case.cwd,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        out = out.decode(errors="replace") if isinstance(out, bytes) else out
        log_path.write_text(out)
        return False, f"{label} timed out after {timeout}s", out
    log_path.write_text(proc.stdout)
    if proc.returncode != 0:
        return False, f"{label} exited {proc.returncode}", proc.stdout
    return True, "ok", proc.stdout


def run_case(case: Case, log_dir: Path, prep_state: dict[tuple[str, str], tuple[bool, str, str]],
             install_timeout: int | None, setup_timeout: int | None) -> Result:
    log_path = log_dir / f"{case.name}.log"
    started = time.monotonic()

    def done(status: str, reason: str, text: str = "") -> Result:
        return Result(case, status, reason, time.monotonic() - started, log_path,
                      tail_of(text) if status != "PASS" else "")

    # Prep steps run once per directory: the shared requirements (repo root), then this app's
    # extra requirements, then the model download.
    steps = []
    if install_timeout is not None:
        pip_install = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        steps.append((ROOT_CASE, "pip install", pip_install, install_timeout))
        if case.has_requirements:
            steps.append((case, "pip install", pip_install, install_timeout))
    if setup_timeout is not None and case.has_setup:
        steps.append((case, "setup_demo.py", [sys.executable, "setup_demo.py"], setup_timeout))
    setup_failed = ""
    setup_text = ""
    for step_case, label, cmd, timeout in steps:
        key = (step_case.cwd, label)
        if key not in prep_state:
            where = "top-level" if step_case is ROOT_CASE else step_case.cwd
            print(f"    running {where}: {label} ...", flush=True)
            log = log_dir / f"{'top-level' if step_case is ROOT_CASE else step_case.cwd}.{label.split()[0]}.log"
            prep_state[key] = run_step(cmd, step_case, log, timeout)
        ok, why, out = prep_state[key]
        if not ok and not setup_failed:
            setup_failed, setup_text = f"{label}: {why}", out
    missing = [r for r in case.requires if not (REPO / case.cwd / r).exists()]
    if missing:
        if setup_failed:
            return done("SETUP", f"{setup_failed}; still missing {', '.join(missing)}", setup_text)
        return done("SETUP", f"missing {', '.join(missing)} (run the app's setup_demo.py)")
    if setup_failed:
        print(f"    warning: {setup_failed}; models are present, continuing", flush=True)

    with open(log_path, "w") as log:
        log.write(f"$ cd {case.cwd} && {sys.executable} {case.script} {' '.join(case.args)}\n\n")
        proc, cap = start(case, log)
        text = ""
        try:
            if case.kind == "oneshot":
                deadline = time.monotonic() + case.timeout
                next_beat = time.monotonic() + HEARTBEAT_SECONDS
                while proc.poll() is None:
                    now = time.monotonic()
                    if now >= deadline:
                        kill_group(proc, signal.SIGKILL)
                        cap.join()
                        return done("FAIL", f"timed out after {case.timeout}s", cap.text())
                    if now >= next_beat:
                        print(f"    ... {now - started:.0f}s elapsed (limit {case.timeout}s) | "
                              f"{last_line(cap.text())}", flush=True)
                        next_beat = now + HEARTBEAT_SECONDS
                    time.sleep(0.5)
                cap.join()
                text = cap.text()
                if proc.returncode != 0:
                    bad = find(SETUP_PATTERNS, text)
                    if bad:
                        return done("SETUP", f"cannot run here: {bad}", text)
                    return done("FAIL", f"exit code {proc.returncode}", text)
                bad = find(ERROR_PATTERNS + case.forbid, text)
                if bad:
                    return done("FAIL", f"error in output: {bad!r}", text)
                for pat in case.expect:
                    if not re.search(pat, text):
                        return done("FAIL", f"expected output not found: {pat!r}", text)
                return done("PASS", "ok", text)

            # service: wait for ready, then confirm it stays up
            deadline = time.monotonic() + case.timeout
            next_beat = time.monotonic() + HEARTBEAT_SECONDS
            ready = False
            while time.monotonic() < deadline and proc.poll() is None:
                text = cap.text()
                if find(ERROR_PATTERNS + case.forbid, text):
                    break
                if case.ready_regex and re.search(case.ready_regex, text):
                    ready = True
                    break
                if case.ready_url and url_ok(case.ready_url):
                    ready = True
                    break
                if time.monotonic() >= next_beat:
                    print(f"    ... waiting for ready, {time.monotonic() - started:.0f}s elapsed "
                          f"(limit {case.timeout}s) | {last_line(text)}", flush=True)
                    next_beat = time.monotonic() + HEARTBEAT_SECONDS
                time.sleep(1)
            if ready:
                time.sleep(case.settle)
            text = cap.text()  # snapshot before we signal it, so shutdown noise is ignored
            alive = proc.poll() is None
            bad = find(ERROR_PATTERNS + case.forbid, text)
            if bad:
                return done("FAIL", f"error in output: {bad!r}", text)
            if not ready:
                if not alive:
                    bad = find(SETUP_PATTERNS, text)
                    if bad:
                        return done("SETUP", f"cannot run here: {bad}", text)
                    return done("FAIL", f"exited early with code {proc.returncode}", text)
                return done("FAIL", f"not ready after {case.timeout}s", text)
            if not alive:
                return done("FAIL", f"exited with code {proc.returncode} after becoming ready", text)
            return done("PASS", f"ready and still running after {case.settle}s", text)
        finally:
            kill_group(proc)
            cap.join()


def main() -> int:
    names = [c.name for c in CASES]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list cases and exit")
    ap.add_argument("--only", nargs="+", choices=names, metavar="NAME", help="run only these cases")
    ap.add_argument("--skip", nargs="+", choices=names, metavar="NAME", default=[], help="skip these cases")
    ap.add_argument("--skip-install", action="store_true",
                    help="do not pip install the requirements files first (offline runs)")
    ap.add_argument("--install-timeout", type=int, default=1800,
                    help="seconds allowed for each pip install (default: %(default)s)")
    ap.add_argument("--skip-setup", action="store_true",
                    help="do not run each app's setup_demo.py first (offline runs)")
    ap.add_argument("--setup-timeout", type=int, default=1800,
                    help="seconds allowed for each setup_demo.py (default: %(default)s)")
    ap.add_argument("--strict", action="store_true", help="treat SETUP results as failures")
    ap.add_argument("--log-dir", type=Path, default=None, help="where to write per-case logs (default: a temp dir)")
    args = ap.parse_args()

    if args.list:
        for c in CASES:
            print(f"{c.name:<26} {c.kind:<8} {c.cwd}/{c.script} {' '.join(c.args)}")
        return 0

    selected = [c for c in CASES if (not args.only or c.name in args.only) and c.name not in args.skip]
    log_dir = args.log_dir or Path(tempfile.mkdtemp(prefix="sl2610_smoke_"))
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(selected)} case(s) with {sys.executable}\nLogs: {log_dir}\n")

    free_gb = shutil.disk_usage(REPO).free / 1e9
    if free_gb < MIN_FREE_GB and not (args.skip_install and args.skip_setup):
        print(f"ERROR: only {free_gb:.1f} GB free on the filesystem holding {REPO}; the install and "
              f"model-download steps need about {MIN_FREE_GB:.1f} GB and can fill the disk and hang the "
              "board. Free some space, or pass --skip-install --skip-setup.")
        return 2

    results: list[Result] = []
    prep_state: dict[tuple[str, str], tuple[bool, str, str]] = {}
    install_timeout = None if args.skip_install else args.install_timeout
    setup_timeout = None if args.skip_setup else args.setup_timeout
    for i, case in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {case.name} ...", flush=True)
        try:
            res = run_case(case, log_dir, prep_state, install_timeout, setup_timeout)
        except KeyboardInterrupt:
            print("\ninterrupted")
            break
        results.append(res)
        print(f"    {res.status}  {res.reason}  ({res.seconds:.1f}s)")
        if res.tail:
            print("    --- last output ---")
            print("\n".join("    " + ln for ln in res.tail.splitlines()))
        print(flush=True)

    print("=" * 72)
    for r in results:
        print(f"{r.status:<6} {r.case.name:<26} {r.seconds:6.1f}s  {r.reason}")
    counts = {s: sum(r.status == s for r in results) for s in ("PASS", "FAIL", "SETUP")}
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SETUP']} need setup")
    print("Not covered (GUI-only, no CLI)")
    bad = counts["FAIL"] + (counts["SETUP"] if args.strict else 0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
