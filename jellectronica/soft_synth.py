"""
SoftSynth — Pure-Python/NumPy Software Synthesizer
====================================================
Zero system dependencies.  Only needs Python stdlib + NumPy.

Audio output:
  Linux  → pipes raw S16_LE PCM to `aplay` via subprocess (ALSA)
  macOS  → uses `sounddevice` if available, else silent

Provides noteon/noteoff/cc interface matching standard MIDI synthesizer APIs
so MusicEngine can use it directly.

Design priorities:
  1. Clean audio — float64 rendering, modular phase accumulators (no drift)
  2. Performance — all DSP is vectorized NumPy (no per-sample Python loops)
  3. Beauty — pure sine tones with gentle detuning, exponential decay
  4. Reliability — 1MB kernel pipe buffer absorbs GIL stalls from inference
"""

import shutil
import subprocess
import sys
import threading
import time

import numpy as np

# ── Audio constants ────────────────────────────────────────────
SAMPLE_RATE = 44100
CHUNK_SIZE = 4096         # ~93ms per chunk — large to reduce write frequency
MAX_VOICES = 12
TWO_PI = 2.0 * np.pi

# Time increment per sample (computed once)
_DT = 1.0 / SAMPLE_RATE

# Pre-compute frequency table (all 128 MIDI notes)
_FREQ_TABLE = 440.0 * (2.0 ** ((np.arange(128) - 69) / 12.0))

def midi_to_freq(note: int) -> float:
    return float(_FREQ_TABLE[min(max(note, 0), 127)])


# ── Channel Timbre Definitions ─────────────────────────────────
# Each partial: (freq_ratio, amplitude, decay_rate)
# Amplitudes are intentionally low — they stack across voices.
# decay_rate: higher = faster fade (seconds⁻¹)
TIMBRES = {
    # Channel 0: Warm Pad — slow, gentle, evolving
    0: {
        "name": "Warm Pad",
        "partials": [
            (1.0,   0.35, 0.15),    # fundamental — very slow decay
            (2.0,   0.10, 0.25),    # octave — subtle
            (1.003, 0.25, 0.15),    # slight detune = gentle chorus
        ],
        "attack": 0.4,
    },
    # Channel 1: Glass Bell — crystalline, pure
    1: {
        "name": "Glass Bell",
        "partials": [
            (1.0,   0.40, 1.5),
            (2.0,   0.15, 2.5),
            (3.0,   0.05, 4.0),
        ],
        "attack": 0.003,
    },
    # Channel 2: Sub Bass — deep, warm
    2: {
        "name": "Sub Bass",
        "partials": [
            (1.0,   0.45, 0.4),
            (2.0,   0.05, 0.8),
        ],
        "attack": 0.02,
    },
    # Channel 3: Wind Chime — ethereal, sparkling
    3: {
        "name": "Wind Chime",
        "partials": [
            (1.0,   0.30, 2.0),
            (2.0,   0.12, 3.0),
            (3.0,   0.06, 4.5),
            (1.002, 0.20, 2.0),
        ],
        "attack": 0.001,
    },
    # Channel 4: AI Sine — ultra-soft, pure sine for MelodyRNN accompaniment
    4: {
        "name": "AI Sine",
        "partials": [
            (1.0,   0.30, 0.3),    # Pure fundamental, slow decay
            (1.002, 0.15, 0.35),   # Tiny detune for warmth/chorus
        ],
        "attack": 0.5,             # Very slow attack — ethereal
    },
}

DEFAULT_TIMBRE = TIMBRES[0]


# ── Voice ──────────────────────────────────────────────────────
class Voice:
    """A single note. Uses modular phase accumulators for clean audio.

    Key design: phase accumulators wrap at 2π on every render call,
    keeping sin() arguments small. This prevents the float precision
    degradation that causes digital distortion on sustained notes.
    All rendering uses float64 for maximum precision.
    """
    __slots__ = ("channel", "note", "velocity", "freq", "timbre",
                 "_phases", "_elapsed", "release_time", "active",
                 "_attack", "_partial_data")

    def __init__(self, channel: int, note: int, velocity: int, timbre: dict):
        self.channel = channel
        self.note = note
        self.velocity = velocity / 127.0
        self.freq = midi_to_freq(note)
        self.timbre = timbre
        self._attack = timbre["attack"]
        self._elapsed = 0.0
        self.release_time = None
        self.active = True

        # Pre-compute per-partial data for rendering
        partials = timbre["partials"]
        n = len(partials)
        # Phase accumulators — random initial phase for natural variation
        self._phases = np.random.uniform(0, TWO_PI, n)
        # Pre-compute: (frequency, amplitude, decay) arrays
        self._partial_data = np.array(
            [(self.freq * r, a, d) for r, a, d in partials], dtype=np.float64)

    def release(self):
        if self.release_time is None:
            self.release_time = self._elapsed
            self.active = False

    def is_silent(self) -> bool:
        min_decay = float(np.min(self._partial_data[:, 2]))
        level = np.exp(-min_decay * self._elapsed) * self.velocity
        if self.release_time is not None:
            level *= np.exp(-5.0 * (self._elapsed - self.release_time))
        return level < 0.0005

    def render(self, n_samples: int) -> np.ndarray | None:
        """Render n_samples of audio. Returns float64 array or None if silent."""
        if self.is_silent():
            return None

        # Time offset array for this chunk (float64 for precision)
        dt = np.arange(n_samples, dtype=np.float64) * _DT
        elapsed = self._elapsed
        attack = self._attack

        signal = np.zeros(n_samples, dtype=np.float64)

        for i in range(len(self._partial_data)):
            freq_i = self._partial_data[i, 0]
            amp_i = self._partial_data[i, 1]
            decay_i = self._partial_data[i, 2]

            # Phase: start from accumulator, advance by freq * dt
            # Keeping phases modular prevents float64 precision loss
            phase = self._phases[i] + TWO_PI * freq_i * dt
            signal += amp_i * np.sin(phase)

            # Advance phase accumulator (wrap at 2π)
            total_advance = TWO_PI * freq_i * n_samples * _DT
            self._phases[i] = (self._phases[i] + total_advance) % TWO_PI

        # Exponential decay envelope
        env = np.exp(-np.min(self._partial_data[:, 2]) * (elapsed + dt))
        # Per-partial decay weighting (approximate — use slowest)
        signal *= env

        # Attack ramp
        if elapsed < attack:
            attack_env = np.clip((elapsed + dt) / attack, 0.0, 1.0)
            signal *= attack_env

        # Release fade
        if self.release_time is not None:
            rel = self._elapsed - self.release_time
            signal *= np.exp(-5.0 * (rel + dt))

        # Apply velocity and advance elapsed time
        signal *= self.velocity
        self._elapsed += n_samples * _DT

        return signal


# ── Simple Delay Reverb (vectorized) ──────────────────────────
class SimpleReverb:
    """Single-tap delay reverb. Fully vectorized."""

    def __init__(self, delay_ms=200, feedback=0.25, mix=0.20):
        self._delay_len = int(SAMPLE_RATE * delay_ms / 1000.0)
        self._buffer = np.zeros(self._delay_len, dtype=np.float64)
        self._pos = 0
        self._feedback = feedback
        self._mix = mix

    def process(self, samples: np.ndarray) -> np.ndarray:
        n = len(samples)
        wet = np.zeros(n, dtype=np.float64)
        remaining = n
        offset = 0
        while remaining > 0:
            chunk = min(remaining, self._delay_len - self._pos)
            buf_slice = self._buffer[self._pos:self._pos + chunk]
            sig_slice = samples[offset:offset + chunk]
            wet[offset:offset + chunk] = buf_slice
            self._buffer[self._pos:self._pos + chunk] = (
                sig_slice + buf_slice * self._feedback
            )
            self._pos = (self._pos + chunk) % self._delay_len
            offset += chunk
            remaining -= chunk
        return samples * (1.0 - self._mix) + wet * self._mix


# ── Audio Output Backends ──────────────────────────────────────
class _AplayOutput:
    """Stream raw PCM to aplay (Linux ALSA). Zero dependencies.

    Uses a 1MB kernel pipe buffer to absorb Python GIL stalls (>6 sec
    of audio buffered). Outputs stereo S16_LE for USB DAC compatibility.

    If no audio device is available, falls back to silent mode after
    a few retries to avoid log spam.
    """

    PIPE_BUF_BYTES = 1_048_576  # 1MB
    MAX_RESPAWNS = 3            # Give up after this many consecutive failures
    RESPAWN_BACKOFF_S = 2.0     # Seconds between respawn attempts

    def __init__(self, device: str | None = None):
        self._cmd = ["aplay", "-f", "S16_LE", "-r", str(SAMPLE_RATE),
                     "-c", "2", "-t", "raw", "-q",
                     "--buffer-time", "500000",
                     "--period-time", "50000"]
        if device:
            if device.startswith("hw:"):
                device = device.replace("hw:", "plughw:", 1)
            self._cmd.extend(["-D", device])
        self._proc = None
        self._consecutive_failures = 0
        self._gave_up = False
        self._last_respawn = 0.0

        # Probe for audio devices before first spawn.
        # Skip probe when a specific device is provided (user knows what
        # they want — could be bluealsa, pulse, or another virtual sink).
        if device is None and not self._probe_audio():
            print("[SoftSynth] ⚠ No ALSA audio devices found")
            print("[SoftSynth]   Options:")
            print("[SoftSynth]     • Plug in a USB audio DAC (board has no built-in audio)")
            print("[SoftSynth]     • Pair Bluetooth headphones: --alsa-device bluealsa")
            print("[SoftSynth]     • Run without audio: --no-audio")
            self._gave_up = True
            return

        self._spawn()

    @staticmethod
    def _probe_audio() -> bool:
        """Check if any ALSA playback devices exist (hardware or virtual).

        Uses 'aplay -L' which lists both hardware cards and virtual sinks
        (PulseAudio, bluealsa for Bluetooth, PipeWire, etc.)
        """
        try:
            result = subprocess.run(
                ["aplay", "-L"], capture_output=True, text=True, timeout=5)
            # aplay -L always lists 'null' — look for something more real
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line or line.startswith(" ") or line == "null":
                    continue
                # Any real device name (default, hw:*, plughw:*, bluealsa, pulse, etc.)
                if line in ("default",) or line.startswith(("hw:", "plughw:",
                            "sysdefault:", "bluealsa", "pulse")):
                    return True
            # Fallback: also check hardware cards directly
            hw = subprocess.run(
                ["aplay", "-l"], capture_output=True, text=True, timeout=5)
            return "card " in hw.stdout.lower()
        except Exception:
            return False

    def _spawn(self):
        try:
            if self._proc:
                try: self._proc.stdin.close()
                except Exception: pass
                try: self._proc.kill(); self._proc.wait(timeout=1)
                except Exception: pass
            # Capture stderr so we can report why aplay failed
            self._proc = subprocess.Popen(
                self._cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            # Enlarge kernel pipe buffer to absorb GIL stalls
            try:
                import fcntl
                fcntl.fcntl(self._proc.stdin.fileno(), 1031, self.PIPE_BUF_BYTES)
            except Exception:
                pass
            # Pre-fill with ~200ms silence for glitch-free start
            prefill = np.zeros(SAMPLE_RATE // 5 * 2, dtype=np.int16)
            self._proc.stdin.write(prefill.tobytes())

            # Give aplay a moment to fail on bad devices
            time.sleep(0.1)
            if self._proc.poll() is not None:
                stderr = ""
                try:
                    stderr = self._proc.stderr.read().decode(errors="replace").strip()
                except Exception:
                    pass
                raise RuntimeError(
                    f"aplay exited immediately (code {self._proc.returncode})"
                    + (f": {stderr}" if stderr else ""))

            # Success — reset failure counter
            self._consecutive_failures = 0
        except Exception as e:
            print(f"[SoftSynth] aplay spawn failed: {e}")
            self._proc = None
            self._consecutive_failures += 1

    def write(self, samples_f32: np.ndarray):
        if self._gave_up:
            return
        if self._proc is None or self._proc.poll() is not None:
            # Rate-limit respawn attempts
            now = time.time()
            if now - self._last_respawn < self.RESPAWN_BACKOFF_S:
                return
            self._last_respawn = now

            if self._consecutive_failures >= self.MAX_RESPAWNS:
                print(f"[SoftSynth] aplay failed {self.MAX_RESPAWNS} times — giving up")
                print("[SoftSynth]   Check audio device: aplay -l")
                print("[SoftSynth]   Continuing without audio")
                self._gave_up = True
                return

            print(f"[SoftSynth] aplay pipe broken — respawning "
                  f"(attempt {self._consecutive_failures + 1}/{self.MAX_RESPAWNS})...")
            self._spawn()
            if self._proc is None:
                return
        # Convert float64 → int16 stereo with headroom
        # Scale to 30000 (not 32767) to leave DAC headroom
        clipped = np.clip(samples_f32, -1.0, 1.0)
        pcm_mono = (clipped * 30000).astype(np.int16)
        pcm_stereo = np.column_stack((pcm_mono, pcm_mono))
        try:
            self._proc.stdin.write(pcm_stereo.tobytes())
        except (BrokenPipeError, OSError):
            pass

    def close(self):
        if self._proc:
            try: self._proc.stdin.close()
            except Exception: pass
            try: self._proc.stderr.close()
            except Exception: pass
            try: self._proc.terminate(); self._proc.wait(timeout=2)
            except Exception: pass


class _SounddeviceOutput:
    def __init__(self, device=None):
        import sounddevice as sd
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=CHUNK_SIZE, device=device)
        self._stream.start()

    def write(self, samples: np.ndarray):
        try:
            s = np.clip(samples, -1.0, 1.0).astype(np.float32).reshape(-1, 1)
            self._stream.write(s)
        except Exception:
            pass

    def close(self):
        try: self._stream.stop(); self._stream.close()
        except Exception: pass


class _NullOutput:
    def write(self, s): pass
    def close(self): pass


# ── Brickwall limiter with soft knee ──────────────────────────────
def _soft_limit(samples: np.ndarray, threshold: float = 0.4,
                ceiling: float = 0.85) -> np.ndarray:
    """Soft-knee limiter with hard ceiling.

    Transparent below threshold. Smoothly compressed between threshold
    and ceiling. Hard-clipped at ceiling to guarantee no overs.
    """
    abs_s = np.abs(samples)
    mask = abs_s > threshold
    if np.any(mask):
        overshoot = abs_s[mask] - threshold
        # Asymptotic compression: approaches ceiling but never exceeds it
        headroom = ceiling - threshold
        compressed = threshold + headroom * overshoot / (overshoot + headroom)
        samples[mask] = np.sign(samples[mask]) * compressed
    return samples


# ── SoftSynth (main class) ────────────────────────────────────
class SoftSynth:
    """Pure-Python polyphonic synthesizer.

    Provides a standard MIDI synthesizer interface (noteon/noteoff/cc)
    that MusicEngine uses directly.
    """

    def __init__(self, gain: float = 0.3):
        self._gain = gain
        self._voices: list[Voice] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._output = None
        self._reverb = SimpleReverb(delay_ms=200, feedback=0.25, mix=0.20)

        # Per-channel volume (CC 7)
        self._channel_volume = {ch: 100 / 127.0 for ch in range(16)}

    def start(self, driver: str | None = None, device: str | None = None):
        self._output = self._create_output(driver, device)
        self._running = True
        self._thread = threading.Thread(target=self._audio_loop, daemon=True,
                                        name="SoftSynth-audio")
        self._thread.start()

    def _create_output(self, driver, device):
        if driver == "alsa" or (driver is None and sys.platform == "linux"):
            if shutil.which("aplay"):
                try:
                    out = _AplayOutput(device=device)
                    if out._gave_up:
                        # No audio device — fall through to silent
                        print("[SoftSynth] ALSA unavailable — falling back to silent")
                    else:
                        print("[SoftSynth] Audio output: aplay (ALSA stereo)")
                        return out
                except Exception as e:
                    print(f"[SoftSynth] aplay failed: {e}")
            else:
                print("[SoftSynth] aplay not found — is alsa-utils installed?")
        try:
            out = _SounddeviceOutput(device=device)
            print("[SoftSynth] Audio output: sounddevice")
            return out
        except Exception:
            pass
        if sys.platform == "darwin":
            print("[SoftSynth] ⚠ No audio output on macOS — pip install sounddevice")
        print("[SoftSynth] Audio output: none (silent)")
        return _NullOutput()

    def _audio_loop(self):
        """Background thread: mix voices → reverb → limiter → aplay.

        All mixing is done in float64 for maximum precision.
        The 1MB kernel pipe buffer absorbs any GIL stalls from
        the inference thread, preventing audio dropouts.
        """
        silence = np.zeros(CHUNK_SIZE, dtype=np.float64)

        while self._running:
            # Snapshot voices under lock (fast — just a list copy)
            with self._lock:
                voices = list(self._voices)

            if not voices:
                self._output.write(silence)
                time.sleep(0.05)
                continue

            # Render all voices in float64 (outside lock)
            mix = np.zeros(CHUNK_SIZE, dtype=np.float64)
            dead_indices = []
            for i, voice in enumerate(voices):
                samples = voice.render(CHUNK_SIZE)
                if samples is None:
                    dead_indices.append(i)
                    continue
                vol = self._channel_volume.get(voice.channel, 0.8)
                mix += samples * vol

            # Remove dead voices
            if dead_indices:
                dead_set = set(dead_indices)
                with self._lock:
                    self._voices = [v for i, v in enumerate(self._voices)
                                    if i not in dead_set and not v.is_silent()]

            # Apply reverb (float64)
            mix = self._reverb.process(mix)

            # Apply master gain
            mix *= self._gain

            # Soft-knee limiter (transparent below 0.4, ceiling at 0.85)
            mix = _soft_limit(mix, threshold=0.4, ceiling=0.85)

            self._output.write(mix)

    # ── MIDI synthesizer interface ────────────────────────────

    def sfload(self, path: str) -> int:
        return 1

    def program_select(self, channel: int, sfid: int, bank: int, program: int):
        pass

    def noteon(self, channel: int, note: int, velocity: int):
        if velocity <= 0:
            self.noteoff(channel, note)
            return
        timbre = TIMBRES.get(channel, DEFAULT_TIMBRE)
        voice = Voice(channel, note, velocity, timbre)
        with self._lock:
            if len(self._voices) >= MAX_VOICES:
                self._voices.sort(key=lambda v: v._elapsed)
                self._voices.pop(0)
            self._voices.append(voice)

    def noteoff(self, channel: int, note: int):
        with self._lock:
            for voice in self._voices:
                if voice.channel == channel and voice.note == note and voice.active:
                    voice.release()
                    break

    def cc(self, channel: int, cc_num: int, value: int):
        if cc_num == 7:
            self._channel_volume[channel] = value / 127.0

    def delete(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._output:
            self._output.close()
        self._voices.clear()
