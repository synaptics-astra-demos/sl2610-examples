"""
Music Engine — Ambient music driven by jellyfish positions

Design (housey / deep / sine-wave focused):
  - Channel 0: Main synth (rows 1-2) — Warm Pad (smooth, sine-like)
  - Channel 1: Arp synth (row 0) — Bell (mellow)
  - Channel 2: Bass synth (row 3) — Sub Bass (deep house)
  - Channel 3: Clash — Percussive shimmer

Audio is rendered by SoftSynth (pure Python/NumPy) and piped to
ALSA via `aplay`. Zero system dependencies beyond Python + NumPy.
"""

import math
import random
import threading
import time

from soft_synth import SoftSynth

# ── Grid config ────────────────────────────────────────────────
GRID_COLS = 8
GRID_ROWS = 4

# ── Note mapping per row (MIDI note numbers) ──────────────────
NOTE_GRID = [
    [72, 74, 76, 79, 81, 84, 86, 88],   # Row 0 (top): arp
    [45, 48, 50, 52, 55, 57, 60, 62],   # Row 1: pad
    [55, 57, 60, 62, 64, 67, 69, 72],   # Row 2: chords
    [48, 50, 52, 55, 57, 60, 62, 64],   # Row 3 (bottom): bass
]

# Chord voicings for row 2 (intervals in semitones from root)
CHORD_VOICINGS = [
    [0, 4, 7],     # G3 (Major)
    [0, 3, 7],     # A3 (minor)
    [0, 4, 7],     # C4 (Major)
    [0, 3, 7],     # D4 (minor)
    [0, 3, 7],     # E4 (minor)
    [0, 4, 7],     # G4 (Major)
    [0, 3, 7],     # A4 (minor)
    [0, 4, 7],     # C5 (Major)
]

# Arpeggio patterns (scale degree offsets in pentatonic table)
ARP_PATTERNS = [
    [0, 1, 2, 5],
    [0, 2, 3, 4],
    [0, 1, 3, 5],
    [0, 2, 4, 5],
    [0, 1, 2, 4],
    [0, 2, 4, 6],
    [0, 3, 4, 6],
    [0, 2, 5, 7],
]

# Pentatonic MIDI table (C D E G A across octaves 3-8)
PENTATONIC_MIDI = []
for _oct in range(3, 9):
    for _pc in [0, 2, 4, 7, 9]:
        PENTATONIC_MIDI.append(_oct * 12 + _pc)

# Note durations (seconds) per row
NOTE_DURATIONS = {
    0: 0.15,    # Arp — short
    1: 2.0,     # Middle — ambient drone
    2: 6.0,     # Chords — long ambient pads
    3: 2.0,     # Bass — sustained
}

class MusicEngine:
    """Ambient music engine using SoftSynth (pure Python/NumPy).

    Drives 4 channels of synthesized audio through ALSA.
    Zero system dependencies — only Python stdlib + NumPy.
    """

    def __init__(self, audio_driver: str | None = None,
                 alsa_device: str | None = None,
                 **kwargs):
        self.fs = None
        self._ready = False
        self._note_off_timers: list[threading.Timer] = []
        self._audio_driver = audio_driver
        self._alsa_device = alsa_device

        # Clash rate limiting: 2-4 triggers per 3-5 second window
        self._clash_cooldown_end = 0.0
        self._clash_triggers_left = 0

    def init(self) -> None:
        """Initialize SoftSynth audio engine."""
        print("[MusicEngine] Initializing SoftSynth...")

        self.fs = SoftSynth(gain=0.5)
        self.fs.start(driver=self._audio_driver, device=self._alsa_device)

        # Set volumes per channel — keep it subtle
        self.fs.cc(0, 7, 65)    # Main — warm, not too loud
        self.fs.cc(1, 7, 15)    # Arp — very barely audible
        self.fs.cc(2, 7, 75)    # Bass — present but deep
        self.fs.cc(3, 7, 20)    # Clash — subtle

        # Reverb send (CC 91)
        self.fs.cc(0, 91, 60)   # Main — moderate reverb
        self.fs.cc(1, 91, 70)   # Arp — more reverb (ethereal)
        self.fs.cc(2, 91, 30)   # Bass — less reverb
        self.fs.cc(3, 91, 120)  # Clash — massive reverb

        self._ready = True
        print("[MusicEngine] ✓ SoftSynth ready (ambient oscillator synthesis)")

        # Start sound evolution
        self._evolver = SoundEvolver(self)
        self._evolver.start()

    def trigger_cell(self, row: int, col: int) -> dict | None:
        """Trigger note(s) for a grid cell. Returns trigger info or None."""
        if not self._ready or self.fs is None:
            return None

        note = NOTE_GRID[row][col]
        duration = NOTE_DURATIONS.get(row, 0.5)

        if row == 0:
            self._play_arpeggio(col, duration)
        elif row == 2:
            self._play_chord(col, duration)
        elif row == 3:
            velocity = int(40 + 15 * (col / GRID_COLS))
            self._play_note(2, note, velocity, duration)
        else:
            velocity = int(50 + 20 * (1 - row / GRID_ROWS))
            self._play_note(0, note, velocity, duration)

        return {
            "row": row, "col": col, "note": note,
            "x": (col + 0.5) / GRID_COLS,
            "y": (row + 0.5) / GRID_ROWS,
        }

    def play_clash(self) -> None:
        """Trigger a clash chime sound (rate-limited: 2-4 per 3-5s window)."""
        if not self._ready or self.fs is None:
            return

        now = time.time()
        if now < self._clash_cooldown_end:
            if self._clash_triggers_left <= 0:
                return
            self._clash_triggers_left -= 1
        else:
            # New window
            self._clash_triggers_left = random.randint(2, 4) - 1
            self._clash_cooldown_end = now + random.uniform(3.0, 5.0)

        # A quick glissando/chord in the pentatonic scale
        base = random.choice([84, 88, 91]) # C6, E6, G6 approx
        for i, val in enumerate([0, 4, 7]):
            delay = i * 0.04
            def play_hit(n=base+val, v=23-i*3):
                self._play_note(3, n, v, 2.0)
            if delay > 0:
                t = threading.Timer(delay, play_hit)
                t.daemon = True
                t.start()
                self._note_off_timers.append(t)
            else:
                play_hit()

    def _play_note(self, channel: int, note: int, velocity: int, duration: float) -> None:
        """Play a note with automatic note-off after duration."""
        if self.fs is None:
            return
        velocity = max(20, min(127, velocity))
        self.fs.noteon(channel, note, velocity)

        def off():
            if self.fs is not None:
                self.fs.noteoff(channel, note)

        timer = threading.Timer(duration, off)
        timer.daemon = True
        timer.start()
        self._note_off_timers.append(timer)

    def _play_arpeggio(self, col: int, base_duration: float) -> None:
        """Play arpeggiated pattern for top row."""
        base_note = NOTE_GRID[0][col]

        # Find base note in pentatonic table
        base_idx = -1
        for i, m in enumerate(PENTATONIC_MIDI):
            if m == base_note:
                base_idx = i
                break
        if base_idx == -1:
            base_idx = min(range(len(PENTATONIC_MIDI)),
                           key=lambda i: abs(PENTATONIC_MIDI[i] - base_note))

        pattern = ARP_PATTERNS[col % len(ARP_PATTERNS)]
        step_dur = 0.08  # 16th note at ~70 BPM

        for i, offset in enumerate(pattern):
            idx = min(base_idx + offset, len(PENTATONIC_MIDI) - 1)
            arp_note = PENTATONIC_MIDI[idx]
            velocity = max(10, 20 + (5 if i == 0 else 0) - i * 3)
            delay = i * step_dur

            def play(n=arp_note, v=velocity):
                self._play_note(1, n, v, 0.1)

            if delay > 0:
                t = threading.Timer(delay, play)
                t.daemon = True
                t.start()
                self._note_off_timers.append(t)
            else:
                play()

    def _play_chord(self, col: int, duration: float) -> None:
        """Play a chord for row 2."""
        root = NOTE_GRID[2][col]
        voicing = CHORD_VOICINGS[col % len(CHORD_VOICINGS)]
        velocity = 40

        for interval in voicing:
            chord_note = root + interval
            self._play_note(0, chord_note, velocity, duration)

    def dispose(self) -> None:
        """Clean up synth resources."""
        if hasattr(self, '_evolver'):
            self._evolver.stop()
        for timer in self._note_off_timers:
            timer.cancel()
        self._note_off_timers.clear()
        if self.fs:
            self.fs.delete()
            self.fs = None
        self._ready = False


class SoundEvolver:
    """Background thread that slowly evolves synth parameters over time.

    Modulates volume, pan, and reverb send using sine-wave LFOs
    for a slowly shifting ambient soundscape.
    """

    def __init__(self, engine: MusicEngine):
        self.engine = engine
        self._running = False
        self._thread: threading.Thread | None = None
        self._start_time = time.time()
        # Randomized phase offsets for each LFO
        self._phases = {
            'brightness': random.uniform(0, math.tau),
            'mod': random.uniform(0, math.tau),
            'pan': random.uniform(0, math.tau),
            'reverb': random.uniform(0, math.tau),
        }

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._evolve_loop, daemon=True)
        self._thread.start()
        print("[SoundEvolver] Timbre evolution started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _evolve_loop(self) -> None:
        """Slowly modulate synth parameters using sine-wave LFOs."""
        time.sleep(3)  # Let audio settle

        while self._running:
            try:
                self._update_parameters()
            except Exception as e:
                print(f"[SoundEvolver] Error: {e}")

            # Sleep 4-8 seconds between updates (randomized)
            time.sleep(random.uniform(4, 8))

    def _update_parameters(self) -> None:
        """Apply slow-moving parameter changes."""
        fs = self.engine.fs
        if not fs:
            return

        t = time.time() - self._start_time

        # ── Filter Brightness (CC 74) ──
        # Slow sine wave: period ~45 seconds, range 40-100
        brightness = int(70 + 30 * math.sin(t / 45 * math.tau + self._phases['brightness']))
        fs.cc(0, 74, brightness)       # Main pad

        # ── Modulation / Vibrato (CC 1) ──
        # Very slow: period ~60 seconds, range 0-35
        mod = int(17 + 17 * math.sin(t / 60 * math.tau + self._phases['mod']))
        fs.cc(0, 1, mod)       # Main pad — gentle vibrato
        fs.cc(1, 1, mod // 2)  # Arp — less vibrato

        # ── Pan drift (CC 10) ──
        # Channels drift gently: period ~30 seconds, range 44-84
        pan_main = int(64 + 20 * math.sin(t / 30 * math.tau + self._phases['pan']))
        fs.cc(0, 10, pan_main)  # Main pad drifts

        # ── Reverb amount (CC 91) ──
        # Slow breathe: period ~50 seconds, range 40-90
        reverb = int(65 + 25 * math.sin(t / 50 * math.tau + self._phases['reverb']))
        fs.cc(0, 91, reverb)       # Main
