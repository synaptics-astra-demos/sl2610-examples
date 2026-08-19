"""
Real Magenta MelodyRNN — Pure NumPy LSTM Inference

Loads the basic_rnn weights from TF.js checkpoint and runs forward inference
using a 2-layer LSTM. Zero TensorFlow dependency — works on Python 3.x + numpy.

Architecture (from TF.js checkpoint):
  Input: 38-dim one-hot (36 MIDI pitches [48-83] + NOTE_OFF + NO_EVENT)
  LSTM Layer 0: input_dim=38+512=550 → hidden_size=512
  LSTM Layer 1: input_dim=512+512=1024 → hidden_size=512
  Output: fully_connected [512, 38] → 38 logits → softmax → sample

Event encoding (basic_rnn):
  0-35: NOTE_ON for MIDI 48-83
  36: NOTE_OFF
  37: NO_EVENT (time step with no change)
"""

import math
import os
import random
import threading
import time
from pathlib import Path

import numpy as np

# ── Constants ──
WEIGHTS_PATH = Path(__file__).parent / "basic_rnn_weights.npz"
NUM_CLASSES = 38
MIDI_MIN = 48     # basic_rnn pitch range
MIDI_MAX = 83
NOTE_OFF_EVENT = 36
NO_EVENT = 37
STEPS_PER_QUARTER = 4
HIDDEN_SIZE = 512

# Pentatonic pitch classes for snapping output
PENTATONIC_PC = [0, 2, 4, 7, 9]


def snap_to_pentatonic(midi: int) -> int:
    """Snap a MIDI note to nearest pentatonic pitch in C4-C7 range."""
    octave = midi // 12
    pc = midi % 12
    closest = min(PENTATONIC_PC, key=lambda p: min(abs(pc - p), 12 - abs(pc - p)))
    result = octave * 12 + closest
    while result < 60:
        result += 12
    while result > 96:
        result -= 12
    return result


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


class LSTMCell:
    """Single LSTM cell operating on numpy arrays."""

    def __init__(self, kernel: np.ndarray, bias: np.ndarray, hidden_size: int):
        self.kernel = kernel   # [input_dim + hidden_size, 4 * hidden_size]
        self.bias = bias       # [4 * hidden_size]
        self.hidden_size = hidden_size

    def forward(self, x: np.ndarray, h: np.ndarray, c: np.ndarray):
        """One step of LSTM. Returns (h_new, c_new)."""
        # Concatenate input and hidden state
        xh = np.concatenate([x, h])

        # Linear projection
        gates = xh @ self.kernel + self.bias

        # Split into 4 gates (TF order: i, j/g, f, o — but TF uses: i, c_in, f, o)
        i = sigmoid(gates[0:self.hidden_size])
        j = np.tanh(gates[self.hidden_size:2*self.hidden_size])
        f = sigmoid(gates[2*self.hidden_size:3*self.hidden_size] + 1.0)  # forget gate bias
        o = sigmoid(gates[3*self.hidden_size:4*self.hidden_size])

        c_new = f * c + i * j
        h_new = o * np.tanh(c_new)
        return h_new, c_new


class MelodyRNN:
    """Real Magenta MelodyRNN using pre-trained basic_rnn weights.

    Generates melodic continuations given a seed sequence.
    Uses a 2-layer LSTM with temperature-based sampling.
    """

    def __init__(self):
        self._ready = False
        self._lstm0: LSTMCell | None = None
        self._lstm1: LSTMCell | None = None
        self._fc_w: np.ndarray | None = None
        self._fc_b: np.ndarray | None = None

        # LSTM states (persistent across calls for continuity)
        self._h0 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        self._c0 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        self._h1 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        self._c1 = np.zeros(HIDDEN_SIZE, dtype=np.float32)

        # Generation state
        self._running = False
        self._thread: threading.Thread | None = None
        self._jelly_count = 0
        self._note_buffer: list[int] = []  # Recent jellyfish MIDI notes
        self._lock = threading.Lock()
        self._recent_events: list[int] = []  # Track recent generated events for repetition penalty

        # Callback for playing notes
        self.on_note: callable | None = None  # (midi, velocity, duration)
        # Callback for UI visualization
        self.ai_note_callback: callable | None = None

    def load_weights(self, path: Path | None = None) -> bool:
        """Load pre-trained basic_rnn weights."""
        path = path or WEIGHTS_PATH
        if not path.exists():
            print(f"[MelodyRNN] ⚠ Weights not found: {path}")
            return False

        try:
            w = np.load(path)
            self._lstm0 = LSTMCell(
                w["rnn_multi_rnn_cell_cell_0_basic_lstm_cell_kernel"],
                w["rnn_multi_rnn_cell_cell_0_basic_lstm_cell_bias"],
                HIDDEN_SIZE,
            )
            self._lstm1 = LSTMCell(
                w["rnn_multi_rnn_cell_cell_1_basic_lstm_cell_kernel"],
                w["rnn_multi_rnn_cell_cell_1_basic_lstm_cell_bias"],
                HIDDEN_SIZE,
            )
            self._fc_w = w["fully_connected_weights"]
            self._fc_b = w["fully_connected_biases"]
            self._ready = True
            print("[MelodyRNN] ✓ Loaded basic_rnn weights (2-layer LSTM, 38 classes)")
            return True
        except Exception as e:
            print(f"[MelodyRNN] ✗ Failed to load weights: {e}")
            return False

    def _event_to_onehot(self, event: int) -> np.ndarray:
        """Convert event index to 38-dim one-hot vector."""
        x = np.zeros(NUM_CLASSES, dtype=np.float32)
        x[event] = 1.0
        return x

    def _midi_to_event(self, midi: int) -> int:
        """Convert MIDI note to basic_rnn event index."""
        return max(0, min(35, midi - MIDI_MIN))

    def _event_to_midi(self, event: int) -> int:
        """Convert basic_rnn event to MIDI note."""
        return event + MIDI_MIN

    def _step(self, event: int) -> np.ndarray:
        """Run one LSTM step and return logits (38-dim)."""
        x = self._event_to_onehot(event)
        self._h0, self._c0 = self._lstm0.forward(x, self._h0, self._c0)
        self._h1, self._c1 = self._lstm1.forward(self._h0, self._h1, self._c1)
        logits = self._h1 @ self._fc_w + self._fc_b
        return logits

    def _sample(self, logits: np.ndarray, temperature: float = 1.0) -> int:
        """Sample from logits with temperature and repetition penalty."""
        # Apply repetition penalty: strongly suppress recently played events
        adjusted = logits.copy()
        for i, evt in enumerate(reversed(self._recent_events[-12:])):
            decay = 0.7 ** i  # Most recent = strongest penalty
            if 0 <= evt <= 35:  # Only penalize NOTE_ON events
                adjusted[evt] -= 3.0 * decay  # Big penalty for exact repeats
                # Also mildly penalize notes within ±2 semitones
                for neighbor in range(max(0, evt - 2), min(36, evt + 3)):
                    if neighbor != evt:
                        adjusted[neighbor] -= 1.0 * decay

        probs = softmax(adjusted / temperature)
        choice = int(np.random.choice(NUM_CLASSES, p=probs))
        self._recent_events.append(choice)
        if len(self._recent_events) > 24:
            self._recent_events = self._recent_events[-24:]
        return choice

    def _get_temperature(self) -> float:
        """Temperature based on jellyfish density — raised baseline to add variety."""
        count = self._jelly_count
        if count <= 2:
            return 1.2   # was 0.8, now more exploratory
        elif count <= 5:
            return 1.4   # was 1.1
        elif count <= 8:
            return 1.6   # was 1.4
        else:
            return 1.8   # was 1.7

    def continue_sequence(self, seed_events: list[int], num_steps: int = 32,
                          temperature: float = 1.0) -> list[dict]:
        """Generate a continuation from a seed sequence.

        Args:
            seed_events: List of event indices (0-37) as the seed
            num_steps: Number of steps to generate
            temperature: Sampling temperature (higher = more random)

        Returns:
            List of {event, midi, step} dicts
        """
        if not self._ready:
            return []

        # Reset state so the seed is processed in a clean context
        self._h0 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        self._c0 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        self._h1 = np.zeros(HIDDEN_SIZE, dtype=np.float32)
        self._c1 = np.zeros(HIDDEN_SIZE, dtype=np.float32)

        # Feed seed through LSTM (don't sample, just condition)
        for i, event in enumerate(seed_events):
            self._step(event)
            # Yield GIL every 4 steps so main thread can render
            if i % 4 == 3:
                time.sleep(0)

        # Generate new steps
        last_event = seed_events[-1] if seed_events else NO_EVENT
        generated = []

        for step in range(num_steps):
            logits = self._step(last_event)
            event = self._sample(logits, temperature)
            last_event = event

            if event <= 35:  # NOTE_ON
                midi = self._event_to_midi(event)
                generated.append({"event": event, "midi": midi, "step": step})

            # Yield GIL every 4 steps so main thread can render
            if step % 4 == 3:
                time.sleep(0)

        return generated

    # ── Feed from tracker ──
    def feed_note(self, midi: int) -> None:
        """Feed a jellyfish-triggered note into the buffer."""
        with self._lock:
            self._note_buffer.append(max(MIDI_MIN, min(MIDI_MAX, midi)))
            # Keep last 16 notes
            if len(self._note_buffer) > 16:
                self._note_buffer = self._note_buffer[-16:]

    def feed_activity(self, count: int) -> None:
        """Update jellyfish count for temperature modulation."""
        self._jelly_count = count

    # ── Two-thread pipeline: generator → queue → player ──
    def start(self) -> None:
        """Start continuous melody generation with two threads."""
        if not self._ready:
            print("[MelodyRNN] Cannot start — weights not loaded")
            return

        import queue
        # Small queue size = low latency (max ~8 seconds of music ahead)
        self._note_queue: queue.Queue = queue.Queue(maxsize=10)
        self._running = True

        # Thread 1: generates notes into queue (GIL-heavy, but doesn't block playback)
        self._gen_thread = threading.Thread(target=self._generator_loop, daemon=True)
        # Thread 2: plays notes from queue (lightweight, never blocked by inference)
        self._play_thread = threading.Thread(target=self._player_loop, daemon=True)

        self._gen_thread.start()
        self._play_thread.start()
        print("[MelodyRNN] 🎵 Melody generation started (real Magenta basic_rnn)")

    def stop(self) -> None:
        self._running = False
        for t in [getattr(self, '_gen_thread', None), getattr(self, '_play_thread', None)]:
            if t:
                t.join(timeout=3)
        self._gen_thread = None
        self._play_thread = None

    def _generator_loop(self) -> None:
        """Continuously generate phrases and push notes into the queue."""
        time.sleep(0.5)  # Brief settle

        while self._running:
            # If queue is nearly full, wait a bit
            if self._note_queue.qsize() > 5:
                time.sleep(1.0)
                continue

            try:
                self._generate_into_queue()
            except Exception as e:
                print(f"[MelodyRNN] Generator error: {e}")
                time.sleep(1.0)

            # Small pause — the queue buffers ahead so this is fine
            time.sleep(0.1)

    def _generate_into_queue(self) -> None:
        """Build seed, run LSTM inference, push timed notes into queue."""
        with self._lock:
            recent = list(self._note_buffer)

        if len(recent) < 2:
            # Provide a more diverse random seed
            recent = random.sample([60, 62, 64, 67, 69, 72, 74, 76, 79, 81], k=3)

        # Diversify the seed — pick a varied subset to avoid monotone seeding
        unique_recent = []
        seen = set()
        for n in reversed(recent):
            if n not in seen:
                unique_recent.append(n)
                seen.add(n)
            if len(unique_recent) >= 6:
                break
        unique_recent.reverse()
        if not unique_recent:
            unique_recent = [random.choice([60, 64, 67, 72, 76])]

        # Convert to events with rhythmic variety
        seed_events: list[int] = []
        for midi in unique_recent:
            event = self._midi_to_event(midi)
            seed_events.append(event)
            # Add rhythmic gaps
            if random.random() < 0.4:
                seed_events.append(NO_EVENT)

        temperature = self._get_temperature()
        # Generate shorter phrases more frequently for better responsiveness
        generated = self.continue_sequence(seed_events, num_steps=32, temperature=temperature)

        if not generated:
            return

        # Pre-compute all note data and push into queue
        bpm = 35  # Slightly faster for more content
        step_sec = 60.0 / bpm / STEPS_PER_QUARTER

        last_step = -1
        for note in generated:
            if not self._running:
                break

            midi = snap_to_pentatonic(note["midi"])
            step = note["step"]
            
            # Use step delta for timing instead of random wait
            if last_step == -1:
                wait = 0.05
            else:
                steps_passed = max(1, step - last_step)
                wait = steps_passed * step_sec
            
            last_step = step

            is_downbeat = step % 4 == 0
            base_vel = 70 if is_downbeat else 55
            velocity = base_vel + random.randint(-5, 10)
            velocity = max(40, min(110, velocity))
            duration = step_sec * 2.0 # More sustain

            # Block if queue is full — ensures we don't generate miles of stale music
            try:
                self._note_queue.put({
                    "midi": midi,
                    "velocity": velocity,
                    "duration": duration,
                    "wait": wait,
                }, timeout=2.0)
            except Exception:
                break # Stop generating this phrase if player is stalled

        count = len(generated)
        if count > 0:
            print(f"[MelodyRNN] 🎵 Generated {count} notes (temp={temperature:.1f}, jelly={self._jelly_count})")

    def _player_loop(self) -> None:
        """Continuously play notes from the queue — lightweight, never freezes."""
        while self._running:
            try:
                # Wait for next note
                note = self._note_queue.get(timeout=0.5)
            except Exception:
                continue  # Queue empty, wait for generator

            # Wait the intended duration before playing the note
            time.sleep(note["wait"])

            midi = note["midi"]
            velocity = note["velocity"]
            duration = note["duration"]

            if self.on_note:
                self.on_note(midi, velocity, duration)
            if self.ai_note_callback:
                self.ai_note_callback(midi, velocity, duration)
