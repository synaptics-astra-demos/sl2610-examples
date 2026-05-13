"""
SoftSynth — Pygame Mixer Implementation
====================================================
Replaces the pure-Python audio engine with Pygame's highly optimized
C-based mixer to completely eliminate Python GIL audio stalling.
"""

import math
import numpy as np
import pygame

# ── Audio constants ────────────────────────────────────────────
SAMPLE_RATE = 22050
TWO_PI = 2.0 * np.pi
_DT = 1.0 / SAMPLE_RATE

# Pre-compute frequency table (all 128 MIDI notes)
_FREQ_TABLE = 440.0 * (2.0 ** ((np.arange(128) - 69) / 12.0))

def midi_to_freq(note: int) -> float:
    return float(_FREQ_TABLE[min(max(note, 0), 127)])

# ── Channel Timbre Definitions ─────────────────────────────────
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

class SoftSynth:
    """Pygame-based synthesizer. 
    
    Generates notes on-the-fly and plays them using Pygame's C-based mixer 
    to completely avoid Python GIL stalls.
    """

    def __init__(self, gain: float = 0.3):
        self._gain = gain
        self._cache = {}
        self._channel_volume = {ch: 100 / 127.0 for ch in range(16)}
        self._active_channels = {}  # Tracks playing sounds for noteoff
        self._initialized = False

    def start(self, driver: str | None = None, device: str | None = None):
        import os
        os.environ["SDL_AUDIODRIVER"] = "alsa"
        if device:
            if device.startswith("hw:"):
                device = device.replace("hw:", "plughw:", 1)
            os.environ["AUDIODEV"] = device
            print(f"[SoftSynth] Exported AUDIODEV={device}")

        try:
            # Init mixer with small buffer for low latency
            pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, 1024)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(32)
            self._initialized = True
            print("[SoftSynth] ✓ Pygame audio engine initialized")
        except Exception as e:
            print(f"[SoftSynth] ⚠ Pygame audio failed to initialize: {e}")
            return

        print("[SoftSynth] Audio output: pygame ALSA")

    def _generate_sound(self, channel: int, note: int) -> pygame.mixer.Sound:
        timbre = TIMBRES.get(channel, DEFAULT_TIMBRE)
        freq = midi_to_freq(note)
        
        # Max duration 4 seconds
        duration = 4.0
        n_samples = int(duration * SAMPLE_RATE)
        dt = np.arange(n_samples, dtype=np.float64) * _DT
        attack = timbre["attack"]
        
        signal = np.zeros(n_samples, dtype=np.float64)
        
        for r, a, d in timbre["partials"]:
            phase = TWO_PI * freq * r * dt
            signal += a * np.sin(phase) * np.exp(-d * dt)
            
        # Attack ramp
        attack_env = np.clip(dt / attack, 0.0, 1.0)
        signal *= attack_env
        
        # Apply master gain and soft clip
        signal *= self._gain * 1.5
        signal = np.clip(signal, -1.0, 1.0)
        
        # Convert to 16-bit stereo
        pcm_mono = (signal * 32767).astype(np.int16)
        pcm_stereo = np.column_stack((pcm_mono, pcm_mono))
        
        # Create Pygame Sound
        sound = pygame.sndarray.make_sound(pcm_stereo)
        return sound

    def _get_sound(self, channel: int, note: int) -> pygame.mixer.Sound:
        key = (channel, note)
        if key not in self._cache:
            self._cache[key] = self._generate_sound(channel, note)
        return self._cache[key]

    def sfload(self, path: str) -> int:
        return 1

    def program_select(self, channel: int, sfid: int, bank: int, program: int):
        pass

    def noteon(self, channel: int, note: int, velocity: int):
        if not self._initialized:
            return
        if velocity <= 0:
            self.noteoff(channel, note)
            return

        try:
            sound = self._get_sound(channel, note)
        except Exception as e:
            print(f"[SoftSynth] noteon error ch={channel} note={note}: {e}")
            return
        vol = self._channel_volume.get(channel, 0.8) * (velocity / 127.0)
        sound.set_volume(vol)
        
        ch = pygame.mixer.find_channel()
        if ch:
            ch.play(sound)
            self._active_channels[(channel, note)] = ch

    def noteoff(self, channel: int, note: int):
        # We could use fadeout here, but simple stop works fine for Pygame
        key = (channel, note)
        if key in self._active_channels:
            ch = self._active_channels[key]
            if ch.get_sound() == self._cache.get(key):
                ch.fadeout(200) # 200ms fadeout
            self._active_channels.pop(key, None)

    def cc(self, channel: int, cc_num: int, value: int):
        if cc_num == 7:
            self._channel_volume[channel] = value / 127.0

    def dispose(self):
        try:
            pygame.mixer.quit()
        except:
            pass
