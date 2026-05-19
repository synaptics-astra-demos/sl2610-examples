#!/usr/bin/env python3
"""
SoftSynth Audio Test — Run on the Coral board to verify audio output.

Plays a short ambient melody through SoftSynth → aplay → USB DAC.
No dependencies beyond Python stdlib + NumPy.

Usage:
    python3 test_softsynth.py
    python3 test_softsynth.py --device hw:1,0   # specify ALSA device
"""

import argparse
import sys
import time

# Ensure we can import from the same directory
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soft_synth import SoftSynth

# Pentatonic scale notes (C major pentatonic across octaves)
MELODY = [
    # (midi_note, velocity, duration, pause_after)
    (60, 80, 1.5, 0.1),   # C4 — pad
    (64, 70, 1.2, 0.1),   # E4
    (67, 75, 1.0, 0.2),   # G4
    (72, 85, 1.5, 0.1),   # C5
    (76, 65, 1.0, 0.3),   # E5
    (79, 70, 2.0, 0.5),   # G5
]

CHIME_NOTES = [72, 76, 79, 84]  # C5, E5, G5, C6
BASS_NOTES = [36, 43, 48]       # C2, G2, C3


def main():
    parser = argparse.ArgumentParser(description="SoftSynth Audio Test")
    parser.add_argument("--device", default=None, help="ALSA device (e.g. hw:1,0)")
    args = parser.parse_args()

    print("╔═══════════════════════════════════════╗")
    print("║  🔊 SoftSynth Audio Test              ║")
    print("╚═══════════════════════════════════════╝")
    print()

    # Check aplay is available
    import shutil
    if not shutil.which("aplay"):
        print("✗ aplay not found! ALSA is required.")
        sys.exit(1)

    # List ALSA devices
    import subprocess
    print("[1/3] Available ALSA playback devices:")
    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5)
        if result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                if "card" in line.lower() or "device" in line.lower():
                    print(f"  {line.strip()}")
        else:
            print("  (none found — is USB DAC plugged in?)")
    except Exception as e:
        print(f"  Error listing devices: {e}")
    print()

    # Start synth
    print("[2/3] Starting SoftSynth...")
    synth = SoftSynth(gain=0.6)
    synth.start(driver="alsa", device=args.device)
    time.sleep(0.5)  # let audio thread settle
    print()

    # Play test sequence
    print("[3/3] Playing test melody...")
    print()

    # ── Phase 1: Warm pad notes ──
    print("  ♪ Warm Pad (channel 0)...")
    synth.cc(0, 7, 80)  # volume
    for note, vel, dur, pause in MELODY[:3]:
        print(f"    Note {note} (vel={vel}, dur={dur}s)")
        synth.noteon(0, note, vel)
        time.sleep(dur)
        synth.noteoff(0, note)
        time.sleep(pause)

    time.sleep(0.5)

    # ── Phase 2: Chime melody ──
    print("  ♪ Chime (channel 3)...")
    synth.cc(3, 7, 60)
    for i, note in enumerate(CHIME_NOTES):
        vel = 90 - i * 10
        print(f"    Note {note} (vel={vel})")
        synth.noteon(3, note, vel)
        time.sleep(0.6)
        synth.noteoff(3, note)
        time.sleep(0.15)

    time.sleep(0.5)

    # ── Phase 3: Bass ──
    print("  ♪ Sub Bass (channel 2)...")
    synth.cc(2, 7, 90)
    for note in BASS_NOTES:
        print(f"    Note {note}")
        synth.noteon(2, note, 80)
        time.sleep(1.0)
        synth.noteoff(2, note)
        time.sleep(0.2)

    time.sleep(0.5)

    # ── Phase 4: Everything together ──
    print("  ♪ All together...")
    synth.noteon(2, 48, 70)   # Bass C3
    synth.noteon(0, 60, 60)   # Pad C4
    synth.noteon(0, 64, 55)   # Pad E4
    synth.noteon(0, 67, 50)   # Pad G4
    time.sleep(1.0)
    synth.noteon(3, 84, 75)   # Chime C6
    time.sleep(0.4)
    synth.noteon(3, 88, 60)   # Chime E6
    time.sleep(2.0)

    # Release all
    for ch in [0, 2, 3]:
        for note in range(128):
            synth.noteoff(ch, note)

    # Let reverb tail ring out
    print()
    print("  Letting reverb tail decay...")
    time.sleep(3.0)

    # Cleanup
    synth.delete()
    print()
    print("✓ Test complete! Did you hear audio?")
    print()
    print("  If not:")
    print("    1. Check USB DAC is plugged in: aplay -l")
    print("    2. Try specifying device: python3 test_softsynth.py --device hw:1,0")
    print("    3. Check volume: amixer set Master 100%")


if __name__ == "__main__":
    main()
