from __future__ import annotations

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import threading
import time
import logging

from utils.log import configure_logging
from utils.npu import enable_npu_clock
from utils.speech import (
    MoonshineTranscriber,
    SileroSpeechSegmenter,
    SoundDeviceAudioSource,
    SpeechRecognizer,
    query_input_devices,
)

SAMPLING_RATE = 16_000
CHUNK_SIZE = 512

logger = logging.getLogger("live_caption")

running = False


def choose_audio_device(device_arg: str | None) -> int | str | None:
    if device_arg is None:
        print("List of Audio input devices:")
        print(query_input_devices())
        device_arg = input("Enter input device to listen on [default]: ").strip()
    if device_arg == "":
        return None
    try:
        return int(device_arg)
    except ValueError:
        return device_arg


def start_audio_thread(recognizer: SpeechRecognizer, stop_event: threading.Event):
    global running
    running = True
    print("Moonshine is running. Press Ctrl+C to quit.\n")

    with recognizer:
        while not stop_event.is_set():
            transcript = recognizer.listen_once(stop_event=stop_event)
            if transcript is None:
                break
            print(transcript.text, flush=True)

    logger.debug("Audio thread exiting.")
    running = False


#  NPU Clock
def enable_npu_clock_local():
    import subprocess
    try:
        subprocess.run(["devmem", "0xf7e104b0", "32", "0x216"],
                       capture_output=True, timeout=5)
        print("[NPU] Clock enabled")
    except Exception as e:
        print(f"[NPU] Clock enable failed: {e}")


# ---------------------- CLI / Entry ----------------------

if __name__ == "__main__":
    configure_logging("INFO")

    parser = argparse.ArgumentParser(description="Moonshine live speech-to-text caption")
    parser.add_argument("--moonshine-model", type=str, default=None,
                        help="Path to Moonshine model directory. Defaults to HF download.")
    parser.add_argument("--audio-device", type=str, default=None,
                        help="Audio input device index or name.")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--silence-ms", type=int, default=150)
    parser.add_argument("--max-speech-secs", type=float, default=10.0)
    parser.add_argument("--min-segment-secs", type=float, default=0.20)
    parser.add_argument("--no-npu-clock", action="store_true")
    args = parser.parse_args()

    if not args.no_npu_clock:
        ok, message = enable_npu_clock()
        print(f"[NPU] {message}")

    audio_device = choose_audio_device(args.audio_device)

    transcriber = MoonshineTranscriber(args.moonshine_model)
    source = SoundDeviceAudioSource(
        device=audio_device,
        sample_rate=SAMPLING_RATE,
        chunk_size=CHUNK_SIZE,
    )
    segmenter = SileroSpeechSegmenter(
        sample_rate=SAMPLING_RATE,
        chunk_size=CHUNK_SIZE,
        threshold=args.vad_threshold,
        min_silence_duration_ms=args.silence_ms,
        max_speech_secs=args.max_speech_secs,
        min_segment_secs=args.min_segment_secs,
    )
    recognizer = SpeechRecognizer(
        transcriber=transcriber,
        source=source,
        segmenter=segmenter,
    )

    stop_event = threading.Event()
    audio_thread = threading.Thread(
        target=start_audio_thread, args=(recognizer, stop_event), daemon=True
    )
    audio_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        logger.debug("Shutting down...")
        audio_thread.join(timeout=3)

    logger.debug("Live caption closed.")
