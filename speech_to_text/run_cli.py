from __future__ import annotations

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app_utils.demo_utils import DependencyChecker, IncompatibleVersionError, MissingRequirementsError

try:
    DependencyChecker(
        os.path.join(os.path.dirname(__file__), "requirements.txt"),
        version_map={"torq.runtime": ">=2.3.0"},
    ).run()
except (MissingRequirementsError, IncompatibleVersionError) as e:
    print(f"{e}\nRun: python setup_demo.py", file=sys.stderr)
    sys.exit(1)

import argparse
import numpy as np
import threading
import time
import logging

from app_utils.log import configure_logging
from app_utils.npu import cleanup_npu_after_inference, setup_npu_for_inference
from app_utils.speech import (
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


def load_wav_16k(path: str) -> np.ndarray:
    """Read a WAV file as mono float32 at SAMPLING_RATE (linear resample if needed)."""
    try:
        import soundfile as sf
    except ImportError:
        print("Error: soundfile is required for --wav. Install it with: pip install soundfile",
              file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(path):
        print(f"Error: WAV file {path} not found.", file=sys.stderr)
        sys.exit(1)
    data, rate = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if rate != SAMPLING_RATE:
        n_out = int(round(len(data) * SAMPLING_RATE / rate))
        data = np.interp(np.linspace(0, len(data) - 1, n_out), np.arange(len(data)), data)
        logger.info("Resampled %d Hz -> %d Hz", rate, SAMPLING_RATE)
    return np.ascontiguousarray(data, dtype=np.float32)


def transcribe_wav(path: str, transcriber: MoonshineTranscriber, segmenter: SileroSpeechSegmenter) -> int:
    """Run a WAV file through VAD + Moonshine, the same path the microphone takes.

    Returns the process exit code (0 when at least one utterance was transcribed).
    """
    audio = load_wav_16k(path)
    audio_s = len(audio) / SAMPLING_RATE
    print(f"Transcribing {path} ({audio_s:.1f}s)...", flush=True)

    # Trailing silence lets the VAD close the last utterance, as it would live.
    padded = np.concatenate([audio, np.zeros(SAMPLING_RATE, dtype=np.float32)])
    segmenter.reset()
    transcripts = []
    for start in range(0, len(padded) - CHUNK_SIZE + 1, CHUNK_SIZE):
        segment = segmenter.feed(padded[start:start + CHUNK_SIZE])
        if segment is None:
            continue
        transcript = transcriber.transcribe(segment)
        if transcript.text.strip():
            transcripts.append(transcript)
            print(f"[{len(transcripts)}] {transcript.text}  ({transcript.stats.fmt(verbose=True)})", flush=True)

    if not transcripts:
        print("Error: no speech was transcribed from the WAV file", file=sys.stderr)
        return 1

    infer_ms = sum(t.stats.total_time_ms for t in transcripts)
    speech_s = sum(t.duration_s for t in transcripts)
    print("\nTranscript: " + " ".join(t.text for t in transcripts))
    print(f"(utterances {len(transcripts)} | speech {speech_s:.1f}s of {audio_s:.1f}s audio | "
          f"inference {infer_ms:.0f}ms | RTF {infer_ms / 1000 / speech_s:.2f})")
    return 0


# ---------------------- CLI / Entry ----------------------

if __name__ == "__main__":
    configure_logging("INFO")

    parser = argparse.ArgumentParser(description="Moonshine live speech-to-text caption")
    parser.add_argument("--moonshine-model", type=str, default=None,
                        help="Path to Moonshine model directory. Defaults to HF download.")
    parser.add_argument("--audio-device", type=str, default=None,
                        help="Audio input device index or name.")
    parser.add_argument("--wav", type=str, default=None,
                        help="Transcribe this WAV file and exit, instead of listening to the microphone.")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--silence-ms", type=int, default=150)
    parser.add_argument("--max-speech-secs", type=float, default=10.0)
    parser.add_argument("--min-segment-secs", type=float, default=0.20)
    parser.add_argument("--tda", type=str, choices=["cpu", "dmabuf"], default="dmabuf",
                        help="Allocator backing Torq device buffers (default: %(default)s).")
    parser.add_argument("--no-npu-clock", action="store_true")
    args = parser.parse_args()

    if not args.no_npu_clock:
        setup_npu_for_inference()

    wav_mode = args.wav is not None
    audio_device = None if wav_mode else choose_audio_device(args.audio_device)

    transcriber = MoonshineTranscriber(args.moonshine_model, tda=args.tda)
    if wav_mode:
        segmenter = SileroSpeechSegmenter(
            sample_rate=SAMPLING_RATE,
            chunk_size=CHUNK_SIZE,
            threshold=args.vad_threshold,
            min_silence_duration_ms=args.silence_ms,
            max_speech_secs=args.max_speech_secs,
            min_segment_secs=args.min_segment_secs,
        )
        try:
            code = transcribe_wav(args.wav, transcriber, segmenter)
        finally:
            if not args.no_npu_clock:
                cleanup_npu_after_inference()
        sys.exit(code)

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
    finally:
        if not args.no_npu_clock:
            cleanup_npu_after_inference()

    logger.debug("Live caption closed.")
