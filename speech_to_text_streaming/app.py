# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Moonshine streaming microphone demo (2-Split VMFB).

Captures live microphone audio, runs a VAD (self-calibrating energy detector,
or optionally Silero's neural VAD) to split utterances, and transcribes them
with committed-prefix incremental decode for a real-time live preview. See
``runner.py`` for the inference engine.
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import logging
import queue
import re
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError):
    # sounddevice/PortAudio not installed on this machine — fine for --wav-only usage.
    sd = None

try:
    from tokenizers import Tokenizer
except ImportError:
    print("Error: tokenizers is not installed. Please run:", file=sys.stderr)
    print("  pip install tokenizers", file=sys.stderr)
    sys.exit(1)

from app_utils.torq_examples.moonshine_streaming.src.runner import MoonshineStaticStreamingModel, find_asset
from app_utils.log import add_logging_args, configure_logging
from app_utils.npu import enable_npu_clock
from profiling import WorkerProfiler
from vad import EnergyVAD, SileroVAD

logger = logging.getLogger("speech_to_text_streaming")

# Sentinel put on the audio queue to mark end-of-input in --wav mode (never
# emitted by the live mic path, which just keeps streaming until Ctrl+C).
_END_OF_STREAM = object()


# ── Terminal renderer ─────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r'\033\[[0-9;]*m')


class TerminalListener:
    """ANSI terminal renderer supporting clean wrapped multi-line overwriting."""
    def __init__(self):
        self.prev_rows = 1
        self._last_live_draw = 0.0

    def draw(self, text):
        try:
            cols = os.get_terminal_size().columns
        except OSError:
            cols = 80
        if self.prev_rows > 1:
            sys.stdout.write(f"\033[{self.prev_rows - 1}A")
        sys.stdout.write("\r")
        # Paint the new frame over the old one in place (unchanged characters
        # are simply overwritten, never blanked), then trim whatever trails
        # off past the new content. Erasing *before* writing (the previous
        # order) blanks the whole block on every redraw, which is what
        # produced the visible flicker at the ~12.5 Hz this is called during
        # speech.
        sys.stdout.write(text)
        sys.stdout.write("\033[J")
        sys.stdout.flush()
        visible = _ANSI_RE.sub('', text)
        rows = sum(max(1, (len(line) + cols - 1) // cols) for line in visible.split('\n'))
        self.prev_rows = max(1, rows)

    def draw_live(self, text, min_interval=0.1):
        """Throttled variant of draw() for the high-frequency, mostly-cosmetic
        live indicator (vol/buffer bars), which is otherwise redrawn on every
        80 ms audio chunk (~12.5 Hz) even when nothing meaningful changed.
        Skips the redraw if the previous one was less than min_interval ago."""
        now = time.monotonic()
        if now - self._last_live_draw < min_interval:
            return
        self._last_live_draw = now
        self.draw(text)

    def complete_line(self):
        sys.stdout.write("\n")
        sys.stdout.flush()
        self.prev_rows = 1


# ── Utilities ─────────────────────────────────────────────────────────────────

def resample(audio, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return audio
    duration           = len(audio) / orig_sr
    num_target_samples = int(duration * target_sr)
    indices            = np.linspace(0, len(audio) - 1, num_target_samples)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def _vol_bar(rms, threshold, width=10):
    fill = min(int(rms / max(threshold * 2, 1e-9) * width), width)
    bar  = '=' * fill + ' ' * (width - fill)
    col  = '\033[32m' if rms > threshold else '\033[2m'
    return f"[{col}{bar}\033[0m]"


def _buf_bar(fill_frames, max_frames, width=10):
    pct      = fill_frames / max(max_frames, 1)
    fill     = min(int(pct * width), width)
    bar      = '=' * fill + ' ' * (width - fill)
    col      = '\033[31m' if pct > 0.8 else '\033[33m' if pct > 0.5 else '\033[32m'
    secs     = fill_frames * 0.020
    max_secs = int(max_frames * 0.020)
    return f"[{col}{bar}\033[0m] {secs:.1f}/{max_secs}s"


# ── Entry point ───────────────────────────────────────────────────────────────

def choose_audio_device(device_arg: str | None) -> int | str | None:
    """Resolve the --device argument to what sounddevice expects.

    If not given on the command line, list the available input devices and
    prompt for one interactively (mirrors speech_to_text/live_caption.py) —
    device indices are not stable across boards/OS images, so a hardcoded
    default is liable to point at a non-capture device and fail with a
    PortAudio "Invalid number of channels" error.
    """
    if device_arg is None:
        print("List of Audio input devices:")
        print(sd.query_devices())
        device_arg = input("Enter input device to listen on [default]: ").strip()
    if device_arg == "":
        return None
    try:
        return int(device_arg)
    except ValueError:
        return device_arg


def main(args: argparse.Namespace):
    configure_logging(args.logging)

    if args.list_devices:
        if sd is None:
            logger.error("sounddevice/PortAudio is not available — cannot list audio devices.")
            sys.exit(1)
        print("\nAvailable Audio Devices:")
        print(sd.query_devices())
        sys.exit(0)

    wav_mode = bool(args.wav)
    if not wav_mode and sd is None:
        logger.error(
            "sounddevice/PortAudio is not available. Install it (see README) or pass "
            "--wav <file> to transcribe a pre-recorded file instead of the microphone."
        )
        sys.exit(1)

    model_dir = args.model_dir

    if not os.path.isdir(model_dir):
        logger.error("Model directory %s not found.", model_dir)
        sys.exit(1)

    logger.info("Using model dir:  %s", model_dir)
    if args.full_decode:
        logger.info("Decode mode:      full re-decode from BOS (baseline)")
    else:
        logger.info(
            "Decode mode:      incremental committed-prefix "
            "(LocalAgreement-%d, commit-delay %.1fs)",
            args.commit_agreement, args.commit_delay,
        )

    ok, message = enable_npu_clock()
    print(f"[NPU] {message}")


    try:
        model     = MoonshineStaticStreamingModel(model_dir,
                                                   runtime_flags=args.runtime_flags)
        tokenizer = Tokenizer.from_file(find_asset(model.model_dir, "tokenizer.json"))
    except Exception as e:
        logger.error("Error initializing models: %s", e)
        sys.exit(1)

    wav_audio = None
    device = None
    if wav_mode:
        try:
            import soundfile as sf
        except ImportError:
            logger.error("soundfile is required for --wav. Install it with: pip install soundfile")
            sys.exit(1)
        if not os.path.isfile(args.wav):
            logger.error("WAV file %s not found.", args.wav)
            sys.exit(1)
        data, input_sample_rate = sf.read(args.wav, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        wav_audio = data.astype(np.float32)
        logger.info(
            "Transcribing WAV file:  %s (%.1fs @ %d Hz)",
            args.wav, len(wav_audio) / input_sample_rate, input_sample_rate,
        )
    else:
        device = choose_audio_device(args.device)
        logger.info("Setting up microphone stream...")
        try:
            device_info       = sd.query_devices(device, "input")
            input_sample_rate = int(device_info.get("default_samplerate", 16000))
        except Exception:
            input_sample_rate = 16000

    audio_queue = queue.Queue()
    running     = True

    def audio_callback(in_data, frames, time_info, status):
        if not running:
            return
        if in_data is not None:
            audio_queue.put(in_data.copy().astype(np.float32).flatten())

    sd_stream = None
    if not wav_mode:
        try:
            sd_stream = sd.InputStream(
                samplerate=input_sample_rate,
                blocksize=4096,
                latency="high",
                device=device,
                channels=1,
                dtype="float32",
                callback=audio_callback,
            )
        except sd.PortAudioError as e:
            logger.error("Error opening audio device: %s", e)
            sys.exit(1)

    vad_threshold = args.vad_threshold
    if vad_threshold is None:
        vad_threshold = 0.5 if args.vad_backend == "silero" else 0.010
    if args.vad_backend == "silero":
        logger.info("VAD backend:      silero (%s, threshold %.2f)",
                     args.vad_model or "bundled silero-vad-notorch model", vad_threshold)
        vad = SileroVAD(args.vad_model, threshold=vad_threshold, silence_duration=args.vad_silence,
                         report_calibration=args.profile)
    else:
        logger.info("VAD backend:      energy (self-calibrating, floor %.4f)", vad_threshold)
        vad = EnergyVAD(threshold=vad_threshold, silence_duration=args.vad_silence,
                         report_calibration=args.profile)
    terminal = TerminalListener()
    state    = model.create_state()

    prof = WorkerProfiler(model.chunk_len / 16000 * 1000) if args.profile else None
    if prof:
        logger.info("[profile] enabled — chunk budget %.1f ms", prof.chunk_budget_ms)

    def worker():
        tokens              = []
        utterance_count     = 0
        resampled_buffer    = np.array([], dtype=np.float32)
        chunks_since_decode = 0

        # Pre-speech look-behind buffer: rolls over every "silence" chunk so that
        # when speech_start fires, we have a few chunks of real audio (ambient
        # noise + whatever soft onset the VAD hadn't crossed threshold on yet)
        # to replay instead of losing that window to the encoder's warmup
        # discard (see the replay below). Defaults to warmup_chunks so the
        # replay exactly covers the window that would otherwise be thrown away.
        lookback_chunks = args.vad_lookback if args.vad_lookback is not None else model.warmup_chunks
        lookback_buffer = deque(maxlen=max(lookback_chunks, 0))

        def _decode():
            if args.full_decode:
                return model.decode(state)
            return model.decode_incremental(state, args.commit_delay, args.commit_agreement)

        while running:
            if prof:
                prof.queue_depth.append(
                    (time.perf_counter() - prof.t_start, audio_queue.qsize()))
            try:
                chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if chunk is _END_OF_STREAM:
                # --wav mode: input exhausted. Finalize whatever utterance is
                # still in flight (the file may not end in enough trailing
                # silence for the VAD to have called speech_end itself).
                if state.cross_kv_fill > 0 or vad.is_speaking:
                    terminal.draw(f"\033[34m◉\033[0m Utterance #{utterance_count}: processing...")
                    model.encode(state, is_final=True)
                    if prof:
                        _t_dec = time.perf_counter()
                    tokens = _decode()
                    if prof:
                        prof.decode_ms.append((time.perf_counter() - _t_dec) * 1000)
                        prof.decode_steps.append(state.last_decode_steps)
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                    terminal.draw(f"\033[32m✓\033[0m Utterance #{utterance_count}: {text if text else '(empty)'}")
                    terminal.complete_line()
                audio_queue.task_done()
                break

            chunk_16k        = resample(chunk, input_sample_rate, 16000)
            resampled_buffer = np.concatenate([resampled_buffer, chunk_16k])

            chunk_size = model.chunk_len
            while len(resampled_buffer) >= chunk_size:
                audio_chunk_1280 = resampled_buffer[:chunk_size]
                resampled_buffer = resampled_buffer[chunk_size:]

                if prof:
                    _t_chunk    = time.perf_counter()
                    _had_decode = False

                vad_status = vad.process_chunk(audio_chunk_1280)

                if vad_status == "speech_start":
                    state.reset()
                    tokens = []
                    chunks_since_decode = 0
                    utterance_count += 1
                    terminal.draw(f"\033[32m●\033[0m Utterance #{utterance_count}: [Listening...]")

                    # Replay the buffered pre-speech chunks through the fresh state
                    # before the triggering chunk below. chunk_idx is 0 right after
                    # reset(), so these calls land in the encoder's warmup window
                    # (their cross-KV output is discarded either way, see
                    # process_audio_chunk) — we're just choosing to spend that
                    # discarded window on real pre-onset audio instead of on the
                    # first spoken syllables.
                    for lb_chunk in lookback_buffer:
                        if prof:
                            _t_lb = time.perf_counter()
                        model.process_audio_chunk(state, lb_chunk)
                        model.encode(state, is_final=False)
                        if prof:
                            prof.encode_ms.append((time.perf_counter() - _t_lb) * 1000)
                    lookback_buffer.clear()

                if vad_status in ("speech", "speech_start"):
                    if prof:
                        _t_enc = time.perf_counter()
                    model.process_audio_chunk(state, audio_chunk_1280)
                    model.encode(state, is_final=False)
                    if prof:
                        prof.encode_ms.append((time.perf_counter() - _t_enc) * 1000)
                    chunks_since_decode += 1

                    # Auto-finalize when cross-KV buffer is full
                    if state.cross_kv_fill >= model.max_memory_len:
                        buf_secs = int(model.max_memory_len * 0.020)
                        terminal.draw(
                            f"\033[31m⚠\033[0m Utterance #{utterance_count}:"
                            f" buffer full ({buf_secs}s limit) — finalizing..."
                        )
                        model.encode(state, is_final=True)
                        if prof:
                            _t_dec = time.perf_counter()
                        tokens = _decode()
                        if prof:
                            prof.decode_ms.append((time.perf_counter() - _t_dec) * 1000)
                            prof.decode_steps.append(state.last_decode_steps)
                            _had_decode = True
                        text   = tokenizer.decode(tokens, skip_special_tokens=True)
                        terminal.draw(f"\033[32m✓\033[0m Utterance #{utterance_count}: {text if text else '(empty)'}")
                        terminal.complete_line()
                        state.reset()
                        tokens = []
                        chunks_since_decode = 0
                        utterance_count += 1
                        terminal.draw(f"\033[32m●\033[0m Utterance #{utterance_count}: [Listening...]")
                        if prof:
                            prof.record_chunk((time.perf_counter() - _t_chunk) * 1000, _had_decode)
                        continue

                    # Periodic live preview decode
                    if chunks_since_decode >= args.preview_every and state.cross_kv_fill > 0:
                        if prof:
                            _t_dec = time.perf_counter()
                        tokens = _decode()
                        if prof:
                            prof.decode_ms.append((time.perf_counter() - _t_dec) * 1000)
                            prof.decode_steps.append(state.last_decode_steps)
                            _had_decode = True
                        chunks_since_decode = 0

                    text = tokenizer.decode(tokens, skip_special_tokens=True) if tokens else ""
                    dot = "\033[33m●\033[0m" if vad.silence_remaining_sec > 0 else "\033[32m●\033[0m"
                    indicator = (
                        f"{dot} Utterance #{utterance_count}"
                        f"  vol {_vol_bar(vad.last_score, vad.threshold)}"
                        f"  buf {_buf_bar(state.cross_kv_fill, model.max_memory_len)}"
                    )
                    if vad.silence_remaining_sec > 0:
                        # Constant label, only the trailing seconds count changes.
                        indicator += f"  \033[33mfinalizing {vad.silence_remaining_sec:.1f}s\033[0m"
                    terminal.draw_live(f"{indicator}\n{text if text else '...'}")

                elif vad_status == "speech_end":
                    terminal.draw(f"\033[34m◉\033[0m Utterance #{utterance_count}: processing...")
                    model.process_audio_chunk(state, audio_chunk_1280)
                    model.encode(state, is_final=True)
                    if prof:
                        _t_dec = time.perf_counter()
                    tokens = _decode()
                    if prof:
                        prof.decode_ms.append((time.perf_counter() - _t_dec) * 1000)
                        prof.decode_steps.append(state.last_decode_steps)
                        _had_decode = True
                    text   = tokenizer.decode(tokens, skip_special_tokens=True)
                    terminal.draw(f"\033[32m✓\033[0m Utterance #{utterance_count}: {text if text else '(empty)'}")
                    terminal.complete_line()
                    chunks_since_decode = 0

                elif vad_status == "silence":
                    lookback_buffer.append(audio_chunk_1280.copy())

                if prof:
                    prof.record_chunk((time.perf_counter() - _t_chunk) * 1000, _had_decode)

            audio_queue.task_done()

    def print_profile_summary():
        if not prof:
            return
        out = args.profile_out or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profile_results")
        try:
            prof.summary(out_dir=out)
        except Exception as e:
            print(f"[profile] summary failed: {e}", file=sys.stderr)

    def feed_wav_to_queue():
        """Push the WAV file onto audio_queue in mic-sized blocks, then signal
        end-of-stream. A synthetic silence lead-in is prepended so the VAD's
        self-calibration (~1s of ambient noise) has something to sample even
        when the file starts talking immediately."""
        lead_in = np.zeros(int(1.0 * input_sample_rate), dtype=np.float32)
        full    = np.concatenate([lead_in, wav_audio])
        block   = 4096
        pos     = 0
        while pos < len(full) and running:
            end = min(pos + block, len(full))
            audio_queue.put(full[pos:end])
            if args.realtime:
                time.sleep((end - pos) / input_sample_rate)
            pos = end
        audio_queue.put(_END_OF_STREAM)

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    # Hide the terminal cursor while the live transcript is being redrawn in
    # place — draw() writes "\r" (parking the cursor at column 0, right on
    # the leading "●") before overwriting it with new text, so at the redraw
    # rate used here the terminal's own blinking block cursor visibly flashes
    # over that character. Always restored in the finally below.
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()
    try:
        if wav_mode:
            if args.vad_backend == "energy":
                print("\n[VAD] Calibrating noise floor from synthetic silence lead-in...", file=sys.stderr)
            print(f">>> Transcribing {args.wav} (Static 2-Split VMFB)... <<<\n", file=sys.stderr)
            try:
                feed_wav_to_queue()
                worker_thread.join()
            except KeyboardInterrupt:
                print("\n\nInterrupted...", file=sys.stderr)
                running = False
                worker_thread.join(timeout=1.0)
            finally:
                print_profile_summary()
        else:
            if args.vad_backend == "energy":
                print("\n[VAD] Calibrating noise floor — please remain silent...", file=sys.stderr)
            sd_stream.start()
            time.sleep(1.0)

            print(
                ">>> Listening (Static 2-Split VMFB). Start speaking! Press Ctrl+C to exit. <<<\n",
                file=sys.stderr,
            )

            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n\nExiting...", file=sys.stderr)
            finally:
                running = False
                try:
                    sd_stream.stop()
                    sd_stream.close()
                except Exception:
                    pass
                worker_thread.join(timeout=1.0)
                print_profile_summary()
    finally:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Moonshine Static Streaming Microphone Demo (2-Split VMFB)"
    )
    parser.add_argument("--device",        type=str,   default=None,           help="Input device index or name. If omitted, lists devices and prompts interactively.")
    parser.add_argument("--wav",           type=str,   default=None,           help="Transcribe a WAV file instead of the live microphone")
    parser.add_argument("--realtime",      action="store_true",               help="With --wav, pace the feed to match real-time playback speed (default: feed as fast as possible)")
    parser.add_argument("-m", "--model-dir", type=str, default="../models/Synaptics/moonshine-streaming-tiny-torq", metavar="DIR", help="Path to the flat moonshine-streaming-tiny model dir (required)")
    parser.add_argument("--vad-backend",   type=str,   default="silero",
                        choices=["energy", "silero"],
                        help="VAD implementation: self-calibrating RMS energy, or Silero's neural VAD (default: energy)")
    parser.add_argument("--vad-model",     type=str,   default=None,           help="Path to a custom Silero VAD onnx file (only used with --vad-backend silero; default: bundled model from silero-vad-notorch, no download needed)")
    parser.add_argument("--vad-threshold", type=float, default=None,           help="VAD trigger threshold: RMS floor for 'energy' (default: 0.010), speech probability for 'silero' (default: 0.5)")
    parser.add_argument("--vad-silence",   type=float, default=2.5,            help="Silence gap to split utterances in seconds (default: 2.5)")
    parser.add_argument("--vad-lookback",  type=int,   default=None,           help="Pre-speech chunks to replay into the encoder on speech_start, to avoid clipping word onsets (default: model.warmup_chunks; 0 disables)")
    parser.add_argument("--preview-every", type=int,   default=5,              help="Chunks the decoder waits between live preview decodes (default: 5)")
    parser.add_argument("--commit-agreement", type=int, default=2,             help="LocalAgreement-N: commit a token only if stable across the last N hypotheses (default: 2)")
    parser.add_argument("--commit-delay",  type=float, default=3.0,            help="Only commit tokens at least this many seconds of audio behind the live frontier (default: 3.0)")
    parser.add_argument("--full-decode",   action="store_true",               help="Disable incremental decode; re-decode from BOS each time (baseline behaviour)")
    parser.add_argument("--profile",       action="store_true",               help="Record per-chunk worker timing, missed-real-time count, decode/encode latency and queue depth; print + dump on exit")
    parser.add_argument("--profile-out",   type=str,   default=None,           help="Directory for --profile dumps (default: speech_to_text_streaming/profile_results)")
    parser.add_argument("--list-devices",  "-l", action="store_true",         help="List audio devices and exit")
    parser.add_argument("--no-refresh",    action="store_true", default=False, help="Skip the Hugging Face check for updated models (offline/airgapped runs)")
    add_logging_args(parser)
    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument(
        "--runtime-flags",
        nargs=argparse.REMAINDER,
        default=None,
        metavar="FLAG",
        help=(
            "[Advanced] Extra flags for the Torq runtime (e.g. --torq_hw_type=sim). "
            "Must be specified last; all remaining arguments are forwarded."
        ),
    )
    args = parser.parse_args()
    # --list-devices is a standalone utility (it is how you find the --device
    # index), so it must still work without a model dir.
    if args.model_dir is None and not args.list_devices:
        parser.error("-m/--model-dir is required")
    main(args)
