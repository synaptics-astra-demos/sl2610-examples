from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import json
import re
from typing import List
import numpy as np
import time
import threading
import subprocess
from queue import Empty, Queue
from tokenizers import Tokenizer
import logging
from utils.cli import TerminalMode, install_cli_shutdown_handlers
from utils.log import add_logging_args, configure_logging
from utils.moonshine import MoonshineRunner
from utils.gemma import GemmaBackend, load_gemma
import sounddevice as sd
from sounddevice import InputStream
from silero_vad_notorch import VADIterator, load_silero_vad

from utils.download import download_from_hf

ADD_STATS = True
CONF_GATE = 0.7

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PATH = (_THIS_DIR / ".." / "data" / "2610.txt").resolve()
GEMMA_LLAMA_MODEL_PATH = (_THIS_DIR / ".." / "models" / "gemma-3-270m-it-Q8_0.gguf").resolve()
MOONSHINE_MODEL_PATH = (_THIS_DIR / ".." / "models" / "Synaptics" / "moonshine-tiny-bf16-torq").resolve()

configure_logging("INFO")
logger = logging.getLogger("Translate App")

LANGUAGES = {
    "1": "Spanish",
    "2": "French",
    "3": "Russian",
    "4": "Thai",
    "5": "Hindi",
    "6": "Chinese"
}


class TranslateCLIAppState:
    """Thread-safe mutable state shared by the CLI worker threads."""

    def __init__(self, language: str = "Spanish"):
        self.audio_query_q = Queue()
        self.shutdown_event = threading.Event()
        self._lock = threading.RLock()
        self._language = language
        self._voice_on = False
        self._query_processing = False
        self._translation = None

    @property
    def language(self):
        with self._lock:
            return self._language

    def set_language(self, language: str):
        with self._lock:
            self._language = language

    @property
    def translation(self):
        with self._lock:
            return self._translation

    def set_translation(self, translation):
        with self._lock:
            self._translation = translation

    def set_voice_on(self, enabled: bool):
        with self._lock:
            self._voice_on = enabled

    def set_query_processing(self, enabled: bool):
        with self._lock:
            self._query_processing = enabled

    def can_record_audio(self):
        with self._lock:
            return (
                self._voice_on
                and not self._query_processing
                and not self.shutdown_event.is_set()
            )

    def request_shutdown(self):
        with self._lock:
            self._voice_on = False
            self._query_processing = False
        self.shutdown_event.set()

    @property
    def shutdown_requested(self):
        return self.shutdown_event.is_set()


# ---------------------- Language Translation ----------------------
class LanguageTranslation:
    """Wraps a GemmaBackend to build translation prompts and stream responses."""

    def __init__(self, backend: GemmaBackend, state: TranslateCLIAppState):
        self.backend = backend
        self.state = state
        logger.info("LanguageTranslation ready (backend=%s)", type(backend).__name__)

    def stream_response(self, query: str):
        language = self.state.language
        user_prompt = (
            f"Translate the text in quotes to {language}. Output only the translated text.\n\"{query}\"\n"
        )
        logger.debug(user_prompt)

        last_partial = ""
        for partial in self.backend.stream_response(user_prompt):
            last_partial = partial
            yield partial

        yield last_partial.strip()


# ---------------------- CLI Window ----------------------
class CliWindow:
    """Minimal CLI replacement for ChatWindow. Provides the same interface
    used by the audio and LLM worker threads."""

    def __init__(self, state: TranslateCLIAppState):
        self.state = state
        self._terminal = TerminalMode(log=logger)

    def restore_terminal(self):
        self._terminal.restore()

    def shutdown(self):
        self.state.request_shutdown()
        self.restore_terminal()

    def enter_keyboard_mode(self):
        return self._terminal.enter_cbreak()

    def start_keyboard_listener(self):
        """Read single keypresses for language switching."""
        if not self.enter_keyboard_mode():
            return

        try:
            while not self.state.shutdown_requested:
                ch = self._terminal.read_key(timeout=0.1)
                if ch is None:
                    continue
                if ch in LANGUAGES:
                    self.state.set_language(LANGUAGES[ch])
                    print(f"\n[Language changed to: {self.state.language}]", flush=True)
                elif ch == "\x03":  # Ctrl+C
                    self.shutdown()
                    os.kill(os.getpid(), 2)
                    break
        finally:
            self.restore_terminal()

    def show(self):
        print("\n=== Astra SL2610 Voice Translation Engine ===")
        print("Press 1-6 to change language at any time:")
        for key, lang in LANGUAGES.items():
            print(f"  {key}: {lang}")
        print("Speak to translate. Press Ctrl+C to exit.\n")

    def update_user_text(self, text, replace=False):
        if replace:
            print(f"\r[You] {text:<80}", end="", flush=True)
        else:
            print(f"\n[You] {text}")

    def update_response_text(self, text, replace=False):
        if replace:
            print(f"\r[Translation] {text:<80}", end="", flush=True)
        else:
            print(f"\n[Translation] {text}")

    def update_stt_stats(self, infer_time, n_tokens_gen):
        """Update the stt stats display box with new values."""
        tokens_per_sec = n_tokens_gen / infer_time
        print(f"\n\r[Moonshine] {n_tokens_gen} tokens {infer_time:.3f}s {tokens_per_sec:.1f} tok/s")
        
    def update_llm_stats(self, infer_time, n_tokens_in, n_tokens_gen):
        """Update the llm stats display box with new values."""
        tokens_per_sec = n_tokens_gen / infer_time
        print(f"\n\r[Gemma] input={n_tokens_in} output={n_tokens_gen} tokens | {tokens_per_sec:.1f} tok/s | total={infer_time:.3f}s")


def start_llm_input(state: TranslateCLIAppState, window: CliWindow):
    state.set_voice_on(True)
    while not state.shutdown_requested:
        try:
            query = state.audio_query_q.get(timeout=0.1)
        except Empty:
            continue
        if (query != ""):
            state.set_query_processing(True)
            #window.update_response_text(" ")

            # --- Normal streaming ---
            try:
                translation = state.translation
                if translation is None:
                    raise RuntimeError("Translation model not loaded")
                for partial in translation.stream_response(query):
                    window.update_response_text(str(partial), replace=True)
                print()  # newline after full response
                window.update_llm_stats(
                    translation.backend.last_infer_time_ms / 1000,
                    translation.backend.last_n_input_tokens,
                    translation.backend.last_n_output_tokens,
                )
            except Exception as e:
                errstr = f"Error: {e}"
                window.update_response_text(errstr, replace=True)
                logger.info("response: %s", errstr)
            finally:
                state.set_query_processing(False)


def start_audio_thread(state: TranslateCLIAppState, window: CliWindow, audio_device):
    os.environ["PA_ALSA_PLUGHW"] = "1"

    SAMPLING_RATE = 16000
    CHUNK_SIZE = 512  # Silero VAD requirement with sampling rate 16000.
    LOOKBACK_CHUNKS = 5
    MAX_LINE_LENGTH = 80
    # These affect live caption updating - adjust for your platform speed and model.
    MAX_SPEECH_SECS = 10
    MIN_SPEECH_SECS = 1
    MIN_REFRESH_SECS = 2
    MIN_SILENCE_DURATION_MS = 400

    INPUT_LEN = 5  # input len in seconds for moonshine model
    TOKENS_PER_SEC = 6

    caption_cache = []

    class Transcriber(object):
        def __init__(self):
            logger.info("Loading Moonshine model...")
            self.runner = MoonshineRunner(MOONSHINE_MODEL_PATH)
            try:
                self.tokenizer = Tokenizer.from_file(f"{MOONSHINE_MODEL_PATH}/tokenizer.json")
            except (FileNotFoundError, OSError):
                tokenizer_file = download_from_hf("UsefulSensors/moonshine-tiny", "tokenizer.json")
                self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
            logger.info("Moonshine model loaded successfully!")

            self.rate = 16000
            self.inference_secs = 0
            self.number_inferences = 0
            self.speech_secs = 0
            self.__call__(np.zeros(int(self.rate), dtype=np.float32))  # Warmup.

        def __call__(self, speech):
            """Returns string containing Moonshine transcription of speech."""
            self.number_inferences += 1
            self.speech_secs += len(speech) / self.rate
            start_time = time.time()

            tokens = self.runner.run(speech[np.newaxis, :].astype(np.float32))
            infer_time = self.runner.last_infer_time
            n_tokens_gen = self.runner.generated_tokens
            text = self.tokenizer.decode_batch(tokens, skip_special_tokens=True)[0]

            self.inference_secs += time.time() - start_time
            return text, infer_time, n_tokens_gen

    def create_input_callback(q):
        """Callback method for sounddevice InputStream."""
        def input_callback(data, frames, time, status):
            if status:
                logger.debug(status)
            q.put((data.copy().flatten(), status))
        return input_callback

    def end_recording(speech, do_print=True):
        """Transcribes, prints and caches the caption then clears speech buffer."""
        text, infer_time, n_tokens_gen = transcribe(speech)
        if do_print:
            logger.debug(text)
        speech *= 0.0
        return text, infer_time, n_tokens_gen

    def print_captions(text):
        """Prints right justified on same line, prepending cached captions."""
        if len(text) < MAX_LINE_LENGTH:
            for caption in caption_cache[::-1]:
                text = caption + " " + text
                if len(text) > MAX_LINE_LENGTH:
                    break
        if len(text) > MAX_LINE_LENGTH:
            text = text[-MAX_LINE_LENGTH:]
        else:
            text = " " * (MAX_LINE_LENGTH - len(text)) + text
        print("\r" + (" " * MAX_LINE_LENGTH) + "\r" + text, end="", flush=True)

    def soft_reset(vad_iterator):
        """Soft resets Silero VADIterator without affecting VAD model state."""
        vad_iterator.triggered = False
        vad_iterator.temp_end = 0
        vad_iterator.current_sample = 0

    def auto_correct(query):
        """Auto-corrects common mis-transcriptions of important keywords."""
        query = re.sub(r"\bastro\b", "astra", query, flags=re.IGNORECASE)
        for keyword in ["synaptic", "synoptics", "synoptic", "symmetics", "synapix", "synapse", "symaptix"]:
            query = re.sub(r"\b" + re.escape(keyword) + r"\b", "synaptics", query, flags=re.IGNORECASE)
        return query

    # function of the audio thread starts here
    transcribe = Transcriber()

    vad_model = load_silero_vad(onnx=True)
    vad_iterator = VADIterator(
        model=vad_model,
        sampling_rate=SAMPLING_RATE,
        threshold=0.5,
        min_silence_duration_ms=150,
    )

    inputStreamQ = Queue()
    stream = InputStream(
        samplerate=SAMPLING_RATE,
        channels=1,
        device=audio_device,
        blocksize=CHUNK_SIZE,
        dtype=np.float32,
        callback=create_input_callback(inputStreamQ),
    )

    lookback_size = LOOKBACK_CHUNKS * CHUNK_SIZE
    speech = np.empty(0, dtype=np.float32)
    recording = False

    # Start llm listener thread
    llm_thread = threading.Thread(target=start_llm_input, args=(state, window), daemon=True)
    llm_thread.start()

    logger.info("Audio thread initialized")
    logger.debug("Starting Audio stream...")
    try:
        stream.start()
        window.show()
        new_query = 1
        while not state.shutdown_requested:
            try:
                chunk, status = inputStreamQ.get(timeout=0.1)
                if status:
                    logger.debug(status)

                if state.can_record_audio():
                    speech = np.concatenate((speech, chunk))
                    if not recording:
                        speech = speech[-lookback_size:]

                    speech_dict = vad_iterator(chunk)
                    if speech_dict:
                        logger.debug("speech_dict returned %s", str(speech_dict))
                        if "start" in speech_dict and not recording:
                            recording = True
                            start_time = time.time()
                            logger.debug("Started recording at %s", str(start_time))

                        if "end" in speech_dict and recording:
                            logger.debug("Got end at %s", str(time.time()))
                            if (time.time() - start_time) > MIN_SPEECH_SECS:
                                recording = False
                                audio_query, infer_time, n_tokens_gen = end_recording(speech)
                                #Do quick auto-correct on important keywords
                                audio_query = auto_correct(audio_query)
                                if (new_query == 1):
                                    window.update_user_text(audio_query)
                                    if ADD_STATS:
                                        window.update_stt_stats(infer_time, n_tokens_gen)
                                    new_query = 0
                                else:
                                    window.update_user_text(audio_query, replace=True)
                                #if there is a valid query, then run gemma
                                try:
                                    if (len(audio_query.split()) >= 3):
                                        logger.debug("Sending query to LLM %s", str(audio_query))
                                        state.audio_query_q.put_nowait(audio_query)
                                        new_query = 1
                                        for i in range(1, inputStreamQ.qsize()):
                                            inputStreamQ.get()
                                except AttributeError:
                                    pass
                    elif recording:
                        # Possible speech truncation can cause hallucination.
                        if (len(speech) / SAMPLING_RATE) > MAX_SPEECH_SECS:
                            logger.debug("Timeout: ended recording at %s", str(time.time()))
                            recording = False
                            audio_query, infer_time, n_tokens_gen = end_recording(speech)
                            #if there is a valid query, then run gemma
                            try:
                                if (len(audio_query.split()) >= 3):
                                    #Do quick auto-correct on important keywords
                                    audio_query = auto_correct(audio_query)
                                    logger.debug("Sending query to LLM %s", str(audio_query))
                                    state.audio_query_q.put_nowait(audio_query)
                                    # LLM response is handled asynchronously by
                                    # start_llm_input() via update_response_text().
                                    logger.debug("flushing %d elements from the queue", inputStreamQ.qsize())
                                    for i in range(1, inputStreamQ.qsize()):
                                        inputStreamQ.get()
                                soft_reset(vad_iterator)
                            except AttributeError:
                                pass
                else:
                    speech *= 0.0
            except Empty:
                continue
            except KeyboardInterrupt:
                window.shutdown()
    finally:
        logger.debug("Closing Audio stream...")
        stream.close()

    del transcribe
    import gc
    gc.collect()

#  NPU Clock 
def enable_npu_clock():
    """Enable NPU clock via devmem (required before Torq inference)."""
    try:
        subprocess.run(["devmem", "0xf7e104b0", "32", "0x216"],
                       capture_output=True, timeout=5)
        print("[NPU] Clock enabled")
    except Exception as e:
        print(f"[NPU] Clock enable failed: {e}")

# ---------------------- CLI / Entry ----------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Astra Language Translation with Moonshine and Gemma")
    parser.add_argument("--context", type=str, default=str(DEFAULT_PATH))
    parser.add_argument(
        "--use-llama-gemma", action="store_true",
        help="Use llama.cpp (GGUF) backend instead of the default torq VMFB backend.",
    )
    parser.add_argument(
        "--gemma-model", type=str, default=None,
        help="Path to the Gemma model file (.vmfb for torq, .gguf for llama). "
             "Defaults to HF download for torq or the bundled GGUF for llama.",
    )
    parser.add_argument(
        "--non-instruct-model", action="store_true", default=False,
        help="Not an instruct model",
    )
    args = parser.parse_args()
    state = TranslateCLIAppState()
    window = CliWindow(state)
    install_cli_shutdown_handlers(window.shutdown)

    # Set NPU clock
    enable_npu_clock()

    gemma_model_path = args.gemma_model
    if gemma_model_path is None and args.use_llama_gemma:
        gemma_model_path = str(GEMMA_LLAMA_MODEL_PATH)

    gemma_backend = load_gemma(
        use_llama=args.use_llama_gemma,
        model_path=gemma_model_path,
        instruct_model=not args.non_instruct_model,
    )
    translation = LanguageTranslation(gemma_backend, state)
    state.set_translation(translation)

    # Select audio device before starting keyboard listener (which sets raw terminal mode)
    print("List of Audio input devices:")
    print(sd.query_devices())
    audio_device = int(input("Enter input device to listen on: "))

    # Start keyboard listener thread for language switching (starts after input() is done)
    kb_thread = threading.Thread(target=window.start_keyboard_listener, daemon=True)
    kb_thread.start()

    # Start audio listener thread
    audio_thread = threading.Thread(target=start_audio_thread, args=(state, window, audio_device))
    audio_thread.start()

    try:
        audio_thread.join()
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        window.shutdown()
        state.audio_query_q.put_nowait("")
        audio_thread.join(timeout=2)
        state.set_translation(None)
