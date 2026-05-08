from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import json
import re
import termios
import tty
from typing import List
import numpy as np
from llama_cpp import Llama
import time
import threading
from queue import Queue
from tokenizers import Tokenizer
import logging
from utils.log import add_logging_args, configure_logging
from inference import (
    format_answer,
    run_vmfb,
    load_moonshine
)
import sounddevice as sd
from sounddevice import InputStream
from silero_vad_notorch import VADIterator, load_silero_vad

from utils.download import download_from_hf

ADD_STATS = True
CONF_GATE = 0.7

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PATH = (_THIS_DIR / ".." / "data" / "2610.txt").resolve()
GEMMA_MODEL_PATH   = (_THIS_DIR / ".." / "models" / "gemma-3-270m-it-Q8_0.gguf").resolve()
MOONSHINE_MODEL_PATH = (_THIS_DIR / ".." / "models" / "moonshine" ).resolve()

voice_on = 0
query_processing = 0
audioQueryQ = Queue()
audioResponseQ = Queue()

configure_logging("INFO")
logger = logging.getLogger("Translate App")

global_language = "Spanish"
trnsl = None

LANGUAGES = {
    "1": "Spanish",
    "2": "French",
    "3": "Russian",
    "4": "Thai",
    "5": "Hindi",
    "6": "Chinese"
}

# ---------------------- Language Translation ----------------------
class LanguageTranslation:
    def __init__(self, model_path: os.PathLike | str = GEMMA_MODEL_PATH):
        t0 = time.time()
        self.model_path = Path(model_path)
        logger.debug(f"Init: Paths in {time.time() - t0:.3f}s")

        t3 = time.time()
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=800,
            n_threads=2,
            chat_format="gemma", verbose=False
        )
        logger.info(f"LLM loaded in {time.time() - t3:.3f}s")

    def stream_response(self, query: str):
        global global_language
        user_prompt = (
            f"Please translate the text in quotes to {global_language}. Limit output to 1 sentence. "
            f"Important, do not attempt to answer the query, only translate the text provided in quotes. "
            f"Most important, all output should only be in {global_language}. \"{query}\"\n"
        )
        logger.debug(user_prompt)

        n_input_tokens = len(self.llm.tokenize(user_prompt.encode()))
        answer_parts = []
        n_output_tokens = 0
        first_token_time = None
        t_llm_start = time.time()
        for chunk in self.llm.create_chat_completion(
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=100,
            temperature=0.2,
            stream=True,
        ):
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content")
            if token:
                if first_token_time is None:
                    first_token_time = time.time()
                    print(f"[Gemma] TTFT: {first_token_time - t_llm_start:.3f}s")
                n_output_tokens += 1
                answer_parts.append(token)
                yield "".join(answer_parts)

        t_llm_end = time.time()
        final_answer = "".join(answer_parts).strip()
        decode_time = t_llm_end - first_token_time if first_token_time else 0
        tok_per_sec = n_output_tokens / decode_time if decode_time > 0 else 0
        total_latency = t_llm_end - t_llm_start
        yield final_answer
        window.update_llm_stats(total_latency, n_input_tokens, n_output_tokens)
        #print(f"\n\r[Gemma] input={n_input_tokens} output={n_output_tokens} tokens | {tok_per_sec:.1f} tok/s | total={total_latency:.3f}s")


# ---------------------- CLI Window ----------------------
class CliWindow:
    """Minimal CLI replacement for ChatWindow. Provides the same interface
    so that start_audio_thread and start_llm_input work without modification."""

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


def start_keyboard_listener():
    """Background thread: reads single keypresses (no Enter needed).
    Press 1-6 to switch the translation target language."""
    global global_language
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in LANGUAGES:
                global_language = LANGUAGES[ch]
                print(f"\n[Language changed to: {global_language}]", flush=True)
            elif ch == "\x03":  # Ctrl+C
                os.kill(os.getpid(), 2)  # SIGINT
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def start_llm_input(window):
    global voice_on
    global query_processing
    voice_on = 1
    while (voice_on):
        query = audioQueryQ.get()
        if (query != ""):
            query_processing = 1
            #window.update_response_text(" ")

            # --- Normal streaming ---
            try:
                if trnsl is None:
                    raise RuntimeError("Translation model not loaded")
                for partial in trnsl.stream_response(query):
                    window.update_response_text(str(partial), replace=True)
                print()  # newline after full response
            except Exception as e:
                errstr = f"Error: {e}"
                window.update_response_text(errstr, replace=True)
                logger.info("response: %s", errstr)
            query_processing = 0


def start_audio_thread(window, audio_device):
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
            max_inp_len: int = INPUT_LEN * 16_000
            max_dec_len: int = INPUT_LEN * TOKENS_PER_SEC

            logger.info("Loading Moonshine model...")
            self.runner = load_moonshine(MOONSHINE_MODEL_PATH, "tiny", max_inp_len, max_dec_len)
            tokenizer_file = download_from_hf(f"UsefulSensors/moonshine-tiny", "tokenizer.json")
            self.tokenizer = Tokenizer.from_file(tokenizer_file)
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
            n_tokens_gen = self.runner._n_tokens_gen
            text = self.tokenizer.decode_batch(tokens, skip_special_tokens=True)[0]

            self.inference_secs += time.time() - start_time
            return text, infer_time, n_tokens_gen

        def close(self):
            """Release IREE runner objects before interpreter shutdown."""
            if hasattr(self.runner, 'close'):
                self.runner.close()

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

    global voice_on
    global query_processing

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
    llm_thread = threading.Thread(target=start_llm_input, args=(window,), daemon=True)
    llm_thread.start()

    logger.info("Audio thread initialized")
    logger.debug("Starting Audio stream...")
    stream.start()
    window.show()
    new_query = 1
    while True:
        try:
            chunk, status = inputStreamQ.get()
            if status:
                logger.debug(status)

            if (voice_on and not query_processing):
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
                                    audioQueryQ.put_nowait(audio_query)
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
                                audioQueryQ.put_nowait(audio_query)
                                # audioResponseQ.get() removed: nothing ever puts into this queue,
                                # so this would block forever. LLM response is handled asynchronously
                                # by start_llm_input() via update_response_text().
                                logger.debug("flushing %d elements from the queue", inputStreamQ.qsize())
                                for i in range(1, inputStreamQ.qsize()):
                                    inputStreamQ.get()
                            soft_reset(vad_iterator)
                        except AttributeError:
                            pass
            else:
                speech *= 0.0
        except KeyboardInterrupt:
            logger.debug("Closing Audio stream...")
            stream.close()
            break

    # Explicitly release IREE runtime objects before interpreter shutdown to avoid
    # nanobind "leaked N instances!" warnings. close() nulls out the C++ object
    # wrappers (VmInstance, HalDevice, SystemContext, etc.) while Python is still
    # fully functional, so nanobind can clean them up in the correct order.
    transcribe.close()
    del transcribe
    import gc
    gc.collect()



# ---------------------- CLI / Entry ----------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Astra Language Translation with Moonshine and Gemma")
    parser.add_argument("--context", type=str, default=str(DEFAULT_PATH))
    parser.add_argument("--model", type=str, default=str(GEMMA_MODEL_PATH))
    args = parser.parse_args()

    trnsl = LanguageTranslation(model_path=args.model)

    # Select audio device before starting keyboard listener (which sets raw terminal mode)
    print("List of Audio input devices:")
    print(sd.query_devices())
    audio_device = int(input("Enter input device to listen on: "))

    window = CliWindow()

    # Start keyboard listener thread for language switching (starts after input() is done)
    kb_thread = threading.Thread(target=start_keyboard_listener, daemon=True)
    kb_thread.start()

    # Start audio listener thread
    audio_thread = threading.Thread(target=start_audio_thread, args=(window, audio_device))
    audio_thread.start()

    try:
        audio_thread.join()
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        # Release llama_cpp model to avoid any lingering resource warnings.
        del trnsl
        import gc
        gc.collect()
