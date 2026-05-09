from __future__ import annotations

import sys, os

# ---------------------- system ----------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["PA_ALSA_PLUGHW"] = "4"

import time
import threading
import numpy as np
import sounddevice as sd
import logging
import subprocess

from pathlib import Path
from tokenizers import Tokenizer
from queue import Queue

from sounddevice import InputStream
from silero_vad_notorch import VADIterator, load_silero_vad
from utils.download import download_from_hf
from utils.log import add_logging_args, configure_logging
from inference import (
    format_answer,
    run_vmfb,
    load_moonshine
)
# ---------------------- paths ----------------------
_THIS_DIR = Path(__file__).resolve().parent
MOONSHINE_MODEL_PATH = (_THIS_DIR / ".." / "models" / "moonshine" ).resolve()

# ---------------------- sound device config ----------------------
SAMPLING_RATE = 16000
CHUNK_SIZE = 512  # Silero VAD requirement with sampling rate 16000.
LOOKBACK_CHUNKS = 5
MAX_LINE_LENGTH = 80
# These affect live caption updating - adjust for your platform speed and model.
MAX_SPEECH_SECS = 10
MIN_SPEECH_SECS = 1
MIN_REFRESH_SECS = 2
MIN_SILENCE_DURATION_MS = 400

# ---------------------- moonshine config ----------------------
INPUT_LEN = 5 # input len in seconds for moonshine model
TOKENS_PER_SEC = 6

running = False

def start_audio_thread():

    class Transcriber(object):
        def __init__(self):
            max_inp_len: int = INPUT_LEN  * 16_000
            max_dec_len: int = INPUT_LEN  * TOKENS_PER_SEC

            # Initialize Moonshine components
            print("Loading Moonshine model...")
            self.runner = load_moonshine( MOONSHINE_MODEL_PATH, "tiny", max_inp_len, max_dec_len)
            #tokenizer_file = "tokenizer.json"
            tokenizer_file = download_from_hf(f"UsefulSensors/moonshine-tiny", "tokenizer.json")
            self.tokenizer = Tokenizer.from_file(tokenizer_file)
            print("Moonshine model loaded successfully!")

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
            text = self.tokenizer.decode_batch(tokens, skip_special_tokens=True)[0]
            #text = "Tell me something about the SL2610"

            self.inference_secs += time.time() - start_time
            return text


    def create_input_callback(q):
        """Callback method for sounddevice InputStream."""

        def input_callback(data, frames, time, status):
            if status:
                print(status)
            q.put((data.copy().flatten(), status))

        return input_callback


    def end_recording(speech, do_print=True):
        """Transcribes, prints and caches the caption then clears speech buffer."""
        text = transcribe(speech)
        if do_print:
            print(text)
            #print_captions(text)
        #caption_cache.append(text)
        speech *= 0.0
        return text


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
        #print(textwrap.wrap(text, width=40))


    def soft_reset(vad_iterator):
        """Soft resets Silero VADIterator without affecting VAD model state."""
        vad_iterator.triggered = False
        vad_iterator.temp_end = 0
        vad_iterator.current_sample = 0


    # function of the audio thread starts here
    configure_logging("INFO")
    logger = logging.getLogger("live_caption")
    transcribe = Transcriber()
    global running


    vad_model = load_silero_vad(onnx=True)
    vad_iterator = VADIterator(
        model=vad_model,
        sampling_rate=SAMPLING_RATE,
        threshold=0.5,
        min_silence_duration_ms=150,
    )

    #Ask user for input device
    print("List of Audio input devices:")
    print(sd.query_devices())
    audio_device = int(input("Enter input device to listen on: "))

    inputStreamQ = Queue()
    stream = InputStream(
        samplerate=SAMPLING_RATE,
        channels=1,
        device=audio_device,
        blocksize=CHUNK_SIZE,
        dtype=np.float32,
        callback=create_input_callback(inputStreamQ),
    )

    caption_cache = []
    lookback_size = LOOKBACK_CHUNKS * CHUNK_SIZE
    speech = np.empty(0, dtype=np.float32)

    recording = False

    logger.info("Audio thread initialized")
    logger.debug("Starting Audio stream...")
    stream.start()
    running = True

    print("Moonshine is running. Press Ctrl+C to quit Moonshine.\n")

    while running:
        chunk, status = inputStreamQ.get()
        if status:
            print(status)


        speech = np.concatenate((speech, chunk))
        if not recording:
            speech = speech[-lookback_size:]

        speech_dict = vad_iterator(chunk)
        if speech_dict:
            logger.debug("speech_dict returned %s",str(speech_dict))
            if "start" in speech_dict and not recording:
                recording = True
                start_time = time.time()
                logger.debug("Started recording at %s",str(start_time))

            if "end" in speech_dict and recording:
                logger.debug("Got end at %s",str(time.time()))
                if  (time.time() - start_time) > MIN_SPEECH_SECS:
                    recording = False
                    audio_query= end_recording(speech)
                    #if there is a valid query, then run gemma
                    if (len(audio_query.split()) >= 3):
                        logger.debug("flushing",inputStreamQ.qsize(),"elements from the queue")
                        for i in range(1, inputStreamQ.qsize()):
                            inputStreamQ.get()
                    
        elif recording:
            # Possible speech truncation can cause hallucination.

            if (len(speech) / SAMPLING_RATE) > MAX_SPEECH_SECS:
                logger.debug("Timeout: ended recording at %s",str(time.time()))
                recording = False
                audio_query= end_recording(speech)
                #if there is a valid query, then run gemma
                if (len(audio_query.split()) >= 3):
                    logger.debug("flushing",inputStreamQ.qsize(),"elements from the queue")
                    for i in range(1, inputStreamQ.qsize()):
                        inputStreamQ.get()
                soft_reset(vad_iterator)

    logger.debug("Closing Audio stream...")
    stream.close()

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

    # Set NPU clock
    enable_npu_clock()

    # Start audio listener thread for Moonshine
    audio_thread = threading.Thread(target=start_audio_thread)
    audio_thread.start()
    
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            running = False
            logger.debug("Closing Moonthine...\n")
            audio_thread.join()
            break

    logger.debug("Moonthine Speech-To-Text Example is closed...\n")



  
