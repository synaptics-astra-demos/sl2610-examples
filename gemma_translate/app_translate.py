from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import json
import re
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

# UI constants
BUBBLE_FONT_SIZE = 24  # px
CHAT_WIDTH = 800 #1700
CHAT_HEIGHT = 400 #900
ADD_STATS = True
CONF_GATE = 0.7

_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_PATH = (_THIS_DIR / ".." / "data" / "2610.txt").resolve()
GEMMA_MODEL_PATH   = (_THIS_DIR / ".." / "models" / "gemma-3-270m-it-Q8_0.gguf").resolve()
MOONSHINE_MODEL_PATH = (_THIS_DIR / ".." / "models" / "moonshine" ).resolve()

HINDI_FONT_PATH="./fonts/NotoSansDevanagari-VariableFont_wdth,wght.ttf"
CHINESE_FONT_PATH="./fonts/NotoSansSC-VariableFont_wght.ttf"
THAI_FONT_PATH="./fonts/NotoSansThai-VariableFont_wdth,wght.ttf"
voice_on = 0
query_processing = 0
audioQueryQ = Queue()
audioResponseQ = Queue()

configure_logging("INFO")
logger = logging.getLogger("Translate App")

global_language = "Spanish"
trnsl = None


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
        if ADD_STATS:
            window.update_llm_stats(total_latency, n_input_tokens, n_output_tokens)
                                

def start_llm_input(window):
    global voice_on
    global query_processing
    voice_on = 1
    while (voice_on):
        query = audioQueryQ.get()
        if (query != ""):
            query_processing = 1
            window.update_response_text(" ")

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
            #tokenizer_file = "tokenizer.json"
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
    llm_thread = threading.Thread(target=start_llm_input, args=(window,))
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
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QScrollArea, QFrame, QSizePolicy, QPushButton, QStyle)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QSize
from PyQt5.QtGui import QIcon, QFontDatabase

# --- Signal Management ---
class CommSignals(QObject):
    """Signals to bridge background threads to the Main UI thread."""
    # Signals carry: (text: str, replace: bool)
    update_user = pyqtSignal(str, bool)
    update_resp = pyqtSignal(str, bool)

class MessageBubble(QFrame):
    def __init__(self, text, is_user=True):
        super().__init__()
        self.vbox = QVBoxLayout(self)
        self.label = QLabel()
        self.label.setWordWrap(True)

        # Color palette
        bg_color = "#007AFF" if is_user else "#E1E1E1"
        text_color = "white" if is_user else "#1C1C1E"

        self.setStyleSheet(f"""
            MessageBubble {{
                background-color: {bg_color};
                border-radius: 14px;
                margin: 4px;
            }}
            QLabel {{
                color: {text_color};
                padding: 8px;
                font-size: {BUBBLE_FONT_SIZE}px;
                font-family: 'Segoe UI', sans-serif;
            }}
        """)
        self.vbox.addWidget(self.label)
        self.label.setText(text+"\n")
        self.label.adjustSize()
        self.label.updateGeometry()
        self.adjustSize()
        #logger.info("labelText: ", text)

    def set_text(self, new_text):
        self.label.setText(new_text+"\n")
        self.label.adjustSize()
        self.label.updateGeometry()
        self.adjustSize()
        #logger.info("labelText: ", new_text)

class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.active_user_bubble = None
        self.active_resp_bubble = None

        # Initialize thread-safe signals
        self.signals = CommSignals()
        self.signals.update_user.connect(self._handle_user_update)
        self.signals.update_resp.connect(self._handle_resp_update)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Astra SL2610 Voice Translation Engine")
        self.resize(CHAT_WIDTH, CHAT_HEIGHT)

        layout = QVBoxLayout(self)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)

        self.container = QWidget()
        self.chat_layout = QVBoxLayout(self.container)
        self.chat_layout.addStretch() # Keeps messages pinned to the top

        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)

        # Stats display box at the bottom
        self.stats_stt_label = QLabel("")
        self.stats_stt_label.setStyleSheet("font-size: 18px; color: #333; background: #f5f5f5; padding: 6px; border-radius: 8px;")
        layout.addWidget(self.stats_stt_label, alignment=Qt.AlignRight)

        # Stats display box at the bottom
        self.stats_llm_label = QLabel("")
        self.stats_llm_label.setStyleSheet("font-size: 18px; color: #333; background: #f5f5f5; padding: 6px; border-radius: 8px;")
        layout.addWidget(self.stats_llm_label, alignment=Qt.AlignRight)

        # Add buttons at the bottom, aligned to bottom right
        button_row = QHBoxLayout()

        # 1. Create and add the Dropdown (QComboBox) to the left
        self.language_dropdown = QComboBox()
        self.language_dropdown.addItems(["Spanish", "French", "Russian", "Thai", "Hindi", "Chinese"])
        self.language_dropdown.setMinimumHeight(48) # Matching the height of your buttons
        self.language_dropdown.setStyleSheet("font-size: 16px; padding: 5px;")
        # Connect the callback
        self.language_dropdown.currentTextChanged.connect(self.on_language_change)
        button_row.addWidget(self.language_dropdown)

        # 2. Add Stretch to push everything else to the right
        button_row.addStretch()

        self.settings_button = QPushButton()
        self.settings_button.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))  # gear-like icon
        self.settings_button.setMinimumSize(80, 48)
        self.settings_button.setMaximumSize(80, 48)
        self.settings_button.setIconSize(QSize(40, 40))
        self.settings_button.clicked.connect(self.open_settings)
        button_row.addWidget(self.settings_button)
        self.clear_button = QPushButton()
        self.clear_button.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        self.clear_button.setMinimumSize(80, 48)
        self.clear_button.setMaximumSize(80, 48)
        self.clear_button.setIconSize(QSize(40, 40))
        self.clear_button.clicked.connect(self.clear_chat)
        button_row.addWidget(self.clear_button)
        layout.addLayout(button_row)

    def on_language_change(self, selected_language):
        global global_language
        logger.info(f"Language changed to: {selected_language}")
        if (selected_language == "Chinese"):
            selected_language = "Simplified Chinese"
        global_language = selected_language

    def open_settings(self):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "Settings", "Settings not implemented yet.")

    # --- Internal UI Thread Handlers ---
    def _handle_user_update(self, text, replace):
        if replace and self.active_user_bubble:
            self.active_user_bubble.set_text(text)
        else:
            self.active_user_bubble = MessageBubble(text, is_user=True)
            self.chat_layout.insertWidget(self.chat_layout.count() - 1,
                                        self.active_user_bubble,
                                        alignment=Qt.AlignRight)
        self._scroll_to_bottom()

    def _handle_resp_update(self, text, replace):
        if replace and self.active_resp_bubble:
            self.active_resp_bubble.set_text(text)
        else:
            self.active_resp_bubble = MessageBubble(text, is_user=False)
            self.chat_layout.insertWidget(self.chat_layout.count() - 1,
                                        self.active_resp_bubble,
                                        alignment=Qt.AlignLeft)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        # Small delay to allow layout to update before scrolling
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

    def clear_chat(self):
        """Clear all message bubbles, flush queues, and reset processing."""
        # Remove all message bubbles
        for i in reversed(range(self.chat_layout.count())):
            item = self.chat_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), MessageBubble):
                widget = item.widget()
                self.chat_layout.removeWidget(widget)
                widget.setParent(None)
                widget.deleteLater()
        
        # Reset active bubbles
        self.active_user_bubble = None
        self.active_resp_bubble = None
        self.active_stt_bubble = None
        self.active_llm_bubble = None
        # Flush queues
        global audioQueryQ, audioResponseQ
        while not audioQueryQ.empty():
            try:
                audioQueryQ.get_nowait()
            except:
                pass
        while not audioResponseQ.empty():
            try:
                audioResponseQ.get_nowait()
            except:
                pass
        
        # Reset processing flags
        global query_processing
        query_processing = 0
    def update_user_text(self, text, replace=False):
        """Call to add or replace the latest User bubble."""
        if replace:
            print(f"\r[You] {text:<80}", end="", flush=True)
        else:
            print(f"\n[You] {text}")
        self.signals.update_user.emit(text, replace)

    def update_response_text(self, text, replace=False):
        """Call to add or replace the latest Response bubble."""
        if replace:
            print(f"\r[Translation] {text:<80}", end="", flush=True)
        else:
            print(f"\n[Translation] {text}")
        self.signals.update_resp.emit(text, replace)

    def update_stt_stats(self, infer_time, n_tokens_gen):
        """Update the stt stats display box with new values."""
        tokens_per_sec = n_tokens_gen / infer_time
        self.stats_stt_label.setText(f"Moonshine: {n_tokens_gen} tokens {infer_time:.3f} s {tokens_per_sec:.1f} tokens/s")
        print(f"\n\r[Moonshine] {n_tokens_gen} tokens {infer_time:.3f}s {tokens_per_sec:.1f} tok/s")
        
    def update_llm_stats(self, infer_time, n_tokens_in, n_tokens_gen):
        """Update the llm stats display box with new values."""
        tokens_per_sec = n_tokens_gen / infer_time
        self.stats_llm_label.setText(f"Gemma: in={n_tokens_in} out={n_tokens_gen} tokens {infer_time:.3f} s {tokens_per_sec:.1f} tokens/s")
        print(f"\n\r[Gemma] input={n_tokens_in} output={n_tokens_gen} tokens | {tokens_per_sec:.1f} tok/s | total={infer_time:.3f}s")


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

    os.environ["XDG_RUNTIME_DIR"] = "/var/run/user/0"
    os.environ["WESTON_DISABLE_GBM_MODIFIERS"] = "true"
    os.environ["WAYLAND_DISPLAY"] = "wayland-1"
    os.environ["QT_QPA_PLATFORM"] = "wayland"
    app = QApplication(sys.argv)
    QFontDatabase.addApplicationFont(HINDI_FONT_PATH)
    QFontDatabase.addApplicationFont(CHINESE_FONT_PATH)
    QFontDatabase.addApplicationFont(THAI_FONT_PATH)
    window = ChatWindow()

    window.update_response_text("Welcome to the Voice Translation implemented with Moonshine and Gemma3")
    
    # Start audio listener thread
    audio_thread = threading.Thread(target=start_audio_thread, args=(window, audio_device))
    audio_thread.start()

    sys.exit(app.exec_())


