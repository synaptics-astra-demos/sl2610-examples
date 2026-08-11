# Moonshine Streaming Demo

This guide describes how to convert audible speech to text (streaming word by word) on Astra SL2610-series processors. 

This example uses the following models: 

- [Moonshine V2 Streaming Model](https://github.com/moonshine-ai/moonshine), a modern speech-to-text (automatic speech recognition) model designed specifically for efficient, real-time, and low-latency operation. 
- [Silero VAD (Voice Activity Detection)](https://github.com/snakers4/silero-vad) (Optional), a lightweight, high-performance model designed to detect the presence of human speech in audio streams.

This example performs real-time audio transcription with Moonshine-tiny English using a 2-split
Torq VMFB model (fused encoder + KV decoder). It uses a VAD (self-calibrating energy
detector by default, or optionally Silero's neural VAD) splits utterances and
a committed-prefix incremental decoder gives a live preview.

## Hardware Setup

This example is compatible with the following hardware:
- Astra Machina SL2610 Dev Kit
- Synaptics Coralboard

Machina Dev Kit
- For setup instructions, see the [Setting up the hardware guide](https://synaptics-astra.github.io/doc/v/latest/quickstart/hw_setup.html)

Coralboard
- For setup instructions, see the [Synaptics Coralboard Site](https://developers.google.com/coral/products/SL2610-dev-board)

## Prerequisites
Ensure your board has the following installed:

**Astra SDK "OOBE" Image** (Default):
- [SL2619 OOBE Image](https://github.com/synaptics-astra/sdk/releases)
- The image includes important software components such as `git` and `python3`

## Example Setup

Attach a USB microphone to the board. 

Connect from a PC using ADB or SSH.

Optionally connect a display and USB keyboard/mouse and open a terminal directly. 

## 🔧 Installation
 
### Setup the base environment

Clone the repository including submodules, run setup scripts, and install base Python dependencies according to the [Top Level Readme Installation Section](../README.md#installation)

### Install example-specific dependencies

```sh
cd speech_to_text_stream
pip install -r requirements.txt
```

### Install the PortAudio system libraries for microphone input:

```bash
../setup/install_portaudio.sh
```


### Download Models

```sh
python setup_demo.py
```

This downloads the default model files from our HuggingFace repo to: `models/Synaptics/moonshine-streaming-tiny-torq/`

## Running

Run the demo from the `speech_to_text_stream` directory:

```sh
python app.py
```

**Then start speaking!** 

Speak phrases in English. The app will capture your speech and show the text on the terminal word-by-word as they stream out of the model with retroactive updates. 

Press `Ctrl+C` to exit.


## Detailed Usage

You can optionally pass in the location to the model file: 
```sh
cd speech_to_text_stream
python app.py -m ../models/Synaptics/moonshine-streaming-tiny-torq
```

If you don't pass `--device`, the app lists the available audio input devices
and prompts you to pick one before it starts listening — device indices
aren't stable across boards/OS images, so there's no universal default.
Press Enter to accept the system default device, or pass `--device <N>` (or
a device name) up front to skip the prompt, e.g. for scripted runs:

```sh
python app.py -m ../models/Synaptics/moonshine-streaming-tiny-torq --device 1
```

The other defaults are tuned for the board (`--vad-silence 2.5`,
`--vad-threshold 0.010`, `--preview-every 5`), so the first command above is
equivalent to:

```sh
python app.py -m ../models/Synaptics/moonshine-streaming-tiny-torq \
    --vad-silence 2.5 --vad-threshold 0.010 --preview-every 5
```

> [!TIP]
> **Tune the VAD parameters for your own setup — the defaults are a starting
> point, not a good fit for every room, mic or speaker.** Transcription quality
> depends far more on where utterances get split than on anything else in the
> pipeline.
>
> `--vad-silence` matters most. It sets how long a pause must last before the
> current utterance is closed and flushed. Too low and the VAD cuts mid-sentence
> on natural pauses, so the decoder loses the context that makes the rest of the
> sentence accurate; too high and utterances run together and the transcript
> lags. If your output is fragmented, raise it; if it feels sluggish or
> sentences merge, lower it.
>
> Also worth adjusting:
> - `--vad-threshold` — raise it in a noisy room so background noise does not
>   trigger speech, lower it for a quiet mic or a soft speaker.
> - `--vad-lookback` — raise it if word onsets are clipped at the start of
>   utterances.
> - `--vad-backend silero` — the neural VAD is more robust than the energy
>   detector when the noise floor is uneven.

To try the Silero neural VAD instead of the default energy detector (see the
VAD step under [Step by step](#step-by-step) below for the tradeoff), install
`onnxruntime` and pass `--vad-backend silero`:

```sh
pip install onnxruntime
python app.py -m ../models/Synaptics/moonshine-streaming-tiny-torq --vad-backend silero
```

The Silero onnx model (~2 MB) is auto-downloaded on first use.

List audio input devices and pick a different one:

```sh
python app.py --list-devices
python app.py -m ../models/Synaptics/moonshine-streaming-tiny-torq --device 3
```

Run `python app.py -h` to see all available options (VAD thresholds, decode
mode, runtime flags, profiling).

## Transcribing a WAV file

For testing against pre-recorded audio instead of a live mic (e.g. a real
conversation or speech sample), pass `--wav`:

```sh
python app.py -m ../models/Synaptics/moonshine-streaming-tiny-torq --wav /path/to/sample.wav
```

This reads the file with `soundfile` (mixed to mono if stereo, resampled like any
other input source), feeds it through the same VAD/encode/decode pipeline as the
mic path, and exits automatically once the file is fully transcribed — no `Ctrl+C`
needed. A silent 1 s lead-in is synthesized ahead of the file so the VAD has
something to calibrate against, since a recording — unlike a live mic — often
starts talking immediately.

By default the file is fed as fast as possible (for quick batch testing). Pass
`--realtime` to pace the feed to match the file's real playback speed instead, so
you can watch the live preview update the way it would from a mic:

```sh
python app.py -m ../models/Synaptics/moonshine-streaming-tiny-torq --wav /path/to/sample.wav --realtime
```

`--vad-silence` (default `2.5`s) still governs where the file is split into
utterances — lower it for tightly-paced conversational audio. `--wav` mode doesn't
need `sounddevice`/PortAudio at all, so it also works on machines without a mic or
audio drivers set up.

## How it works

This is a **real-time microphone transcriber**: it consumes a live audio stream
chunk-by-chunk, detects when you are speaking, and updates the transcript *as you
talk* — freezing words once it is confident about them. The two ideas that make it
real-time are (a) an encoder that turns streaming audio into an incrementally-grown
cross-attention memory, and (b) a decoder that resumes from a frozen token prefix
with its KV tensors pinned on the NPU, so each preview is cheap.

### Code layout

| File | Role |
|------|------|
| `runner.py` | the **engine**: the model, its pre-allocated state, and the thin VMFB session wrapper. No audio, no UI — "give me audio chunks, I return tokens". |
| `app.py` | the **app**: microphone capture, VAD, the worker thread, decode triggering, terminal rendering, the CLI, and the profiler. |
| `setup_demo.py` | downloads/verifies the model files (reuses `utils/`). |

### The model (2-split) and its files

A 2-split Moonshine-tiny: a **fused `encoder`** (audio → cross-attention memory) and
a **`decoder`** (memory → tokens, autoregressively). The model directory holds 7
flat files:

| File | Role |
|------|------|
| `encoder.vmfb`, `decoder.vmfb` | the compiled models that run on the NPU |
| `streaming_config.json` | the streaming knobs (below) |
| `config.json` | model config |
| `adapter_pos_emb.npy` | position-embedding table, looked up host-side per chunk |
| `decoder_token_embeddings.npy` | token-embedding table (the decoder is fed embeddings, not token IDs) |
| `tokenizer.json` | token IDs → text |

A VMFB exposes its inputs *positionally* — it has no argument names. The dict-based
feed interface needs names, so they are **hardcoded** as `ENCODER_INPUT_ORDER` /
`DECODER_INPUT_ORDER` in `runner.py` (input shapes + dtypes come from the runner's
`inputs_info`). This is why the demo needs neither `onnx` nor any sidecar files at
runtime; `_Session` validates the hardcoded arity against `inputs_info` at load and
errors loudly if the model is re-exported with a different number of inputs.

The streaming knobs (`streaming_config.json`) drive everything downstream:

```
chunk_len        = 1280 samples  → 80 ms of audio per chunk @ 16 kHz
feature_stride   = 4 frames/chunk → each active chunk adds 4 cross-KV frames
                                     (80 ms / 4 = 20 ms per frame)
total_lookahead  = 16 frames
warmup_chunks    = 4
max_memory_len   = 400 frames    → 400 × 20 ms = 8 s cross-KV buffer
max_tokens       = 48
layers=6  heads=8  head_dim=40  hidden=320  BOS=1  EOS=2
```

A useful invariant: **`cross_kv_fill` frames × 20 ms = seconds of audio captured**.

### Data flow

```
mic ──callback──> audio_queue ──> worker thread ──> resample to 16 kHz
                                        │
                                        ▼  (slice into fixed 1280-sample chunks)
                                   per-chunk loop:
                                        │
                          ┌─────────────┴──────────────┐
                          ▼                             ▼
                   EnergyVAD/SileroVAD          if speaking:
                   (speech? silence?)       process_audio_chunk()  ← ENCODER
                          │                        │ grows cross-KV
                          │                        ▼
                          │                  every N chunks OR on speech-end:
                          │                  decode_incremental()  ← DECODER
                          │                        │ produces tokens
                          ▼                        ▼
                   TerminalListener  <──────  tokenizer.decode()
```

Two threads: the **audio callback** only does `audio_queue.put(chunk)` (it must
return fast or audio drops); the **worker thread** does resample → VAD → encode →
decode → render. This decoupling keeps capture glitch-free while inference runs.

### Step by step

**1. Capture.** `sounddevice` delivers `blocksize=4096` samples at the device's
native rate. The callback just queues a copy.

**2. Normalize to fixed chunks.** The worker resamples each block to 16 kHz
(linear interpolation), accumulates into a buffer, and slices off **exactly 1280
samples (80 ms)** at a time. So the whole pipeline sees uniform chunks regardless of
the mic's native rate/blocksize; leftover samples carry to the next block.

**3. Pre-allocated state** (`MoonshineStaticStreamingState`, allocated once, reused
across utterances):

| Buffer | Shape | Meaning |
|--------|-------|---------|
| `conv1_buffer` / `conv2_buffer` | `(1,320,4)` / `(1,640,4)` | rolling conv left-context |
| `features_buffer` | `(1,16,320)` | rolling feature/lookahead window |
| `enc_bufs` | list of `(1,16,320)` | encoder internal layer state |
| `k_cross` / `v_cross` | `(6,1,8,400,40)` | **the cross-attention memory** |
| `k_self` / `v_self` | `(6,1,8,48,40)` | decoder self-attention KV |
| `pos_offset` | `[0]` | running index into the position-embedding table |

Nothing is reallocated during streaming. `reset()` (on `speech_start`) clears
`cross_kv_fill`, `chunk_idx`, `committed_tokens`, `recent_hyps`.

**4. VAD** — two interchangeable backends, selected with `--vad-backend`, sharing
the same speech/silence endpointing state machine (`_HangoverVAD`): per chunk
they emit `speech_start` / `speech` / `speech_end` (after `--vad-silence` s of
quiet) / `silence`.

- `energy` (default, `EnergyVAD`): a self-calibrating RMS energy detector. It
  samples the room for the first ~12 chunks (~1 s) and sets
  `threshold = max(mean + 4·std, --vad-threshold)`. (The `[VAD Calibration]`
  line only prints with `--profile`.) Zero extra dependencies, but it can't
  tell speech from any other sound of similar loudness.
- `silero` (`SileroVAD`): runs [Silero's](https://github.com/snakers4/silero-vad)
  small neural VAD (ONNX, ~2 MB) instead of raw RMS — actually models
  speech's spectral/temporal structure, so it's far more robust to
  non-stationary background noise (coughs, TV, keyboard, HVAC). Each 1280-sample
  pipeline chunk is split into 512-sample windows fed through the model in
  sequence (max probability across windows = the chunk's score), compared
  against `--vad-threshold` (default `0.5`, a probability). Requires
  `onnxruntime` (`pip install onnxruntime`); the onnx model itself is
  auto-downloaded on first use into `models/silero_vad/silero_vad.onnx`, or
  point `--vad-model` at a local copy.

**5. Encoder step** (`process_audio_chunk`, runs on every speech chunk). Builds a
feed dict keyed by the encoder's input names (audio, the three rolling buffers, a
position-embedding slice looked up from `pos_offset`, and `buf_*`), runs the VMFB,
and unpacks outputs as
`[k_cross_new, v_cross_new, conv1_out, conv2_out, features_out, *enc_buf_outs]`. The
rolling buffers are always updated. Then:

- **Warmup** (`chunk_idx < 4`): discard the new cross-KV and encoder-buffer updates —
  not enough context yet.
- **Active** (`chunk_idx ≥ 4`): append the 4 new cross-KV frames at `cross_kv_fill`,
  advance `cross_kv_fill += 4` and `pos_offset += 4`.

So each active 80 ms chunk grows the memory by 4 frames (80 ms). `encode(is_final=
False)` is a no-op — all the work lives in `process_audio_chunk`.

**6. Finalize / flush** (`encode(is_final=True)`): because the frontend has 16
frames of lookahead, the last audio hasn't fully propagated when you stop. Finalize
feeds **4 zero chunks** through the same path so the tail of your sentence is pushed
out of the lookahead pipeline (otherwise the last word or two is lost).

**7. When the decoder runs.** Three triggers in the worker:
1. **Live preview** — `chunks_since_decode ≥ --preview-every` (≈ every 5 × 80 ms =
   400 ms while speaking).
2. **Buffer full** — `cross_kv_fill ≥ 400` (8 s): force-finalize, start a new utterance.
3. **Speech end** — VAD said so: flush + final decode + commit the line.

**8. Incremental decode** (`decode_incremental`). First it computes
`max_tokens = min(ceil(seconds × 6.5), 48)` (speech is ~4–6.5 tok/s) and a
`cross_attn_bias` that masks the empty tail of the 400-frame memory (`-1e9` beyond
`cross_kv_fill`) so the decoder only attends to real audio. Then it **resumes from
the committed prefix** instead of from BOS:

```
C = len(committed_tokens)
start at position C with current_token = committed[-1]   (or BOS at position 0 if C==0)
loop: feed embedding(current_token) + bias + position + self/cross-KV
      → logits → argmax → next_token; stop on EOS(2) or max_tokens
```

A preview that already committed 12 tokens and finds 4 new ones runs ~5 decoder
passes, not 16 — that is the O(tail) speedup vs. O(T²) re-decode-from-BOS.

**9. KV residency** (the per-token cost saver). The decoder session is created with
`device_outputs=True`, and:
- **P1** (`_upload_cross_kv`): cross-KV is constant for a whole decode call → uploaded
  once as device buffers and reused every token (instead of re-uploading 6×8×400×40
  values per token).
- **P2** (`_ensure_self_kv_device`): self-KV stays resident — each step's self-KV
  *output* handle becomes the next step's *input*, no host round-trip. Only the
  (tiny) logits are copied back, for the argmax. `_Session.run_raw` returns the raw
  `DeviceArray`s that make this possible.

**10. Commit decision** (what gets frozen on screen). After decoding `result_tokens`
(length `T`), a token is committed only when **both** gates pass:

```
la_len  = longest prefix shared by the last `--commit-agreement` (2) hypotheses
          → "the model has stopped changing its mind"
delay_frames = --commit-delay (3 s) / 0.020 = 150 frames
age_len = int(T × max(0, (cross_kv_fill − 150) / cross_kv_fill))
          → tokens at least 3 s of audio behind the live frontier
commit_len = max(C, min(la_len, age_len))     # monotonic — never un-commit
```

*Worked example:* `cross_kv_fill = 250` (5 s), `T = 20`, previously `C = 5`. Then
`age_len = int(20 × (250−150)/250) = 8`; if the last two hypotheses agree on 12
tokens, `commit_len = max(5, min(12, 8)) = 8` → freeze 8 tokens. **LocalAgreement**
stops you committing a word still being revised; **commit-delay** stops you
committing a word so recent that more audio could still change it; `max(C, …)`
guarantees on-screen committed text never rewrites itself. `--full-decode` bypasses
all of this (re-decode from BOS every time: correct, but O(T²) and it flickers).

**11. Render & lifecycle.** `TerminalListener` overwrites the current line(s) in
place (ANSI) with the live transcript plus volume/buffer bars. On `speech_end` the
final line is locked with `complete_line()` and the terminal sits on it; the
utterance counter only ticks and `state.reset()` only clears the memory +
committed prefix lazily, on the *next* `speech_start` (see step 3) — there's no
"Listening…" in between. Buffer-full is the one case that resets immediately: it
must start a fresh utterance while you're still mid-sentence, so it locks the
line, resets state, ticks the counter, and redraws "Listening…" all at once
before continuing.

## Profiling

`--profile` records per-chunk worker timing, prints a real-time keep-up summary on
exit, and dumps the raw arrays as `.npy` files to `profile_results/`; override with `--profile-out`:

```sh
python app.py -m ../models/Synaptics/moonshine-streaming-tiny-torq --wav sample.wav --profile
```

Then turn those dumps into plots with `plot_profile.py`:

```sh
python plot_profile.py
```

This reads `profile_results/*.npy` and writes PNGs to `profile_results/plots/`
(override with `--profile-dir` / `--out-dir`):

| Plot | Shows |
|------|-------|
| `chunk_times.png` | per-chunk worker time, cheap vs. with-decode, against the real-time budget line |
| `encode_times.png` | encoder latency distribution (p50/p95/p99) |
| `decode_times.png` | decode-call latency distribution (p50/p95/p99) |
| `decode_steps.png` | decoder forward passes per decode call |
| `queue_depth.png` | audio queue depth over time, with a linear trend line (sustained growth ⇒ falling behind) |
| `realtime_factor.png` | running work/audio ratio over the session (>1.0x ⇒ cannot keep up) |

Requires `matplotlib`, which is not in `requirements.txt` since it is only
needed for plotting: `pip install matplotlib`.

## Notes

- `--full-decode` restores the baseline re-decode-from-BOS behaviour (instead of
  the default committed-prefix incremental decode).
- `--profile` records per-chunk worker timing and prints a real-time keep-up
  summary on exit (and shows the VAD calibration line); see
  [Profiling](#profiling) to turn the dumps into plots.
- `--runtime-flags` forwards flags straight to the Torq runtime and must come
  last, since every remaining argument is passed through (e.g. `--runtime-flags
  --torq_hw_type=sim` for software simulation).
- `--preview-every`, `--commit-agreement`, `--commit-delay` tune the live-preview
  cadence and how eagerly tokens are frozen; `--vad-threshold` / `--vad-silence`
  tune speech detection and utterance splitting.
- `--wav <file>` transcribes a pre-recorded file instead of the mic (see
  [Transcribing a WAV file](#transcribing-a-wav-file)); `--realtime` paces the feed
  to match playback speed.

# Citations

Useful Sensors. “Moonshine: On-Device Speech Recognition.” 2024.
https://github.com/usefulsensors/moonshine

Useful Sensors. “Ergodic streaming encoder asr for latency-critical speech applications.” 2026. 
https://github.com/usefulsensors/moonshine


Silero Team. “Silero VAD: Voice Activity Detection.” 2021.
https://github.com/snakers4/silero-vad
