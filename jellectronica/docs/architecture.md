# Architecture

## Overview

Jellectronica runs **entirely on the Coral Dev Board**. No computation happens on the connected laptop — it's used only as an optional monitoring display.

```
┌────────────────────────────────────────────────────────────────────────┐
│                    CORAL DEV BOARD (SL2619)                            │
│                                                                        │
│   Video Source                                                         │
│   (YouTube live / local .mp4)                                          │
│         │                                                              │
│         ▼                                                              │
│   ┌─────────────────────┐    ┌──────────────────────┐                  │
│   │   cv2.VideoCapture  │───▶│  Torq NPU Inference  │                  │
│   │   (frame decode)    │    │  YOLOv8 320×320 int8 │                  │
│   └─────────────────────┘    │  ~32ms / 31 FPS      │                  │
│                              └──────────┬───────────┘                  │
│                                         │                              │
│                              ┌──────────▼───────────┐                  │
│                              │   Tracker + Grid     │                  │
│                              │   8×4 musical grid   │                  │
│                              │   cell transitions   │                  │
│                              └──────────┬───────────┘                  │
│                                         │                              │
│           ┌─────────────────────────────┼────────────────┐             │
│           │                             │                │             │
│   ┌───────▼───────┐          ┌──────────▼──────┐   ┌────▼───────┐      │
│   │   SoftSynth   │          │   MelodyRNN     │   │  Display   │      │
│   │   5 channels  │◀─────────│   LSTM AI       │   │  DSI/MJPEG │      │
│   │   → aplay     │  notes   │   (optional)    │   │  + overlay │      │
│   │   → USB DAC   │          │   NumPy only    │   └────────────┘      │
│   └───────────────┘          └─────────────────┘                       │
│                                                                        │
│   Ports (server mode only):                                            │
│     :5002 → Flask HTTP (MJPEG stream + monitoring page)                │
│     :5003 → WebSocket (real-time trigger events + AI notes)            │
└────────────────────────────────────────────────────────────────────────┘
```

## Components

### Detection (`detector.py`)

YOLOv8 jellyfish detection:

| Backend | Model File | Performance | Notes |
|---------|-----------|-------------|-------|
| **Torq NPU** (primary) | `moon320.vmfb` | ~32ms / 31 FPS | INT8 quantized, IREE FlatBuffer |

The detector handles letterbox padding, int8 quantization, NMS postprocessing, and outputs normalized bounding boxes.

### Tracking (`tracker.py`)

Persistent multi-object tracker using exponential moving average (EMA) smoothing:
- Matches detections to existing tracks by distance
- Generates **cell transition triggers** when a jellyfish crosses a grid boundary
- Grace period for lost tracks (15 frames before removal)

### Music Engine (`music_engine.py` + `soft_synth.py`)

SoftSynth — a pure Python/NumPy software synthesizer with 5 channels:

| Channel | Row | Timbre | Role |
|---------|-----|--------|------|
| 0 | 1-2 | Warm Pad (sine + detune) | Ambient chords |
| 1 | 0 | Bell (triangle + harmonics) | Arpeggiated patterns |
| 2 | 3 | Sub Bass (fundamental sine) | Deep bass notes |
| 3 | — | AI Sine (pure sine, 0.5s attack) | MelodyRNN accompaniment (optional) |
| 4 | — | Clash (noise + partials) | Collision chime effects |

Uses additive synthesis with ADSR envelopes, 24-voice polyphony, and feedback-delay reverb. Audio is streamed to ALSA via `aplay` subprocess pipe — zero system dependencies beyond Python + NumPy.

Includes a `SoundEvolver` that slowly modulates filter brightness, pan, reverb, and program changes for evolving timbre. The AI channel receives independent pan drift and extra reverb for an ethereal, spatial quality.

### AI Accompaniment (`melody_rnn.py`) — Optional

Jellectronica includes an optional **MelodyRNN** AI melody generator — a real pre-trained [Magenta](https://magenta.tensorflow.org/) basic_rnn neural network that generates evolving melodies in real time.

1. Jellyfish-triggered MIDI notes are fed into the MelodyRNN as seed input
2. A 2-layer LSTM (512-unit hidden state) generates melodic continuations
3. Output is snapped to the pentatonic scale for guaranteed consonance
4. Notes play through a soft, ethereal sine-wave synth patch (Channel 3)
5. Temperature modulation: more jellyfish = more exploratory melodies

The model weights (`basic_rnn_weights.npz`, ~12MB) are pre-extracted from a TensorFlow.js checkpoint. Inference runs in **pure NumPy** — no TensorFlow, no GPU, no additional dependencies. It runs entirely on the Cortex-A55 CPU alongside everything else.

Real Magenta MelodyRNN running **pure NumPy LSTM inference** — zero TensorFlow dependency.

**Architecture** (from TF.js basic_rnn checkpoint):
- Input: 38-dim one-hot (36 MIDI pitches [48-83] + NOTE_OFF + NO_EVENT)
- LSTM Layer 0: input_dim=550 → hidden_size=512
- LSTM Layer 1: input_dim=1024 → hidden_size=512
- Output: fully_connected [512, 38] → softmax → temperature sampling

**How it integrates**:
1. Every `trigger_cell()` call feeds the triggered MIDI note to MelodyRNN via `feed_note()`
2. Jellyfish count updates temperature via `feed_activity()` — more jellyfish = more exploratory
3. A generator thread runs LSTM inference, pushing timed notes into a queue
4. A player thread pulls notes from the queue and calls `_play_ai_note()` on the music engine
5. Notes are snapped to the pentatonic scale and played on Channel 3 (AI Sine) with humanized timing

**Performance**: LSTM inference for 48 steps takes ~50ms on the Cortex-A55. The two-thread pipeline (generator → queue → player) ensures melody playback never blocks the main rendering loop.

**Weights**: `model/basic_rnn_weights.npz` (~12MB) — pre-extracted from a Magenta TF.js checkpoint. Contains LSTM kernels, biases, and the fully-connected output layer.

When disabled:
- The `--no-ai` flag cleanly skips MelodyRNN loading
- All other audio (pads, arpeggios, bass, clash chimes) continue normally
- The DSI overlay shows "AI ACCOMPANIMENT [OFF]"
- Zero performance impact — no weights are loaded

When enabled:
- MelodyRNN loads in ~1 second (12MB weights)
- Two background threads handle generation and playback
- CPU usage is minimal (~5% for continuous generation)
- DSI overlay shows "AI ACCOMPANIMENT" with a pulsing indicator and live note names
- Web monitor shows real-time AI note events

---

### 8×4 Musical Grid

The video frame is divided into an 8×4 grid. Each cell maps to a MIDI note:

```
Row 0 (Arp):   C5  D5  E5  G5  A5  C6  D6  E6
Row 1 (Pad):   A2  C3  D3  E3  G3  A3  C4  D4
Row 2 (Chord): G3  A3  C4  D4  E4  G4  A4  C5
Row 3 (Bass):  C3  D3  E3  G3  A3  C4  D4  E4
```

When a tracked jellyfish moves from one cell to another, the corresponding note triggers. This note is also fed to MelodyRNN (if enabled) to seed the generative melody.

## Display Modes

### Display Mode (`app.py`)
- Renders to the Waveshare 7" DSI display via GStreamer `waylandsink`
- OpenCV draws detection overlay, converts to BGRx, pipes to GStreamer
- AI Accompaniment visualization bar at the bottom of the display
- Fullscreen, standalone — no laptop needed

### Server Mode (`server.py`)
- Headless: annotates frames with OpenCV, JPEG-encodes, streams as MJPEG
- Flask serves a monitoring dashboard at `:5002`
- WebSocket at `:5003` pushes real-time trigger events and AI note events
- The browser is a **passive monitor** — all computation is on-device
