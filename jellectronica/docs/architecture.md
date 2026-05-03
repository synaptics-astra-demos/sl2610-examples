# Architecture

## Overview

Jellectronica runs **entirely on the Coral Board**. No computation happens on the connected laptop — it's used only as an optional monitoring display.

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
│                    ┌─────────────────────────────────────────┐         │
│                    │                    │                    │         │
│         ┌──────────▼──────┐                    ┌───────▼──────┐        │
│         │   SoftSynth     │                    │  Display     │        │
│         │   → aplay/ALSA  │                    │  DSI / MJPEG │        │
│         │   → USB DAC     │                    │  + overlay   │        │
│         └─────────────────┘                    └──────────────┘        │
│                                                                        │
│   Ports (server mode only):                                            │
│     :5002 → Flask HTTP (MJPEG stream + monitoring page)                │
│     :5003 → WebSocket (real-time trigger events)                       │
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
| 3 | — | Chime (inharmonic partials) | AI generative accompaniment |
| 4 | — | Clash (noise + partials) | Collision chime effects |

Uses additive synthesis with ADSR envelopes, 24-voice polyphony, and feedback-delay reverb. Audio is streamed to ALSA via `aplay` subprocess pipe — zero system dependencies beyond Python + NumPy.

Includes a `SoundEvolver` that slowly modulates filter brightness, pan, reverb, and program changes for evolving timbre.

### 8×4 Musical Grid

The video frame is divided into an 8×4 grid. Each cell maps to a MIDI note:

```
Row 0 (Arp):   C5  D5  E5  G5  A5  C6  D6  E6
Row 1 (Pad):   A2  C3  D3  E3  G3  A3  C4  D4
Row 2 (Chord): G3  A3  C4  D4  E4  G4  A4  C5
Row 3 (Bass):  C3  D3  E3  G3  A3  C4  D4  E4
```

When a tracked jellyfish moves from one cell to another, the corresponding note triggers.

## Display Modes

### Kiosk Mode (`kiosk_dsi.py`)
- Renders to the Waveshare 5" DSI display via GStreamer `waylandsink`
- OpenCV draws detection overlay, converts to BGRx, pipes to GStreamer
- Fullscreen, standalone — no laptop needed

### Server Mode (`server.py`)
- Headless: annotates frames with OpenCV, JPEG-encodes, streams as MJPEG
- Flask serves a monitoring dashboard at `:5002`
- WebSocket at `:5003` pushes real-time trigger events
- The browser is a **passive monitor** — all computation is on-device
