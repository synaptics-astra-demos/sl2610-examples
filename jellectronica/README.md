# 🪼 Jellectronica Lite: Coralboard Native Edition

**Turn any video stream into ambient music using real-time AI object detection**, running entirely on the [Synaptics Coralboard SL2619](https://coral.ai/products/).

**Jellectronica *Lite*** watches a video source — by default, a [live jellyfish stream from Monterey Bay Aquarium](https://www.youtube.com/watch?v=7N9-FODmuBA) — detects objects using a YOLOv8 model on the **Torq NPU at 30 FPS**, and maps their positions to a musical grid. As creatures move through the frame, they trigger notes, chords, and arpeggios, turning motion into evolving ambient soundscapes.

An optional **MelodyRNN AI accompaniment** layer listens to the triggered notes and generates real-time generative melodies using a pre-trained Magenta LSTM neural network — all running on-device in pure Python/NumPy.

**Everything runs on-device** — NPU inference, AI melody generation, audio synthesis, and visual rendering. No cloud, no host processing.

### Make It Your Own

The detection model and video source are fully configurable. Swap in your own YOLOv8 model and any livestream to turn *anything* into music:

- 🐕 **Your dog at home** — Train a dog detection model, point it at a Google Nest camera. Sit the Coral on your desk — every time your dog wanders through the living room, music plays. Your dog is composing for you while you’re at work.
- 🐦 **Birds in your backyard** — Point a camera at a bird feeder with a bird classification model. Each species triggers a different timbre.
- 🚗 **Street traffic** — Mount a camera at a window. Cars become bass notes, pedestrians become chimes, cyclists become arpeggios.
- 🌊 **Waves at the beach** — Any object, any stream, any sound palette.

See [docs/model-conversion.md](docs/model-conversion.md) for how to convert your own YOLOv8 model for the NPU.

---

## Prerequisites

### Hardware

| Component | Required |
|-----------|----------|
| **Synaptics Astra SL2619** | Coralboard or Machina Kit |
| **USB Audio** | Any USB speaker or headset |
| **Display** | Waveshare 7" DSI LCD (optional) |
| **Network** | For installation and YouTube livestream (optional — local video fallback included) |
| **Power** | USB Type C |

### Board Firmware

- Astra SDK v2.0+ (Yocto scarthgap, Python 3.12)

### Host Machine Software

- Android Debug Bridge (ADB) from [Android SDK Platform Tools](https://developer.android.com/tools/releases/platform-tools) (recommended)

---


## Set Up

- Connect to the network

The Synaptics Coralboards support network sharing over USB. Connect to a host machine and enable network sharing. See [Coralboard Documentation](https://developers.google.com/coral/products/SL2610-dev-board) for details. 

Optionally, if you have the supported WiFi/BT module, you can use WiFi for the network and Bluetooth for the audio device. This module comes standard with the Machina Kit, but sold separately with the Coralboard.


- Attach USB audio device such as speaker or headset.


### Choose an installation mdethod

1. Indirect - clone on your host machine than copy the files to the Coralboard

2. Direct - clone directly onto the coralboard


### 🔧 Installation (Indirect Method)


- Clone the repositiory on your host machine.

```bash
git clone --recurse-submodules https://github.com/synaptics-astra-demos/sl2610-examples
cd sl2610-examples
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

- Copy over the files.

```bash
# Push the files to the target

adb push . /home/root/sl2610-examples

# Or use SCP (if network configured)
scp -r . root@<board-ip>:/home/root/sl2610-examples
```

- Connect to the target 

```bash
# Use ADB
adb shell

# or us SSH
ssh root@<board-ip>
```

- On the target, navigate to the repository directory:

```bash
cd /home/root/sl2610-examples
```

### 🔧 Installation (Direct Method)
 
- Connect to the target 

```bash
# Use ADB
adb shell

# or us SSH
ssh root@<board-ip>
```

- Clone the repository using the following command:

```bash
git clone --recurse-submodules https://github.com/synaptics-astra-demos/sl2610-examples
```

- Navigate to the repository directory:

```bash
cd sl2610-examples
```

If you already cloned without submodules, run `git submodule update --init --recursive`.

### Setup Python Environment

- To get started, set up your Python environment. This step ensures all required dependencies are installed and isolated within a virtual environment:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

- Install general dependencies for sl2610-examples

```bash
pip install -r requirements.txt
```

- Install specific dependencies for this example

[!WARNING] Please note that different examples require different versions of the Python Torq runtime. If using a shared virtual environment, always re-run installation of example-specific dependencies when switching between examples.


```bash
cd jellectronica
pip install -r requirements.txt
```

#### Optionally Pair a Bluetooth Device

- If you have the WiFi/BT module, follow the [bluetooth guide](https://synaptics-astra.github.io/doc/v/latest/linux/index.html#using-bluetooth) to pair a bluetooth headset or speaker. 



### 3. Choose Your Display Mode

- Headless - stream to browswer 
- Display - If you have a DSI display connected, use that.


### 4. Start the App

#### For headless mode

Start the app: 

 ```bash
 python3 server.py
 ```

Open `http://<board-ip>:5002` in any browser on the same sub-network. 

Audio plays on the board via USB speaker.

#### For display mode

Set the following environment variables for using the display. 

```bash
export XDG_RUNTIME_DIR=/var/run/user/0
export WAYLAND_DISPLAY=wayland-1
export WESTON_DISABLE_GBM_MODIFIERS=true
```

For portrait mode on 800x480 display:
```bash
export ORIENTATION=portrait
export DISPLAY_HEIGHT=800
export DISPLAY_WIDTH=480
```

Start the app: 

```bash
python3 app.py
```

It renders directly to the attached DSI display. 

Audio plays through the USB speaker.


---

## How It Works

```
YouTube Live ──→ yt-dlp ──→ cv2.VideoCapture
  (or local video)                │
                          ┌───────▼────────┐
                          │   Torq NPU     │
                          │  YOLOv8 int8   │
                          │  320×320       │
                          │  ~32ms/frame   │
                          └───────┬────────┘
                                  │ detections
                          ┌───────▼────────┐
                          │    Tracker     │
                          │  8×4 grid map  │
                          │  cell triggers │
                          └───────┬────────┘
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
        ┌───────▼──────┐  ┌──────▼───────┐  ┌─────▼────────┐
        │  SoftSynth   │  │  MelodyRNN   │  │   Display    │
        │  → aplay     │  │  LSTM AI     │  │  DSI / MJPEG │
        │  → USB DAC   │  │  (optional)  │  │  + overlay   │
        └──────────────┘  └──────────────┘  └──────────────┘

```

When a detected object crosses a cell boundary in the 8×4 grid, it triggers a note. Four rows map to different instruments — arpeggios, pads, chords, and bass. The more objects in the frame, the richer the sound. If MelodyRNN is enabled, triggered notes also seed the AI to generate evolving melodic accompaniment.


See [docs/architecture.md](docs/architecture.md) for full technical details.

---

## Video Sources

| Source | Config | Notes |
|--------|--------|-------|
| YouTube livestream (default) | `--youtube <URL>` | Requires WiFi + yt-dlp |
| Local video file | `--video video/jellyfish.mp4` | Bundled fallback, no network needed |
| Any HTTP stream | `--video http://...` | HLS, DASH, MJPEG |

When YouTube is unavailable (no network or yt-dlp not installed), the system automatically falls back to the bundled `video/jellyfish.mp4`. If the stream goes black (e.g. aquarium turns off lights), it also falls back automatically.

---

## Customization

### Changing the Video Source

You can point Jellectronica at any video source — a YouTube livestream, a local camera, an IP camera, or a video file.

**Command line**
```bash
# YouTube livestream
python3 server.py --video https://www.youtube.com/watch?v=YOUR_VIDEO_ID

# IP camera (e.g. Google Nest, RTSP, MJPEG)
python3 server.py --video rtsp://192.168.1.100:554/stream

# Local video file
python3 server.py --video /path/to/your/video.mp4
```

## Usage Examples

```bash
# Display mode (DSI display, fullscreen)
python3 app.py
python3 app.py --video ../samples/jellyfish.mp4

# Server mode (headless, MJPEG stream)
python3 server.py
python3 server.py --video ../samples/jellyfish.mp4 --port 5002

# Custom video source
python3 server.py --video https://www.youtube.com/watch?v=YOUR_VIDEO_ID
python3 server.py --video /path/to/your/video.mp4

# Custom detection model
python3 server.py --model model/your_model.vmfb

# Audio configuration
python3 server.py --audio-driver alsa --alsa-device hw:0,0

# Disable AI accompaniment
python3 server.py --no-ai
python3 app.py --no-ai

```


### Seamless Looping (H.264 Elementary Streams)

For a looping application, using standard MP4 files requires the entire application pipeline to briefly restart when the video ends. For perfectly seamless, infinite hardware looping with zero black frames, convert your MP4 to a raw H.264 elementary stream:

```bash
ffmpeg -i your_video.mp4 -vcodec copy -bsf h264_mp4toannexb your_video.h264
```

Then update `DEFAULT_LOCAL_VIDEO` in `app.py` to point to the `.h264` file. GStreamer will continuously loop the raw stream without ever restarting the compositor or dropping a frame.


### Swapping the Detection Model

The default model detects jellyfish, but you can swap in any YOLOv8 model to detect whatever you want.

1. **Train a YOLOv8 model** using [Ultralytics](https://docs.ultralytics.com/) on your chosen objects (dogs, birds, cars, etc.)

2. **Convert it for the NPU** — see [docs/model-conversion.md](docs/model-conversion.md) for the Torq compiler workflow.

3. **Deploy the model**:
   ```bash
   # Copy your converted model to the board
   adb push your_model.vmfb /home/root/sl2610-examples/jellectronica/models/

   # Run with the new model
   python3 server.py --model ../models/your_model.vmfb
   ```

The musical grid mapping works with any single-class detection model. Multi-class models will work too — all detected objects trigger notes regardless of class.

---


## Project Structure

```
jellectronica
├── server.py                 # Headless server (MJPEG + WebSocket)
├── app.py                    # Standalone DSI/HDMI display
├── detector.py               # YOLOv8 — Torq NPU (primary) + ONNX CPU (fallback)
├── tracker.py                # Multi-object tracker with grid mapping
├── music_engine.py           # Audio engine (5 channels + effects)
├── soft_synth.py             # Built-in synthesizer (pure Python/NumPy, zero dependencies)
├── melody_rnn.py             # MelodyRNN AI accompaniment (pure NumPy LSTM inference)
├── requirements.txt          # Python dependencies
├── ../models/moon_jellyfish
│   ├── moon320.vmfb          # Quantized YOLOv8 (int8, NPU)
│   ├── basic_rnn_weights.npz # MelodyRNN weights (Magenta basic_rnn, 12MB)
│   └── moon.json             # Model metadata
├── ../samples/
│   └── jellyfish.mp4            # Fallback jellyfish video (~208MB)
└── docs/
    ├── architecture.md       # Technical architecture deep-dive
    └── model-conversion.md   # How to retrain/convert the YOLOv8 model
 ``

---

## Dependencies

### Pre-installed on Board (Yocto)

- Python 3.12
- OpenCV (with FFMPEG)
- GStreamer 1.0 + plugins
- ALSA (audio output)
- Weston compositor

> **Note**: Audio synthesis and AI melody generation are built-in via SoftSynth and MelodyRNN (pure Python/NumPy). No TensorFlow, no Magenta pip package, no additional AI libraries needed.

---

## Content Licensing

Jellyfish video content included in this project is provided courtesy of the Monterey Bay Aquarium and is licensed for non-commercial use only. For more information on commercial-usage terms, please contact the Monterey Bay Aquarium, and we encourage you to visit and support their conservation and education efforts.

## Credits

- **Jellectronica Coral Board Native Edition**: [jellectronica-coral](https://github.com/raphdixon/jellectronica-coral) by Raphael Dixon
- **Jellyfish Detection Model**: [seaphony-ml](https://github.com/patrickdmiller/seaphony-ml) by Patrick Miller
- **Live Stream**: [Monterey Bay Aquarium](https://www.youtube.com/watch?v=7N9-FODmuBA) Moon Jelly Cam
- **MelodyRNN Weights**: [Magenta](https://magenta.tensorflow.org/) basic_rnn checkpoint by Google Brain


## License

Apache 2.0 — see [LICENSE](LICENSE).
