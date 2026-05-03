# 🪼 Jellectronica: Coralboard Native Edition

**Turn any video stream into ambient music using real-time AI object detection**, running entirely on the [Synaptics Astra SL2619 Coral Dev Board](https://coral.ai/products/).

Jellectronica watches a video source — by default, a [live jellyfish stream from Monterey Bay Aquarium](https://www.youtube.com/watch?v=7N9-FODmuBA) — detects objects using a YOLOv8 model on the **Torq NPU at 31 FPS**, and maps their positions to a musical grid. As creatures move through the frame, they trigger notes, chords, and arpeggios, turning motion into evolving ambient soundscapes.

**Everything runs on-device** — NPU inference, audio synthesis, and visual rendering. No cloud, no host processing.

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
| **Coral Devboard** | Synaptics Astra SL2619 |
| **USB Audio** | Any USB speaker or headset |
| **Display** | Waveshare 7" DSI LCD (optional) |
| **Network** | For YouTube livestream (optional — local video fallback included) |
| **Power** | USB Type C |

### Board Firmware

- Astra SDK v2.0+ (Yocto scarthgap, Python 3.12)
- NPU must be enabled in the device tree

---


## Set Up

Connect the network. The Synaptics Coralboards support network sharing over USB. Connect to a computer and enable network sharing. 

Attach USB audio device such as speaker or headset.

Optionally, if you have the supported WiFi/BT module, you can use WiFi for the network and bluetooth for the audio device. 

See Coralboard documentation for details. 


### Choose an installation mdethod

1. Indirect - clone on your computer than copy the files to the Coralboard

2. Direct - clone directly onto the coralboard

[Warning!] Due to the large files, Git LFS must be installed before cloning this repository. 


### 🔧 Installation (Indirect Method)

Ensure you have git LFS installed. 

Clone the repositiory on your computer and copy over the files.

```bash
git clone https://github.com/synaptics-astra-demos/sl2610-examples 
cd sl2610-examples
```

```bash
# Push the files to the target

adb push . /home/root/sl2610-examples

# Or use SCP (if network configured)
scp -r . root@<board-ip>:/home/root/sl2610-examples
```

```bash
# Use ADB
adb shell
```

On the target, navigate to the repository directory:

```bash
cd /home/root/sl2610-examples
```

### 🔧 Installation (Direct Method)
 
Clone the repository using the following command:

```bash
git clone https://github.com/synaptics-astra-demos/sl2610-examples.git
```
Navigate to the repository directory:

```bash
cd sl2610-examples
```

### Setup Python Environment

To get started, set up your Python environment. This step ensures all required dependencies are installed and isolated within a virtual environment:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

Install dependencies

If online
```bash
pip install -r requirements.txt
```


#### Optionally Pair a Bluetooth Device

If you have the WiFi/BT card installed, follow this guide to pair a bluetooth headset or speaker. 
https://synaptics-astra.github.io/doc/v/latest/linux/index.html#using-bluetooth


### 3. Choose Your Display Mode

The setup script asks which mode to enable:

| Mode | What It Does | Launch |
|------|-------------|--------|
| **Kiosk** | Fullscreen on DSI/HDMI display, standalone | `python3 kiosk_dsi.py` |
| **Server** | Headless, stream to browser for monitoring | `python3 server.py` |

#### Kiosk Mode (Standalone Installation)

Renders directly to the attached DSI. Audio plays through the USB DAC. No laptop needed.

#### Server Mode (Remote Monitoring)

Runs headless on the board. Open `http://<board-ip>:5002` in any browser to see the live detection stream. Audio still plays on the board via USB DAC.

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
                ┌─────────────────┼
                │                 │                 
        ┌───────▼──────┐  ┌──────▼───────┐
        │  SoftSynth   │  │   Display    │
        │  → aplay     │  │  DSI/MJPEG   │
        │  → USB DAC   │  │  + overlay   │
        └──────────────┘  └──────────────┘
```

When a detected object crosses a cell boundary in the 8×4 grid, it triggers a note. Four rows map to different instruments — arpeggios, pads, chords, and bass. The more objects in the frame, the richer the sound.

See [docs/architecture.md](docs/architecture.md) for full technical details.

---

## Video Sources

| Source | Config | Notes |
|--------|--------|-------|
| YouTube livestream (default) | `--youtube <URL>` | Requires WiFi + yt-dlp |
| Local video file | `--video video/moon15.mp4` | Bundled fallback, no network needed |
| Any HTTP stream | `--video http://...` | HLS, DASH, MJPEG |

When YouTube is unavailable (no network or yt-dlp not installed), the system automatically falls back to the bundled `video/moon15.mp4`. If the stream goes black (e.g. aquarium turns off lights), it also falls back automatically.

---

## Customization

### Setting Up WiFi

Newer Coral boards ship with WiFi built in. To connect:

```bash
# SSH or serial into the board, then:
wpa_passphrase "YourNetworkName" "YourPassword" > /etc/wpa_supplicant.conf
wpa_supplicant -i wlan0 -c /etc/wpa_supplicant.conf -B
udhcpc -i wlan0

# Verify connectivity
ping -c 3 google.com
```

To make it persist across reboots, ensure `wpa_supplicant` is enabled as a systemd service or configured in your Yocto network setup.

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


### Swapping the Detection Model

The default model detects jellyfish, but you can swap in any YOLOv8 model to detect whatever you want.

1. **Train a YOLOv8 model** using [Ultralytics](https://docs.ultralytics.com/) on your chosen objects (dogs, birds, cars, etc.)

2. **Convert it for the NPU** — see [docs/model-conversion.md](docs/model-conversion.md) for the full SyNAP Toolkit workflow

3. **Deploy the model**:
   ```bash
   # Copy your converted model to the board
   adb push your_model.vmfb /home/root/jellyphony-native/model/

   # Run with the new model
   python3 server.py --model model/your_model.vmfb
   ```

The musical grid mapping works with any single-class detection model. Multi-class models will work too — all detected objects trigger notes regardless of class.

---

## Manual Usage

```bash
# Kiosk mode (DSI display, fullscreen)
python3 kiosk_dsi.py
python3 kiosk_dsi.py --video video/moon15.mp4

# Server mode (headless, MJPEG stream)
python3 server.py
python3 server.py --video video/moon15.mp4 --port 5002

# Custom video source
python3 server.py --video https://www.youtube.com/watch?v=YOUR_VIDEO_ID
python3 server.py --video /path/to/your/video.mp4

# Custom detection model
python3 server.py --model model/your_model.vmfb

# Audio configuration
python3 server.py --audio-driver alsa --alsa-device hw:0,0

```

---

## Project Structure

```
jellectronica
├── server.py                 # Headless server (MJPEG + WebSocket)
├── kiosk_dsi.py              # Standalone DSI/HDMI kiosk display
├── detector.py               # YOLOv8 — Torq NPU (primary) + ONNX CPU (fallback)
├── tracker.py                # Multi-object tracker with grid mapping
├── music_engine.py           # Audio engine (4 instruments + effects)
├── soft_synth.py             # Built-in synthesizer (pure Python/NumPy, zero dependencies)
├── requirements.txt          # Python dependencies
├── models/
│   ├── moon320.vmfb          # Quantized YOLOv8 (int8, NPU)
│   └── moon.json             # Model metadata
├── video/
│   └── moon15.mp4            # Fallback jellyfish video (~208MB)
└── docs/
    ├── architecture.md       # Technical architecture deep-dive
    ├── serial-console.md     # Serial console wiring & connection
    ├── model-conversion.md   # How to retrain/convert the YOLOv8 model
    ├── dsi-display.md        # Waveshare 5" DSI LCD setup
    └── recovery.md           # Board recovery (bricked boot)
```

---

## Dependencies

### Pre-installed on Board (Yocto)

- Python 3.12
- OpenCV (with FFMPEG)
- GStreamer 1.0 + plugins
- ALSA (audio output)
- Weston compositor


> **Note**: Audio synthesis is built-in via SoftSynth (pure Python/NumPy → `aplay` → ALSA). No additional audio libraries are needed.

---

## Credits

- **Jellectronica Coral Board Native Edition**: [jellectronica-coral](https://github.com/raphdixon/jellectronica-coral) by Raphael Dixon
- **Jellyfish Detection Model**: [seaphony-ml](https://github.com/patrickdmiller/seaphony-ml) by Patrick Miller
- **Live Stream**: [Monterey Bay Aquarium](https://www.youtube.com/watch?v=7N9-FODmuBA) Moon Jelly Cam
- **Hardware**: [Synaptics Astra SL2619](https://coral.ai/products/) Coral Dev Board

## License

Apache 2.0 — see [LICENSE](LICENSE).
