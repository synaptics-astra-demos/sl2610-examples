#!/usr/bin/env python3
"""
Jellectronica — Display App with Detection Overlay (Standalone)
=================================================================

Full standalone entry point for the Coralboard. Renders jellyfish
video with detection overlay on DSI display via GStreamer waylandsink.

Pipeline:
  GStreamer tee fork -> video to compositor -> waylandsink
                     -> frame to fdsink -> Python NPU -> UI layer to compositor -> waylandsink

Audio: SoftSynth (built-in) → ALSA → USB Audio DAC
"""

import argparse
import math
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import random
import re

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import Detector
from tracker import Tracker, TrackedJellyfish
from music_engine import MusicEngine, NOTE_GRID

#  Configuration 
DEFAULT_LOCAL_VIDEO = "../samples/jellyfish.mp4"
DETECT_INTERVAL_S = 0.15
DISPLAY_W = 480
DISPLAY_H = 800
GRID_COLS = 8
GRID_ROWS = 4

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
def midi_to_name(midi): return f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}"

AI_COLOR = (180, 140, 255)  # Soft purple

#  NPU Clock 
def enable_npu_clock():
    """Enable NPU clock via devmem (required before Torq inference)."""
    try:
        subprocess.run(["devmem", "0xf7e104b0", "32", "0x216"],
                       capture_output=True, timeout=5)
        print("[NPU] Clock enabled")
    except Exception as e:
        print(f"[NPU] Clock enable failed: {e}")

#  NeoPixel Control 
NP_TTY = "/dev/ttyACM0"

def init_neopixels():
    if not os.path.exists(NP_TTY):
        return False
    try:
        os.system(f"stty -F {NP_TTY} 115200 raw -echo -hupcl")
        print(f"[NeoPixel] Initialized TTY {NP_TTY}")
        return True
    except Exception as e:
        print(f"[NeoPixel] Initialization failed: {e}")
        return False

def send_npxl_command(cmd):
    import json
    try:
        with open(NP_TTY, "w") as f:
            f.write(json.dumps(cmd) + "\r\n")
    except Exception:
        pass

def npxl_ocean():
    send_npxl_command({
        "on": True,
        "bri": 150,
        "seg": [{"fx": 43, "sx": 30, "ix": 255, "col": [[0, 100, 255], [0, 200, 255], [0, 255, 150]]}]
    })

def create_symlink_for_python():
    """Create symlink for Python 3.12 library (required for some environments)."""
    try:
        target = "/usr/lib/libpython3.12.so.1.0"
        link = "/usr/lib/libpython3.12.so"
        if not os.path.exists(link):
            os.symlink(target, link)
            print(f"[Setup] Created symlink: {link} → {target}")
    except Exception as e:
        print(f"[Setup] Failed to create symlink: {e}")

#  USB Audio Discovery
def find_usb_audio_device():
    """Return the first USB audio ALSA device string (e.g. 'hw:2,0'), or None."""
    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines():
            if re.search(r"\bUSB\b", line, re.IGNORECASE):
                m = re.search(r"card\s+(\d+).*device\s+(\d+)", line, re.IGNORECASE)
                if m:
                    card, dev = m.group(1), m.group(2)
                    device_str = f"hw:{card},{dev}"
                    print(f"[Audio] USB speaker found: {device_str} ({line.strip()})")
                    return device_str
    except Exception as e:
        print(f"[Audio] USB discovery failed: {e}")
    return None

#  Physics Engine 
class PhysicsEngine:
    def __init__(self, cooldown=10.0):
        self.last_clashes = {}
        self.cooldown = cooldown

    def check_collisions(self, tracks):
        new_clashes = []
        now = time.time()
        active = [t for t in tracks if t.age == 0]
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                bb1, bb2 = active[i].bbox, active[j].bbox
                if (bb1["x1"] < bb2["x2"] and bb1["x2"] > bb2["x1"] and
                    bb1["y1"] < bb2["y2"] and bb1["y2"] > bb2["y1"]):
                    pair = tuple(sorted((active[i].id, active[j].id)))
                    if now - self.last_clashes.get(pair, 0.0) > self.cooldown:
                        self.last_clashes[pair] = now
                        mx = (max(bb1["x1"], bb2["x1"]) + min(bb1["x2"], bb2["x2"])) / 2
                        my = (max(bb1["y1"], bb2["y1"]) + min(bb1["y2"], bb2["y2"])) / 2
                        new_clashes.append((mx, my))
        to_del = [k for k, v in self.last_clashes.items() if now - v > self.cooldown * 2]
        for k in to_del:
            del self.last_clashes[k]
        return new_clashes

#  Visual effects 
class Trigger:
    def __init__(self, x, y, row, col, note, t):
        self.x, self.y = x, y
        self.row, self.col = row, col
        self.note = note
        self.t = t

class ClashFX:
    def __init__(self, x, y, t):
        self.x, self.y, self.t = x, y, t

ROW_COLORS = [
    (255, 200, 100),  # Row 0: warm amber (arp)
    (100, 200, 255),  # Row 1: sky blue (pad)
    (200, 120, 255),  # Row 2: purple (chords)
    (100, 255, 180),  # Row 3: teal (bass)
]

class UIOverlay:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.static_layer = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Determine letterbox boundaries once
        self.video_h = int(w * (720 / 1280)) # Assuming 16:9 720p source
        self.top_pad = (h - self.video_h) // 2
        self.bot_pad = self.top_pad + self.video_h
        
        self._precompute_static()

    def _precompute_static(self):
        """Draw elements that never change (labels, backgrounds, grid lines)."""
        # Grid lines (subtle)
        for c in range(1, GRID_COLS):
            x = int(c * self.w / GRID_COLS)
            cv2.line(self.static_layer, (x, self.top_pad), (x, self.bot_pad), (255, 255, 255, 40), 1)
        for r in range(1, GRID_ROWS):
            y = self.top_pad + int(r * self.video_h / GRID_ROWS)
            cv2.line(self.static_layer, (0, y), (self.w, y), (255, 255, 255, 40), 1)

        # --- TOP METRICS AREA ---
        # Subtle header bar
        cv2.rectangle(self.static_layer, (0, 0), (self.w, 60), (20, 20, 30, 160), -1)
        cv2.line(self.static_layer, (0, 60), (self.w, 60), (100, 100, 120, 100), 1)
        
        # Centered main title
        title = 'Generative Audio "Jellectronica"'
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(self.static_layer, title, (self.w // 2 - tw // 2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 240, 255), 2, cv2.LINE_AA)
        
        cv2.putText(self.static_layer, "SYSTEM METRICS", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 180, 255), 1, cv2.LINE_AA)
        
        cv2.putText(self.static_layer, "NPU LATENCY", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120, 255), 1, cv2.LINE_AA)
        cv2.putText(self.static_layer, "FRAME RATE", (260, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120, 255), 1, cv2.LINE_AA)

        # --- BOTTOM METRICS AREA ---
        bot_start = self.h - 265
        
        # Center everything in this section
        mid_x = self.w // 2
        
        # Small centered divider
        cv2.line(self.static_layer, (mid_x - 50, bot_start + 20), (mid_x + 50, bot_start + 20), (100, 255, 150, 150), 1)
        
        # Centered label
        label = "JELLYFISH TRACKED"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(self.static_layer, label, (mid_x - lw // 2, bot_start + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150, 255), 1, cv2.LINE_AA)
        

    def draw(self, frame, tracks, triggers, clashes, trigger_count, jelly_count, fps, npu_ms):
        """Draw dynamic detection overlay on BGRA frame."""
        # Start with static content
        np.copyto(frame, self.static_layer)
        
        w, h = self.w, self.h
        mid_x = w // 2
        video_h, top_pad = self.video_h, self.top_pad
        now = time.time()

        # Create a view of the frame for the video area to enable automatic clipping
        video_roi = frame[top_pad:self.bot_pad, :]

        # Active triggers (ripple)
        alive_triggers = []
        for trig in triggers:
            age = now - trig.t
            if age > 2.0: continue
            alive_triggers.append(trig)
            progress = age / 2.0
            alpha = 1.0 - progress
            color = ROW_COLORS[trig.row % 4]
            cx, cy = int(trig.x * w), int(trig.y * video_h)
            radius = int(15 + 40 * progress)
            ring_color = tuple(int(c * alpha) for c in color) + (255,)
            cv2.circle(video_roi, (cx, cy), radius, ring_color, max(1, int(3 * alpha)))
            if age < 1.0:
                name = midi_to_name(trig.note)
                fa = max(0, 1.0 - age * 1.5)
                text_color = tuple(int(c * fa) for c in color) + (255,)
                cv2.putText(video_roi, name, (cx - 10, cy - radius - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
        triggers[:] = alive_triggers

        # Clash effects
        alive_clashes = []
        for cl in clashes:
            age = now - cl.t
            if age > 1.0: continue
            alive_clashes.append(cl)
            progress = age / 1.0
            fade = 1.0 - progress
            cx, cy = int(cl.x * w), int(cl.y * video_h)
            r, b = int(5 + progress * 60), int(255 * fade)
            cv2.circle(video_roi, (cx, cy), r, (int(b*0.5), int(b*0.9), b, 255), max(1, int(3 * fade)), cv2.LINE_AA)
        clashes[:] = alive_clashes

        # Tracked jellyfish
        for jf in tracks:
            bbox = jf.bbox
            bx1, by1 = int(bbox["x1"] * w), int(bbox["y1"] * video_h)
            bx2, by2 = int(bbox["x2"] * w), int(bbox["y2"] * video_h)
            length = max(4, min(bx2 - bx1, by2 - by1) // 5)
            color = (220, 220, 220, 200)
            # Top-left
            cv2.line(video_roi, (bx1, by1), (bx1 + length, by1), color, 1)
            cv2.line(video_roi, (bx1, by1), (bx1, by1 + length), color, 1)
            # Top-right
            cv2.line(video_roi, (bx2, by1), (bx2 - length, by1), color, 1)
            cv2.line(video_roi, (bx2, by1), (bx2, by1 + length), color, 1)
            # Bottom-left
            cv2.line(video_roi, (bx1, by2), (bx1 + length, by2), color, 1)
            cv2.line(video_roi, (bx1, by2), (bx1, by2 - length), color, 1)
            # Bottom-right
            cv2.line(video_roi, (bx2, by2), (bx2 - length, by2), color, 1)
            cv2.line(video_roi, (bx2, by2), (bx2, by2 - length), color, 1)

            cv2.circle(video_roi, (int(jf.smooth_x * w), int(jf.smooth_y * video_h)), 2, (0, 255, 200, 255), -1)

        # Dynamic Metrics (drawn on full frame)
        cv2.putText(frame, f"{npu_ms:.1f} ms", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 220, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{fps:.1f} FPS", (260, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 255, 255), 2, cv2.LINE_AA)
        
        # Centered counter
        count_str = str(jelly_count)
        (cw, ch), _ = cv2.getTextSize(count_str, cv2.FONT_HERSHEY_SIMPLEX, 2.8, 3)
        cv2.putText(frame, count_str, (mid_x - cw // 2, self.h - 265 + 140), cv2.FONT_HERSHEY_SIMPLEX, 2.8, (120, 255, 150, 255), 3, cv2.LINE_AA)
        
        return frame, self.bot_pad

def draw_ai_bar(frame, ai_notes, ai_active, bot_y):
    """Draw AI Accompaniment visualization bar overlaying the bottom of the video."""
    h, w = frame.shape[:2]
    bar_h = 32
    bar_y = bot_y - bar_h
    now = time.time()

    # Dark, semi-transparent background with a subtle purple tint
    cv2.rectangle(frame, (0, bar_y), (w, bot_y), (25, 15, 35, 210), -1)

    # Glowing top edge pulse
    line_pulse = 0.6 + 0.3 * math.sin(now * 3.0)
    line_color = tuple(int(c * line_pulse) for c in AI_COLOR) + (255,)
    cv2.line(frame, (0, bar_y), (w, bar_y), line_color, 1)
    
    # Faint secondary glow line
    glow_color = tuple(int(c * line_pulse * 0.4) for c in AI_COLOR) + (255,)
    cv2.line(frame, (0, bar_y - 1), (w, bar_y - 1), glow_color, 1)

    # Status indicator dot
    dot_x = 12
    dot_y = bar_y + bar_h // 2
    pulse = 0.5 + 0.5 * math.sin(now * 4.0) if ai_active else 0.2
    dot_color = tuple(int(c * pulse) for c in AI_COLOR) + (255,)
    cv2.circle(frame, (dot_x, dot_y), 4, dot_color, -1)
    if ai_active:
        cv2.circle(frame, (dot_x, dot_y), int(4 + 6 * (1-pulse)), 
                   tuple(int(c * (1-pulse) * 0.5) for c in AI_COLOR) + (255,), 1)

    label = "AI ACCOMPANIMENT" if ai_active else "LIVE PERFORMANCE"
    label_color = AI_COLOR + (255,) if ai_active else (180, 180, 200, 255)
    cv2.putText(frame, label, (22, bar_y + bar_h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)

    if not ai_notes:
        return frame

    note_area_x = 170
    note_spacing = 30
    max_visible = min(len(ai_notes), (w - note_area_x) // note_spacing)

    for i, note_info in enumerate(ai_notes[-max_visible:]):
        age = now - note_info["t"]
        if age > 10.0:
            continue

        # Color based on channel (0-2: Jellyfish rows, 3: AI)
        ch = note_info.get("ch", 3)
        if ch == 3:
            base_color = AI_COLOR
        else:
            # Map channel back to a row color (Channel 0=Pad, 1=Arp, 2=Bass)
            # MusicEngine: Ch0=rows 1/2, Ch1=row 0, Ch2=row 3
            color_idx = 1 if ch == 0 else (0 if ch == 1 else 3)
            base_color = ROW_COLORS[color_idx]

        fade = max(0.0, 1.0 - age / 10.0)
        nx = note_area_x + i * note_spacing
        ny = bar_y + bar_h // 2
        
        # Subtle "float" animation
        drift = math.sin(now * 2.0 + i) * 2
        
        # Initial pop effect
        pop = max(0, 1.0 - age / 0.4)
        radius = int(3 + 5 * pop)
        
        circle_color = tuple(int(c * fade) for c in base_color) + (255,)
        cv2.circle(frame, (nx, int(ny + 4 + drift)), radius, circle_color, -1)

        name = midi_to_name(note_info["midi"])
        text_color = tuple(int(c * fade * 0.9) for c in base_color) + (255,)
        cv2.putText(frame, name, (nx - 8, int(ny - 6 + drift)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)

    return frame

class GstManager:
    def __init__(self, gst_cmd, video_w, video_h, initial_canvas):
        self.gst_cmd = gst_cmd
        self.video_w = video_w
        self.video_h = video_h
        self.initial_canvas = initial_canvas
        self.proc = None
        self.reader_thread = None
        self.writer_thread = None
        self.read_queue = queue.Queue(maxsize=2)
        self.write_queue = queue.Queue(maxsize=1)
        self.running = False
        self.shared_state = None

    def start(self, shared_state):
        self.shared_state = shared_state
        self.running = True
        self._start_pipeline()

    def _start_pipeline(self):
        self.proc = subprocess.Popen(
            self.gst_cmd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, env=os.environ.copy()
        )

        def stdout_reader(proc):
            frame_bytes = self.video_w * self.video_h * 3
            read_frames = 0
            while self.running and proc.poll() is None:
                try:
                    buffer = bytearray()
                    while len(buffer) < frame_bytes and self.running:
                        chunk = proc.stdout.read(frame_bytes - len(buffer))
                        if not chunk: break
                        buffer.extend(chunk)

                    if len(buffer) != frame_bytes:
                        break
                    raw = bytes(buffer)
                    read_frames += 1
                    if self.read_queue.full():
                        try: self.read_queue.get_nowait()
                        except: pass
                    self.read_queue.put((read_frames, raw))
                except Exception:
                    break

        def stdin_writer(proc):
            last_canvas_bytes = self.initial_canvas.tobytes()
            target_dt = 1.0 / 30.0
            try:
                for _ in range(5):
                    proc.stdin.write(last_canvas_bytes)
                    proc.stdin.flush()
            except Exception:
                pass

            while self.running and proc.poll() is None:
                loop_start = time.time()
                try:
                    canvas_bytes = self.write_queue.get_nowait()
                    last_canvas_bytes = canvas_bytes
                except queue.Empty:
                    pass
                
                try:
                    proc.stdin.write(last_canvas_bytes)
                    proc.stdin.flush()
                    self.shared_state["ui_written_count"] = self.shared_state.get("ui_written_count", 0) + 1
                except Exception:
                    break
                    
                elapsed = time.time() - loop_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

        self.reader_thread = threading.Thread(target=stdout_reader, args=(self.proc,), daemon=True)
        self.writer_thread = threading.Thread(target=stdin_writer, args=(self.proc,), daemon=True)
        self.reader_thread.start()
        self.writer_thread.start()

    def check_and_restart(self):
        if self.proc and self.proc.poll() is not None:
            print("[Video] Looping pipeline seamlessly...")
            self.proc.wait()
            self._start_pipeline()

    def write_ui(self, ui_bytes):
        if self.write_queue.full():
            try: self.write_queue.get_nowait()
            except: pass
        try:
            self.write_queue.put_nowait(ui_bytes)
        except queue.Full:
            pass

    def read_video(self):
        try:
            return self.read_queue.get(timeout=0.1)
        except queue.Empty:
            return None, None

    def stop(self):
        self.running = False
        if self.proc:
            try: self.proc.stdin.close()
            except: pass
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=2.0)


# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Jellectronica Display APp (Standalone)")
    parser.add_argument("--video", default=None, help="Local video file")
    parser.add_argument("--youtube", default=None, help="YouTube URL (Disabled)")
    parser.add_argument("--model", default="../models/moon_jellyfish/moon320.vmfb")
    parser.add_argument("--no-audio", action="store_true")
    parser.add_argument("--no-ai", action="store_true", help="Disable MelodyRNN AI accompaniment")

    parser.add_argument("--width", type=int, default=DISPLAY_W)
    parser.add_argument("--height", type=int, default=DISPLAY_H)
    parser.add_argument("--audio-driver", default="alsa")
    parser.add_argument("--alsa-device", default=None)
    args = parser.parse_args()

    W, H = args.width, args.height

    print("============================================")
    print("|   🪼 Jellectronica Display App 🪼         |")
    print("============================================")

    # Wayland env
    os.environ["XDG_RUNTIME_DIR"] = "/var/run/user/0"
    os.environ["WESTON_DISABLE_GBM_MODIFIERS"] = "true"
    os.environ["WAYLAND_DISPLAY"] = "wayland-1"

    enable_npu_clock()
    create_symlink_for_python()
    init_neopixels()
    npxl_ocean()

    #  Resolve source 
    source = args.video if args.video else DEFAULT_LOCAL_VIDEO
    print(f"[Source] Local video: {source}")

    if not os.path.exists(source):
        print(f"ERROR: {source} not found")
        sys.exit(1)

    #  Load detector 
    print("[1/3] Loading detector...")
    detector = Detector(model_path=args.model)
    detector.load()

    print("[2/3] Initializing tracker...")
    tracker = Tracker()
    physics = PhysicsEngine(cooldown=10.0)

    #  Audio
    music = None
    if not args.no_audio:
        print("[3/3] Initializing audio...")
        alsa_device = args.alsa_device or find_usb_audio_device()
        if alsa_device is None:
            print("[Audio] No USB speaker found — using ALSA default")
        try:
            music = MusicEngine(
                audio_driver=args.audio_driver,
                alsa_device=alsa_device,
                ai_enabled=not args.no_ai,
            )
            music.init()
        except Exception as e:
            print(f"[Audio] Failed: {e} — continuing without")
            music = None
    else:
        print("[3/3] Audio disabled")

    video_w, video_h = 480, 270
    uri = "file://" + os.path.abspath(source)

    if source.endswith(".h264"):
        src_elements = [
            "multifilesrc", f"location={source}", "loop=true", "!",
            "h264parse", "!", "avdec_h264", "!",
            "videorate", "!", "video/x-raw,framerate=30/1", "!",
            "identity", "single-segment=true", "!"
        ]
    elif source.endswith(".yuv"):
        src_elements = [
            "multifilesrc", f"location={source}", "loop=true", "!",
            "rawvideoparse", "use-sink-caps=false", "width=1280", "height=720", "format=i420", "framerate=30/1", "!"
        ]
    else:
        src_elements = [
            "filesrc", f"location={source}", "!",
            "qtdemux", "name=demux",
            "demux.video_0", "!", "queue", "!", "h264parse", "!", "decodebin", "!"
        ]

    #  Launch GStreamer display 
    gst_cmd = [
        "gst-launch-1.0", "-q", "-e",
        *src_elements,
        "videoconvert", "!",
        "videoscale", "!", f"video/x-raw,width={video_w},height={video_h},format=BGRA", "!",
        "tee", "name=t",
        
        # Branch 1: Video to Compositor
        "t.", "!", "queue", "max-size-buffers=2", "leaky=downstream", "!",
        "comp.sink_0",

        # Branch 2: Video to Python for NPU
        "t.", "!", "queue", "max-size-buffers=1", "leaky=upstream", "!",
        "videoconvert", "!", f"video/x-raw,format=BGR,width={video_w},height={video_h}", "!",
        "fdsink", "fd=1", "sync=false",

        # Branch 3: UI from Python to Compositor
        "fdsrc", "fd=0", "blocksize=1536000", "!",
        "rawvideoparse", "use-sink-caps=false", f"width={W}", f"height={H}", "format=bgra", "framerate=30/1", "!",
        "queue", "max-size-buffers=1", "!",
        "comp.sink_1",
        
        # Compositor -> Screen
        "compositor", "name=comp", "ignore-inactive-pads=true", "background=black",
        f"sink_0::xpos=0", f"sink_0::ypos={(H - video_h) // 2}", "sink_0::zorder=1",
        "sink_1::xpos=0", "sink_1::ypos=0", "sink_1::zorder=2", "!",
        "waylandsink", "fullscreen=true", "sync=false"
    ]


    # Inference Thread
    shared_state = {
        "frame": None,
        "npu_ms": 0.0,
        "new_triggers": [],
        "new_clashes": [],
        "ui_written_count": 0,
        "ui_dropped_count": 0
    }
    state_lock = threading.Lock()
    _running = True

    def inference_worker():
        while _running:
            try:
                frame_to_proc = None
                with state_lock:
                    if shared_state["frame"] is not None:
                        frame_to_proc = shared_state["frame"].copy()
                        shared_state["frame"] = None

                if frame_to_proc is not None:
                    now = time.time()
                    detections, ms = detector.detect(frame_to_proc)
                    new_trigs = tracker.update(detections)

                    t_fx = []
                    c_fx = []

                    for row, col in new_trigs:
                        note = NOTE_GRID[row][col]
                        x = (col + 0.5) / GRID_COLS
                        y = (row + 0.5) / GRID_ROWS
                        t_fx.append(Trigger(x, y, row, col, note, now))
                        if music:
                            music.trigger_cell(row, col)

                    collisions = physics.check_collisions(tracker.tracks)
                    for cx, cy in collisions:
                        c_fx.append(ClashFX(cx, cy, now))
                        if music:
                            music.play_clash()

                    if music:
                        music.feed_activity(tracker.count)

                    with state_lock:
                        shared_state["npu_ms"] = ms
                        shared_state["new_triggers"].extend(t_fx)
                        shared_state["new_clashes"].extend(c_fx)
                        shared_state["inf_count"] = shared_state.get("inf_count", 0) + 1
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"[Inference Thread] Error: {e}")
                time.sleep(1.0)
                
    inf_thread = threading.Thread(target=inference_worker, daemon=True)
    inf_thread.start()

    print(f"\n[Ready] Jellectronica Display App running!")
    print(f"[Ready] {W}x{H}, local video (GStreamer compositor)")
    print(f"[Ready] Ctrl+C to quit\n")

    triggers = []
    clash_fx = []
    trigger_count = 0
    display_npu_ms = 0.0
    display_fps = 0.0
    display_count = 0
    frame_count = 0
    fps = 0.0
    npu_ms = 0.0
    fps_time = time.time()

    initial_canvas = np.zeros((H, W, 4), dtype=np.uint8)

    while _running:
        print("[Display] Launching GStreamer Pipeline...")
        gst_proc = subprocess.Popen(
            gst_cmd, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=os.environ.copy()
        )

        read_queue = queue.Queue(maxsize=2)
        write_queue = queue.Queue(maxsize=1)
        gst_running = [True]

        def stdout_reader():
            frame_bytes = video_w * video_h * 3
            read_frames = 0
            while gst_running[0] and gst_proc.poll() is None:
                try:
                    buffer = bytearray()
                    while len(buffer) < frame_bytes and gst_running[0]:
                        chunk = gst_proc.stdout.read(frame_bytes - len(buffer))
                        if not chunk:
                            break
                        buffer.extend(chunk)

                    if len(buffer) != frame_bytes:
                        break
                    
                    raw = bytes(buffer)
                    read_frames += 1
                    if read_queue.full():
                        try: read_queue.get_nowait()
                        except: pass
                    read_queue.put((read_frames, raw))
                except Exception:
                    break
            gst_running[0] = False

        reader_thread = threading.Thread(target=stdout_reader, daemon=True)
        reader_thread.start()

        def stdin_writer():
            last_canvas_bytes = initial_canvas.tobytes()
            target_dt = 1.0 / 30.0
            try:
                for _ in range(5):
                    gst_proc.stdin.write(last_canvas_bytes)
                    gst_proc.stdin.flush()
            except Exception:
                pass
                
            while gst_running[0] and gst_proc.poll() is None:
                loop_start = time.time()
                try:
                    canvas_bytes = write_queue.get_nowait()
                    last_canvas_bytes = canvas_bytes
                except queue.Empty:
                    pass
                try:
                    gst_proc.stdin.write(last_canvas_bytes)
                    gst_proc.stdin.flush()
                    with state_lock:
                        shared_state["ui_written_count"] = shared_state.get("ui_written_count", 0) + 1
                except Exception:
                    break
                elapsed = time.time() - loop_start
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
            gst_running[0] = False

        writer_thread = threading.Thread(target=stdin_writer, daemon=True)
        writer_thread.start()

        frame_npu = np.zeros((video_h, video_w, 3), dtype=np.uint8)
        reusable_canvas = np.zeros((H, W, 4), dtype=np.uint8)
        overlay = UIOverlay(W, H)
        last_read_frame = 0
        last_video_time = time.time()

        try:
            while _running and gst_running[0]:
                try:
                    frame_idx, raw = read_queue.get(timeout=0.1)
                    last_read_frame = frame_idx
                    if raw and len(raw) == video_w * video_h * 3:
                        frame_npu = np.frombuffer(raw, dtype=np.uint8).reshape((video_h, video_w, 3))
                        last_video_time = time.time()
                except queue.Empty:
                    pass
                now = time.time()

                if now - last_video_time > 1.0:
                    print("[Video] Video stream ended (no frames for 1s). Restarting...")
                    break

                with state_lock:
                    shared_state["frame"] = frame_npu

                with state_lock:
                    npu_ms = shared_state["npu_ms"]
                    triggers.extend(shared_state["new_triggers"])
                    clash_fx.extend(shared_state["new_clashes"])
                    trigger_count += len(shared_state["new_triggers"])
                    shared_state["new_triggers"].clear()
                    shared_state["new_clashes"].clear()
                    
                if frame_count % 15 == 0:
                    display_npu_ms = npu_ms
                    display_count = tracker.count
                    display_fps = fps

                reusable_canvas.fill(0)
                t_draw_start = time.time()
                canvas, bot_y = overlay.draw(reusable_canvas, tracker.tracks, triggers, clash_fx,
                                             trigger_count, display_count, display_fps, display_npu_ms)
                
                ai_notes = music.get_recent_ai_notes() if music else []
                ai_active = music.ai_enabled if music else False
                canvas = draw_ai_bar(canvas, ai_notes, ai_active, bot_y)
                t_draw_end = time.time()
                draw_ms = (t_draw_end - t_draw_start) * 1000

                ui_bytes = canvas.tobytes()
                dropped = False
                if write_queue.full():
                    try: 
                        write_queue.get_nowait()
                        dropped = True
                    except: pass
                try:
                    write_queue.put_nowait(ui_bytes)
                except queue.Full:
                    dropped = True
                    
                with state_lock:
                    if dropped:
                        shared_state["ui_dropped_count"] = shared_state.get("ui_dropped_count", 0) + 1

                frame_count += 1
                if now - fps_time >= 1.0:
                    fps = frame_count / (now - fps_time)
                    with state_lock:
                        inf_c = shared_state.get("inf_count", 0)
                        written = shared_state.get("ui_written_count", 0)
                        dropped_c = shared_state.get("ui_dropped_count", 0)
                        shared_state["inf_count"] = 0
                        shared_state["ui_written_count"] = 0
                        shared_state["ui_dropped_count"] = 0
                    print(f"[Perf] UI Loop: {fps:.1f} FPS | UI Out: {written} written, {dropped_c} dropped | NPU: {npu_ms:.1f}ms | Draw Overlay: {draw_ms:.1f}ms | GST Output Frames: {last_read_frame} | NPU Iter/sec: {inf_c}")
                    frame_count = 0
                    fps_time = now
                    
        except KeyboardInterrupt:
            print("\n[App] Interrupted")
            _running = False
        except Exception as e:
            print(f"\n[App] Error: {e}")
            import traceback
            traceback.print_exc()
            _running = False

        gst_running[0] = False
        if gst_proc:
            try: gst_proc.stdin.close()
            except: pass
            if gst_proc.poll() is None:
                gst_proc.terminate()
                gst_proc.wait(timeout=1.0)
        
        if _running:
            print("[Video] Stream ended. Restarting GStreamer internally...")
            
    inf_thread.join(timeout=1.0)
    detector.dispose()
    if music:
        music.dispose()
    print("[App] Done.")

if __name__ == "__main__":
    main()
