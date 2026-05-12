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

def draw_overlay(frame, tracks, triggers, clashes, trigger_count, jelly_count, fps, npu_ms):
    """Draw detection overlay on BGRA frame."""
    h, w = frame.shape[:2]

    # Determine letterbox boundaries
    video_h = int(w * (720 / 1280)) # Assuming 16:9 720p source, width is 480
    top_pad = (h - video_h) // 2
    bot_pad = top_pad + video_h

    # Grid lines (subtle)
    for c in range(1, GRID_COLS):
        x = int(c * w / GRID_COLS)
        cv2.line(frame, (x, top_pad), (x, bot_pad), (255, 255, 255, 40), 1)
    for r in range(1, GRID_ROWS):
        y = top_pad + int(r * video_h / GRID_ROWS)
        cv2.line(frame, (0, y), (w, y), (255, 255, 255, 40), 1)

    now = time.time()

    # Active triggers (ripple)
    alive_triggers = []
    for trig in triggers:
        age = now - trig.t
        if age > 2.0:
            continue
        alive_triggers.append(trig)
        progress = age / 2.0
        alpha = 1.0 - progress
        color = ROW_COLORS[trig.row % 4]
        cx = int(trig.x * w)
        cy = top_pad + int(trig.y * video_h)
        radius = int(15 + 40 * progress)
        thickness = max(1, int(3 * alpha))
        ring_color = tuple(int(c * alpha) for c in color) + (255,)
        cv2.circle(frame, (cx, cy), radius, ring_color, thickness)
        if age < 1.0:
            name = midi_to_name(trig.note)
            fa = max(0, 1.0 - age * 1.5)
            text_color = tuple(int(c * fa) for c in color) + (255,)
            cv2.putText(frame, name, (cx - 10, cy - radius - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)
    triggers[:] = alive_triggers

    # Clash effects
    alive_clashes = []
    for cl in clashes:
        age = now - cl.t
        if age > 1.0:
            continue
        alive_clashes.append(cl)
        progress = age / 1.0
        fade = 1.0 - progress
        cx = int(cl.x * w)
        cy = top_pad + int(cl.y * video_h)
        r = int(5 + progress * 60)
        b = int(255 * fade)
        cv2.circle(frame, (cx, cy), r, (int(b*0.5), int(b*0.9), b, 255),
                   max(1, int(3 * fade)), cv2.LINE_AA)
    clashes[:] = alive_clashes

    # Tracked jellyfish
    for jf in tracks:
        bbox = jf.bbox
        bx1 = int(bbox["x1"] * w)
        by1 = top_pad + int(bbox["y1"] * video_h)
        bx2 = int(bbox["x2"] * w)
        by2 = top_pad + int(bbox["y2"] * video_h)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (200, 200, 200, 255), 1)
        cx = int(jf.smooth_x * w)
        cy = top_pad + int(jf.smooth_y * video_h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 200, 255), -1)

    # --- TOP METRICS AREA ---
    cv2.putText(frame, "SYSTEM METRICS", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "NPU LATENCY", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{npu_ms:.1f} ms", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 200, 255, 255), 2, cv2.LINE_AA)
    
    cv2.putText(frame, "FRAME RATE", (260, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.1f} FPS", (260, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 255, 255), 2, cv2.LINE_AA)

    # --- BOTTOM METRICS AREA ---
    bot_start = h - 265
    cv2.putText(frame, "JELLYFISH TRACKED", (20, bot_start + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, str(jelly_count), (20, bot_start + 110), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (100, 255, 100, 255), 3, cv2.LINE_AA)
    
    cv2.putText(frame, "CORAL SL2610 EDGE NPU", (120, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100, 255), 1, cv2.LINE_AA)

    return frame

def draw_ai_bar(frame, ai_notes, ai_active):
    """Draw AI Accompaniment visualization bar at the bottom of the BGRA frame."""
    h, w = frame.shape[:2]
    bar_h = 36
    bar_y = h - bar_h
    now = time.time()

    cv2.rectangle(frame, (0, bar_y), (w, h), (15, 10, 25, 180), -1)

    line_alpha = 0.4 + 0.2 * math.sin(now * 2.0)
    line_color = tuple(int(c * line_alpha) for c in AI_COLOR) + (255,)
    cv2.line(frame, (0, bar_y), (w, bar_y), line_color, 1)

    dot_x = 12
    dot_y = bar_y + bar_h // 2
    pulse = 0.5 + 0.5 * math.sin(now * 3.0)
    dot_color = tuple(int(c * pulse) for c in AI_COLOR) + (255,)
    cv2.circle(frame, (dot_x, dot_y), 4, dot_color, -1)

    label = "AI ACCOMPANIMENT" if ai_active else "AI ACCOMPANIMENT [OFF]"
    label_color = AI_COLOR + (255,) if ai_active else (80, 60, 100, 255)
    cv2.putText(frame, label, (22, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, label_color, 1, cv2.LINE_AA)

    if not ai_active or not ai_notes:
        return frame

    note_area_x = 160
    note_spacing = 28
    max_visible = min(len(ai_notes), (w - note_area_x) // note_spacing)

    for i, note_info in enumerate(ai_notes[-max_visible:]):
        age = now - note_info["t"]
        if age > 4.0:
            continue

        fade = max(0.0, 1.0 - age / 4.0)
        nx = note_area_x + i * note_spacing
        ny = bar_y + bar_h // 2

        radius = int(3 + 4 * max(0, 1.0 - age / 0.5))
        circle_color = tuple(int(c * fade) for c in AI_COLOR) + (255,)
        cv2.circle(frame, (nx, ny + 4), radius, circle_color, -1)

        name = midi_to_name(note_info["midi"])
        text_color = tuple(int(c * fade * 0.8) for c in AI_COLOR) + (255,)
        cv2.putText(frame, name, (nx - 8, ny - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, text_color, 1, cv2.LINE_AA)

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
                    d_start = time.time()
                    now = time.time()
                    detections = detector.detect(frame_to_proc)
                    ms = (time.time() - d_start) * 1000
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
                canvas = draw_overlay(reusable_canvas, tracker.tracks, triggers, clash_fx,
                                      trigger_count, display_count, display_fps, display_npu_ms)
                
                ai_notes = music.get_recent_ai_notes() if music else []
                ai_active = music.ai_enabled if music else False
                canvas = draw_ai_bar(canvas, ai_notes, ai_active)
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