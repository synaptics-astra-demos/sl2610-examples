#!/usr/bin/env python3
"""
Jellectronica — DSI Kiosk with Detection Overlay (Standalone)
=================================================================

Full standalone entry point for the Coral Dev Board. Renders jellyfish
video with detection overlay on DSI display via GStreamer waylandsink.

Pipeline:
  YouTube/local → cv2.VideoCapture → NPU detect → draw overlay →
  raw BGRx → subprocess(gst-launch fdsrc → rawvideoparse → waylandsink)

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

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import Detector
from tracker import Tracker, TrackedJellyfish
from music_engine import MusicEngine, NOTE_GRID


#  Configuration 
DEFAULT_YOUTUBE_URL = "https://www.youtube.com/watch?v=7N9-FODmuBA"
DEFAULT_LOCAL_VIDEO = "../samples/moon15.mp4"
DETECT_INTERVAL_S = 0.15
DISPLAY_W = 800
DISPLAY_H = 480
GRID_COLS = 8
GRID_ROWS = 4

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
def midi_to_name(midi): return f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


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
        else:
            print(f"[Setup] Symlink already exists: {link}")
    except Exception as e:
        print(f"[Setup] Failed to create symlink: {e}")

#  YouTube helpers 
def resolve_youtube_stream(url):
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        print("[YouTube] yt-dlp not found in PATH")
        return None
    try:
        print(f"[YouTube] Resolving: {url}")
        r = subprocess.run(
            [ytdlp, "-f",
             "best[ext=mp4][height<=480]/best[ext=mp4]/best",
             "--get-url", url],
            capture_output=True, text=True, timeout=30)
        u = r.stdout.strip()
        if r.returncode == 0 and u.startswith("http"):
            print(f"[YouTube] ✓ Resolved ({len(u)} chars)")
            return u
        else:
            err = r.stderr.strip()
            print(f"[YouTube] ✗ yt-dlp failed (exit {r.returncode}): {err[:200]}")
    except subprocess.TimeoutExpired:
        print("[YouTube] ✗ yt-dlp timed out (30s)")
    except Exception as e:
        print(f"[YouTube] ✗ Error: {e}")
    return None


#  Buffered Video Reader 
class BufferedVideoReader:
    """Reads frames in background to smooth out network stalls."""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self._q = queue.Queue(maxsize=120)
        self._running = False
        self._thread = None

    def isOpened(self): return self.cap.isOpened()
    def get(self, prop): return self.cap.get(prop)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        for _ in range(40):
            if self._q.qsize() > 5:
                break
            time.sleep(0.05)
        return self

    def _read_loop(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            try:
                self._q.put((ret, frame), timeout=1.0)
            except (queue.Empty, queue.Full):
                pass

    def read(self):
        if not self._running:
            return False, None
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return False, None

    def release(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self.cap.release()


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
        # Cleanup
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


def draw_live_badge(frame):
    """Draw pulsing red LIVE indicator in top-right."""
    h, w = frame.shape[:2]
    pulse = 0.7 + 0.3 * math.sin(time.time() * 3.0)
    badge_text = "LIVE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(badge_text, font, 0.5, 2)
    bx = w - tw - 40
    by = 10
    cv2.rectangle(frame, (bx - 5, by), (bx + tw + 25, by + th + 10), (20, 20, 20), -1)
    dot_x = bx + 5
    dot_y = by + th // 2 + 5
    cv2.circle(frame, (dot_x, dot_y), 4, (0, 0, int(220 * pulse)), -1)
    cv2.putText(frame, badge_text, (dot_x + 10, by + th + 3),
                font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_overlay(frame, tracks, triggers, clashes, trigger_count, jelly_count, fps):
    """Draw detection overlay on frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Grid lines (subtle)
    for c in range(1, GRID_COLS):
        x = int(c * w / GRID_COLS)
        cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 1)
    for r in range(1, GRID_ROWS):
        y = int(r * h / GRID_ROWS)
        cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

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
        cx, cy = int(trig.x * w), int(trig.y * h)
        radius = int(15 + 40 * progress)
        thickness = max(1, int(3 * alpha))
        ring_color = tuple(int(c * alpha) for c in color)
        cv2.circle(frame, (cx, cy), radius, ring_color, thickness)
        if age < 1.0:
            name = midi_to_name(trig.note)
            fa = max(0, 1.0 - age * 1.5)
            cv2.putText(frame, name, (cx - 10, cy - radius - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        tuple(int(c * fa) for c in color), 1, cv2.LINE_AA)
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
        cx, cy = int(cl.x * w), int(cl.y * h)
        r = int(5 + progress * 60)
        b = int(255 * fade)
        cv2.circle(frame, (cx, cy), r, (int(b*0.5), int(b*0.9), b),
                   max(1, int(3 * fade)), cv2.LINE_AA)
    clashes[:] = alive_clashes

    # Tracked jellyfish
    for jf in tracks:
        bbox = jf.bbox
        bx1, by1 = int(bbox["x1"] * w), int(bbox["y1"] * h)
        bx2, by2 = int(bbox["x2"] * w), int(bbox["y2"] * h)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (200, 200, 200), 1)
        cx, cy = int(jf.smooth_x * w), int(jf.smooth_y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 200), -1)

    # HUD
    cv2.putText(frame, "JELLECTRONICA", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{jelly_count} jellyfish | {trigger_count} triggers | {fps:.0f}fps",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (150, 150, 150), 1, cv2.LINE_AA)
    return frame


#  Main 
def main():
    parser = argparse.ArgumentParser(description="Jellectronica DSI Kiosk (Standalone)")
    parser.add_argument("--video", default=None, help="Local video file")
    parser.add_argument("--youtube", default=None, help="YouTube URL")
    parser.add_argument("--model", default="../models/moon320.vmfb")
    parser.add_argument("--soundfont", default=None)  # ignored, kept for CLI compat
    parser.add_argument("--no-audio", action="store_true")

    parser.add_argument("--width", type=int, default=DISPLAY_W)
    parser.add_argument("--height", type=int, default=DISPLAY_H)
    parser.add_argument("--audio-driver", default="alsa")
    parser.add_argument("--alsa-device", default=None)
    args = parser.parse_args()

    W, H = args.width, args.height

    print("============================================")
    print("|   🪼 Jellectronica DSI Kiosk 🪼          |")
    print("============================================")


    # Wayland env
    os.environ["XDG_RUNTIME_DIR"] = "/var/run/user/0"
    os.environ["WESTON_DISABLE_GBM_MODIFIERS"] = "true"
    os.environ["WAYLAND_DISPLAY"] = "wayland-1"

    # Enable NPU clock
    enable_npu_clock()

    # Add symlink for Python library
    create_symlink_for_python()

    #  Resolve source 
    source = None
    is_live = False
    if args.video:
        source = args.video
        print(f"[Source] Local video: {source}")
    elif args.youtube:
        print("[Source] Resolving YouTube...")
        source = resolve_youtube_stream(args.youtube)
        if source:
            is_live = True
            print("[Source] ✓ YouTube stream resolved")
        else:
            source = DEFAULT_LOCAL_VIDEO
            print(f"[Source] YouTube failed, using: {source}")
    else:
        print("[Source] Trying YouTube...")
        source = resolve_youtube_stream(DEFAULT_YOUTUBE_URL)
        if source:
            is_live = True
            print("[Source] ✓ YouTube stream resolved")
        else:
            source = DEFAULT_LOCAL_VIDEO
            print(f"[Source] Using local video: {source}")

    if not is_live and not os.path.exists(source):
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
        try:
            music = MusicEngine(
                audio_driver=args.audio_driver,
                alsa_device=args.alsa_device,
            )
            music.init()
        except Exception as e:
            print(f"[Audio] Failed: {e} — continuing without")
            music = None
    else:
        print("[3/3] Audio disabled")


    #  Open video 
    print(f"[Video] Opening source...")
    if is_live:
        cap = BufferedVideoReader(source)
        if not cap.isOpened():
            print("[Source] Live stream failed — trying local fallback")
            is_live = False
            source = DEFAULT_LOCAL_VIDEO
            cap = cv2.VideoCapture(source)
        else:
            cap.start()
            print("[Video] Buffered reader started")
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Cannot open {source}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0 or math.isnan(src_fps):
        src_fps = 30.0
    print(f"[Video] Source @ {src_fps:.0f}fps → display {W}x{H}")

    #  Launch GStreamer display 
    frame_size = W * H * 4
    gst_cmd = [
        "gst-launch-1.0", "-e",
        "fdsrc", "fd=0", f"blocksize={frame_size}", "!",
        "rawvideoparse", "use-sink-caps=false",
        f"width={W}", f"height={H}",
        "format=bgrx", "framerate=25/1", "!",
        "queue", "max-size-buffers=2", "leaky=downstream", "!",
        "waylandsink", "fullscreen=true", "sync=false",
    ]

    env = os.environ.copy()

    print("[Display] Launching GStreamer...")
    gst_proc = subprocess.Popen(
        gst_cmd, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=env,
    )
    time.sleep(1)
    if gst_proc.poll() is not None:
        print(f"[Display] GStreamer failed (exit code: {gst_proc.returncode})")
        sys.exit(1)
    print(f"[Display] ✓ GStreamer running (PID {gst_proc.pid})")


    #  Main loop 
    triggers = []
    clash_fx = []
    trigger_count = 0
    last_detect = 0
    frame_count = 0
    fps_time = time.time()
    fps = 0.0
    target_dt = 1.0 / 25.0
    _running = True

    # YouTube reconnection state
    yt_fail_count = 0
    last_yt_reconnect = 0

    print(f"\n[Ready] Jellectronica DSI Kiosk running!")
    print(f"[Ready] {W}x{H}, {'YouTube LIVE' if is_live else 'local video'}")
    print(f"[Ready] Ctrl+C to quit\n")

    try:
        while _running:
            loop_start = time.time()
            ret, frame = cap.read()

            if not ret:
                if not is_live:
                    if isinstance(cap, cv2.VideoCapture):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                # Live stream stall — try reconnect
                yt_fail_count += 1
                if yt_fail_count > 150 and time.time() - last_yt_reconnect > 60:
                    print("[YouTube] Stream stalled, reconnecting...")
                    last_yt_reconnect = time.time()
                    cap.release()
                    new_url = resolve_youtube_stream(DEFAULT_YOUTUBE_URL)
                    if new_url:
                        cap = BufferedVideoReader(new_url)
                        if cap.isOpened():
                            cap.start()
                            yt_fail_count = 0
                            print("[YouTube] ✓ Reconnected")
                            continue
                    # Fallback to local
                    print("[YouTube] Reconnect failed, using local video")
                    is_live = False
                    cap = cv2.VideoCapture(DEFAULT_LOCAL_VIDEO)
                time.sleep(0.01)
                continue

            yt_fail_count = 0

            # Resize to display
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)

            now = time.time()

            # Detection
            if (now - last_detect) >= DETECT_INTERVAL_S:
                last_detect = now
                detections = detector.detect(frame)
                new_triggers = tracker.update(detections)

                for row, col in new_triggers:
                    trigger_count += 1
                    note = NOTE_GRID[row][col]
                    x = (col + 0.5) / GRID_COLS
                    y = (row + 0.5) / GRID_ROWS
                    triggers.append(Trigger(x, y, row, col, note, now))

                    if music:
                        music.trigger_cell(row, col)


                # Physics collisions
                collisions = physics.check_collisions(tracker.tracks)
                for cx, cy in collisions:
                    clash_fx.append(ClashFX(cx, cy, now))
                    if music:
                        music.play_clash()



            # Draw overlay
            frame = draw_overlay(frame, tracker.tracks, triggers, clash_fx,
                                 trigger_count, tracker.count, fps)
            if is_live:
                frame = draw_live_badge(frame)

            # Write to GStreamer
            try:
                bgrx = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
                gst_proc.stdin.write(bgrx.tobytes())
                gst_proc.stdin.flush()
            except (BrokenPipeError, OSError):
                print("[Display] GStreamer pipe broken")
                break

            if gst_proc.poll() is not None:
                print(f"[Display] GStreamer exited")
                _running = False

            # FPS
            frame_count += 1
            if now - fps_time >= 1.0:
                fps = frame_count / (now - fps_time)
                frame_count = 0
                fps_time = now

            # Frame pacing
            elapsed = time.time() - loop_start
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    except KeyboardInterrupt:
        print("\n[Kiosk] Interrupted")
    except Exception as e:
        print(f"\n[Kiosk] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Kiosk] Cleaning up...")
        try:
            gst_proc.stdin.close()
        except Exception:
            pass
        if gst_proc.poll() is None:
            gst_proc.terminate()

        if music:
            music.dispose()
        cap.release()
        detector.dispose()
        print("[Kiosk] Done.")


if __name__ == "__main__":
    main()
