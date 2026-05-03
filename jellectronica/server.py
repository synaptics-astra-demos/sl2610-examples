"""
Jellectronica: Coralboard Native Edition — On-Device Server
==================================================
Runs entirely on the Coral board:
  • NPU detection (Torq/IREE moon320.vmfb)
  • SoftSynth audio → ALSA → USB Audio DAC
  • MJPEG stream + WebSocket events served to laptop browser

Usage:
    python3 server.py
    python3 server.py --video video/moon15.mp4
    python3 server.py --audio-driver alsa --alsa-device hw:0,0
"""

import argparse, json, math, os, queue, subprocess, shutil
import socket, sys, threading, time

import cv2
import numpy as np
from flask import Flask, Response

from detector import Detector
from tracker import Tracker, TrackedJellyfish
from music_engine import MusicEngine, NOTE_GRID

# ── Config ─────────────────────────────────────────────────────
GRID_COLS = 8
GRID_ROWS = 4
DETECT_INTERVAL_S = 0.1
DEFAULT_VIDEO = "https://www.youtube.com/watch?v=7N9-FODmuBA"
STREAM_MAX_FPS = 5
STREAM_JPEG_QUALITY = 25
STREAM_MAX_HEIGHT = 400

_frame_queue: queue.Queue = queue.Queue(maxsize=2)
_ws_queues: list = []
_ws_lock = threading.Lock()

app = Flask(__name__)

_NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
def midi_to_name(midi: int) -> str:
    return f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}"

def cell_hue(row: int, col: int) -> float:
    return (180 + (row * GRID_COLS + col) * 22.5) % 360

ROW_COLORS = [
    (255, 200, 100), (100, 200, 255),
    (200, 120, 255), (100, 255, 180),
]

def broadcast(event: dict):
    msg = json.dumps(event)
    with _ws_lock:
        for q in list(_ws_queues):
            try: q.put_nowait(msg)
            except Exception: pass


# ── Physics Engine ─────────────────────────────────────────────
class PhysicsEngine:
    def __init__(self, cooldown=10.0):
        self.last_clashes = {}
        self.cooldown = cooldown

    def check_collisions(self, tracks):
        clashes = []
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
                        clashes.append((mx, my))
        to_del = [k for k, v in self.last_clashes.items() if now - v > self.cooldown * 2]
        for k in to_del: del self.last_clashes[k]
        return clashes


# ── Visual Effects ─────────────────────────────────────────────
class Trigger:
    __slots__ = ("x","y","row","col","note","t")
    def __init__(self, x, y, row, col, note, t):
        self.x, self.y, self.row, self.col, self.note, self.t = x, y, row, col, note, t

class ClashFX:
    __slots__ = ("x","y","t")
    def __init__(self, x, y, t):
        self.x, self.y, self.t = x, y, t



def annotate_frame(frame, tracks, triggers, clashes, trigger_count, jelly_count, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    grid_color = (60, 50, 0)
    for c in range(1, GRID_COLS):
        x = int(w * c / GRID_COLS)
        for y_pos in range(0, h, 12):
            cv2.line(overlay, (x, y_pos), (x, min(h, y_pos+5)), grid_color, 1)
    for r in range(1, GRID_ROWS):
        y = int(h * r / GRID_ROWS)
        for x_pos in range(0, w, 12):
            cv2.line(overlay, (x_pos, y), (min(w, x_pos+5), y), grid_color, 1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

    now = time.time()
    alive = []
    for trig in triggers:
        age = now - trig.t
        if age > 2.0: continue
        alive.append(trig)
        progress = age / 2.0
        alpha = 1.0 - progress
        color = ROW_COLORS[trig.row % 4]
        cx, cy = int(trig.x * w), int(trig.y * h)
        radius = int(15 + 40 * progress)
        ring_color = tuple(int(c * alpha) for c in color)
        cv2.circle(frame, (cx, cy), radius, ring_color, max(1, int(3*alpha)))
        if age < 1.0:
            fa = max(0, 1.0 - age * 1.5)
            cv2.putText(frame, midi_to_name(trig.note), (cx-10, cy-radius-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        tuple(int(c*fa) for c in color), 1, cv2.LINE_AA)
    triggers[:] = alive

    alive_cl = []
    for cl in clashes:
        age = now - cl.t
        if age > 1.0: continue
        alive_cl.append(cl)
        progress = age / 1.0
        fade = 1.0 - progress
        cx, cy = int(cl.x * w), int(cl.y * h)
        r = int(5 + progress * 60)
        b = int(255 * fade)
        cv2.circle(frame, (cx, cy), r, (int(b*0.5), int(b*0.9), b),
                   max(1, int(3*fade)), cv2.LINE_AA)
    clashes[:] = alive_cl

    for jf in tracks:
        bb = jf.bbox
        x1, y1 = int(bb["x1"]*w), int(bb["y1"]*h)
        x2, y2 = int(bb["x2"]*w), int(bb["y2"]*h)
        opacity = max(0.2, 1.0 - jf.age * 0.06)
        col = tuple(int(c*opacity) for c in (200, 180, 0))
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{jf.confidence:.0%}", (x1+4, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

    cv2.putText(frame, "JELLECTRONICA", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{jelly_count} jellyfish | {trigger_count} triggers | {fps:.0f}fps",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (150, 150, 150), 1, cv2.LINE_AA)
    return frame


# ── YouTube resolution ──────────────────────────────────────────
def _resolve_youtube_url(youtube_url: str) -> str:
    print(f"[YouTube] Resolving: {youtube_url}")
    try:
        import yt_dlp
        ydl_opts = {"format": "best[height<=720]", "quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            url = info.get("url")
            if url:
                print(f"[YouTube] ✓ Resolved via API ({len(url)} chars)")
                return url
    except ImportError:
        pass
    except Exception as e:
        print(f"[YouTube] yt-dlp API error: {e}")
    try:
        result = subprocess.run(
            ["yt-dlp", "-f", "best[height<=720]", "--get-url", youtube_url],
            capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            url = result.stdout.strip()
            print(f"[YouTube] ✓ Resolved via CLI ({len(url)} chars)")
            return url
        else:
            err = result.stderr.strip()
            print(f"[YouTube] ✗ CLI failed (exit {result.returncode}): {err[:200]}")
    except FileNotFoundError:
        print("[YouTube] ✗ yt-dlp not found")
    except subprocess.TimeoutExpired:
        print("[YouTube] ✗ yt-dlp timed out (30s)")
    print("[YouTube] ✗ Resolution failed")
    return None

def _is_youtube_url(url: str) -> bool:
    return any(d in url for d in ["youtube.com/watch", "youtu.be/", "youtube.com/live"])


# ── Shared detection state ──────────────────────────────────────
_detect_frame = None
_detect_frame_lock = threading.Lock()
_detect_results_lock = threading.Lock()
_triggers_list = []
_clash_fx_list = []
_trigger_count = 0
_jelly_count = 0
_fps = 0.0

# Global references for music/physics (set in main)
_music = None
_physics = None

# YouTube recheck state
_youtube_recheck_url = None  # Set when YouTube fallback is active


def _detection_thread(detector, tracker):
    global _detect_frame, _trigger_count, _jelly_count
    detect_count = 0
    while True:
        with _detect_frame_lock:
            frame = _detect_frame
            _detect_frame = None
        if frame is None:
            time.sleep(0.01)
            continue

        detections = detector.detect(frame)
        with _detect_results_lock:
            triggers = tracker.update(detections)

        now = time.time()
        for row, col in triggers:
            note = NOTE_GRID[row][col]
            x = (col + 0.5) / GRID_COLS
            y = (row + 0.5) / GRID_ROWS
            _triggers_list.append(Trigger(x, y, row, col, note, now))
            _trigger_count += 1

            if _music:
                _music.trigger_cell(row, col)


            broadcast({"type": "trigger", "row": row, "col": col,
                       "note": note, "noteName": midi_to_name(note),
                       "hue": cell_hue(row, col)})

        # Physics collisions
        if _physics:
            with _detect_results_lock:
                collisions = _physics.check_collisions(tracker.tracks)
            for cx, cy in collisions:
                _clash_fx_list.append(ClashFX(cx, cy, now))
                if _music:
                    _music.play_clash()

        with _detect_results_lock:
            _jelly_count = tracker.count
            broadcast({"type": "count", "count": tracker.count})



        detect_count += 1
        if detect_count % 100 == 0:
            print(f"[Detect] {detect_count} detections completed", flush=True)


def _schedule_youtube_recheck(video_path: str):
    """Background thread: periodically check if YouTube is reachable.

    When the inference loop falls back to local video due to network loss,
    this thread checks every 60 seconds if YouTube is back. When it is,
    the next stream-lost event will re-resolve successfully.
    """
    global _youtube_recheck_url

    def _recheck():
        global _youtube_recheck_url
        while True:
            time.sleep(60)
            try:
                resolved = _resolve_youtube_url(video_path)
                if resolved:
                    _youtube_recheck_url = resolved
                    print(f"[YouTube] ✓ Stream available again — will switch on next cycle",
                          flush=True)
                    return
            except Exception:
                pass
            print("[YouTube] Still unavailable, retrying in 60s...", flush=True)

    t = threading.Thread(target=_recheck, daemon=True, name="youtube-recheck")
    t.start()


def inference_loop(video_path: str, model_path: str):
    global _detect_frame, _fps
    print(f"[Inference] Starting — video: {video_path}, model: {model_path}", flush=True)

    detector = Detector(model_path=model_path)
    detector.load()
    tracker = Tracker()

    det_thread = threading.Thread(
        target=_detection_thread, args=(detector, tracker), daemon=True)
    det_thread.start()

    actual_url = video_path
    is_youtube = _is_youtube_url(video_path)
    is_stream = video_path.startswith("http")

    if is_youtube:
        print(f"[Inference] Resolving YouTube URL...", flush=True)
        resolved = _resolve_youtube_url(video_path)
        if resolved:
            actual_url = resolved
            is_stream = True
        else:
            fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "../samples", "moon15.mp4")
            if os.path.exists(fallback):
                print(f"[Inference] Falling back to {fallback}", flush=True)
                actual_url = fallback
                is_stream = False
            else:
                print("[Inference] No video source available", flush=True)
                return

    stream_retry_count = 0
    MAX_STREAM_RETRIES = 3  # Fall back to local video after this many failures

    while True:
        print(f"[Inference] Opening: {actual_url[:80]}...", flush=True)
        cap = cv2.VideoCapture(actual_url)
        if not cap.isOpened():
            if is_stream:
                stream_retry_count += 1
                if stream_retry_count <= MAX_STREAM_RETRIES:
                    print(f"[Inference] Stream unavailable ({stream_retry_count}/{MAX_STREAM_RETRIES}), "
                          f"retrying in 3s...", flush=True)
                    time.sleep(3)
                    if is_youtube:
                        resolved = _resolve_youtube_url(video_path)
                        if resolved: actual_url = resolved
                    continue
                else:
                    # Fall back to local video
                    fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "../samples", "moon15.mp4")
                    if os.path.exists(fallback):
                        print(f"[Inference] Stream failed after {MAX_STREAM_RETRIES} attempts — "
                              f"falling back to {os.path.basename(fallback)}", flush=True)
                        actual_url = fallback
                        is_stream = False
                        stream_retry_count = 0
                        # Schedule background re-check for YouTube
                        _schedule_youtube_recheck(video_path)
                        continue
                    else:
                        print("[Inference] No video source available", flush=True)
                        return
            else:
                print(f"[Inference] ERROR: Cannot open {video_path}", flush=True)
                return

        stream_retry_count = 0  # Reset on successful open

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 0 or math.isnan(src_fps):
            src_fps = 15.0 if is_stream else 25.0
        out_fps = min(src_fps, STREAM_MAX_FPS)
        frame_skip = max(1, round(src_fps / out_fps))
        frame_duration = 1.0 / out_fps

        print(f"[Inference] ✓ {w}x{h} @ {src_fps:.1f}fps "
              f"({'stream' if is_stream else 'file'})", flush=True)

        fail_count = 0
        black_frame_count = 0
        frame_num = 0
        output_num = 0
        last_detect_submit = 0
        fps_time = time.time()
        fps_count = 0

        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                if is_stream:
                    fail_count += 1
                    if fail_count > 60:
                        print("[Inference] Stream lost, reconnecting...", flush=True)
                        cap.release()
                        time.sleep(2)
                        # Try to re-resolve, but fall back quickly
                        reconnect_ok = False
                        if is_youtube:
                            resolved = _resolve_youtube_url(video_path)
                            if resolved:
                                actual_url = resolved
                                reconnect_ok = True
                        if not reconnect_ok:
                            fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "../samples", "moon15.mp4")
                            if os.path.exists(fallback):
                                print(f"[Inference] Cannot reconnect — falling back to "
                                      f"{os.path.basename(fallback)}", flush=True)
                                actual_url = fallback
                                is_stream = False
                                _schedule_youtube_recheck(video_path)
                        break
                    time.sleep(0.03)
                    continue
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            fail_count = 0
            frame_num += 1

            # Black screen detection — aquarium lights off
            if is_stream:
                brightness = np.mean(frame)
                if brightness < 10:
                    black_frame_count += 1
                    if black_frame_count == 1:
                        print("[Inference] Dark frame detected...", flush=True)
                    if black_frame_count >= 15:
                        fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "../samples", "moon15.mp4")
                        if os.path.exists(fallback):
                            print(f"[Inference] Stream is black (lights off?) — "
                                  f"switching to {os.path.basename(fallback)}", flush=True)
                            cap.release()
                            actual_url = fallback
                            is_stream = False
                            break
                else:
                    black_frame_count = 0

            if frame_skip > 1 and (frame_num % frame_skip) != 0:
                continue
            output_num += 1

            now = time.time()
            if (now - last_detect_submit) >= DETECT_INTERVAL_S:
                with _detect_frame_lock:
                    _detect_frame = frame.copy()
                last_detect_submit = now

            # FPS tracking
            fps_count += 1
            if now - fps_time >= 1.0:
                _fps = fps_count / (now - fps_time)
                fps_count = 0
                fps_time = now

            # Downscale + annotate + stream
            if STREAM_MAX_HEIGHT and frame.shape[0] > STREAM_MAX_HEIGHT:
                scale = STREAM_MAX_HEIGHT / frame.shape[0]
                small = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            else:
                small = frame

            with _detect_results_lock:
                annotated = annotate_frame(small, tracker.tracks,
                    _triggers_list, _clash_fx_list,
                    _trigger_count, _jelly_count, _fps)

            _, jpeg = cv2.imencode(".jpg", annotated,
                                   [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
            frame_bytes = jpeg.tobytes()
            try:
                _frame_queue.put_nowait(frame_bytes)
            except queue.Full:
                try:
                    _frame_queue.get_nowait()
                    _frame_queue.put_nowait(frame_bytes)
                except Exception: pass

            if not is_stream:
                elapsed = time.time() - loop_start
                if elapsed < frame_duration:
                    time.sleep(frame_duration - elapsed)


# ── Flask routes ────────────────────────────────────────────────
def generate_mjpeg():
    while True:
        try:
            frame_bytes = _frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/stream")
def stream():
    return Response(generate_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def run_ws_server(host: str, port: int):
    import asyncio, websockets
    async def handler(ws):
        q = asyncio.Queue(maxsize=50)
        with _ws_lock: _ws_queues.append(q)
        print(f"[WS] Client connected")
        try:
            while True:
                msg = await q.get()
                await ws.send(msg)
        except Exception: pass
        finally:
            with _ws_lock:
                if q in _ws_queues: _ws_queues.remove(q)
            print(f"[WS] Client disconnected")
    async def serve():
        async with websockets.serve(handler, host, port):
            print(f"[WS] WebSocket server on ws://{host}:{port}")
            await asyncio.Future()
    asyncio.run(serve())
PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Jellectronica — Live Monitor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: #050510; color: #e0e0ff;
      font-family: 'Inter', system-ui, sans-serif;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      min-height: 100vh; overflow: hidden;
    }
    h1 {
      font-size: 1.1rem; font-weight: 300;
      letter-spacing: 0.25em; color: #88aaff;
      margin-bottom: 14px; text-transform: uppercase;
    }
    #video-wrap {
      position: relative;
      border: 1px solid rgba(80,120,255,0.2);
      border-radius: 10px; overflow: hidden;
      box-shadow: 0 0 60px rgba(80,120,255,0.12), 0 0 120px rgba(80,120,255,0.05);
    }
    #stream-img {
      display: block; width: 100%; height: auto;
      max-width: 960px; border-radius: 10px;
      background: #111;
    }
    #status {
      margin-top: 12px; font-size: 0.72rem;
      color: #445; letter-spacing: 0.1em;
    }
    #jf-count { color: #88aaff; font-weight: 600; }
    #ws-status { transition: color 0.3s; }
    #audio-badge {
      margin-top: 8px; font-size: 0.65rem;
      color: #4a5; letter-spacing: 0.08em;
      display: flex; align-items: center; gap: 6px;
    }
    #audio-dot {
      width: 6px; height: 6px; background: #4a5;
      border-radius: 50%;
      animation: pulse-dot 2.5s ease-in-out infinite;
    }
    #log {
      position: fixed; bottom: 14px; right: 14px;
      font-size: 0.6rem; color: #3a3a55;
      text-align: right; max-width: 200px;
      pointer-events: none;
      font-family: 'SF Mono', 'Fira Code', monospace;
    }
    #log div { opacity: 0; animation: log-in 0.3s ease forwards; }
    @keyframes log-in { to { opacity: 1; } }
  </style>
</head>
<body>
  <h1>🪼 Jellectronica 🪼</h1>
  <div id="video-wrap">
    <img id="stream-img" src="" alt="Jellectronica Stream">
  </div>
  <div id="status">
    Jellyfish: <span id="jf-count">—</span>
    &nbsp;·&nbsp; WS: <span id="ws-status">connecting…</span>
  </div>
  <div id="audio-badge">
    <div id="audio-dot"></div>
    Audio playing on board (SoftSynth)
  </div>
  <div id="log"></div>
  <script>
    // Lazy-load MJPEG stream after page load (prevents page load blocking)
    document.getElementById('stream-img').src = '/stream';
    const logEl = document.getElementById('log');
    const rowLabels = ['ARP', 'PAD', 'CHORD', 'BASS'];
    const rowColors = ['#f0c864','#64c8f0','#c878ff','#64ffb4'];
    function logEvent(name, row) {
      const d = document.createElement('div');
      d.textContent = rowLabels[row]+': '+name;
      d.style.color = rowColors[row] || '#888';
      logEl.prepend(d);
      while (logEl.children.length > 8) logEl.removeChild(logEl.lastChild);
    }
    function connectWS() {
      const wsUrl = 'ws://'+location.hostname+':'+(parseInt(location.port)+1);
      const ws = new WebSocket(wsUrl);
      document.getElementById('ws-status').textContent = 'connecting…';
      ws.onopen = () => {
        document.getElementById('ws-status').textContent = '✓ live';
        document.getElementById('ws-status').style.color = '#4f4';
      };
      ws.onmessage = (evt) => {
        const e = JSON.parse(evt.data);
        if (e.type === 'trigger') logEvent(e.noteName, e.row);
        else if (e.type === 'count') document.getElementById('jf-count').textContent = e.count;
      };
      ws.onclose = () => {
        document.getElementById('ws-status').textContent = 'reconnecting…';
        document.getElementById('ws-status').style.color = '#f84';
        setTimeout(connectWS, 2000);
      };
    }
    connectWS();
  </script>
</body>
</html>"""

@app.route("/")
def index():
    return PAGE_HTML

# ── Entry point ─────────────────────────────────────────────────
def main():
    global _music, _physics

    parser = argparse.ArgumentParser(description="Jellectronica — Native Server")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Video source (YouTube URL, file, or stream)")
    parser.add_argument("--model", default="../models/moon320.vmfb", help="Model path (.vmfb for NPU, .onnx for CPU)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=5002, help="HTTP port")
    parser.add_argument("--ws-port", type=int, default=5003, help="WebSocket port")
    parser.add_argument("--audio-driver", type=str, default=None, help="Audio driver (alsa, auto)")
    parser.add_argument("--alsa-device", type=str, default=None, help="ALSA device (e.g. hw:0,0)")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio")

    args = parser.parse_args()

    print("╔═══════════════════════════════════════════════════╗")
    print("║  🪼 Jellectronica — Native On-Device AI 🪼         ║")
    print("╚═══════════════════════════════════════════════════╝")
    print(f"  Video  : {args.video}")
    print(f"  Model  : {args.model}")
    print(f"  HTTP   : http://{args.host}:{args.port}")
    print(f"  WS     : ws://{args.host}:{args.ws_port}")


    # ── Initialize audio ──
    if not args.no_audio:
        print("\n[Audio] Initializing SoftSynth...")
        try:
            _music = MusicEngine(
                audio_driver=args.audio_driver,
                alsa_device=args.alsa_device,
            )
            _music.init()
            print("[Audio] ✓ Audio ready")
        except Exception as e:
            print(f"[Audio] ✗ Audio failed: {e} — continuing without audio")
            _music = None
    else:
        print("\n[Audio] Disabled (--no-audio)")



    # ── Physics engine ──
    _physics = PhysicsEngine(cooldown=10.0)

    # ── Start WebSocket server ──
    ws_thread = threading.Thread(
        target=run_ws_server, args=(args.host, args.ws_port), daemon=True)
    ws_thread.start()

    # ── Start inference loop ──
    inf_thread = threading.Thread(
        target=inference_loop, args=(args.video, args.model), daemon=True)
    inf_thread.start()

    # ── Start Flask HTTP server ──
    print(f"\n[Server] Starting at http://{args.host}:{args.port}")
    print(f"[Server] Open in your laptop browser to monitor\n")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
