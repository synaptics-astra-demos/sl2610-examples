# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Label loading, result persistence (JSON/image files), and console logging for run_cli.py."""
import json
import os


def load_labels(path):
    """Load labels.json: either an Ultralytics export ({"names": {...}}) or a flat id->name map."""
    if not path:
        return {}
    with open(path) as f:
        data = json.load(f)
    if "names" in data:
        return {str(k): v for k, v in data["names"].items()}
    return data


class RotatingJsonArrayWriter:
    def __init__(self, path, max_entries):
        self.path = path
        self.max_entries = max_entries
        self.rotated_path = self._build_rotated_path(path)
        self.file = None
        self.first_entry = True
        self.current_entries = 0
        self._open_new_file()

    @staticmethod
    def _build_rotated_path(path):
        base, ext = os.path.splitext(path)
        return f"{base}.1{ext or '.json'}"

    def _open_new_file(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self.file = open(self.path, "w", encoding="utf-8")
        self.file.write("[\n")
        self.file.flush()
        self.first_entry = True
        self.current_entries = 0

    def _close_current_file(self):
        if self.file is None:
            return
        if not self.first_entry:
            self.file.write("\n")
        self.file.write("]\n")
        self.file.flush()
        self.file.close()
        self.file = None

    def _rotate(self):
        self._close_current_file()
        if os.path.exists(self.rotated_path):
            os.remove(self.rotated_path)
        if os.path.exists(self.path):
            os.replace(self.path, self.rotated_path)
        self._open_new_file()

    def append(self, record):
        if self.current_entries >= self.max_entries:
            self._rotate()

        prefix = "" if self.first_entry else ",\n"
        self.file.write(prefix)
        self.file.write(json.dumps(record, separators=(",", ":")))
        self.file.flush()
        self.first_entry = False
        self.current_entries += 1

    def close(self):
        self._close_current_file()


def detection_to_record(label, conf, box):
    """Build the JSON-serializable record for one video-frame detection."""
    x1, y1, w_box, h_box = [float(v) for v in box]
    return {
        "label": label,
        "confidence": float(conf),
        "bounding_box": {
            "origin": {"x": int(round(x1)), "y": int(round(y1))},
            "size": {"x": int(round(w_box)), "y": int(round(h_box))},
        },
    }


def save_result_image(img, path):
    img.save(path)
    print(f"Result image saved to: {path}")
    return path


def log_image_detections(detections):
    print("\n[4/4] Detections:")
    if not detections:
        print("No objects detected.")
    for label, conf, box in detections:
        print(f"  {label:<15} Conf: {conf:.4f}  Box: {box.astype(int)}")


def log_video_frame(frame_count, infer_time_ms, detections, last_labels):
    """Print one live-updating status line. Returns this frame's labels, for the next call."""
    labels = [d[0] for d in detections]
    if set(labels) != set(last_labels):
        print("\n", end="", flush=True)  # objects changed — start a new line
    else:
        print("\r" + " " * 50 + "\r", end="", flush=True)  # clear the current line

    print(f"{frame_count} ({infer_time_ms:.3f} ms)", end="", flush=True)
    for label, conf, _box in detections:
        print(f" {label} {conf:.2f}", end="", flush=True)
    return labels


def log_video_summary(frame_count, output_path, json_path, json_writer, kept_detections):
    print(f"Done. Processed {frame_count} frames. Output: {output_path if output_path else 'not saved'}")
    print(
        f"Detection results saved to: {json_path} "
        f"(previous file: {json_writer.rotated_path if os.path.exists(json_writer.rotated_path) else 'none'})"
    )
    print(f"Kept the last {len(kept_detections)} detections in memory.")
