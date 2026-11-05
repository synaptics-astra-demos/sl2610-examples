# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Shared camera-based face detection and recognition with optional Qt enrollment."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from torq.runtime import VMFBInferenceRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app_utils.camera import Camera, rotate_frame
from app_utils.display import enable_wayland_display
from app_utils.log import add_logging_args, configure_logging
from app_utils.torq_examples.utils.npu import configure_npu_userspace_frequency
from app_utils.torq_examples.utils.object_detection.draw import annotate_frame
from app_utils.torq_examples.utils.runtime import build_runtime_flags, cleanup_npu_after_inference
from app_utils.torq_examples.utils.video import RotatingJsonArrayWriter
from app_utils.paths import MODELS_DIR
from app_utils.torq_examples.Face_ID.setup_demo import download_face_id

from face_id.src.postprocess import decode_face_outputs
from face_id.src.keypoints import (
    decode_keypoints,
    map_keypoints_to_frame,
    preprocess_keypoint_roi,
)
from face_id.src.embedding import decode_embedding, preprocess_embedding
from face_id.src.recognition import FaceDatabase, RecognitionTracker
from face_id.src.box_smoother import BoxSmoother
from face_id.src.batch_enrollment import BatchEnrollmentSession
from face_id.src.live_status import LiveStatus


logger = logging.getLogger(__name__)

MODEL_WIDTH = 1280
MODEL_HEIGHT = 704
FACE_DETECTION_DIR = Path(__file__).resolve().parents[1]
FACE_DATABASE_PATH = FACE_DETECTION_DIR / "face_embeddings.json"
FACE_MODEL_DIR = MODELS_DIR / "Synaptics" / "face-id-torq"
FACE_MODEL_FILENAMES = (
    "face_detection.vmfb",
    "face_keypoint_static.vmfb",
    "face_embeddings_static.vmfb",
)


def ensure_face_models() -> dict[str, Path]:
    """Ensure the shared downloader has prepared the SL2610 model directory."""
    download_face_id(base_dir=MODELS_DIR)
    return {filename: FACE_MODEL_DIR / filename for filename in FACE_MODEL_FILENAMES}


@dataclass(frozen=True)
class AppConfig:
    """Configuration for board-side face detection and recognition."""

    model: str = str(FACE_MODEL_DIR / "face_detection.vmfb")
    keypoint_model: str = str(FACE_MODEL_DIR / "face_keypoint_static.vmfb")
    embedding_model: str = str(FACE_MODEL_DIR / "face_embeddings_static.vmfb")
    face_database: str = str(FACE_DATABASE_PATH)
    camera_device: str = "auto"
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    rtsp_url: str | None = None
    video: str | None = None
    image: str | None = None
    enroll: str | None = None
    remove: list[str] | None = None
    remove_all: bool = False
    tda: str = "dmabuf"
    device_io: bool = True
    device: str = "torq"
    runtime_flags: list[str] | None = None
    confidence_threshold: float = 0.6
    iou_threshold: float = 0.4
    box1_scale: float = 6.7147956
    box1_zero_point: int = -61
    box2_scale: float = 6.6746836
    box2_zero_point: int = -128
    score_scale: float = 1.0 / 256.0
    score_zero_point: int = -128
    keypoint_input_zero_point: int = -128
    keypoint_output_scale: float = 0.2108401
    keypoint_output_zero_point: int = -128
    embedding_output_scale: float = 0.1855205
    embedding_output_zero_point: int = -2
    match_threshold: float = 0.50
    match_hysteresis: float = 0.15
    match_margin: float = 0.06
    switch_confirm_frames: int = 2
    acquire_confirm_frames: int = 4
    recognition_debug: bool = os.environ.get("FACE_ID_RECOGNITION_DEBUG", "0") == "1"
    enroll_frames: int = 10
    recognition_window: int = 5
    recognition_hold_frames: int = 16
    display: bool = False
    raw_yuyv: bool = True
    rotate: int = 0
    display_sink: str = "waylandsink"
    output: str | None = None
    json_results: str = "face_detection_results.json"
    max_frames: int = 0
    logging: str = "INFO"
    log_file: Path | None = None


def parse_args(argv=None) -> AppConfig:
    parser = argparse.ArgumentParser(description="Face detection and recognition with optional Qt enrollment.")
    add_logging_args(parser)
    default_verbosity = os.environ.get(
        "FACE_ID_LOG_LEVEL", "DEBUG" if AppConfig.recognition_debug else "INFO",
    ).upper()
    if default_verbosity in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        parser.set_defaults(logging=default_verbosity)
    parser.add_argument(
        "--tda", choices=["cpu", "dmabuf"], default="dmabuf",
        help="Torq device allocator (default: %(default)s)",
    )
    parser.add_argument(
        "--device-io", action=argparse.BooleanOptionalAction, default=True,
        help="Use device-backed inference I/O (default: %(default)s)",
    )
    parser.add_argument(
        "--camera-device", default="auto",
        help="Camera device, e.g. /dev/video0, or 'auto' (default: %(default)s)",
    )
    parser.add_argument(
        "--image", default=None, metavar="PATH",
        help="Run on this image file instead of the camera, print the result and exit "
             "(headless run_cli.py only; useful for testing)",
    )
    parser.add_argument(
        "--enroll", default=None, metavar="NAME",
        help="Register the first face seen (camera or --image) under NAME, then exit "
             "(headless run_cli.py only)",
    )
    parser.add_argument(
        "--remove", nargs="+", default=None, metavar="NAME",
        help="Delete the stored embeddings of these enrolled names, then exit "
             "(headless run_cli.py only)",
    )
    parser.add_argument(
        "--remove-all", action="store_true",
        help="Delete every enrolled face, then exit (headless run_cli.py only)",
    )
    parser.add_argument(
        "--rotate", type=int, choices=[0, 90, 180, 270], default=0,
        help="Rotate camera feed (degrees clockwise, default: %(default)s)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after this many frames; 0 runs until interrupted (default: %(default)s)",
    )
    parser.add_argument(
        "--face-database", default=str(FACE_DATABASE_PATH),
        help="Face embedding database (default: %(default)s)",
    )
    parser.add_argument(
        "--json-results", default="face_detection_results.json",
        help="Detection results JSON path (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative")
    if args.enroll is not None and not args.enroll.strip():
        parser.error("--enroll needs a non-empty name")
    if args.enroll and (args.remove or args.remove_all):
        parser.error("--enroll cannot be combined with --remove or --remove-all")
    if args.remove and args.remove_all:
        parser.error("use either --remove NAME... or --remove-all, not both")
    return AppConfig(**vars(args))


def run_image_inference(args, process_fn):
    """Run the pipeline on one image file and log the usual status line once."""
    bgr_frame = cv2.imread(args.image)
    if bgr_frame is None:
        logger.error("Could not read image %s", args.image)
        raise SystemExit(1)
    bgr_frame = rotate_frame(bgr_frame, args.rotate)
    logger.info("Processing %s with Torq Python runtime...", args.image)
    # Recognition only settles after several consistent frames, so feed the same
    # image enough times for an enrolled face to be acquired.
    for _ in range(args.acquire_confirm_frames + 1):
        _annotated, frame_detections, infer_time, log_str = process_fn(bgr_frame)
    logger.info("%s (%.3f ms) %s", Path(args.image).name, infer_time, log_str)
    with closing(RotatingJsonArrayWriter(args.json_results, 60)) as json_writer:
        json_writer.append({"image": args.image, "detections": frame_detections})


ENROLL_TIMEOUT_SECONDS = 30


def run_remove(args) -> None:
    """Delete enrolled identities from the face database. Needs no models or camera."""
    database = FaceDatabase(args.face_database)
    enrolled = sorted(database.embeddings)
    names = enrolled if args.remove_all else args.remove
    missing = [name for name in names if name not in database.embeddings]
    if missing:
        logger.error(
            "Not enrolled: %s. Enrolled names: %s", ", ".join(missing),
            ", ".join(enrolled) or "(none)",
        )
        raise SystemExit(1)
    if not names:
        logger.info("[REMOVE] No enrolled faces in %s", database.path)
        return
    for name in names:
        database.remove(name)
        logger.info("[REMOVE] Removed '%s'", name)
    logger.info("[REMOVE] %d enrolled identit%s left in %s",
                len(database.embeddings), "y" if len(database.embeddings) == 1 else "ies", database.path)


def _enrollment_frames(args):
    """Yield BGR frames from --image (repeated) or the camera."""
    if args.image:
        bgr_frame = cv2.imread(args.image)
        if bgr_frame is None:
            logger.error("Could not read image %s", args.image)
            raise SystemExit(1)
        for _ in range(args.enroll_frames * 3):
            yield rotate_frame(bgr_frame, args.rotate)
        return
    with Camera(
        args.camera_device, width=args.camera_width,
        height=args.camera_height, fps=args.camera_fps,
    ) as camera:
        for bgr_frame in camera.stream():
            yield rotate_frame(bgr_frame, args.rotate)


def run_enrollment(args, process_fn, latest_embeddings, database) -> None:
    """Register the first face seen as ``args.enroll``: average several frames, save, exit."""
    name = args.enroll.strip()
    logger.info("[ENROLLMENT] Looking for a face to register as '%s' (%d frames)...",
                name, args.enroll_frames)
    vectors = []
    deadline = time.monotonic() + ENROLL_TIMEOUT_SECONDS
    frames = _enrollment_frames(args)
    try:
        for bgr_frame in frames:
            latest_embeddings.clear()
            process_fn(bgr_frame)
            if latest_embeddings:
                if len(latest_embeddings) > 1 and not vectors:
                    logger.warning("[ENROLLMENT] %d faces in view; using the first one",
                                   len(latest_embeddings))
                vectors.append(np.asarray(latest_embeddings[0], dtype=np.float32).reshape(-1))
                if len(vectors) >= args.enroll_frames:
                    break
            if time.monotonic() > deadline:
                break
    finally:
        frames.close()

    if len(vectors) < args.enroll_frames:
        logger.error("[ENROLLMENT] Only saw a face in %d of the %d frames needed within %ds; nothing saved",
                     len(vectors), args.enroll_frames, ENROLL_TIMEOUT_SECONDS)
        raise SystemExit(1)

    matched_name, score = database.match(np.mean(vectors, axis=0))
    if matched_name is not None and matched_name != name and score >= args.match_threshold:
        logger.error("[ENROLLMENT] This face is already registered as '%s' (similarity %.2f); nothing saved",
                     matched_name, score)
        raise SystemExit(1)
    updated = name in database.embeddings
    database.enroll_many(name, vectors)
    logger.info("[ENROLLMENT] Successfully %s '%s' with %d frames", "updated" if updated else "enrolled",
                name, len(vectors))


def run_camera_inference_loop(args, process_fn):
    with Camera(
        args.camera_device, width=args.camera_width,
        height=args.camera_height, fps=args.camera_fps,
    ) as camera, closing(RotatingJsonArrayWriter(args.json_results, 60)) as json_writer:
        logger.info("Processing %s with Torq Python runtime... Press Ctrl+C to stop.", camera.device)
        # One status line, updated in place; enrolment/removal messages still print on their own lines.
        with LiveStatus(logger) as status:
            try:
                for frame_count, bgr_frame in enumerate(camera.stream()):
                    bgr_frame = rotate_frame(bgr_frame, args.rotate)
                    _annotated, frame_detections, infer_time, log_str = process_fn(bgr_frame)
                    status.update(f"{frame_count} ({infer_time:.3f} ms)", log_str)
                    json_writer.append({"frame": frame_count, "detections": frame_detections})
                    if args.max_frames and frame_count + 1 >= args.max_frames:
                        break
            except KeyboardInterrupt:
                status.finish()
                logger.info("Interrupted by user.")


def preprocess_frame(
    bgr_frame: np.ndarray,
    raw_yuyv: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int, float]]:
    """Letterbox camera frame to model input, preserving aspect ratio."""
    gray_frame = raw_yuyv[:, :, 0] if raw_yuyv is not None else cv2.cvtColor(
        bgr_frame, cv2.COLOR_BGR2GRAY,
    )
    source_height, source_width = gray_frame.shape

    # Calculate scale to fit camera frame into model dimensions while preserving aspect ratio
    scale = min(MODEL_WIDTH / source_width, MODEL_HEIGHT / source_height)
    new_width = int(source_width * scale)
    new_height = int(source_height * scale)

    # Resize preserving aspect ratio
    resized = cv2.resize(gray_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Create letterboxed frame (pad to MODEL_WIDTH x MODEL_HEIGHT)
    letterboxed = np.full((MODEL_HEIGHT, MODEL_WIDTH), 0, dtype=np.uint8)
    offset_x = (MODEL_WIDTH - new_width) // 2
    offset_y = (MODEL_HEIGHT - new_height) // 2
    letterboxed[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized

    quantized = np.clip(letterboxed.astype(np.int16) - 128, -128, 127).astype(np.int8)
    return quantized[np.newaxis, :, :, np.newaxis], (offset_x, offset_y, new_width, new_height, scale)


def map_detections_to_frame(detections, letterbox_info, frame_shape):
    """Map model-space xywh detections back to original camera frame."""
    offset_x, offset_y, letterbox_width, letterbox_height, scale = letterbox_info
    frame_height, frame_width = frame_shape[:2]
    mapped = []
    for label, confidence, box in detections:
        x, y, width, height = box
        # Remove letterbox offset and scale back to original camera coordinates
        mapped_box = np.array([
            (x - offset_x) / scale,
            (y - offset_y) / scale,
            width / scale,
            height / scale,
        ], dtype=np.float32)
        # Clip to frame bounds
        mapped_box[[0, 2]] = np.clip(
            [mapped_box[0], mapped_box[0] + mapped_box[2]], 0, frame_width - 1,
        )
        mapped_box[[1, 3]] = np.clip(
            [mapped_box[1], mapped_box[1] + mapped_box[3]], 0, frame_height - 1,
        )
        mapped_box[2] -= mapped_box[0]
        mapped_box[3] -= mapped_box[1]
        mapped.append((label, confidence, mapped_box))
    return mapped


def run_model(runner, model_input, device_io):
    if device_io:
        model_input = runner.allocate_device_array(model_input)
    outputs = runner.infer([model_input])
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    return [output.to_host() if hasattr(output, "to_host") else output for output in outputs]


def make_process_fn(
    runner, args, keypoint_runner=None, embedding_runner=None, face_database=None,
    batch_session=None, qt_ui=None, embedding_sink=None,
):
    recognition_tracker = RecognitionTracker(
        args.recognition_window,
        hold_frames=args.recognition_hold_frames,
        hysteresis=args.match_hysteresis,
        switch_confirm_frames=args.switch_confirm_frames,
        acquire_confirm_frames=args.acquire_confirm_frames,
        min_margin=args.match_margin,
        debug=args.recognition_debug,
    )
    box_smoother = BoxSmoother()

    def process_fn(bgr_frame, raw_yuyv=None):
        recognition_tracker.begin_frame()
        recognized_names = []
        frame_embeddings = []
        model_input, letterbox_info = preprocess_frame(bgr_frame, raw_yuyv)
        outputs = run_model(runner, model_input, args.device_io)

        detections = decode_face_outputs(
            outputs,
            box1_scale=args.box1_scale,
            box1_zero_point=args.box1_zero_point,
            box2_scale=args.box2_scale,
            box2_zero_point=args.box2_zero_point,
            score_scale=args.score_scale,
            score_zero_point=args.score_zero_point,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
            image_width=MODEL_WIDTH,
            image_height=MODEL_HEIGHT,
        )
        detections = map_detections_to_frame(detections, letterbox_info, bgr_frame.shape)
        if batch_session is not None:
            batch_session.update_faces(detections)
            detections = batch_session.order_detections(detections)
        if not detections:
            recognition_tracker.clear()
            box_smoother.clear()
        # Smoothed boxes are only for the drawn overlay (and anything derived
        # from frame_detections, like the card thumbnail crop) - keypoint and
        # embedding extraction below still use the raw `detections` boxes.
        annotated, frame_detections = annotate_frame(
            bgr_frame, box_smoother.smooth(detections), show_labels=False,
        )
        if keypoint_runner is not None:
            batch_embeddings = []
            for _label, _confidence, box in detections:
                keypoint_input, roi = preprocess_keypoint_roi(
                    bgr_frame, box, args.keypoint_input_zero_point,
                )
                keypoint_outputs = run_model(keypoint_runner, keypoint_input, args.device_io)
                keypoints = decode_keypoints(
                    keypoint_outputs,
                    args.keypoint_output_scale,
                    args.keypoint_output_zero_point,
                )
                keypoints = map_keypoints_to_frame(keypoints, roi)
                if embedding_runner is not None:
                    embedding_input, landmarks = preprocess_embedding(bgr_frame, keypoints)
                    embedding_outputs = run_model(embedding_runner, embedding_input, args.device_io)
                    embedding = decode_embedding(
                        embedding_outputs[0],
                        args.embedding_output_scale,
                        args.embedding_output_zero_point,
                    )
                    frame_embeddings.append(embedding)
                    if batch_session is not None:
                        batch_embeddings.append(embedding)
                    if face_database is not None:
                        matched_name, similarity = recognition_tracker.update(
                            box, embedding, face_database, args.match_threshold,
                        )
                        recognition_text = matched_name if matched_name is not None else "unknown"
                        recognized_names.append(recognition_text)
                        if recognition_text:
                            cv2.putText(
                                annotated, recognition_text, (int(box[0]), max(int(box[1]) - 28, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
                            )
            if batch_session is not None:
                batch_session.update_preview_embeddings(batch_embeddings)
                batch_session.submit(batch_embeddings)
        if qt_ui is not None:
            qt_ui.update_source_frame(annotated, frame_detections)
        if embedding_sink is not None:
            embedding_sink(frame_embeddings)
        if not detections:
            user = "none"
        else:
            user = ",".join(recognized_names) or "unknown"
        log_str = f"faces: {len(detections)} user: {user}"
        progress = batch_session.capture_progress() if batch_session is not None else None
        if progress is not None:
            name, captured, needed = progress
            log_str += f" enrolling: {name or '?'} {captured}/{needed}"
        return annotated, frame_detections, runner.infer_time_ms, log_str

    return process_fn


def main(*, headless: bool = False) -> None:
    args = parse_args()
    if (args.image or args.enroll or args.remove or args.remove_all) and not headless:
        raise SystemExit("--image, --enroll, --remove and --remove-all are only supported by the headless run_cli.py")
    configure_logging(args.logging, args.log_file)
    if args.remove or args.remove_all:
        run_remove(args)
        return
    if args.enroll and not args.embedding_model:
        raise SystemExit("--enroll needs the face embedding model")
    ensure_face_models()
    runtime_flags = build_runtime_flags(args.tda, args.runtime_flags)

    if not headless:
        enable_wayland_display()

    ok, message = configure_npu_userspace_frequency("max")
    logger.info("[NPU] %s", message)

    runner = VMFBInferenceRunner(
        args.model,
        device_uri=args.device,
        function="main",
        load_model_to_mem=True,
        runtime_flags=runtime_flags,
        device_outputs=args.device_io,
    )
    keypoint_runner = None
    if args.keypoint_model:
        keypoint_runner = VMFBInferenceRunner(
            args.keypoint_model,
            device_uri=args.device,
            function="main",
            load_model_to_mem=True,
            runtime_flags=runtime_flags,
            device_outputs=args.device_io,
        )
    embedding_runner = None
    if args.embedding_model:
        embedding_runner = VMFBInferenceRunner(
            args.embedding_model,
            device_uri=args.device,
            function="main",
            load_model_to_mem=True,
            runtime_flags=runtime_flags,
            device_outputs=args.device_io,
        )
    face_database = FaceDatabase(args.face_database)
    batch_session = None
    qt_ui = None
    if not headless:
        from face_id.src.qt_ui import QtEnrollmentUI

        batch_session = BatchEnrollmentSession(face_database, args.enroll_frames)
        qt_ui = QtEnrollmentUI(batch_session)

    try:
        latest_embeddings = []
        process_fn = make_process_fn(
            runner, args, keypoint_runner, embedding_runner, face_database,
            batch_session, qt_ui,
            embedding_sink=(lambda embeddings: latest_embeddings.extend(embeddings)) if args.enroll else None,
        )
        if args.enroll:
            run_enrollment(args, process_fn, latest_embeddings, face_database)
        elif args.image:
            run_image_inference(args, process_fn)
        else:
            run_camera_inference_loop(args, process_fn)
    finally:
        if qt_ui is not None:
            qt_ui.stop()
        cleanup_npu_after_inference()


if __name__ == "__main__":
    main()