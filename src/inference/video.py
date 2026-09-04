import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from uuid import uuid4

import cv2
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "results" / "yolo" / "baseline" / "weights" / "best.pt"
IMAGE_SIZE = 640
FALLBACK_FPS = 30.0

CLASS_COLORS = {
    "person": (40, 180, 255),
    "tool": (255, 150, 40),
    "helmet": (50, 220, 50),
    "safety-vest": (0, 215, 255),
    "gloves": (220, 80, 220),
    "glasses": (255, 200, 80),
    "face-mask": (80, 120, 255),
}
DEFAULT_COLOR = (200, 200, 200)


def _create_video_writer(path, fps, frame_size):
    """Prefer browser-compatible H.264 and fall back to MPEG-4 Part 2."""
    candidates = []
    if sys.platform == "win32":
        candidates.append(("h264", cv2.CAP_MSMF, "H264"))
    candidates.extend(
        [
            ("h264", cv2.CAP_ANY, "avc1"),
            ("mp4v", cv2.CAP_ANY, "mp4v"),
        ]
    )

    for codec_name, backend, fourcc_name in candidates:
        writer = cv2.VideoWriter(
            str(path),
            backend,
            cv2.VideoWriter_fourcc(*fourcc_name),
            fps,
            frame_size,
        )
        if writer.isOpened():
            return writer, codec_name
        writer.release()

    raise RuntimeError(
        "OpenCV could not create an MP4 output with H.264 or mp4v."
    )


def _class_name(names, class_id):
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _extract_detections(result):
    """Convert an Ultralytics result into plain Python dictionaries."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    coordinates = boxes.xyxy.detach().cpu().tolist()
    class_ids = boxes.cls.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()

    detections = []
    for box, class_id, confidence in zip(
        coordinates,
        class_ids,
        confidences,
    ):
        class_id = int(class_id)
        detections.append(
            {
                "class_id": class_id,
                "class_name": _class_name(result.names, class_id),
                "confidence": float(confidence),
                "box": [float(value) for value in box],
            }
        )
    return detections


def _draw_detections(frame, detections):
    height, width = frame.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = (round(value) for value in detection["box"])
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        class_name = detection["class_name"]
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
        label = f"{class_name} {detection['confidence']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1,
        )
        label_top = max(0, y1 - text_height - baseline - 8)
        label_right = min(width - 1, x1 + text_width + 8)
        label_bottom = min(height - 1, label_top + text_height + baseline + 8)
        cv2.rectangle(
            frame,
            (x1, label_top),
            (label_right, label_bottom),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_bottom - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return frame


def process_video(
    input_path,
    output_path,
    confidence=0.30,
    frame_interval=5,
):
    """Annotate a video, running YOLO once every ``frame_interval`` frames."""
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if input_path == output_path:
        raise ValueError("Input and output paths must be different.")
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("Output path must use the .mp4 extension.")
    if not 0.0 < confidence <= 1.0:
        raise ValueError("confidence must be greater than 0 and at most 1.")
    if frame_interval < 1:
        raise ValueError("frame_interval must be at least 1.")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {MODEL_PATH}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(
        f".{output_path.stem}.{uuid4().hex}.processing.mp4"
    )

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open the input video: {input_path}")

    reported_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    reported_width = max(0, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    reported_height = max(0, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = source_fps if math.isfinite(source_fps) and source_fps > 0 else FALLBACK_FPS

    first_frame_read, first_frame = capture.read()
    if not first_frame_read or first_frame is None:
        capture.release()
        raise RuntimeError(f"The input video contains no readable frames: {input_path}")
    decoded_height, decoded_width = first_frame.shape[:2]
    width = reported_width or decoded_width
    height = reported_height or decoded_height

    writer = None
    completed = False
    started_at = time.perf_counter()
    try:
        device = 0 if torch.cuda.is_available() else "cpu"
        device_name = "cuda:0" if device == 0 else "cpu"
        model = YOLO(str(MODEL_PATH))

        writer, output_codec = _create_video_writer(
            temporary_output,
            fps,
            (width, height),
        )

        frame_index = 0
        inference_frames = 0
        detections_on_inference_frames = Counter()
        latest_detections = []
        frame = first_frame

        while frame is not None:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            if frame_index % frame_interval == 0:
                result = model.predict(
                    source=frame,
                    imgsz=IMAGE_SIZE,
                    conf=confidence,
                    device=device,
                    verbose=False,
                )[0]
                latest_detections = _extract_detections(result)
                inference_frames += 1
                detections_on_inference_frames.update(
                    detection["class_name"] for detection in latest_detections
                )

            writer.write(_draw_detections(frame, latest_detections))
            frame_index += 1

            frame_read, next_frame = capture.read()
            frame = next_frame if frame_read else None

        writer.release()
        writer = None
        temporary_output.replace(output_path)
        completed = True

        return {
            "frames": frame_index,
            "reported_frames": reported_frames,
            "inference_frames": inference_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "processing_seconds": time.perf_counter() - started_at,
            "device": device_name,
            "codec": output_codec,
            "browser_compatible": output_codec == "h264",
            "detections_on_inference_frames": dict(
                sorted(detections_on_inference_frames.items())
            ),
            "output_path": str(output_path),
        }
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not completed and temporary_output.exists():
            temporary_output.unlink()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate an industrial video with the trained YOLO11n model."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input video path")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Run inference every N frames",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = process_video(
        input_path=args.input,
        output_path=args.output,
        confidence=args.conf,
        frame_interval=args.interval,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
