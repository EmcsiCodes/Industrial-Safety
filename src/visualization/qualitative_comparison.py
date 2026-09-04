import argparse
import random
import sys
from multiprocessing import freeze_support
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from ultralytics import YOLO


SCRATCH_DIR = Path(__file__).resolve().parents[1] / "scratch"
sys.path.insert(0, str(SCRATCH_DIR))

from evaluate import box_iou, decode_predictions  # noqa: E402
from model import SafetyNetDetector  # noqa: E402


DATA_ROOT = Path("data/processed/SH17_safety_1920")
IMAGE_DIR = DATA_ROOT / "images" / "val"
LABEL_DIR = DATA_ROOT / "labels" / "val"
YOLO_WEIGHTS = Path("results/yolo/baseline/weights/best.pt")
SAFETYNET_RUN_DIR = Path("results/scratch/v2")
SAFETYNET_CHECKPOINT_METRICS = (
    SAFETYNET_RUN_DIR / "evaluation" / "checkpoint_comparison.csv"
)
SAFETYNET_WEIGHTS_DIR = SAFETYNET_RUN_DIR / "weights"
OUTPUT_DIR = Path("results/final_figures/model_comparison/qualitative")

CLASS_NAMES = [
    "person",
    "tool",
    "helmet",
    "safety-vest",
    "gloves",
    "glasses",
    "face-mask",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAFETYNET_IMAGE_SIZE = 416
PANEL_MAX_SIZE = (720, 520)


def validate_inputs():
    required = [
        IMAGE_DIR,
        LABEL_DIR,
        YOLO_WEIGHTS,
        SAFETYNET_CHECKPOINT_METRICS,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path.resolve()}")


def read_ground_truth(image_path):
    label_path = LABEL_DIR / f"{image_path.stem}.txt"
    boxes = []
    labels = []
    areas = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])
            if class_id not in range(len(CLASS_NAMES)):
                continue
            boxes.append(
                [
                    x - width / 2,
                    y - height / 2,
                    x + width / 2,
                    y + height / 2,
                ]
            )
            labels.append(class_id)
            areas.append(width * height)
    return {
        "boxes": (
            torch.tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32)
        ),
        "labels": (
            torch.tensor(labels, dtype=torch.long)
            if labels
            else torch.zeros((0,), dtype=torch.long)
        ),
        "areas": areas,
    }


def build_image_records():
    records = []
    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for image_path in image_paths:
        target = read_ground_truth(image_path)
        if len(target["boxes"]):
            records.append({"image_path": image_path, "target": target})
    if not records:
        raise RuntimeError(f"No annotated validation images found in {IMAGE_DIR}")
    return records


def count_class(record, class_id):
    return int((record["target"]["labels"] == class_id).sum())


def select_representative_images(records, num_images, seed):
    categories = [
        (
            "easy_person",
            lambda record: count_class(record, 0) >= 1
            and len(record["target"]["labels"]) <= 3
            and any(
                label == 0 and area >= 0.12
                for label, area in zip(
                    record["target"]["labels"].tolist(),
                    record["target"]["areas"],
                )
            ),
        ),
        ("helmet", lambda record: count_class(record, 2) >= 1),
        (
            "glasses_or_face_mask",
            lambda record: count_class(record, 5) + count_class(record, 6) >= 1,
        ),
        ("tools", lambda record: count_class(record, 1) >= 1),
        ("multiple_workers", lambda record: count_class(record, 0) >= 3),
        (
            "small_ppe",
            lambda record: any(
                label in {2, 3, 4, 5, 6} and area <= 0.005
                for label, area in zip(
                    record["target"]["labels"].tolist(),
                    record["target"]["areas"],
                )
            ),
        ),
        ("cluttered_scene", lambda record: len(record["target"]["labels"]) >= 8),
        (
            "mixed_ppe",
            lambda record: len(
                set(record["target"]["labels"].tolist()) & {2, 3, 4, 5, 6}
            )
            >= 2,
        ),
    ]
    random_generator = random.Random(seed)
    selected = []
    selected_paths = set()

    for category, predicate in categories:
        if len(selected) >= num_images:
            break
        candidates = [
            record
            for record in records
            if record["image_path"] not in selected_paths and predicate(record)
        ]
        if not candidates:
            continue
        record = random_generator.choice(candidates)
        selected.append({**record, "category": category})
        selected_paths.add(record["image_path"])

    remaining = [record for record in records if record["image_path"] not in selected_paths]
    random_generator.shuffle(remaining)
    for record in remaining[: max(0, num_images - len(selected))]:
        selected.append({**record, "category": "deterministic_random"})
    return selected


def select_device(requested_device):
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def run_yolo(image_paths, confidence, device):
    model = YOLO(str(YOLO_WEIGHTS))
    model_names = [model.names[index] for index in sorted(model.names)]
    if model_names != CLASS_NAMES:
        raise RuntimeError(
            "YOLO class order does not match the prepared dataset.\n"
            f"Expected: {CLASS_NAMES}\nFound: {model_names}"
        )
    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=confidence,
        imgsz=640,
        device=0 if device.type == "cuda" else "cpu",
        verbose=False,
        save=False,
    )
    predictions = []
    for result in results:
        boxes = result.boxes
        predictions.append(
            {
                "boxes": boxes.xyxyn.detach().cpu(),
                "scores": boxes.conf.detach().cpu(),
                "labels": boxes.cls.detach().cpu().long(),
            }
        )
    del results, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions


def load_safetynet(device):
    metrics = pd.read_csv(SAFETYNET_CHECKPOINT_METRICS)
    required = {"checkpoint", "epoch", "mAP50-95"}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise RuntimeError(
            f"Missing columns in {SAFETYNET_CHECKPOINT_METRICS}:\n"
            + "\n".join(missing)
        )
    best_row = metrics.loc[metrics["mAP50-95"].idxmax()]
    expected_epoch = int(best_row["epoch"])
    best_map_path = SAFETYNET_WEIGHTS_DIR / "best_map.pt"
    checkpoint_path = (
        best_map_path
        if best_map_path.exists()
        else SAFETYNET_WEIGHTS_DIR / str(best_row["checkpoint"])
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best SafetyNet detection checkpoint not found: {checkpoint_path.resolve()}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    if checkpoint_epoch != expected_epoch:
        raise RuntimeError(
            f"{checkpoint_path} is epoch {checkpoint_epoch}, but stored evaluation "
            f"selects epoch {expected_epoch}."
        )
    model = SafetyNetDetector(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint_path, checkpoint_epoch


def run_safetynet(image_paths, confidence, device):
    model, checkpoint_path, checkpoint_epoch = load_safetynet(device)
    tensors = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = image.resize(
            (SAFETYNET_IMAGE_SIZE, SAFETYNET_IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )
        tensors.append((TF.to_tensor(image) - 0.5) / 0.5)
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        output = model(batch)
    predictions = decode_predictions(
        output,
        confidence_threshold=confidence,
        nms_iou_threshold=0.50,
    )
    del output, batch, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions, checkpoint_path, checkpoint_epoch


def count_matches(prediction, target, iou_threshold=0.5):
    matched = torch.zeros(len(target["boxes"]), dtype=torch.bool)
    true_positives = 0
    order = torch.argsort(prediction["scores"], descending=True)
    for prediction_index in order:
        class_id = prediction["labels"][prediction_index]
        available = (target["labels"] == class_id) & ~matched
        if not available.any():
            continue
        available_indices = torch.where(available)[0]
        ious = box_iou(
            prediction["boxes"][prediction_index].unsqueeze(0),
            target["boxes"][available_indices],
        ).squeeze(0)
        best_iou, best_position = torch.max(ious, dim=0)
        if float(best_iou) >= iou_threshold:
            matched[available_indices[best_position]] = True
            true_positives += 1
    false_positives = len(prediction["boxes"]) - true_positives
    missed = len(target["boxes"]) - true_positives
    return true_positives, false_positives, missed


def draw_detections(image, detections, color, show_scores):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_width = max(2, int(max(width, height) / 350))
    scores = detections.get("scores")
    for index, (box, label) in enumerate(
        zip(detections["boxes"], detections["labels"])
    ):
        x1 = max(0, min(width - 1, int(float(box[0]) * width)))
        y1 = max(0, min(height - 1, int(float(box[1]) * height)))
        x2 = max(0, min(width - 1, int(float(box[2]) * width)))
        y2 = max(0, min(height - 1, int(float(box[3]) * height)))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        caption = CLASS_NAMES[int(label)]
        if show_scores and scores is not None:
            caption += f" {float(scores[index]):.2f}"
        text_box = draw.textbbox((0, 0), caption)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_y = max(0, y1 - text_height - 6)
        draw.rectangle(
            [x1, label_y, x1 + text_width + 6, label_y + text_height + 6],
            fill=color,
        )
        draw.text((x1 + 3, label_y + 3), caption, fill="white")
    return image


def add_header(image, title):
    header_height = 36
    panel = Image.new("RGB", (image.width, image.height + header_height), "white")
    panel.paste(image, (0, header_height))
    draw = ImageDraw.Draw(panel)
    text_box = draw.textbbox((0, 0), title)
    text_width = text_box[2] - text_box[0]
    draw.text(((image.width - text_width) / 2, 11), title, fill="black")
    return panel


def create_comparison_image(
    image_path,
    target,
    yolo_prediction,
    safety_prediction,
    yolo_confidence,
    safety_confidence,
):
    original = Image.open(image_path).convert("RGB")
    original.thumbnail(PANEL_MAX_SIZE, Image.Resampling.LANCZOS)
    ground_truth = draw_detections(original.copy(), target, "green", show_scores=False)
    yolo = draw_detections(original.copy(), yolo_prediction, "blue", show_scores=True)
    safety = draw_detections(
        original.copy(),
        safety_prediction,
        "darkorange",
        show_scores=True,
    )
    panels = [
        add_header(ground_truth, "GROUND TRUTH"),
        add_header(yolo, f"YOLO11n (conf >= {yolo_confidence:.2f})"),
        add_header(safety, f"SAFETYNET v2 (conf >= {safety_confidence:.2f})"),
    ]
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
        "white",
    )
    x_offset = 0
    for panel in panels:
        canvas.paste(panel, (x_offset, 0))
        x_offset += panel.width
    return canvas


def main(args):
    if args.num_images < 1:
        raise ValueError("--num-images must be at least 1.")
    if not 0 <= args.yolo_conf <= 1 or not 0 <= args.safetynet_conf <= 1:
        raise ValueError("Confidence thresholds must be between 0 and 1.")
    validate_inputs()
    device = select_device(args.device)
    records = build_image_records()
    selected = select_representative_images(records, args.num_images, args.seed)
    image_paths = [record["image_path"] for record in selected]

    print(f"Device: {device}")
    print(f"Selected {len(selected)} of {len(records)} annotated validation images.")
    yolo_predictions = run_yolo(image_paths, args.yolo_conf, device)
    safety_predictions, checkpoint_path, checkpoint_epoch = run_safetynet(
        image_paths,
        args.safetynet_conf,
        device,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for index, (record, yolo_prediction, safety_prediction) in enumerate(
        zip(selected, yolo_predictions, safety_predictions),
        start=1,
    ):
        target = record["target"]
        filename = f"{index:02d}_{record['category']}_{record['image_path'].stem}.jpg"
        comparison = create_comparison_image(
            record["image_path"],
            target,
            yolo_prediction,
            safety_prediction,
            args.yolo_conf,
            args.safetynet_conf,
        )
        comparison.save(OUTPUT_DIR / filename, quality=92)
        yolo_tp, yolo_fp, yolo_missed = count_matches(yolo_prediction, target)
        safety_tp, safety_fp, safety_missed = count_matches(safety_prediction, target)
        manifest_rows.append(
            {
                "Filename": filename,
                "Source Image": record["image_path"].name,
                "Selection Category": record["category"],
                "Ground Truth Instances": len(target["boxes"]),
                "YOLO Detections": len(yolo_prediction["boxes"]),
                "YOLO True Positives": yolo_tp,
                "YOLO False Positives": yolo_fp,
                "YOLO Missed": yolo_missed,
                "SafetyNet Detections": len(safety_prediction["boxes"]),
                "SafetyNet True Positives": safety_tp,
                "SafetyNet False Positives": safety_fp,
                "SafetyNet Missed": safety_missed,
                "YOLO Confidence": args.yolo_conf,
                "SafetyNet Confidence": args.safetynet_conf,
                "SafetyNet Checkpoint": checkpoint_path.name,
                "SafetyNet Epoch": checkpoint_epoch,
            }
        )
        print(
            f"{index:02d}/{len(selected)} {record['category']}: "
            f"{record['image_path'].name}"
        )

    pd.DataFrame(manifest_rows).to_csv(
        OUTPUT_DIR / "qualitative_manifest.csv",
        index=False,
    )
    print(f"SafetyNet checkpoint: {checkpoint_path} (epoch {checkpoint_epoch})")
    print(f"Qualitative comparisons saved to {OUTPUT_DIR.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare YOLO11n and SafetyNet on the same validation images."
    )
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--yolo-conf", type=float, default=0.30)
    parser.add_argument("--safetynet-conf", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    freeze_support()
    main(parse_args())
