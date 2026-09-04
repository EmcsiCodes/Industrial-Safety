import argparse
import random
from multiprocessing import freeze_support
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

from evaluate import CLASS_NAMES, decode_predictions
from model import SafetyNetDetector


DATA_ROOT = Path("data/processed/SH17_safety_1920")
IMAGE_SIZE = 416
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_ground_truth(image_path, label_dir):
    label_path = label_dir / f"{image_path.stem}.txt"
    boxes = []
    labels = []

    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])
            boxes.append(
                [
                    x - width / 2,
                    y - height / 2,
                    x + width / 2,
                    y + height / 2,
                ]
            )
            labels.append(class_id)

    if not boxes:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
        )
    return (
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def preprocess(image):
    resized = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )
    return (TF.to_tensor(resized) - 0.5) / 0.5


def load_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )
    model = SafetyNetDetector(num_classes=len(CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"Loaded: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', '?')}")
    print(f"Validation loss: {checkpoint.get('val_loss', '?')}")
    return model


def draw_box(draw, box, label, width, height, outline, score=None):
    x1 = int(box[0] * width)
    y1 = int(box[1] * height)
    x2 = int(box[2] * width)
    y2 = int(box[3] * height)
    draw.rectangle([x1, y1, x2, y2], outline=outline, width=2)
    caption = CLASS_NAMES[int(label)]
    if score is not None:
        caption += f" {score:.2f}"
    draw.text((x1 + 2, max(0, y1 - 12)), caption, fill=outline)


def create_visualization(
    original,
    ground_truth_boxes,
    ground_truth_labels,
    prediction,
    confidence,
):
    left = original.copy()
    right = original.copy()
    left_draw = ImageDraw.Draw(left)
    right_draw = ImageDraw.Draw(right)
    width, height = original.size

    for box, label in zip(ground_truth_boxes, ground_truth_labels):
        draw_box(
            left_draw,
            box.tolist(),
            label,
            width,
            height,
            outline="lime",
        )
    for box, label, score in zip(
        prediction["boxes"],
        prediction["labels"],
        prediction["scores"],
    ):
        draw_box(
            right_draw,
            box.tolist(),
            label,
            width,
            height,
            outline="red",
            score=float(score),
        )

    left_draw.text((10, 10), "GROUND TRUTH", fill="lime")
    right_draw.text(
        (10, 10),
        f"PREDICTIONS conf >= {confidence:.2f}",
        fill="red",
    )
    canvas = Image.new("RGB", (width * 2, height))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (width, 0))
    return canvas


def print_percentiles(title, values):
    percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
    results = np.percentile(np.asarray(values, dtype=float), percentiles)
    print(f"\n{title}")
    for percentile, value in zip(percentiles, results):
        print(f"  p{percentile:>3}: {value:.5f}")


def main(args):
    random.seed(args.seed)
    checkpoint_path = Path(args.checkpoint)
    image_dir = DATA_ROOT / "images" / args.split
    label_dir = DATA_ROOT / "labels" / args.split
    output_dir = checkpoint_path.parent.parent / "visualization" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SAFETYNET VISUAL DIAGNOSTIC")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Split: {args.split}")
    model = load_model(checkpoint_path)

    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    # Smoke training used the first 64 sorted training images.
    if args.pool_limit is not None:
        image_paths = image_paths[: args.pool_limit]

    usable_images = []
    for image_path in image_paths:
        ground_truth_boxes, _ = read_ground_truth(image_path, label_dir)
        if len(ground_truth_boxes) > 0:
            usable_images.append(image_path)
    if not usable_images:
        raise RuntimeError("No annotated images found in the selected image pool.")

    selected_images = random.sample(
        usable_images,
        min(args.num_images, len(usable_images)),
    )
    print(f"\nImage pool: {len(image_paths)}")
    print(f"Images selected: {len(selected_images)}")

    all_objectness = []
    all_class_confidence = []
    all_scores = []
    all_predicted_widths = []
    all_predicted_heights = []
    all_ground_truth_widths = []
    all_ground_truth_heights = []
    confidence_thresholds = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    prediction_counts = {threshold: 0 for threshold in confidence_thresholds}

    for index, image_path in enumerate(selected_images, start=1):
        original = Image.open(image_path).convert("RGB")
        tensor = preprocess(original).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(tensor)

        boxes_raw = torch.sigmoid(output[..., :4])
        objectness = torch.sigmoid(output[..., 4])
        class_probabilities = torch.softmax(output[..., 5:], dim=-1)
        class_confidence, _ = class_probabilities.max(dim=-1)
        final_scores = objectness * class_confidence

        all_objectness.extend(objectness.detach().cpu().flatten().tolist())
        all_class_confidence.extend(
            class_confidence.detach().cpu().flatten().tolist()
        )
        all_scores.extend(final_scores.detach().cpu().flatten().tolist())
        all_predicted_widths.extend(
            boxes_raw[..., 2].detach().cpu().flatten().tolist()
        )
        all_predicted_heights.extend(
            boxes_raw[..., 3].detach().cpu().flatten().tolist()
        )
        for threshold in confidence_thresholds:
            prediction_counts[threshold] += int((final_scores >= threshold).sum().item())

        ground_truth_boxes, ground_truth_labels = read_ground_truth(
            image_path,
            label_dir,
        )
        ground_truth_widths = ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]
        ground_truth_heights = ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
        all_ground_truth_widths.extend(ground_truth_widths.tolist())
        all_ground_truth_heights.extend(ground_truth_heights.tolist())

        decoded = decode_predictions(
            output,
            confidence_threshold=args.conf,
            nms_iou_threshold=0.50,
        )[0]
        canvas = create_visualization(
            original,
            ground_truth_boxes,
            ground_truth_labels,
            decoded,
            args.conf,
        )
        filename = f"{index:02d}_{image_path.stem}.jpg"
        canvas.save(output_dir / filename, quality=90)
        print(
            f"{index:02d}/{len(selected_images)}  {image_path.name}: "
            f"{len(ground_truth_boxes)} GT, {len(decoded['boxes'])} predictions"
        )

    print("\n" + "=" * 70)
    print("RAW OUTPUT STATISTICS")
    print("=" * 70)
    print_percentiles("OBJECTNESS", all_objectness)
    print_percentiles("CLASS CONFIDENCE", all_class_confidence)
    print_percentiles("FINAL SCORE (objectness x class confidence)", all_scores)
    print_percentiles("PREDICTED WIDTH", all_predicted_widths)
    print_percentiles("GROUND-TRUTH WIDTH", all_ground_truth_widths)
    print_percentiles("PREDICTED HEIGHT", all_predicted_heights)
    print_percentiles("GROUND-TRUTH HEIGHT", all_ground_truth_heights)

    print("\nRAW PREDICTIONS ABOVE CONFIDENCE THRESHOLDS")
    image_count = len(selected_images)
    for threshold, count in prediction_counts.items():
        print(
            f"  >= {threshold:.2f}: {count:6d} total "
            f"({count / image_count:.1f}/image)"
        )
    print("\nVisualizations saved to:")
    print(output_dir.resolve())


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SafetyNet predictions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--pool-limit", type=int)
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    freeze_support()
    main(parse_args())
