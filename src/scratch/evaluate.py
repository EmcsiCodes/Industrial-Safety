import time
from multiprocessing import freeze_support
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import SafetyNetDetector


DATA_ROOT = Path("data/processed/SH17_safety_1920")
RUN_DIR = Path("results/scratch/v2")
WEIGHTS_DIR = RUN_DIR / "weights"
OUTPUT_DIR = RUN_DIR / "evaluation"

CLASS_NAMES = [
    "person",
    "tool",
    "helmet",
    "safety-vest",
    "gloves",
    "glasses",
    "face-mask",
]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 416
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# AP needs low-confidence predictions across the confidence range.
PREDICTION_THRESHOLD = 0.01
NMS_IOU_THRESHOLD = 0.50
MAX_DETECTIONS = 300
IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DetectionEvaluationDataset(Dataset):
    """Validation images paired with their original, unencoded annotations."""

    def __init__(self, root, split="val", image_size=416):
        self.root = Path(root)
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.image_size = image_size
        self.image_paths = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        print(f"Validation images: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def _read_labels(self, image_path):
        label_path = self.label_dir / f"{image_path.stem}.txt"
        boxes = []
        labels = []

        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                x, y, width, height = map(float, parts[1:])

                # Convert normalized YOLO xywh to normalized xyxy.
                boxes.append(
                    [
                        x - width / 2,
                        y - height / 2,
                        x + width / 2,
                        y + height / 2,
                    ]
                )
                labels.append(class_id)

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
        }

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )
        image = (TF.to_tensor(image) - 0.5) / 0.5
        return image, self._read_labels(image_path)


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def box_iou(boxes1, boxes2):
    """Calculate pairwise IoU for two sets of xyxy boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]),
            device=boxes1.device,
        )

    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    width_height = (bottom_right - top_left).clamp(min=0)
    intersection = width_height[..., 0] * width_height[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1e-9)


def nms(boxes, scores, iou_threshold):
    """Apply non-maximum suppression to one class of predictions."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = torch.argsort(scores, descending=True)
    keep = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious = box_iou(
            boxes[current].unsqueeze(0),
            boxes[remaining],
        ).squeeze(0)
        order = remaining[ious <= iou_threshold]
    return torch.stack(keep)


def decode_predictions(predictions, confidence_threshold, nms_iou_threshold):
    """Decode grid predictions into normalized boxes, scores, and class IDs."""
    batch_size = predictions.shape[0]
    grid_size = predictions.shape[1]
    box_values = torch.sigmoid(predictions[..., :4])
    objectness = torch.sigmoid(predictions[..., 4])
    class_probabilities = torch.softmax(predictions[..., 5:], dim=-1)
    class_confidence, class_ids = torch.max(class_probabilities, dim=-1)

    # Final detection confidence is P(object) * P(class | object).
    scores = objectness * class_confidence
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_size, device=predictions.device),
        torch.arange(grid_size, device=predictions.device),
        indexing="ij",
    )
    center_x = (grid_x[None] + box_values[..., 0]) / grid_size
    center_y = (grid_y[None] + box_values[..., 1]) / grid_size
    width = box_values[..., 2]
    height = box_values[..., 3]
    boxes = torch.stack(
        [
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ],
        dim=-1,
    ).clamp(0, 1)

    results = []
    for batch_index in range(batch_size):
        image_boxes = boxes[batch_index].reshape(-1, 4)
        image_scores = scores[batch_index].reshape(-1)
        image_classes = class_ids[batch_index].reshape(-1)
        mask = image_scores >= confidence_threshold
        image_boxes = image_boxes[mask]
        image_scores = image_scores[mask]
        image_classes = image_classes[mask]

        if image_boxes.numel() == 0:
            results.append(
                {
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.long),
                }
            )
            continue

        # NMS is class-aware so boxes from different classes do not suppress
        # one another.
        kept_indices = []
        for class_id in torch.unique(image_classes):
            indices = torch.where(image_classes == class_id)[0]
            class_keep = nms(
                image_boxes[indices],
                image_scores[indices],
                nms_iou_threshold,
            )
            kept_indices.append(indices[class_keep])

        kept_indices = torch.cat(kept_indices)
        order = torch.argsort(image_scores[kept_indices], descending=True)
        kept_indices = kept_indices[order][:MAX_DETECTIONS]
        results.append(
            {
                "boxes": image_boxes[kept_indices].detach().cpu(),
                "scores": image_scores[kept_indices].detach().cpu(),
                "labels": image_classes[kept_indices].detach().cpu(),
            }
        )
    return results


def calculate_ap(recall, precision):
    """Calculate COCO-style 101-point interpolated average precision."""
    if len(recall) == 0:
        return 0.0

    ap = 0.0
    for recall_threshold in np.linspace(0, 1, 101):
        valid = recall >= recall_threshold
        if np.any(valid):
            ap += np.max(precision[valid])
    return ap / 101


def evaluate_class_at_iou(
    predictions,
    ground_truths,
    class_id,
    iou_threshold,
):
    """Greedily match one class at one IoU threshold and calculate AP."""
    ground_truth_by_image = {}
    total_ground_truth = 0
    for image_index, target in enumerate(ground_truths):
        boxes = target["boxes"][target["labels"] == class_id]
        ground_truth_by_image[image_index] = boxes
        total_ground_truth += len(boxes)

    if total_ground_truth == 0:
        return {
            "ap": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "confidence": np.nan,
        }

    detections = []
    for image_index, prediction in enumerate(predictions):
        mask = prediction["labels"] == class_id
        for box, score in zip(
            prediction["boxes"][mask],
            prediction["scores"][mask],
        ):
            detections.append((float(score), image_index, box))
    detections.sort(key=lambda item: item[0], reverse=True)

    if not detections:
        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confidence": 0.0,
        }

    matched = {
        image_index: torch.zeros(len(boxes), dtype=torch.bool)
        for image_index, boxes in ground_truth_by_image.items()
    }
    true_positives = np.zeros(len(detections))
    false_positives = np.zeros(len(detections))
    confidences = np.zeros(len(detections))

    for detection_index, (score, image_index, predicted_box) in enumerate(detections):
        confidences[detection_index] = score
        ground_truth_boxes = ground_truth_by_image[image_index]
        if len(ground_truth_boxes) == 0:
            false_positives[detection_index] = 1
            continue

        available = ~matched[image_index]
        if not available.any():
            false_positives[detection_index] = 1
            continue

        available_indices = torch.where(available)[0]
        ious = box_iou(
            predicted_box.unsqueeze(0),
            ground_truth_boxes[available_indices],
        ).squeeze(0)
        best_iou, best_position = torch.max(ious, dim=0)
        if float(best_iou) >= iou_threshold:
            matched_index = available_indices[best_position]
            matched[image_index][matched_index] = True
            true_positives[detection_index] = 1
        else:
            false_positives[detection_index] = 1

    cumulative_true_positives = np.cumsum(true_positives)
    cumulative_false_positives = np.cumsum(false_positives)
    recall = cumulative_true_positives / total_ground_truth
    precision = cumulative_true_positives / np.maximum(
        cumulative_true_positives + cumulative_false_positives,
        1e-12,
    )
    ap = calculate_ap(recall, precision)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_index = int(np.argmax(f1))
    return {
        "ap": float(ap),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(f1[best_index]),
        "confidence": float(confidences[best_index]),
    }


def calculate_metrics(predictions, ground_truths):
    class_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        aps = [
            evaluate_class_at_iou(
                predictions,
                ground_truths,
                class_id,
                threshold,
            )["ap"]
            for threshold in IOU_THRESHOLDS
        ]
        result_50 = evaluate_class_at_iou(
            predictions,
            ground_truths,
            class_id,
            0.50,
        )
        ground_truth_count = sum(
            int((target["labels"] == class_id).sum())
            for target in ground_truths
        )
        class_rows.append(
            {
                "class_id": class_id,
                "class": class_name,
                "instances": ground_truth_count,
                "precision": result_50["precision"],
                "recall": result_50["recall"],
                "f1": result_50["f1"],
                "best_confidence": result_50["confidence"],
                "mAP50": aps[0],
                "mAP50-95": float(np.nanmean(aps)),
            }
        )

    class_df = pd.DataFrame(class_rows)
    overall = {
        "precision": class_df["precision"].mean(),
        "recall": class_df["recall"].mean(),
        "f1": class_df["f1"].mean(),
        "mAP50": class_df["mAP50"].mean(),
        "mAP50-95": class_df["mAP50-95"].mean(),
    }
    return class_df, overall


def load_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )
    model = SafetyNetDetector(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, checkpoint.get("epoch", -1), checkpoint.get("val_loss", np.nan)


def run_inference(model, loader):
    predictions = []
    ground_truths = []
    total_inference_time = 0.0
    image_count = 0

    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(loader, start=1):
            images = images.to(DEVICE)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = model(images)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            total_inference_time += time.perf_counter() - start

            decoded = decode_predictions(
                outputs,
                confidence_threshold=PREDICTION_THRESHOLD,
                nms_iou_threshold=NMS_IOU_THRESHOLD,
            )
            predictions.extend(decoded)
            ground_truths.extend(targets)
            image_count += len(images)
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(f"    {batch_index}/{len(loader)} batches")

    inference_ms = total_inference_time / image_count * 1000
    return predictions, ground_truths, inference_ms


def get_checkpoints():
    checkpoints = [
        WEIGHTS_DIR / name
        for name in ("best.pt", "last.pt")
        if (WEIGHTS_DIR / name).exists()
    ]
    checkpoints.extend(
        sorted(
            WEIGHTS_DIR.glob("epoch_*.pt"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
    )
    return checkpoints


def create_checkpoint_plot(comparison_df):
    plot_df = comparison_df.sort_values("epoch")
    x = np.arange(len(plot_df))
    width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, plot_df["mAP50"], width, label="mAP@0.5")
    plt.bar(x + width / 2, plot_df["mAP50-95"], width, label="mAP@0.5:0.95")
    labels = [
        f"{row.checkpoint}\nepoch {row.epoch}"
        for row in plot_df.itertuples()
    ]
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Average Precision")
    plt.title("SafetyNet Checkpoint Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "checkpoint_map_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def main():
    print("=" * 70)
    print("SAFETYNET DETECTOR - EVALUATION")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = DetectionEvaluationDataset(
        DATA_ROOT,
        split="val",
        image_size=IMAGE_SIZE,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=DEVICE.type == "cuda",
    )
    checkpoints = get_checkpoints()
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in:\n{WEIGHTS_DIR.resolve()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nCheckpoints:")
    for checkpoint in checkpoints:
        print(f"  - {checkpoint.name}")

    comparison_rows = []
    checkpoint_class_results = {}
    for checkpoint_path in checkpoints:
        print("\n" + "=" * 70)
        print(f"EVALUATING {checkpoint_path.name}")
        print("=" * 70)
        model, epoch, val_loss = load_model(checkpoint_path)
        print(f"Epoch: {epoch}")
        print(f"Saved validation loss: {val_loss:.4f}")
        print("Running inference...")
        predictions, ground_truths, inference_ms = run_inference(model, loader)
        print("Calculating detection metrics...")
        class_df, overall = calculate_metrics(predictions, ground_truths)
        checkpoint_class_results[checkpoint_path.name] = class_df
        comparison_rows.append(
            {
                "checkpoint": checkpoint_path.name,
                "epoch": epoch,
                "val_loss": val_loss,
                "precision": overall["precision"],
                "recall": overall["recall"],
                "f1": overall["f1"],
                "mAP50": overall["mAP50"],
                "mAP50-95": overall["mAP50-95"],
                "inference_ms_per_image": inference_ms,
            }
        )
        print("\nRESULT")
        print(f"Precision:   {overall['precision']:.4f}")
        print(f"Recall:      {overall['recall']:.4f}")
        print(f"F1:          {overall['f1']:.4f}")
        print(f"mAP50:       {overall['mAP50']:.4f}")
        print(f"mAP50-95:    {overall['mAP50-95']:.4f}")
        print(f"Inference:   {inference_ms:.2f} ms/image")

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(OUTPUT_DIR / "checkpoint_comparison.csv", index=False)
    best_index = comparison_df["mAP50-95"].idxmax()
    best_row = comparison_df.loc[best_index]
    best_checkpoint = best_row["checkpoint"]
    checkpoint_class_results[best_checkpoint].to_csv(
        OUTPUT_DIR / "best_per_class_metrics.csv",
        index=False,
    )
    create_checkpoint_plot(comparison_df)

    lines = [
        "SAFETYNET DETECTOR EVALUATION",
        "=" * 60,
        "",
        "Architecture:",
        "  Custom single-scale grid detector",
        "  Random initialization",
        "  26 x 26 prediction grid",
        "  One object per grid cell",
        "  7 classes",
        "",
        "Best checkpoint by mAP@0.5:0.95:",
        f"  {best_checkpoint}",
        f"  Epoch: {int(best_row['epoch'])}",
        "",
        "Best checkpoint metrics:",
        f"  Precision:   {best_row['precision']:.4f}",
        f"  Recall:      {best_row['recall']:.4f}",
        f"  F1:          {best_row['f1']:.4f}",
        f"  mAP@0.5:     {best_row['mAP50']:.4f}",
        f"  mAP@0.5:0.95: {best_row['mAP50-95']:.4f}",
        f"  Inference:   {best_row['inference_ms_per_image']:.2f} ms/image",
    ]
    summary = "\n".join(lines)
    (OUTPUT_DIR / "evaluation_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + "=" * 70)
    print(summary)
    print("\nResults saved to:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    freeze_support()
    main()
