from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


DATASET_ROOT = Path("data/raw/SH17")
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"
TRAIN_FILE = DATASET_ROOT / "train_files.txt"
VAL_FILE = DATASET_ROOT / "val_files.txt"
OUTPUT_DIR = Path("results/dataset_analysis")

CLASS_NAMES = {
    0: "person",
    1: "ear",
    2: "ear-mufs",
    3: "face",
    4: "face-guard",
    5: "face-mask",
    6: "foot",
    7: "tool",
    8: "glasses",
    9: "gloves",
    10: "helmet",
    11: "hands",
    12: "head",
    13: "medical-suit",
    14: "shoes",
    15: "safety-suit",
    16: "safety-vest",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_split(path):
    """Return the image stems listed in a dataset split."""
    return {
        Path(line.strip().replace("\\", "/")).stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 60)
    print("SH17 DATASET INSPECTION")
    print("=" * 60)

    train_stems = read_split(TRAIN_FILE)
    val_stems = read_split(VAL_FILE)
    image_files = sorted(
        path
        for path in IMAGES_DIR.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    label_files = sorted(LABELS_DIR.rglob("*.txt"))

    class_counts = Counter()
    train_counts = Counter()
    val_counts = Counter()
    images_per_class = Counter()
    bounding_box_areas = defaultdict(list)
    objects_per_image = []
    widths = []
    heights = []
    corrupt_images = 0
    invalid_annotations = 0

    print("\nReading images...")
    for image_path in image_files:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                widths.append(width)
                heights.append(height)
        except Exception:
            corrupt_images += 1

    print("Reading labels...")
    for label_path in label_files:
        object_count = 0
        classes_in_image = set()
        lines = label_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                invalid_annotations += 1
                continue
            try:
                class_id = int(parts[0])
                x, y, width, height = map(float, parts[1:])
            except ValueError:
                invalid_annotations += 1
                continue
            if class_id not in CLASS_NAMES or not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                invalid_annotations += 1
                continue

            class_counts[class_id] += 1
            classes_in_image.add(class_id)
            bounding_box_areas[class_id].append(width * height)
            if label_path.stem in train_stems:
                train_counts[class_id] += 1
            elif label_path.stem in val_stems:
                val_counts[class_id] += 1
            object_count += 1

        objects_per_image.append(object_count)
        for class_id in classes_in_image:
            images_per_class[class_id] += 1

    rows = []
    for class_id, class_name in CLASS_NAMES.items():
        areas = bounding_box_areas[class_id]
        rows.append(
            {
                "class_id": class_id,
                "class": class_name,
                "annotations": class_counts[class_id],
                "images": images_per_class[class_id],
                "train_annotations": train_counts[class_id],
                "val_annotations": val_counts[class_id],
                "median_bbox_area_percent": (
                    np.median(areas) * 100 if areas else 0
                ),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    statistics = pd.DataFrame(rows)
    statistics.to_csv(OUTPUT_DIR / "class_statistics.csv", index=False)

    plot_data = statistics.sort_values("annotations")
    plt.figure(figsize=(11, 7))
    plt.barh(plot_data["class"], plot_data["annotations"])
    plt.title("SH17 Class Distribution")
    plt.xlabel("Number of annotations")
    plt.ylabel("")
    save_plot("class_distribution.png")

    plt.figure(figsize=(11, 7))
    plt.barh(
        plot_data["class"],
        plot_data["train_annotations"],
        label="Train",
    )
    plt.barh(
        plot_data["class"],
        plot_data["val_annotations"],
        left=plot_data["train_annotations"],
        label="Validation",
    )
    plt.title("Class Distribution by Dataset Split")
    plt.xlabel("Number of annotations")
    plt.ylabel("")
    plt.legend()
    save_plot("train_val_distribution.png")

    plot_data = statistics.sort_values("median_bbox_area_percent")
    plt.figure(figsize=(11, 7))
    plt.barh(plot_data["class"], plot_data["median_bbox_area_percent"])
    plt.title("Median Bounding Box Size by Class")
    plt.xlabel("Median bounding box area (% of image)")
    plt.ylabel("")
    save_plot("bbox_size_by_class.png")

    plt.figure(figsize=(9, 6))
    plt.scatter(widths, heights, alpha=0.25)
    plt.title("SH17 Image Resolutions")
    plt.xlabel("Image width (pixels)")
    plt.ylabel("Image height (pixels)")
    save_plot("image_resolutions.png")

    plt.figure(figsize=(9, 6))
    plt.hist(objects_per_image, bins=35)
    plt.title("Objects per Image")
    plt.xlabel("Number of annotated objects")
    plt.ylabel("Number of images")
    save_plot("objects_per_image.png")

    summary = f"""
SH17 DATASET SUMMARY
====================

Images:
    {len(image_files)}

Annotations:
    {sum(class_counts.values())}

Classes:
    {len(CLASS_NAMES)}

Official split:
    Train:       {len(train_stems)}
    Validation:  {len(val_stems)}

Image resolution:
    Width:   {min(widths)} - {max(widths)} px
    Height:  {min(heights)} - {max(heights)} px

Data quality:
    Corrupt images:       {corrupt_images}
    Invalid annotations:  {invalid_annotations}

Average objects per image:
    {np.mean(objects_per_image):.2f}

Median objects per image:
    {np.median(objects_per_image):.0f}
""".strip()
    (OUTPUT_DIR / "dataset_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to:\n{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
