from pathlib import Path
from collections import Counter, defaultdict

from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("data/raw/SH17")
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"

TRAIN_FILE = DATASET_ROOT / "train_files.txt"
VAL_FILE = DATASET_ROOT / "val_files.txt"

OUTPUT_DIR = Path("results/dataset_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


# ============================================================
# HELPERS
# ============================================================

def read_split(path):
    """Return image stems belonging to a split."""

    stems = set()

    for line in path.read_text(encoding="utf-8").splitlines():

        line = line.strip()

        if line:
            stems.add(
                Path(line.replace("\\", "/")).stem
            )

    return stems


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SH17 DATASET INSPECTION")
    print("=" * 60)

    train_stems = read_split(TRAIN_FILE)
    val_stems = read_split(VAL_FILE)

    image_files = sorted([
        path
        for path in IMAGES_DIR.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ])

    label_files = sorted(
        LABELS_DIR.rglob("*.txt")
    )

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    class_counts = Counter()

    train_counts = Counter()
    val_counts = Counter()

    images_per_class = Counter()

    bbox_areas = defaultdict(list)

    objects_per_image = []

    widths = []
    heights = []

    corrupt_images = 0
    invalid_annotations = 0

    # ========================================================
    # IMAGE INSPECTION
    # ========================================================

    print("\nReading images...")

    for image_path in image_files:

        try:

            with Image.open(image_path) as image:

                width, height = image.size

                widths.append(width)
                heights.append(height)

        except Exception:

            corrupt_images += 1

    # ========================================================
    # LABEL INSPECTION
    # ========================================================

    print("Reading labels...")

    for label_path in label_files:

        stem = label_path.stem

        lines = (
            label_path
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        )

        object_count = 0
        classes_in_image = set()

        for line in lines:

            parts = line.split()

            if len(parts) != 5:
                invalid_annotations += 1
                continue

            try:

                class_id = int(parts[0])

                x = float(parts[1])
                y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

            except ValueError:

                invalid_annotations += 1
                continue

            if class_id not in CLASS_NAMES:
                invalid_annotations += 1
                continue

            if not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                invalid_annotations += 1
                continue

            class_counts[class_id] += 1
            classes_in_image.add(class_id)

            bbox_areas[class_id].append(
                width * height
            )

            if stem in train_stems:
                train_counts[class_id] += 1

            elif stem in val_stems:
                val_counts[class_id] += 1

            object_count += 1

        objects_per_image.append(object_count)

        for class_id in classes_in_image:
            images_per_class[class_id] += 1

    # ========================================================
    # CREATE STATISTICS TABLE
    # ========================================================

    rows = []

    for class_id, class_name in CLASS_NAMES.items():

        areas = bbox_areas[class_id]

        median_area = (
            np.median(areas)
            if areas else 0
        )

        rows.append({
            "class_id": class_id,
            "class": class_name,

            "annotations":
                class_counts[class_id],

            "images":
                images_per_class[class_id],

            "train_annotations":
                train_counts[class_id],

            "val_annotations":
                val_counts[class_id],

            "median_bbox_area_percent":
                median_area * 100,
        })

    df = pd.DataFrame(rows)

    df.to_csv(
        OUTPUT_DIR / "class_statistics.csv",
        index=False
    )

    # ========================================================
    # 1. CLASS DISTRIBUTION
    # ========================================================

    plot_df = df.sort_values(
        "annotations",
        ascending=True
    )

    plt.figure(figsize=(11, 7))

    plt.barh(
        plot_df["class"],
        plot_df["annotations"]
    )

    plt.title(
        "SH17 Class Distribution"
    )

    plt.xlabel(
        "Number of annotations"
    )

    plt.ylabel("")

    save_plot(
        "class_distribution.png"
    )

    # ========================================================
    # 2. TRAIN / VALIDATION DISTRIBUTION
    # ========================================================

    plot_df = df.sort_values(
        "annotations",
        ascending=True
    )

    plt.figure(figsize=(11, 7))

    plt.barh(
        plot_df["class"],
        plot_df["train_annotations"],
        label="Train"
    )

    plt.barh(
        plot_df["class"],
        plot_df["val_annotations"],
        left=plot_df["train_annotations"],
        label="Validation"
    )

    plt.title(
        "Class Distribution by Dataset Split"
    )

    plt.xlabel(
        "Number of annotations"
    )

    plt.ylabel("")

    plt.legend()

    save_plot(
        "train_val_distribution.png"
    )

    # ========================================================
    # 3. BOUNDING BOX SIZE BY CLASS
    # ========================================================

    plot_df = df.sort_values(
        "median_bbox_area_percent",
        ascending=True
    )

    plt.figure(figsize=(11, 7))

    plt.barh(
        plot_df["class"],
        plot_df["median_bbox_area_percent"]
    )

    plt.title(
        "Median Bounding Box Size by Class"
    )

    plt.xlabel(
        "Median bounding box area (% of image)"
    )

    plt.ylabel("")

    save_plot(
        "bbox_size_by_class.png"
    )

    # ========================================================
    # 4. IMAGE RESOLUTION
    # ========================================================

    plt.figure(figsize=(9, 6))

    plt.scatter(
        widths,
        heights,
        alpha=0.25
    )

    plt.title(
        "SH17 Image Resolutions"
    )

    plt.xlabel(
        "Image width (pixels)"
    )

    plt.ylabel(
        "Image height (pixels)"
    )

    save_plot(
        "image_resolutions.png"
    )

    # ========================================================
    # 5. OBJECTS PER IMAGE
    # ========================================================

    plt.figure(figsize=(9, 6))

    plt.hist(
        objects_per_image,
        bins=35
    )

    plt.title(
        "Objects per Image"
    )

    plt.xlabel(
        "Number of annotated objects"
    )

    plt.ylabel(
        "Number of images"
    )

    save_plot(
        "objects_per_image.png"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    total_annotations = sum(
        class_counts.values()
    )

    summary = f"""
SH17 DATASET SUMMARY
====================

Images:
    {len(image_files)}

Annotations:
    {total_annotations}

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

    (
        OUTPUT_DIR /
        "dataset_summary.txt"
    ).write_text(
        summary,
        encoding="utf-8"
    )

    print("\n" + summary)

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()