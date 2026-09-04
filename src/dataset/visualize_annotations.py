from pathlib import Path
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path("data/raw/SH17")

IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"

OUTPUT_DIR = Path("results/dataset_samples")
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


# Classes we currently want to inspect closely
INSPECT_CLASSES = [
    "person",
    "head",
    "hands",
    "tool",
    "glasses",
    "gloves",
    "helmet",
    "shoes",
    "face-mask",
    "safety-vest",
]


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# Maximum resolution used ONLY for presentation images.
# Original SH17 files are never modified.
MAX_DISPLAY_SIZE = (1400, 1000)

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# IMAGE INDEX
# ============================================================

def build_image_index():
    """
    Create:
        stem -> image path

    This avoids repeatedly searching the images folder.
    """

    image_index = {}

    for path in IMAGES_DIR.rglob("*"):

        if path.suffix.lower() in IMAGE_EXTENSIONS:
            image_index[path.stem] = path

    return image_index


# ============================================================
# READ YOLO ANNOTATIONS
# ============================================================

def read_annotations(label_path):
    """
    Read YOLO annotations:

        class_id x_center y_center width height

    All coordinates are normalized to [0, 1].
    """

    annotations = []

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():

        parts = line.split()

        if len(parts) != 5:
            continue

        try:

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])

            box_width = float(parts[3])
            box_height = float(parts[4])

        except ValueError:
            continue

        if class_id not in CLASS_NAMES:
            continue

        annotations.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],

            "x_center": x_center,
            "y_center": y_center,

            "width": box_width,
            "height": box_height,
        })

    return annotations


# ============================================================
# DRAW ANNOTATIONS
# ============================================================

def draw_annotations(
    image_path,
    annotations,
    target_class=None
):
    """
    Draw YOLO annotations on a resized COPY of the image.

    target_class:
        None -> draw all annotations
        name -> draw only this class

    The original dataset image is never changed.
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # Resize for visualization
    # --------------------------------------------------------

    image.thumbnail(
        MAX_DISPLAY_SIZE,
        Image.Resampling.LANCZOS
    )

    image_width, image_height = image.size

    draw = ImageDraw.Draw(image)

    line_width = max(
        2,
        int(
            max(
                image_width,
                image_height
            ) / 500
        )
    )

    # --------------------------------------------------------
    # Draw boxes
    # --------------------------------------------------------

    for annotation in annotations:

        class_name = annotation[
            "class_name"
        ]

        # For class-specific sheets,
        # ignore all unrelated classes.
        if (
            target_class is not None
            and class_name != target_class
        ):
            continue

        x_center = annotation[
            "x_center"
        ]

        y_center = annotation[
            "y_center"
        ]

        box_width = annotation[
            "width"
        ]

        box_height = annotation[
            "height"
        ]

        # Convert normalized YOLO values
        # into pixel coordinates.

        x1 = (
            x_center - box_width / 2
        ) * image_width

        y1 = (
            y_center - box_height / 2
        ) * image_height

        x2 = (
            x_center + box_width / 2
        ) * image_width

        y2 = (
            y_center + box_height / 2
        ) * image_height

        # Keep coordinates inside image
        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(
            image_width - 1,
            x2
        )

        y2 = min(
            image_height - 1,
            y2
        )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=line_width
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        text_box = draw.textbbox(
            (0, 0),
            class_name
        )

        text_width = (
            text_box[2] -
            text_box[0]
        )

        text_height = (
            text_box[3] -
            text_box[1]
        )

        label_y = max(
            0,
            y1 - text_height - 8
        )

        draw.rectangle(
            [
                x1,
                label_y,
                x1 + text_width + 8,
                label_y + text_height + 8
            ],
            fill="red"
        )

        draw.text(
            (
                x1 + 4,
                label_y + 4
            ),
            class_name,
            fill="white"
        )

    return image


# ============================================================
# BUILD DATASET INDEX BY CLASS
# ============================================================

def build_class_index(image_index):

    class_images = {
        class_name: []
        for class_name in CLASS_NAMES.values()
    }

    all_samples = []

    print(
        "Indexing SH17 annotations..."
    )

    for label_path in sorted(
        LABELS_DIR.glob("*.txt")
    ):

        stem = label_path.stem

        image_path = image_index.get(
            stem
        )

        if image_path is None:
            continue

        annotations = read_annotations(
            label_path
        )

        if not annotations:
            continue

        sample = (
            image_path,
            annotations
        )

        all_samples.append(
            sample
        )

        classes_here = {
            annotation["class_name"]
            for annotation in annotations
        }

        for class_name in classes_here:

            class_images[
                class_name
            ].append(
                sample
            )

    return class_images, all_samples


# ============================================================
# CLASS-SPECIFIC MONTAGES
# ============================================================

def create_class_sheets(
    class_images
):

    print(
        "Creating class sample sheets..."
    )

    for class_name in INSPECT_CLASSES:

        samples = class_images.get(
            class_name,
            []
        )

        if not samples:

            print(
                f"No samples found for "
                f"{class_name}"
            )

            continue

        selected = random.sample(
            samples,
            min(
                6,
                len(samples)
            )
        )

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(15, 9)
        )

        axes = axes.flatten()

        # Hide unused plots
        for axis in axes:
            axis.axis("off")

        for (
            axis,
            sample
        ) in zip(
            axes,
            selected
        ):

            image_path, annotations = sample

            annotated = draw_annotations(
                image_path,
                annotations,
                target_class=class_name
            )

            axis.imshow(
                np.asarray(
                    annotated,
                    dtype=np.uint8
                )
            )

            axis.axis("off")

        fig.suptitle(
            f"SH17 — {class_name} examples",
            fontsize=20
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.95]
        )

        output_path = (
            OUTPUT_DIR /
            f"{class_name}_samples.png"
        )

        plt.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"  Created: "
            f"{output_path.name}"
        )


# ============================================================
# GENERAL DATASET MONTAGE
# ============================================================

def create_overview(
    all_samples
):

    print(
        "Creating dataset overview..."
    )

    selected = random.sample(
        all_samples,
        min(
            9,
            len(all_samples)
        )
    )

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(16, 14)
    )

    axes = axes.flatten()

    for axis in axes:
        axis.axis("off")

    for (
        axis,
        sample
    ) in zip(
        axes,
        selected
    ):

        image_path, annotations = sample

        # General dataset overview:
        # show ALL annotation classes.
        annotated = draw_annotations(
            image_path,
            annotations,
            target_class=None
        )

        axis.imshow(
            np.asarray(
                annotated,
                dtype=np.uint8
            )
        )

        axis.axis("off")

    fig.suptitle(
        "SH17 Industrial Safety Dataset — Annotated Examples",
        fontsize=22
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    output_path = (
        OUTPUT_DIR /
        "dataset_examples.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  Created: "
        f"{output_path.name}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SH17 ANNOTATION VISUALIZATION")
    print("=" * 60)

    image_index = (
        build_image_index()
    )

    print(
        f"\nImages indexed: "
        f"{len(image_index)}"
    )

    (
        class_images,
        all_samples
    ) = build_class_index(
        image_index
    )

    print(
        f"Annotated samples: "
        f"{len(all_samples)}"
    )

    print()

    create_class_sheets(
        class_images
    )

    print()

    create_overview(
        all_samples
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "VISUALIZATION COMPLETE"
    )

    print("=" * 60)

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()