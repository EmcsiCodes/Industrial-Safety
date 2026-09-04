from pathlib import Path
from collections import Counter
import os
import shutil
import csv
import yaml


# ============================================================
# CONFIGURATION
# ============================================================

RAW_ROOT = Path("data/raw/SH17")

RAW_IMAGES = RAW_ROOT / "images"
RAW_LABELS = RAW_ROOT / "labels"

TRAIN_FILE = RAW_ROOT / "train_files.txt"
VAL_FILE = RAW_ROOT / "val_files.txt"


OUTPUT_ROOT = Path("data/processed/SH17_safety_unresized")

OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_LABELS = OUTPUT_ROOT / "labels"


# ============================================================
# SELECTED CLASSES
# ============================================================

# Original SH17 class IDs
SH17_CLASSES = {
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


# Mapping:
# original SH17 ID -> our new ID
CLASS_MAPPING = {
    0: 0,    # person
    7: 1,    # tool
    10: 2,   # helmet
    16: 3,   # safety-vest
    9: 4,    # gloves
    8: 5,    # glasses
    5: 6,    # face-mask
}


NEW_CLASS_NAMES = {
    0: "person",
    1: "tool",
    2: "helmet",
    3: "safety-vest",
    4: "gloves",
    5: "glasses",
    6: "face-mask",
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# "hardlink" saves disk space.
# If it fails, the script falls back to copying.
IMAGE_MODE = "hardlink"


# ============================================================
# HELPERS
# ============================================================

def read_split_file(path):
    """
    Read SH17 train_files.txt / val_files.txt
    and return the image stems.
    """

    stems = set()

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if not line:
            continue

        normalized = line.replace("\\", "/")

        stems.add(
            Path(normalized).stem
        )

    return stems


def build_image_index():
    """
    Map:
        image stem -> full image path

    Example:
        abc123 -> data/raw/SH17/images/abc123.jpg
    """

    index = {}

    for image_path in RAW_IMAGES.rglob("*"):

        if (
            image_path.is_file()
            and image_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            stem = image_path.stem

            if stem in index:

                raise RuntimeError(
                    f"Duplicate image stem found: {stem}\n"
                    f"{index[stem]}\n"
                    f"{image_path}"
                )

            index[stem] = image_path

    return index


def build_label_index():
    """
    Map:
        image stem -> YOLO label path
    """

    index = {}

    for label_path in RAW_LABELS.rglob("*.txt"):

        stem = label_path.stem

        if stem in index:

            raise RuntimeError(
                f"Duplicate label stem found: {stem}"
            )

        index[stem] = label_path

    return index


def create_image_link_or_copy(
    source,
    destination
):
    """
    Prefer hard links to avoid duplicating
    gigabytes of SH17 image data.

    If hard-link creation is unavailable,
    fall back to a normal file copy.
    """

    if destination.exists():
        return "existing"

    if IMAGE_MODE == "hardlink":

        try:

            os.link(
                source,
                destination
            )

            return "hardlink"

        except OSError:

            shutil.copy2(
                source,
                destination
            )

            return "copy"

    shutil.copy2(
        source,
        destination
    )

    return "copy"


def filter_label_file(
    source_label,
    destination_label
):
    """
    Keep only selected SH17 classes and
    remap their IDs.

    Bounding-box coordinates remain unchanged.

    Returns:
        Counter containing output annotation
        counts per NEW class ID.
    """

    counts = Counter()

    output_lines = []

    if source_label is not None:

        for line in source_label.read_text(
            encoding="utf-8"
        ).splitlines():

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:

                raise ValueError(
                    f"Invalid annotation in "
                    f"{source_label}:\n{line}"
                )

            old_class_id = int(
                parts[0]
            )

            # Discard classes we do not want
            if old_class_id not in CLASS_MAPPING:
                continue

            new_class_id = (
                CLASS_MAPPING[
                    old_class_id
                ]
            )

            # Keep the original YOLO coordinates
            new_line = " ".join([
                str(new_class_id),
                parts[1],
                parts[2],
                parts[3],
                parts[4],
            ])

            output_lines.append(
                new_line
            )

            counts[
                new_class_id
            ] += 1

    # Empty label files are intentional:
    # they represent negative/background images.

    destination_label.write_text(
        "\n".join(output_lines)
        + (
            "\n"
            if output_lines
            else ""
        ),
        encoding="utf-8"
    )

    return counts


# ============================================================
# CREATE OUTPUT STRUCTURE
# ============================================================

def prepare_directories():

    directories = [
        OUTPUT_IMAGES / "train",
        OUTPUT_IMAGES / "val",
        OUTPUT_LABELS / "train",
        OUTPUT_LABELS / "val",
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    split_name,
    stems,
    image_index,
    label_index
):

    image_output_dir = (
        OUTPUT_IMAGES /
        split_name
    )

    label_output_dir = (
        OUTPUT_LABELS /
        split_name
    )


    class_counts = Counter()

    images_processed = 0
    negative_images = 0

    hardlinks_created = 0
    files_copied = 0
    existing_images = 0


    print(
        f"\nProcessing {split_name} split..."
    )


    for number, stem in enumerate(
        sorted(stems),
        start=1
    ):

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        source_image = (
            image_index.get(stem)
        )

        if source_image is None:

            raise FileNotFoundError(
                f"No image found for split entry: "
                f"{stem}"
            )


        destination_image = (
            image_output_dir /
            source_image.name
        )


        method = create_image_link_or_copy(
            source_image,
            destination_image
        )


        if method == "hardlink":
            hardlinks_created += 1

        elif method == "copy":
            files_copied += 1

        else:
            existing_images += 1


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        source_label = (
            label_index.get(stem)
        )


        destination_label = (
            label_output_dir /
            f"{stem}.txt"
        )


        counts = filter_label_file(
            source_label,
            destination_label
        )


        class_counts.update(
            counts
        )


        if sum(counts.values()) == 0:
            negative_images += 1


        images_processed += 1


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            number % 500 == 0
            or number == len(stems)
        ):

            print(
                f"  {number}/{len(stems)} images"
            )


    return {
        "images": images_processed,
        "negative_images": negative_images,

        "class_counts": class_counts,

        "hardlinks": hardlinks_created,
        "copies": files_copied,
        "existing": existing_images,
    }


# ============================================================
# CREATE YAML
# ============================================================

def create_yaml():

    yaml_data = {
        "path": str(
            OUTPUT_ROOT.resolve()
        ),

        "train": "images/train",
        "val": "images/val",

        "names": NEW_CLASS_NAMES,
    }


    yaml_path = (
        OUTPUT_ROOT /
        "data.yaml"
    )


    with yaml_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        yaml.safe_dump(
            yaml_data,
            file,
            sort_keys=False
        )


    return yaml_path


# ============================================================
# SAVE CLASS MAPPING
# ============================================================

def save_class_mapping():

    output_path = (
        OUTPUT_ROOT /
        "class_mapping.csv"
    )


    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow([
            "original_id",
            "original_name",
            "new_id",
            "new_name",
        ])


        for old_id, new_id in (
            CLASS_MAPPING.items()
        ):

            writer.writerow([
                old_id,
                SH17_CLASSES[old_id],

                new_id,
                NEW_CLASS_NAMES[new_id],
            ])


# ============================================================
# VERIFY OUTPUT
# ============================================================

def verify_processed_dataset(
    train_stems,
    val_stems
):

    print(
        "\nVerifying processed dataset..."
    )


    errors = []


    for split_name, stems in [
        ("train", train_stems),
        ("val", val_stems),
    ]:

        image_dir = (
            OUTPUT_IMAGES /
            split_name
        )

        label_dir = (
            OUTPUT_LABELS /
            split_name
        )


        output_images = [
            path
            for path in image_dir.iterdir()
            if path.suffix.lower()
            in IMAGE_EXTENSIONS
        ]


        output_labels = list(
            label_dir.glob("*.txt")
        )


        if len(output_images) != len(stems):

            errors.append(
                f"{split_name}: expected "
                f"{len(stems)} images but found "
                f"{len(output_images)}"
            )


        if len(output_labels) != len(stems):

            errors.append(
                f"{split_name}: expected "
                f"{len(stems)} labels but found "
                f"{len(output_labels)}"
            )


        # Check class IDs
        for label_path in output_labels:

            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines():

                if not line.strip():
                    continue

                class_id = int(
                    line.split()[0]
                )

                if class_id not in (
                    NEW_CLASS_NAMES
                ):

                    errors.append(
                        f"Invalid class ID "
                        f"{class_id} in "
                        f"{label_path}"
                    )


    if errors:

        print(
            "\nVERIFICATION FAILED:"
        )

        for error in errors:
            print(
                f"  - {error}"
            )

        raise RuntimeError(
            "Processed dataset verification failed."
        )


    print(
        "Verification successful."
    )


# ============================================================
# SUMMARY
# ============================================================

def print_and_save_summary(
    train_results,
    val_results
):

    train_counts = (
        train_results[
            "class_counts"
        ]
    )

    val_counts = (
        val_results[
            "class_counts"
        ]
    )


    lines = [
        "SH17 SAFETY DATASET",
        "=" * 50,
        "",
        "Selected classes:",
    ]


    for class_id, class_name in (
        NEW_CLASS_NAMES.items()
    ):

        lines.append(
            f"  {class_id}: {class_name}"
        )


    lines.extend([
        "",
        "Dataset split:",
        (
            f"  Train images: "
            f"{train_results['images']}"
        ),
        (
            f"  Validation images: "
            f"{val_results['images']}"
        ),
        (
            f"  Total images: "
            f"{train_results['images'] + val_results['images']}"
        ),
        "",
        "Negative/background images:",
        (
            f"  Train: "
            f"{train_results['negative_images']}"
        ),
        (
            f"  Validation: "
            f"{val_results['negative_images']}"
        ),
        "",
        "Annotations:",
        "",
        (
            f"{'Class':15}"
            f"{'Train':>10}"
            f"{'Val':>10}"
            f"{'Total':>10}"
        ),
        "-" * 45,
    ])


    for class_id, class_name in (
        NEW_CLASS_NAMES.items()
    ):

        train_count = (
            train_counts[class_id]
        )

        val_count = (
            val_counts[class_id]
        )

        total = (
            train_count
            +
            val_count
        )


        lines.append(
            f"{class_name:15}"
            f"{train_count:>10}"
            f"{val_count:>10}"
            f"{total:>10}"
        )


    total_train_annotations = sum(
        train_counts.values()
    )

    total_val_annotations = sum(
        val_counts.values()
    )


    lines.extend([
        "-" * 45,
        (
            f"{'TOTAL':15}"
            f"{total_train_annotations:>10}"
            f"{total_val_annotations:>10}"
            f"{total_train_annotations + total_val_annotations:>10}"
        ),
        "",
        "Image storage:",
        (
            f"  Hard links created: "
            f"{train_results['hardlinks'] + val_results['hardlinks']}"
        ),
        (
            f"  Images copied: "
            f"{train_results['copies'] + val_results['copies']}"
        ),
    ])


    summary = "\n".join(
        lines
    )


    print(
        "\n" + summary
    )


    (
        OUTPUT_ROOT /
        "dataset_summary.txt"
    ).write_text(
        summary,
        encoding="utf-8"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PREPARING SH17 SAFETY DATASET")
    print("=" * 60)


    # --------------------------------------------------------
    # Read official split
    # --------------------------------------------------------

    train_stems = read_split_file(
        TRAIN_FILE
    )

    val_stems = read_split_file(
        VAL_FILE
    )


    print(
        f"\nTrain images:      "
        f"{len(train_stems)}"
    )

    print(
        f"Validation images: "
        f"{len(val_stems)}"
    )


    # Safety check:
    # no image should be in both splits.

    overlap = (
        train_stems &
        val_stems
    )


    if overlap:

        raise RuntimeError(
            f"{len(overlap)} images appear "
            f"in BOTH train and validation."
        )


    # --------------------------------------------------------
    # Index raw files
    # --------------------------------------------------------

    print(
        "\nIndexing raw images..."
    )

    image_index = (
        build_image_index()
    )


    print(
        f"Images indexed: "
        f"{len(image_index)}"
    )


    print(
        "Indexing raw labels..."
    )

    label_index = (
        build_label_index()
    )


    print(
        f"Labels indexed: "
        f"{len(label_index)}"
    )


    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    prepare_directories()


    # --------------------------------------------------------
    # Process splits
    # --------------------------------------------------------

    train_results = process_split(
        "train",
        train_stems,
        image_index,
        label_index
    )


    val_results = process_split(
        "val",
        val_stems,
        image_index,
        label_index
    )


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    yaml_path = create_yaml()

    save_class_mapping()


    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_processed_dataset(
        train_stems,
        val_stems
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_and_save_summary(
        train_results,
        val_results
    )


    print(
        "\n" + "=" * 60
    )

    print(
        "DATASET PREPARATION COMPLETE"
    )

    print("=" * 60)


    print(
        f"\nProcessed dataset:\n"
        f"{OUTPUT_ROOT.resolve()}"
    )


    print(
        f"\nYOLO configuration:\n"
        f"{yaml_path.resolve()}"
    )


if __name__ == "__main__":
    main()
