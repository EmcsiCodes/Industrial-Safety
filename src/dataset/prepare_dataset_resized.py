from pathlib import Path
from collections import Counter
import shutil
import csv

import yaml
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

RAW_ROOT = Path("data/raw/SH17")

RAW_IMAGES = RAW_ROOT / "images"
RAW_LABELS = RAW_ROOT / "labels"

TRAIN_FILE = RAW_ROOT / "train_files.txt"
VAL_FILE = RAW_ROOT / "val_files.txt"


# IMPORTANT:
# New folder. Do NOT modify SH17_safety because those images
# may be hard-linked to the raw dataset.
OUTPUT_ROOT = Path(
    "data/processed/SH17_safety_1920"
)

OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_LABELS = OUTPUT_ROOT / "labels"


# Maximum width OR height.
# Images smaller than this are NOT upscaled.
MAX_SIDE = 1920

JPEG_QUALITY = 90


# ============================================================
# ORIGINAL SH17 CLASSES
# ============================================================

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


# ============================================================
# OUR SELECTED CLASSES
# ============================================================

# original SH17 ID -> new ID
CLASS_MAPPING = {
    0: 0,     # person
    7: 1,     # tool
    10: 2,    # helmet
    16: 3,    # safety-vest
    9: 4,     # gloves
    8: 5,     # glasses
    5: 6,     # face-mask
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


# ============================================================
# HELPERS
# ============================================================

def read_split_file(path):
    """
    Read train_files.txt or val_files.txt and return image stems.
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
    image stem -> original image path
    """

    index = {}

    for image_path in RAW_IMAGES.rglob("*"):

        if (
            image_path.is_file()
            and image_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            if image_path.stem in index:

                raise RuntimeError(
                    f"Duplicate image stem: "
                    f"{image_path.stem}"
                )

            index[
                image_path.stem
            ] = image_path

    return index


def build_label_index():
    """
    image stem -> original YOLO label path
    """

    index = {}

    for label_path in RAW_LABELS.rglob(
        "*.txt"
    ):

        if label_path.stem in index:

            raise RuntimeError(
                f"Duplicate label stem: "
                f"{label_path.stem}"
            )

        index[
            label_path.stem
        ] = label_path

    return index


# ============================================================
# IMAGE RESIZING
# ============================================================

def resize_or_copy_image(
    source,
    destination
):
    """
    Downscale the image if its largest dimension exceeds MAX_SIDE.

    The aspect ratio is preserved.
    Images are NEVER upscaled.

    Returns:
        resized: bool
        original_size: (width, height)
        output_size: (width, height)
    """

    with Image.open(source) as image:

        original_width, original_height = (
            image.size
        )

        largest_side = max(
            original_width,
            original_height
        )

        # ----------------------------------------------------
        # Already small enough
        # ----------------------------------------------------

        if largest_side <= MAX_SIDE:

            shutil.copy2(
                source,
                destination
            )

            return (
                False,
                (original_width, original_height),
                (original_width, original_height),
            )

        # ----------------------------------------------------
        # Calculate new dimensions
        # ----------------------------------------------------

        scale = (
            MAX_SIDE / largest_side
        )

        new_width = round(
            original_width * scale
        )

        new_height = round(
            original_height * scale
        )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = image.convert("RGB")

        resized_image = image.resize(
            (
                new_width,
                new_height
            ),
            Image.Resampling.LANCZOS
        )

        suffix = (
            destination
            .suffix
            .lower()
        )

        # ----------------------------------------------------
        # Save according to format
        # ----------------------------------------------------

        if suffix in {
            ".jpg",
            ".jpeg"
        }:

            resized_image.save(
                destination,
                quality=JPEG_QUALITY,
                optimize=True
            )

        elif suffix == ".png":

            resized_image.save(
                destination,
                optimize=True
            )

        elif suffix == ".webp":

            resized_image.save(
                destination,
                quality=JPEG_QUALITY
            )

        else:
            resized_image.save(
                destination
            )

        return (
            True,
            (original_width, original_height),
            (new_width, new_height),
        )


# ============================================================
# LABEL FILTERING
# ============================================================

def filter_label_file(
    source_label,
    destination_label
):
    """
    Keep only our 7 selected classes.

    YOLO coordinates are normalized, so image resizing requires
    NO changes to bounding-box coordinates.
    """

    output_lines = []
    counts = Counter()

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
                    f"{source_label}:\n"
                    f"{line}"
                )

            old_class_id = int(
                parts[0]
            )

            # Ignore unwanted SH17 classes
            if old_class_id not in CLASS_MAPPING:
                continue

            new_class_id = (
                CLASS_MAPPING[
                    old_class_id
                ]
            )

            output_line = " ".join([
                str(new_class_id),
                parts[1],
                parts[2],
                parts[3],
                parts[4],
            ])

            output_lines.append(
                output_line
            )

            counts[
                new_class_id
            ] += 1

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
# DIRECTORY SETUP
# ============================================================

def create_directories():

    for directory in [
        OUTPUT_IMAGES / "train",
        OUTPUT_IMAGES / "val",
        OUTPUT_LABELS / "train",
        OUTPUT_LABELS / "val",
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    split_name,
    stems,
    image_index,
    label_index
):

    image_dir = (
        OUTPUT_IMAGES /
        split_name
    )

    label_dir = (
        OUTPUT_LABELS /
        split_name
    )

    class_counts = Counter()

    resized_count = 0
    unchanged_count = 0
    negative_count = 0

    original_pixels = 0
    output_pixels = 0


    print(
        f"\nProcessing "
        f"{split_name}..."
    )


    for number, stem in enumerate(
        sorted(stems),
        start=1
    ):

        source_image = (
            image_index.get(stem)
        )

        if source_image is None:

            raise FileNotFoundError(
                f"Image missing for "
                f"{stem}"
            )


        destination_image = (
            image_dir /
            source_image.name
        )


        # ----------------------------------------------------
        # Resize / copy image
        # ----------------------------------------------------

        (
            resized,
            original_size,
            output_size
        ) = resize_or_copy_image(
            source_image,
            destination_image
        )


        if resized:
            resized_count += 1
        else:
            unchanged_count += 1


        original_pixels += (
            original_size[0]
            *
            original_size[1]
        )

        output_pixels += (
            output_size[0]
            *
            output_size[1]
        )


        # ----------------------------------------------------
        # Process label
        # ----------------------------------------------------

        source_label = (
            label_index.get(stem)
        )

        destination_label = (
            label_dir /
            f"{stem}.txt"
        )


        counts = filter_label_file(
            source_label,
            destination_label
        )


        class_counts.update(
            counts
        )


        if sum(
            counts.values()
        ) == 0:

            negative_count += 1


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            number % 250 == 0
            or number == len(stems)
        ):

            print(
                f"  {number:4d} / "
                f"{len(stems)}"
            )


    return {
        "images": len(stems),

        "resized": resized_count,
        "unchanged": unchanged_count,
        "negative": negative_count,

        "class_counts": class_counts,

        "original_pixels":
            original_pixels,

        "output_pixels":
            output_pixels,
    }


# ============================================================
# YAML
# ============================================================

def create_yaml():

    data = {
        "path": str(
            OUTPUT_ROOT.resolve()
        ),

        "train":
            "images/train",

        "val":
            "images/val",

        "names":
            NEW_CLASS_NAMES,
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
            data,
            file,
            sort_keys=False
        )


    return yaml_path


# ============================================================
# CLASS MAPPING
# ============================================================

def create_class_mapping_csv():

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

        for (
            old_id,
            new_id
        ) in CLASS_MAPPING.items():

            writer.writerow([
                old_id,
                SH17_CLASSES[old_id],

                new_id,
                NEW_CLASS_NAMES[
                    new_id
                ],
            ])


# ============================================================
# VERIFY
# ============================================================

def verify_dataset(
    train_stems,
    val_stems
):

    print(
        "\nVerifying output..."
    )

    errors = []


    for (
        split_name,
        stems
    ) in [
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


        images = [
            p
            for p in image_dir.iterdir()
            if p.suffix.lower()
            in IMAGE_EXTENSIONS
        ]

        labels = list(
            label_dir.glob(
                "*.txt"
            )
        )


        if len(images) != len(stems):

            errors.append(
                f"{split_name}: "
                f"expected {len(stems)} images, "
                f"found {len(images)}"
            )


        if len(labels) != len(stems):

            errors.append(
                f"{split_name}: "
                f"expected {len(stems)} labels, "
                f"found {len(labels)}"
            )


        for label_path in labels:

            for line in (
                label_path
                .read_text(
                    encoding="utf-8"
                )
                .splitlines()
            ):

                if not line.strip():
                    continue

                class_id = int(
                    line.split()[0]
                )

                if class_id not in (
                    NEW_CLASS_NAMES
                ):

                    errors.append(
                        f"Bad class ID "
                        f"{class_id} in "
                        f"{label_path}"
                    )


    if errors:

        print(
            "\nVerification FAILED:"
        )

        for error in errors:
            print(
                f"  - {error}"
            )

        raise RuntimeError(
            "Dataset verification failed."
        )


    print(
        "Verification successful."
    )


# ============================================================
# SUMMARY
# ============================================================

def create_summary(
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


    original_pixels = (
        train_results[
            "original_pixels"
        ]
        +
        val_results[
            "original_pixels"
        ]
    )

    output_pixels = (
        train_results[
            "output_pixels"
        ]
        +
        val_results[
            "output_pixels"
        ]
    )


    pixel_reduction = (
        1
        -
        output_pixels
        / original_pixels
    ) * 100


    lines = [
        "SH17 SAFETY DATASET - RESIZED",
        "=" * 55,
        "",
        f"Maximum image side: {MAX_SIDE} px",
        f"JPEG quality:       {JPEG_QUALITY}",
        "",
        "Dataset split:",
        f"  Train:       {train_results['images']}",
        f"  Validation:  {val_results['images']}",
        "",
        "Image resizing:",
        (
            f"  Resized:     "
            f"{train_results['resized'] + val_results['resized']}"
        ),
        (
            f"  Unchanged:   "
            f"{train_results['unchanged'] + val_results['unchanged']}"
        ),
        (
            f"  Pixel reduction: "
            f"{pixel_reduction:.1f}%"
        ),
        "",
        "Negative/background images:",
        (
            f"  Train:       "
            f"{train_results['negative']}"
        ),
        (
            f"  Validation:  "
            f"{val_results['negative']}"
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
    ]


    for (
        class_id,
        class_name
    ) in NEW_CLASS_NAMES.items():

        train_count = (
            train_counts[
                class_id
            ]
        )

        val_count = (
            val_counts[
                class_id
            ]
        )

        lines.append(
            f"{class_name:15}"
            f"{train_count:>10}"
            f"{val_count:>10}"
            f"{train_count + val_count:>10}"
        )


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
    print("PREPARING RESIZED SH17 SAFETY DATASET")
    print("=" * 60)


    # Safety check:
    # don't accidentally overwrite an existing prepared dataset.

    if OUTPUT_ROOT.exists():

        raise RuntimeError(
            f"\nOutput folder already exists:\n"
            f"{OUTPUT_ROOT}\n\n"
            f"Delete it manually if you want "
            f"to regenerate the dataset."
        )


    train_stems = read_split_file(
        TRAIN_FILE
    )

    val_stems = read_split_file(
        VAL_FILE
    )


    overlap = (
        train_stems &
        val_stems
    )

    if overlap:

        raise RuntimeError(
            f"{len(overlap)} images are "
            f"in both train and val."
        )


    print(
        f"\nTrain images: "
        f"{len(train_stems)}"
    )

    print(
        f"Validation images: "
        f"{len(val_stems)}"
    )


    # --------------------------------------------------------
    # Index
    # --------------------------------------------------------

    print(
        "\nIndexing raw images..."
    )

    image_index = (
        build_image_index()
    )

    print(
        f"Images found: "
        f"{len(image_index)}"
    )


    print(
        "Indexing labels..."
    )

    label_index = (
        build_label_index()
    )

    print(
        f"Labels found: "
        f"{len(label_index)}"
    )


    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    create_directories()


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

    create_class_mapping_csv()


    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_dataset(
        train_stems,
        val_stems
    )


    create_summary(
        train_results,
        val_results
    )


    print(
        "\n" + "=" * 60
    )

    print(
        "RESIZED DATASET READY"
    )

    print("=" * 60)


    print(
        f"\nDataset:\n"
        f"{OUTPUT_ROOT.resolve()}"
    )

    print(
        f"\nYOLO YAML:\n"
        f"{yaml_path.resolve()}"
    )


if __name__ == "__main__":
    main()