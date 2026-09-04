import csv
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


RAW_ROOT = Path("data/raw/SH17")
RAW_IMAGES = RAW_ROOT / "images"
RAW_LABELS = RAW_ROOT / "labels"
TRAIN_FILE = RAW_ROOT / "train_files.txt"
VAL_FILE = RAW_ROOT / "val_files.txt"

# This is separate from the older hard-linked dataset because those links may
# share storage with the raw images.
OUTPUT_ROOT = Path("data/processed/SH17_safety_1920")
OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_LABELS = OUTPUT_ROOT / "labels"
MAX_SIDE = 1920
JPEG_QUALITY = 90

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

# Original SH17 class ID -> selected dataset class ID.
CLASS_MAPPING = {
    0: 0,
    7: 1,
    10: 2,
    16: 3,
    9: 4,
    8: 5,
    5: 6,
}
CLASS_NAMES = {
    0: "person",
    1: "tool",
    2: "helmet",
    3: "safety-vest",
    4: "gloves",
    5: "glasses",
    6: "face-mask",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_split_file(path):
    """Read a SH17 split file and return its image stems."""
    return {
        Path(line.strip().replace("\\", "/")).stem
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def build_image_index():
    """Map each image stem to its raw image path."""
    index = {}
    for image_path in RAW_IMAGES.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if image_path.stem in index:
            raise RuntimeError(f"Duplicate image stem: {image_path.stem}")
        index[image_path.stem] = image_path
    return index


def build_label_index():
    """Map each image stem to its raw YOLO label path."""
    index = {}
    for label_path in RAW_LABELS.rglob("*.txt"):
        if label_path.stem in index:
            raise RuntimeError(f"Duplicate label stem: {label_path.stem}")
        index[label_path.stem] = label_path
    return index


def resize_or_copy_image(source, destination):
    """Downscale one image to MAX_SIDE without upscaling or changing its ratio."""
    with Image.open(source) as image:
        original_size = image.size
        largest_side = max(original_size)
        if largest_side <= MAX_SIDE:
            shutil.copy2(source, destination)
            return False, original_size, original_size

        scale = MAX_SIDE / largest_side
        output_size = (
            round(original_size[0] * scale),
            round(original_size[1] * scale),
        )
        resized_image = image.convert("RGB").resize(
            output_size,
            Image.Resampling.LANCZOS,
        )
        suffix = destination.suffix.lower()
        if suffix in {".jpg", ".jpeg"}:
            resized_image.save(destination, quality=JPEG_QUALITY, optimize=True)
        elif suffix == ".png":
            resized_image.save(destination, optimize=True)
        elif suffix == ".webp":
            resized_image.save(destination, quality=JPEG_QUALITY)
        else:
            resized_image.save(destination)
        return True, original_size, output_size


def filter_label_file(source_label, destination_label):
    """Keep the seven selected classes and remap their IDs."""
    output_lines = []
    counts = Counter()
    if source_label is not None:
        for line in source_label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"Invalid annotation in {source_label}:\n{line}")
            old_class_id = int(parts[0])
            if old_class_id not in CLASS_MAPPING:
                continue
            new_class_id = CLASS_MAPPING[old_class_id]
            # Resizing needs no label adjustment because YOLO xywh is normalized.
            output_lines.append(" ".join([str(new_class_id), *parts[1:]]))
            counts[new_class_id] += 1

    destination_label.write_text(
        "\n".join(output_lines) + ("\n" if output_lines else ""),
        encoding="utf-8",
    )
    return counts


def create_directories():
    for directory in (
        OUTPUT_IMAGES / "train",
        OUTPUT_IMAGES / "val",
        OUTPUT_LABELS / "train",
        OUTPUT_LABELS / "val",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def process_split(split_name, stems, image_index, label_index):
    image_dir = OUTPUT_IMAGES / split_name
    label_dir = OUTPUT_LABELS / split_name
    class_counts = Counter()
    resized_count = 0
    unchanged_count = 0
    negative_count = 0
    original_pixels = 0
    output_pixels = 0

    print(f"\nProcessing {split_name}...")
    for number, stem in enumerate(sorted(stems), start=1):
        source_image = image_index.get(stem)
        if source_image is None:
            raise FileNotFoundError(f"Image missing for {stem}")

        resized, original_size, output_size = resize_or_copy_image(
            source_image,
            image_dir / source_image.name,
        )
        if resized:
            resized_count += 1
        else:
            unchanged_count += 1
        original_pixels += original_size[0] * original_size[1]
        output_pixels += output_size[0] * output_size[1]

        counts = filter_label_file(
            label_index.get(stem),
            label_dir / f"{stem}.txt",
        )
        class_counts.update(counts)
        if sum(counts.values()) == 0:
            negative_count += 1
        if number % 250 == 0 or number == len(stems):
            print(f"  {number:4d} / {len(stems)}")

    return {
        "images": len(stems),
        "resized": resized_count,
        "unchanged": unchanged_count,
        "negative": negative_count,
        "class_counts": class_counts,
        "original_pixels": original_pixels,
        "output_pixels": output_pixels,
    }


def create_yaml():
    yaml_path = OUTPUT_ROOT / "data.yaml"
    data = {
        "path": str(OUTPUT_ROOT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }
    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
    return yaml_path


def create_class_mapping_csv():
    with (OUTPUT_ROOT / "class_mapping.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["original_id", "original_name", "new_id", "new_name"])
        for old_id, new_id in CLASS_MAPPING.items():
            writer.writerow(
                [old_id, SH17_CLASSES[old_id], new_id, CLASS_NAMES[new_id]]
            )


def verify_dataset(train_stems, val_stems):
    print("\nVerifying output...")
    errors = []
    for split_name, stems in (("train", train_stems), ("val", val_stems)):
        image_dir = OUTPUT_IMAGES / split_name
        label_dir = OUTPUT_LABELS / split_name
        images = [
            path
            for path in image_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        labels = list(label_dir.glob("*.txt"))
        if len(images) != len(stems):
            errors.append(
                f"{split_name}: expected {len(stems)} images, found {len(images)}"
            )
        if len(labels) != len(stems):
            errors.append(
                f"{split_name}: expected {len(stems)} labels, found {len(labels)}"
            )

        for label_path in labels:
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    class_id = int(line.split()[0])
                    if class_id not in CLASS_NAMES:
                        errors.append(f"Bad class ID {class_id} in {label_path}")

    if errors:
        print("\nVerification FAILED:")
        for error in errors:
            print(f"  - {error}")
        raise RuntimeError("Dataset verification failed.")
    print("Verification successful.")


def create_summary(train_results, val_results):
    train_counts = train_results["class_counts"]
    val_counts = val_results["class_counts"]
    original_pixels = (
        train_results["original_pixels"] + val_results["original_pixels"]
    )
    output_pixels = train_results["output_pixels"] + val_results["output_pixels"]
    pixel_reduction = (1 - output_pixels / original_pixels) * 100
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
        f"  Resized:     {train_results['resized'] + val_results['resized']}",
        f"  Unchanged:   {train_results['unchanged'] + val_results['unchanged']}",
        f"  Pixel reduction: {pixel_reduction:.1f}%",
        "",
        "Negative/background images:",
        f"  Train:       {train_results['negative']}",
        f"  Validation:  {val_results['negative']}",
        "",
        "Annotations:",
        "",
        f"{'Class':15}{'Train':>10}{'Val':>10}{'Total':>10}",
        "-" * 45,
    ]
    for class_id, class_name in CLASS_NAMES.items():
        train_count = train_counts[class_id]
        val_count = val_counts[class_id]
        lines.append(
            f"{class_name:15}{train_count:>10}{val_count:>10}"
            f"{train_count + val_count:>10}"
        )

    summary = "\n".join(lines)
    print("\n" + summary)
    (OUTPUT_ROOT / "dataset_summary.txt").write_text(summary, encoding="utf-8")


def main():
    print("=" * 60)
    print("PREPARING RESIZED SH17 SAFETY DATASET")
    print("=" * 60)
    # Regeneration is deliberately explicit so existing processed data cannot
    # be overwritten accidentally.
    if OUTPUT_ROOT.exists():
        raise RuntimeError(
            f"\nOutput folder already exists:\n{OUTPUT_ROOT}\n\n"
            "Delete it manually if you want to regenerate the dataset."
        )

    train_stems = read_split_file(TRAIN_FILE)
    val_stems = read_split_file(VAL_FILE)
    overlap = train_stems & val_stems
    if overlap:
        raise RuntimeError(f"{len(overlap)} images are in both train and val.")
    print(f"\nTrain images: {len(train_stems)}")
    print(f"Validation images: {len(val_stems)}")

    print("\nIndexing raw images...")
    image_index = build_image_index()
    print(f"Images found: {len(image_index)}")
    print("Indexing labels...")
    label_index = build_label_index()
    print(f"Labels found: {len(label_index)}")

    create_directories()
    train_results = process_split("train", train_stems, image_index, label_index)
    val_results = process_split("val", val_stems, image_index, label_index)
    yaml_path = create_yaml()
    create_class_mapping_csv()
    verify_dataset(train_stems, val_stems)
    create_summary(train_results, val_results)

    print("\n" + "=" * 60)
    print("RESIZED DATASET READY")
    print("=" * 60)
    print(f"\nDataset:\n{OUTPUT_ROOT.resolve()}")
    print(f"\nYOLO YAML:\n{yaml_path.resolve()}")


if __name__ == "__main__":
    main()
