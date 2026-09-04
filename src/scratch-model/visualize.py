from pathlib import Path
from multiprocessing import freeze_support

import argparse
import random

import numpy as np
import torch

from PIL import (
    Image,
    ImageDraw,
)

import torchvision.transforms.functional as TF

from model import SafetyNetDetector

from evaluate import (
    decode_predictions,
    CLASS_NAMES,
)


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path(
    "data/processed/SH17_safety_1920"
)

IMAGE_SIZE = 416

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# GROUND TRUTH
# ============================================================

def read_ground_truth(
    image_path,
    label_dir,
):
    label_path = (
        label_dir
        /
        f"{image_path.stem}.txt"
    )


    boxes = []
    labels = []


    if not label_path.exists():

        return (
            torch.zeros(
                (0, 4),
                dtype=torch.float32,
            ),
            torch.zeros(
                (0,),
                dtype=torch.long,
            ),
        )


    for line in (
        label_path
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    ):

        line = line.strip()

        if not line:
            continue


        parts = line.split()

        if len(parts) != 5:
            continue


        class_id = int(parts[0])

        x = float(parts[1])
        y = float(parts[2])

        width = float(parts[3])
        height = float(parts[4])


        x1 = x - width / 2
        y1 = y - height / 2

        x2 = x + width / 2
        y2 = y + height / 2


        boxes.append([
            x1,
            y1,
            x2,
            y2,
        ])

        labels.append(
            class_id
        )


    if not boxes:

        return (
            torch.zeros(
                (0, 4),
                dtype=torch.float32,
            ),
            torch.zeros(
                (0,),
                dtype=torch.long,
            ),
        )


    return (
        torch.tensor(
            boxes,
            dtype=torch.float32,
        ),
        torch.tensor(
            labels,
            dtype=torch.long,
        ),
    )


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(image):

    resized = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        Image.Resampling.BILINEAR,
    )


    tensor = TF.to_tensor(
        resized
    )


    tensor = (
        tensor - 0.5
    ) / 0.5


    return tensor


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )


    model = SafetyNetDetector(
        num_classes=len(CLASS_NAMES)
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.to(DEVICE)
    model.eval()


    print(
        f"Loaded: {checkpoint_path}"
    )

    print(
        f"Epoch: "
        f"{checkpoint.get('epoch', '?')}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint.get('val_loss', '?')}"
    )


    return model


# ============================================================
# DRAW BOX
# ============================================================

def draw_box(
    draw,
    box,
    label,
    width,
    height,
    outline,
    score=None,
):

    x1 = int(box[0] * width)
    y1 = int(box[1] * height)

    x2 = int(box[2] * width)
    y2 = int(box[3] * height)


    draw.rectangle(
        [x1, y1, x2, y2],
        outline=outline,
        width=2,
    )


    caption = CLASS_NAMES[
        int(label)
    ]


    if score is not None:

        caption += (
            f" {score:.2f}"
        )


    draw.text(
        (
            x1 + 2,
            max(0, y1 - 12),
        ),
        caption,
        fill=outline,
    )


# ============================================================
# SIDE-BY-SIDE IMAGE
# ============================================================

def create_visualization(
    original,
    gt_boxes,
    gt_labels,
    prediction,
    confidence,
):

    left = original.copy()
    right = original.copy()


    left_draw = ImageDraw.Draw(
        left
    )

    right_draw = ImageDraw.Draw(
        right
    )


    width, height = original.size


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    for box, label in zip(
        gt_boxes,
        gt_labels,
    ):

        draw_box(
            left_draw,
            box.tolist(),
            label,
            width,
            height,
            outline="lime",
        )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

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


    left_draw.text(
        (10, 10),
        "GROUND TRUTH",
        fill="lime",
    )


    right_draw.text(
        (10, 10),
        f"PREDICTIONS conf >= {confidence:.2f}",
        fill="red",
    )


    canvas = Image.new(
        "RGB",
        (
            width * 2,
            height,
        ),
    )


    canvas.paste(
        left,
        (0, 0),
    )

    canvas.paste(
        right,
        (width, 0),
    )


    return canvas


# ============================================================
# PERCENTILES
# ============================================================

def print_percentiles(
    title,
    values,
):

    array = np.asarray(
        values,
        dtype=float,
    )


    percentiles = [
        0,
        25,
        50,
        75,
        90,
        95,
        99,
        100,
    ]


    results = np.percentile(
        array,
        percentiles,
    )


    print(
        f"\n{title}"
    )


    for percentile, value in zip(
        percentiles,
        results,
    ):

        print(
            f"  p{percentile:>3}: "
            f"{value:.5f}"
        )


# ============================================================
# MAIN
# ============================================================

def main(args):

    random.seed(
        args.seed
    )


    checkpoint_path = Path(
        args.checkpoint
    )


    image_dir = (
        DATA_ROOT
        /
        "images"
        /
        args.split
    )


    label_dir = (
        DATA_ROOT
        /
        "labels"
        /
        args.split
    )


    experiment_name = (
        checkpoint_path
        .parent
        .parent
        .name
    )


    output_dir = (
        checkpoint_path
        .parent
        .parent
        /
        "visualization"
        /
        args.split
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    print("=" * 70)
    print("SAFETYNET VISUAL DIAGNOSTIC")
    print("=" * 70)


    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Split: {args.split}"
    )


    model = load_model(
        checkpoint_path
    )


    # ========================================================
    # IMAGE POOL
    # ========================================================

    image_paths = sorted([
        path
        for path in image_dir.iterdir()
        if path.suffix.lower()
        in IMAGE_EXTENSIONS
    ])


    # Critical for smoke test:
    # training used the FIRST 64 sorted training images.
    if args.pool_limit is not None:

        image_paths = (
            image_paths[
                :args.pool_limit
            ]
        )


    usable_images = []


    for image_path in image_paths:

        gt_boxes, _ = (
            read_ground_truth(
                image_path,
                label_dir,
            )
        )

        if len(gt_boxes) > 0:

            usable_images.append(
                image_path
            )


    selected_images = random.sample(
        usable_images,
        min(
            args.num_images,
            len(usable_images),
        ),
    )


    print(
        f"\nImage pool: "
        f"{len(image_paths)}"
    )

    print(
        f"Images selected: "
        f"{len(selected_images)}"
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    all_objectness = []
    all_class_confidence = []
    all_scores = []

    all_predicted_widths = []
    all_predicted_heights = []

    all_gt_widths = []
    all_gt_heights = []


    confidence_thresholds = [
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.30,
        0.50,
    ]


    prediction_counts = {
        threshold: 0
        for threshold
        in confidence_thresholds
    }


    # ========================================================
    # IMAGES
    # ========================================================

    for index, image_path in enumerate(
        selected_images,
        start=1,
    ):

        original = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        tensor = preprocess(
            original
        )


        tensor = (
            tensor
            .unsqueeze(0)
            .to(DEVICE)
        )


        with torch.no_grad():

            output = model(
                tensor
            )


        # ----------------------------------------------------
        # Raw network output
        # ----------------------------------------------------

        boxes_raw = torch.sigmoid(
            output[..., 0:4]
        )


        objectness = torch.sigmoid(
            output[..., 4]
        )


        class_probabilities = torch.softmax(
            output[..., 5:],
            dim=-1,
        )


        class_confidence, _ = (
            class_probabilities.max(
                dim=-1
            )
        )


        final_scores = (
            objectness
            *
            class_confidence
        )


        all_objectness.extend(
            objectness
            .detach()
            .cpu()
            .flatten()
            .tolist()
        )


        all_class_confidence.extend(
            class_confidence
            .detach()
            .cpu()
            .flatten()
            .tolist()
        )


        all_scores.extend(
            final_scores
            .detach()
            .cpu()
            .flatten()
            .tolist()
        )


        all_predicted_widths.extend(
            boxes_raw[..., 2]
            .detach()
            .cpu()
            .flatten()
            .tolist()
        )


        all_predicted_heights.extend(
            boxes_raw[..., 3]
            .detach()
            .cpu()
            .flatten()
            .tolist()
        )


        # ----------------------------------------------------
        # Confidence counts BEFORE NMS
        # ----------------------------------------------------

        for threshold in (
            confidence_thresholds
        ):

            prediction_counts[
                threshold
            ] += int(
                (
                    final_scores
                    >=
                    threshold
                )
                .sum()
                .item()
            )


        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        gt_boxes, gt_labels = (
            read_ground_truth(
                image_path,
                label_dir,
            )
        )


        if len(gt_boxes) > 0:

            gt_widths = (
                gt_boxes[:, 2]
                -
                gt_boxes[:, 0]
            )


            gt_heights = (
                gt_boxes[:, 3]
                -
                gt_boxes[:, 1]
            )


            all_gt_widths.extend(
                gt_widths.tolist()
            )


            all_gt_heights.extend(
                gt_heights.tolist()
            )


        # ----------------------------------------------------
        # Decode + NMS
        # ----------------------------------------------------

        decoded = decode_predictions(
            output,
            confidence_threshold=
                args.conf,
            nms_iou_threshold=0.50,
        )[0]


        # ----------------------------------------------------
        # Save visualization
        # ----------------------------------------------------

        canvas = create_visualization(
            original,
            gt_boxes,
            gt_labels,
            decoded,
            args.conf,
        )


        filename = (
            f"{index:02d}_"
            f"{image_path.stem}.jpg"
        )


        canvas.save(
            output_dir / filename,
            quality=90,
        )


        print(
            f"{index:02d}/"
            f"{len(selected_images)}  "
            f"{image_path.name}: "
            f"{len(gt_boxes)} GT, "
            f"{len(decoded['boxes'])} predictions"
        )


    # ========================================================
    # REPORT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RAW OUTPUT STATISTICS"
    )

    print(
        "=" * 70
    )


    print_percentiles(
        "OBJECTNESS",
        all_objectness,
    )


    print_percentiles(
        "CLASS CONFIDENCE",
        all_class_confidence,
    )


    print_percentiles(
        "FINAL SCORE "
        "(objectness x class confidence)",
        all_scores,
    )


    print_percentiles(
        "PREDICTED WIDTH",
        all_predicted_widths,
    )


    print_percentiles(
        "GROUND-TRUTH WIDTH",
        all_gt_widths,
    )


    print_percentiles(
        "PREDICTED HEIGHT",
        all_predicted_heights,
    )


    print_percentiles(
        "GROUND-TRUTH HEIGHT",
        all_gt_heights,
    )


    print(
        "\nRAW PREDICTIONS ABOVE "
        "CONFIDENCE THRESHOLDS"
    )


    image_count = len(
        selected_images
    )


    for threshold, count in (
        prediction_counts.items()
    ):

        average = (
            count
            /
            image_count
        )


        print(
            f"  >= {threshold:.2f}: "
            f"{count:6d} total "
            f"({average:.1f}/image)"
        )


    print(
        "\nVisualizations saved to:"
    )

    print(
        output_dir.resolve()
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )


    parser.add_argument(
        "--split",
        type=str,
        choices=[
            "train",
            "val",
        ],
        default="val",
    )


    parser.add_argument(
        "--pool-limit",
        type=int,
        default=None,
    )


    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
    )


    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )


    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    freeze_support()

    arguments = parse_args()

    main(arguments)