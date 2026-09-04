from pathlib import Path
from multiprocessing import freeze_support
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from model import SafetyNetDetector


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    "data/processed/SH17_safety_1920"
)

RUN_DIR = Path(
    "results/scratch/safetynet_v2"
)

WEIGHTS_DIR = RUN_DIR / "weights"

OUTPUT_DIR = RUN_DIR / "evaluation"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
GRID_SIZE = 26

BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# Keep a very low threshold for AP evaluation.
# AP needs predictions across the confidence range.
PREDICTION_THRESHOLD = 0.01

NMS_IOU_THRESHOLD = 0.50

MAX_DETECTIONS = 300


IOU_THRESHOLDS = np.arange(
    0.50,
    0.96,
    0.05
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# VALIDATION DATASET
# ============================================================

class DetectionEvaluationDataset(Dataset):

    def __init__(
        self,
        root,
        split="val",
        image_size=416
    ):

        self.root = Path(root)

        self.image_dir = (
            self.root /
            "images" /
            split
        )

        self.label_dir = (
            self.root /
            "labels" /
            split
        )

        self.image_size = image_size

        self.image_paths = sorted([
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower()
            in IMAGE_EXTENSIONS
        ])

        print(
            f"Validation images: "
            f"{len(self.image_paths)}"
        )


    def __len__(self):

        return len(
            self.image_paths
        )


    def _read_labels(
        self,
        image_path
    ):

        label_path = (
            self.label_dir /
            f"{image_path.stem}.txt"
        )

        boxes = []
        labels = []

        if not label_path.exists():

            return {
                "boxes":
                    torch.zeros(
                        (0, 4),
                        dtype=torch.float32
                    ),

                "labels":
                    torch.zeros(
                        (0,),
                        dtype=torch.long
                    )
            }


        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(
                parts[0]
            )

            x = float(parts[1])
            y = float(parts[2])

            width = float(parts[3])
            height = float(parts[4])


            # -----------------------------------------------
            # YOLO xywh -> normalized xyxy
            # -----------------------------------------------

            x1 = x - width / 2
            y1 = y - height / 2

            x2 = x + width / 2
            y2 = y + height / 2


            boxes.append([
                x1,
                y1,
                x2,
                y2
            ])

            labels.append(
                class_id
            )


        return {
            "boxes":
                torch.tensor(
                    boxes,
                    dtype=torch.float32
                )
                if boxes
                else torch.zeros(
                    (0, 4),
                    dtype=torch.float32
                ),

            "labels":
                torch.tensor(
                    labels,
                    dtype=torch.long
                )
                if labels
                else torch.zeros(
                    (0,),
                    dtype=torch.long
                )
        }


    def __getitem__(
        self,
        index
    ):

        image_path = (
            self.image_paths[index]
        )


        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        # Same preprocessing used during
        # scratch-model training.

        image = image.resize(
            (
                self.image_size,
                self.image_size
            ),
            Image.Resampling.BILINEAR
        )


        image = TF.to_tensor(
            image
        )


        image = (
            image - 0.5
        ) / 0.5


        target = (
            self._read_labels(
                image_path
            )
        )


        return (
            image,
            target,
            image_path.name
        )


# ============================================================
# COLLATE
# ============================================================

def collate_fn(
    batch
):

    images = torch.stack([
        item[0]
        for item in batch
    ])


    targets = [
        item[1]
        for item in batch
    ]


    names = [
        item[2]
        for item in batch
    ]


    return (
        images,
        targets,
        names
    )


# ============================================================
# IoU
# ============================================================

def box_iou(
    boxes1,
    boxes2
):

    if (
        boxes1.numel() == 0
        or
        boxes2.numel() == 0
    ):

        return torch.zeros(
            (
                boxes1.shape[0],
                boxes2.shape[0]
            ),
            device=boxes1.device
        )


    # Intersection

    top_left = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2]
    )

    bottom_right = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:]
    )


    wh = (
        bottom_right
        -
        top_left
    ).clamp(
        min=0
    )


    intersection = (
        wh[..., 0]
        *
        wh[..., 1]
    )


    # Areas

    area1 = (
        boxes1[:, 2]
        -
        boxes1[:, 0]
    ).clamp(
        min=0
    ) * (
        boxes1[:, 3]
        -
        boxes1[:, 1]
    ).clamp(
        min=0
    )


    area2 = (
        boxes2[:, 2]
        -
        boxes2[:, 0]
    ).clamp(
        min=0
    ) * (
        boxes2[:, 3]
        -
        boxes2[:, 1]
    ).clamp(
        min=0
    )


    union = (
        area1[:, None]
        +
        area2[None, :]
        -
        intersection
    )


    return (
        intersection
        /
        union.clamp(
            min=1e-9
        )
    )


# ============================================================
# CUSTOM NMS
# ============================================================

def nms(
    boxes,
    scores,
    iou_threshold
):

    if boxes.numel() == 0:

        return torch.empty(
            (0,),
            dtype=torch.long,
            device=boxes.device
        )


    order = torch.argsort(
        scores,
        descending=True
    )


    keep = []


    while order.numel() > 0:

        current = order[0]

        keep.append(
            current
        )


        if order.numel() == 1:
            break


        remaining = (
            order[1:]
        )


        ious = box_iou(

            boxes[
                current
            ].unsqueeze(0),

            boxes[
                remaining
            ]

        ).squeeze(0)


        order = remaining[
            ious <= iou_threshold
        ]


    return torch.stack(
        keep
    )


# ============================================================
# DECODE MODEL OUTPUT
# ============================================================

def decode_predictions(
    predictions,
    confidence_threshold,
    nms_iou_threshold
):

    batch_size = (
        predictions.shape[0]
    )

    S = (
        predictions.shape[1]
    )


    # --------------------------------------------------------
    # RAW OUTPUT
    # --------------------------------------------------------

    box_values = torch.sigmoid(
        predictions[..., 0:4]
    )


    objectness = torch.sigmoid(
        predictions[..., 4]
    )


    class_probabilities = torch.softmax(
        predictions[..., 5:],
        dim=-1
    )


    class_confidence, class_ids = (
        torch.max(
            class_probabilities,
            dim=-1
        )
    )


    # Final confidence:
    #
    # probability that object exists
    # ×
    # probability of predicted class

    scores = (
        objectness
        *
        class_confidence
    )


    # --------------------------------------------------------
    # GRID COORDINATES
    # --------------------------------------------------------

    grid_y, grid_x = torch.meshgrid(

        torch.arange(
            S,
            device=predictions.device
        ),

        torch.arange(
            S,
            device=predictions.device
        ),

        indexing="ij"
    )


    # Center relative to complete image

    center_x = (
        grid_x[None]
        +
        box_values[..., 0]
    ) / S


    center_y = (
        grid_y[None]
        +
        box_values[..., 1]
    ) / S


    width = (
        box_values[..., 2]
    )

    height = (
        box_values[..., 3]
    )


    x1 = (
        center_x
        -
        width / 2
    )

    y1 = (
        center_y
        -
        height / 2
    )

    x2 = (
        center_x
        +
        width / 2
    )

    y2 = (
        center_y
        +
        height / 2
    )


    boxes = torch.stack(
        [
            x1,
            y1,
            x2,
            y2
        ],
        dim=-1
    ).clamp(
        0,
        1
    )


    # ========================================================
    # PROCESS EACH IMAGE
    # ========================================================

    results = []


    for batch_index in range(
        batch_size
    ):

        image_boxes = (
            boxes[
                batch_index
            ].reshape(
                -1,
                4
            )
        )


        image_scores = (
            scores[
                batch_index
            ].reshape(
                -1
            )
        )


        image_classes = (
            class_ids[
                batch_index
            ].reshape(
                -1
            )
        )


        # ----------------------------------------------------
        # CONFIDENCE FILTERING
        # ----------------------------------------------------

        mask = (
            image_scores
            >=
            confidence_threshold
        )


        image_boxes = (
            image_boxes[
                mask
            ]
        )

        image_scores = (
            image_scores[
                mask
            ]
        )

        image_classes = (
            image_classes[
                mask
            ]
        )


        if image_boxes.numel() == 0:

            results.append({
                "boxes":
                    torch.zeros(
                        (0, 4)
                    ),

                "scores":
                    torch.zeros(
                        (0,)
                    ),

                "labels":
                    torch.zeros(
                        (0,),
                        dtype=torch.long
                    )
            })

            continue


        # ----------------------------------------------------
        # CLASS-AWARE NMS
        # ----------------------------------------------------

        kept_indices = []


        for class_id in torch.unique(
            image_classes
        ):

            class_mask = (
                image_classes
                ==
                class_id
            )


            indices = torch.where(
                class_mask
            )[0]


            class_keep = nms(

                image_boxes[
                    indices
                ],

                image_scores[
                    indices
                ],

                nms_iou_threshold
            )


            kept_indices.append(
                indices[
                    class_keep
                ]
            )


        kept_indices = torch.cat(
            kept_indices
        )


        # Sort final results by confidence

        order = torch.argsort(

            image_scores[
                kept_indices
            ],

            descending=True
        )


        kept_indices = (
            kept_indices[
                order
            ][:MAX_DETECTIONS]
        )


        results.append({
            "boxes":
                image_boxes[
                    kept_indices
                ]
                .detach()
                .cpu(),

            "scores":
                image_scores[
                    kept_indices
                ]
                .detach()
                .cpu(),

            "labels":
                image_classes[
                    kept_indices
                ]
                .detach()
                .cpu()
        })


    return results


# ============================================================
# AP CALCULATION
# ============================================================

def calculate_ap(
    recall,
    precision
):

    if len(recall) == 0:
        return 0.0


    # COCO-style 101-point interpolation

    recall_points = np.linspace(
        0,
        1,
        101
    )


    ap = 0.0


    for recall_threshold in (
        recall_points
    ):

        valid = (
            recall
            >=
            recall_threshold
        )


        if np.any(
            valid
        ):

            ap += np.max(
                precision[
                    valid
                ]
            )


    return (
        ap /
        101
    )


# ============================================================
# EVALUATE ONE CLASS + IoU
# ============================================================

def evaluate_class_at_iou(
    predictions,
    ground_truths,
    class_id,
    iou_threshold
):

    # --------------------------------------------------------
    # Collect ground truth
    # --------------------------------------------------------

    gt_by_image = {}

    total_gt = 0


    for image_index, target in enumerate(
        ground_truths
    ):

        mask = (
            target["labels"]
            ==
            class_id
        )


        boxes = (
            target["boxes"][
                mask
            ]
        )


        gt_by_image[
            image_index
        ] = boxes


        total_gt += len(
            boxes
        )


    if total_gt == 0:

        return {
            "ap": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "confidence": np.nan,
        }


    # --------------------------------------------------------
    # Collect predictions
    # --------------------------------------------------------

    detections = []


    for image_index, prediction in enumerate(
        predictions
    ):

        mask = (
            prediction["labels"]
            ==
            class_id
        )


        boxes = (
            prediction["boxes"][
                mask
            ]
        )


        scores = (
            prediction["scores"][
                mask
            ]
        )


        for box, score in zip(
            boxes,
            scores
        ):

            detections.append(
                (
                    float(score),
                    image_index,
                    box
                )
            )


    detections.sort(
        key=lambda item:
            item[0],
        reverse=True
    )


    if len(detections) == 0:

        return {
            "ap": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "confidence": 0.0,
        }


    # --------------------------------------------------------
    # Greedy matching
    # --------------------------------------------------------

    matched = {
        image_index:
            torch.zeros(
                len(boxes),
                dtype=torch.bool
            )

        for image_index, boxes
        in gt_by_image.items()
    }


    tp = np.zeros(
        len(detections)
    )

    fp = np.zeros(
        len(detections)
    )

    confidences = np.zeros(
        len(detections)
    )


    for detection_index, (
        score,
        image_index,
        predicted_box
    ) in enumerate(
        detections
    ):

        confidences[
            detection_index
        ] = score


        gt_boxes = (
            gt_by_image[
                image_index
            ]
        )


        if len(gt_boxes) == 0:

            fp[
                detection_index
            ] = 1

            continue


        available = (
            ~matched[
                image_index
            ]
        )


        if not available.any():

            fp[
                detection_index
            ] = 1

            continue


        available_indices = (
            torch.where(
                available
            )[0]
        )


        ious = box_iou(

            predicted_box
            .unsqueeze(0),

            gt_boxes[
                available_indices
            ]

        ).squeeze(0)


        best_iou, best_position = (
            torch.max(
                ious,
                dim=0
            )
        )


        if (
            float(best_iou)
            >=
            iou_threshold
        ):

            matched_gt_index = (
                available_indices[
                    best_position
                ]
            )


            matched[
                image_index
            ][
                matched_gt_index
            ] = True


            tp[
                detection_index
            ] = 1

        else:

            fp[
                detection_index
            ] = 1


    # --------------------------------------------------------
    # Precision / Recall curve
    # --------------------------------------------------------

    cumulative_tp = (
        np.cumsum(
            tp
        )
    )

    cumulative_fp = (
        np.cumsum(
            fp
        )
    )


    recall = (
        cumulative_tp /
        total_gt
    )


    precision = (
        cumulative_tp
        /
        np.maximum(
            cumulative_tp
            +
            cumulative_fp,
            1e-12
        )
    )


    ap = calculate_ap(
        recall,
        precision
    )


    # --------------------------------------------------------
    # Best F1 operating point
    # --------------------------------------------------------

    f1 = (
        2
        *
        precision
        *
        recall
        /
        np.maximum(
            precision
            +
            recall,
            1e-12
        )
    )


    best_index = int(
        np.argmax(
            f1
        )
    )


    return {
        "ap":
            float(ap),

        "precision":
            float(
                precision[
                    best_index
                ]
            ),

        "recall":
            float(
                recall[
                    best_index
                ]
            ),

        "f1":
            float(
                f1[
                    best_index
                ]
            ),

        "confidence":
            float(
                confidences[
                    best_index
                ]
            ),
    }


# ============================================================
# COMPLETE METRIC CALCULATION
# ============================================================

def calculate_metrics(
    predictions,
    ground_truths
):

    class_rows = []


    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):

        aps = []


        # ----------------------------------------------------
        # AP over IoU 0.50 → 0.95
        # ----------------------------------------------------

        for threshold in (
            IOU_THRESHOLDS
        ):

            result = (
                evaluate_class_at_iou(

                    predictions,
                    ground_truths,

                    class_id,
                    threshold
                )
            )


            aps.append(
                result["ap"]
            )


        # Precision / Recall / F1
        # at IoU 0.50

        result_50 = (
            evaluate_class_at_iou(

                predictions,
                ground_truths,

                class_id,
                0.50
            )
        )


        gt_count = sum(

            int(
                (
                    target["labels"]
                    ==
                    class_id
                ).sum()
            )

            for target in (
                ground_truths
            )
        )


        class_rows.append({
            "class_id":
                class_id,

            "class":
                class_name,

            "instances":
                gt_count,

            "precision":
                result_50[
                    "precision"
                ],

            "recall":
                result_50[
                    "recall"
                ],

            "f1":
                result_50[
                    "f1"
                ],

            "best_confidence":
                result_50[
                    "confidence"
                ],

            "mAP50":
                aps[0],

            "mAP50-95":
                float(
                    np.nanmean(
                        aps
                    )
                ),
        })


    class_df = pd.DataFrame(
        class_rows
    )


    overall = {
        "precision":
            class_df[
                "precision"
            ].mean(),

        "recall":
            class_df[
                "recall"
            ].mean(),

        "f1":
            class_df[
                "f1"
            ].mean(),

        "mAP50":
            class_df[
                "mAP50"
            ].mean(),

        "mAP50-95":
            class_df[
                "mAP50-95"
            ].mean(),
    }


    return (
        class_df,
        overall
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model(
    checkpoint_path
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )


    model = SafetyNetDetector(
        num_classes=NUM_CLASSES
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model.to(
        DEVICE
    )

    model.eval()


    epoch = checkpoint.get(
        "epoch",
        -1
    )


    val_loss = checkpoint.get(
        "val_loss",
        np.nan
    )


    return (
        model,
        epoch,
        val_loss
    )


# ============================================================
# RUN INFERENCE
# ============================================================

def run_inference(
    model,
    loader
):

    predictions = []
    ground_truths = []

    total_inference_time = 0.0
    image_count = 0


    with torch.no_grad():

        for batch_index, (
            images,
            targets,
            names
        ) in enumerate(
            loader,
            start=1
        ):

            images = images.to(
                DEVICE
            )


            if DEVICE.type == "cuda":

                torch.cuda.synchronize()


            start = time.perf_counter()


            outputs = model(
                images
            )


            if DEVICE.type == "cuda":

                torch.cuda.synchronize()


            total_inference_time += (
                time.perf_counter()
                -
                start
            )


            decoded = decode_predictions(

                outputs,

                confidence_threshold=
                    PREDICTION_THRESHOLD,

                nms_iou_threshold=
                    NMS_IOU_THRESHOLD
            )


            predictions.extend(
                decoded
            )


            ground_truths.extend(
                targets
            )


            image_count += (
                len(images)
            )


            if (
                batch_index % 20 == 0
                or
                batch_index == len(loader)
            ):

                print(
                    f"    "
                    f"{batch_index}/"
                    f"{len(loader)} batches"
                )


    inference_ms = (

        total_inference_time
        /
        image_count
        *
        1000
    )


    return (
        predictions,
        ground_truths,
        inference_ms
    )


# ============================================================
# CHECKPOINT DISCOVERY
# ============================================================

def get_checkpoints():

    candidates = []


    for name in [
        "best.pt",
        "last.pt",
    ]:

        path = (
            WEIGHTS_DIR /
            name
        )

        if path.exists():

            candidates.append(
                path
            )


    epoch_paths = sorted(
        WEIGHTS_DIR.glob(
            "epoch_*.pt"
        ),

        key=lambda path:
            int(
                path.stem.split(
                    "_"
                )[-1]
            )
    )


    candidates.extend(
        epoch_paths
    )


    # Remove accidental duplicates by path

    unique = []

    seen = set()


    for path in candidates:

        resolved = (
            path.resolve()
        )

        if resolved not in seen:

            seen.add(
                resolved
            )

            unique.append(
                path
            )


    return unique


# ============================================================
# PRESENTATION PLOT
# ============================================================

def create_checkpoint_plot(
    comparison_df
):

    plot_df = (
        comparison_df
        .sort_values(
            "epoch"
        )
    )


    x = np.arange(
        len(plot_df)
    )


    width = 0.35


    plt.figure(
        figsize=(10, 6)
    )


    plt.bar(
        x - width / 2,

        plot_df[
            "mAP50"
        ],

        width,

        label="mAP@0.5"
    )


    plt.bar(
        x + width / 2,

        plot_df[
            "mAP50-95"
        ],

        width,

        label="mAP@0.5:0.95"
    )


    labels = [
        f"{row.checkpoint}\n"
        f"epoch {row.epoch}"

        for row in (
            plot_df.itertuples()
        )
    ]


    plt.xticks(
        x,
        labels
    )


    plt.ylim(
        0,
        1
    )


    plt.ylabel(
        "Average Precision"
    )


    plt.title(
        "SafetyNet Checkpoint Performance"
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        OUTPUT_DIR /
        "checkpoint_map_comparison.png",

        dpi=200,
        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SAFETYNET DETECTOR — EVALUATION")
    print("=" * 70)


    print(
        f"\nDevice: "
        f"{DEVICE}"
    )


    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )


    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = (
        DetectionEvaluationDataset(

            DATA_ROOT,

            split="val",

            image_size=IMAGE_SIZE
        )
    )


    loader = DataLoader(

        dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        collate_fn=collate_fn,

        pin_memory=(
            DEVICE.type ==
            "cuda"
        )
    )


    # --------------------------------------------------------
    # CHECKPOINTS
    # --------------------------------------------------------

    checkpoints = (
        get_checkpoints()
    )


    if not checkpoints:

        raise RuntimeError(
            f"No checkpoints found in:\n"
            f"{WEIGHTS_DIR.resolve()}"
        )


    print(
        "\nCheckpoints:"
    )


    for checkpoint in checkpoints:

        print(
            f"  - "
            f"{checkpoint.name}"
        )


    comparison_rows = []

    checkpoint_class_results = {}


    # ========================================================
    # EVALUATE EACH CHECKPOINT
    # ========================================================

    for checkpoint_path in (
        checkpoints
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            f"EVALUATING "
            f"{checkpoint_path.name}"
        )

        print(
            "=" * 70
        )


        (
            model,
            epoch,
            val_loss
        ) = load_model(
            checkpoint_path
        )


        print(
            f"Epoch: "
            f"{epoch}"
        )

        print(
            f"Saved validation loss: "
            f"{val_loss:.4f}"
        )


        print(
            "Running inference..."
        )


        (
            predictions,
            ground_truths,
            inference_ms
        ) = run_inference(

            model,
            loader
        )


        print(
            "Calculating detection metrics..."
        )


        (
            class_df,
            overall
        ) = calculate_metrics(

            predictions,
            ground_truths
        )


        checkpoint_class_results[
            checkpoint_path.name
        ] = class_df


        comparison_rows.append({
            "checkpoint":
                checkpoint_path.name,

            "epoch":
                epoch,

            "val_loss":
                val_loss,

            "precision":
                overall[
                    "precision"
                ],

            "recall":
                overall[
                    "recall"
                ],

            "f1":
                overall[
                    "f1"
                ],

            "mAP50":
                overall[
                    "mAP50"
                ],

            "mAP50-95":
                overall[
                    "mAP50-95"
                ],

            "inference_ms_per_image":
                inference_ms,
        })


        print(
            "\nRESULT"
        )

        print(
            f"Precision:   "
            f"{overall['precision']:.4f}"
        )

        print(
            f"Recall:      "
            f"{overall['recall']:.4f}"
        )

        print(
            f"F1:          "
            f"{overall['f1']:.4f}"
        )

        print(
            f"mAP50:       "
            f"{overall['mAP50']:.4f}"
        )

        print(
            f"mAP50-95:    "
            f"{overall['mAP50-95']:.4f}"
        )

        print(
            f"Inference:   "
            f"{inference_ms:.2f} ms/image"
        )


    # ========================================================
    # SAVE CHECKPOINT COMPARISON
    # ========================================================

    comparison_df = pd.DataFrame(
        comparison_rows
    )


    comparison_df.to_csv(
        OUTPUT_DIR /
        "checkpoint_comparison.csv",

        index=False
    )


    # --------------------------------------------------------
    # SELECT ACTUAL BEST DETECTOR
    # --------------------------------------------------------

    best_index = (
        comparison_df[
            "mAP50-95"
        ].idxmax()
    )


    best_row = (
        comparison_df.loc[
            best_index
        ]
    )


    best_checkpoint = (
        best_row[
            "checkpoint"
        ]
    )


    best_class_df = (
        checkpoint_class_results[
            best_checkpoint
        ]
    )


    best_class_df.to_csv(
        OUTPUT_DIR /
        "best_per_class_metrics.csv",

        index=False
    )


    # --------------------------------------------------------
    # PRESENTATION GRAPH
    # --------------------------------------------------------

    create_checkpoint_plot(
        comparison_df
    )


    # ========================================================
    # SUMMARY
    # ========================================================

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
        (
            f"  {best_checkpoint}"
        ),
        (
            f"  Epoch: "
            f"{int(best_row['epoch'])}"
        ),
        "",
        "Best checkpoint metrics:",
        (
            f"  Precision:   "
            f"{best_row['precision']:.4f}"
        ),
        (
            f"  Recall:      "
            f"{best_row['recall']:.4f}"
        ),
        (
            f"  F1:          "
            f"{best_row['f1']:.4f}"
        ),
        (
            f"  mAP@0.5:     "
            f"{best_row['mAP50']:.4f}"
        ),
        (
            f"  mAP@0.5:0.95:"
            f" {best_row['mAP50-95']:.4f}"
        ),
        (
            f"  Inference:   "
            f"{best_row['inference_ms_per_image']:.2f}"
            f" ms/image"
        ),
    ]


    summary = "\n".join(
        lines
    )


    (
        OUTPUT_DIR /
        "evaluation_summary.txt"
    ).write_text(
        summary,
        encoding="utf-8"
    )


    print(
        "\n" + "=" * 70
    )

    print(
        summary
    )

    print(
        "\nResults saved to:"
    )

    print(
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":

    freeze_support()

    main()