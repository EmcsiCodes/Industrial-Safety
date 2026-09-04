from multiprocessing import freeze_support
from pathlib import Path

import argparse
import csv
import json
import random
import time

import numpy as np
import torch

from torch.utils.data import DataLoader

from dataset import (
    SH17GridDataset,
    CLASS_NAMES,
)

from model import (
    SafetyNetDetector,
    detection_loss,
)


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = Path(
    "data/processed/SH17_safety_1920"
)

RESULTS_ROOT = Path(
    "results/scratch"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    val_loss,
    config,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": config,
            "classes": CLASS_NAMES,
        },
        path,
    )


# ============================================================
# ONE TRAINING / VALIDATION EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()


    totals = {
        "total": 0.0,
        "box": 0.0,
        "objectness": 0.0,
        "classification": 0.0,
    }

    sample_count = 0


    # --------------------------------------------------------
    # Objectness diagnostics
    # --------------------------------------------------------

    positive_objectness_sum = 0.0
    positive_objectness_count = 0

    background_objectness_sum = 0.0
    background_objectness_count = 0


    for batch_index, (images, targets) in enumerate(
        loader,
        start=1,
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )


        if training:
            optimizer.zero_grad(
                set_to_none=True
            )


        with torch.set_grad_enabled(training):

            predictions = model(images)


            if predictions.shape[1:3] != targets.shape[1:3]:
                raise RuntimeError(
                    "\nModel grid and target grid do not match.\n"
                    f"Model:  {predictions.shape}\n"
                    f"Target: {targets.shape}"
                )


            losses = detection_loss(
                predictions,
                targets,
            )


            if training:
                losses["total"].backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=10.0,
                )

                optimizer.step()


        # ====================================================
        # LOSS ACCUMULATION
        # ====================================================

        batch_size = images.size(0)

        sample_count += batch_size


        for key in totals:
            totals[key] += (
                losses[key].item()
                *
                batch_size
            )


        # ====================================================
        # OBJECTNESS DIAGNOSTIC
        # ====================================================

        with torch.no_grad():

            object_probabilities = torch.sigmoid(
                predictions[..., 4]
            )

            object_mask = (
                targets[..., 4] > 0.5
            )

            background_mask = ~object_mask


            if object_mask.any():

                positive_objectness_sum += (
                    object_probabilities[
                        object_mask
                    ].sum().item()
                )

                positive_objectness_count += int(
                    object_mask.sum().item()
                )


            if background_mask.any():

                background_objectness_sum += (
                    object_probabilities[
                        background_mask
                    ].sum().item()
                )

                background_objectness_count += int(
                    background_mask.sum().item()
                )


        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            training
            and (
                batch_index % 50 == 0
                or
                batch_index == len(loader)
            )
        ):
            print(
                f"    {batch_index:4d}/{len(loader)}  "
                f"loss={losses['total'].item():.4f}"
            )


    # ========================================================
    # AVERAGES
    # ========================================================

    for key in totals:
        totals[key] /= sample_count


    positive_objectness = (
        positive_objectness_sum
        /
        positive_objectness_count
        if positive_objectness_count > 0
        else 0.0
    )


    background_objectness = (
        background_objectness_sum
        /
        background_objectness_count
        if background_objectness_count > 0
        else 0.0
    )


    totals["positive_objectness"] = (
        positive_objectness
    )

    totals["background_objectness"] = (
        background_objectness
    )


    return totals


# ============================================================
# HISTORY CSV
# ============================================================

def save_history(
    history,
    output_path,
):
    if not history:
        return


    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )

        writer.writeheader()
        writer.writerows(history)


# ============================================================
# TRAINING
# ============================================================

def train(args):

    set_seed(args.seed)


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print("=" * 65)
    print("CUSTOM SAFETYNET OBJECT DETECTOR — V2")
    print("=" * 65)

    print(
        f"\nDevice: {device}"
    )


    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )


    # ========================================================
    # SMOKE / OVERFIT MODE
    # ========================================================

    if args.smoke:

        print(
            "\nSMOKE / OVERFIT TEST — V2"
        )

        print(
            "Using the first 64 training images."
        )

        print(
            "Validation uses the same 64 images."
        )


        epochs = 15
        batch_size = 8
        workers = 0


        train_dataset = SH17GridDataset(
            DATA_ROOT,
            split="train",
            image_size=args.imgsz,
            grid_size=args.grid,
            augment=False,
            limit=64,
        )


        # Deliberately the exact same images.
        val_dataset = train_dataset


        experiment_name = "smoke_v2"


    # ========================================================
    # FULL TRAINING
    # ========================================================

    else:

        epochs = args.epochs
        batch_size = args.batch
        workers = args.workers


        train_dataset = SH17GridDataset(
            DATA_ROOT,
            split="train",
            image_size=args.imgsz,
            grid_size=args.grid,
            augment=True,
        )


        val_dataset = SH17GridDataset(
            DATA_ROOT,
            split="val",
            image_size=args.imgsz,
            grid_size=args.grid,
            augment=False,
        )


        experiment_name = args.name


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = SafetyNetDetector(
        num_classes=len(CLASS_NAMES)
    ).to(device)


    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )


    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


    print(
        f"\nParameters: {parameter_count:,}"
    )

    print(
        f"Trainable:  {trainable_count:,}"
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )


    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-5,
        )
    )


    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    output_dir = (
        RESULTS_ROOT
        /
        experiment_name
    )


    weights_dir = (
        output_dir
        /
        "weights"
    )


    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    weights_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # CONFIG
    # ========================================================

    config = {
        "version": "SafetyNet v2",
        "model": "SafetyNetDetector",
        "pretrained": False,

        "image_size": args.imgsz,
        "grid_size": args.grid,

        "classes": CLASS_NAMES,

        "batch": batch_size,
        "epochs": epochs,

        "learning_rate": args.lr,

        "workers": workers,
        "seed": args.seed,

        "parameter_count":
            parameter_count,

        "objectness_loss":
            "BCEWithLogits with dynamic positive weight",

        "box_loss":
            "SmoothL1 xy + sqrt-wh SmoothL1",
    }


    (
        output_dir
        /
        "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=4,
        ),
        encoding="utf-8",
    )


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    history = []

    best_val_loss = float("inf")

    epochs_without_improvement = 0


    print(
        f"\nTraining for {epochs} epochs..."
    )


    for epoch in range(
        1,
        epochs + 1,
    ):

        epoch_start = time.time()


        print(
            "\n"
            + "=" * 65
        )

        print(
            f"EPOCH {epoch}/{epochs}"
        )

        print(
            "=" * 65
        )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        with torch.no_grad():

            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
            )


        scheduler.step()


        epoch_minutes = (
            time.time()
            -
            epoch_start
        ) / 60.0


        learning_rate = (
            optimizer
            .param_groups[0]["lr"]
        )


        # ====================================================
        # RECORD HISTORY
        # ====================================================

        record = {
            "epoch": epoch,

            "train_total":
                train_metrics["total"],

            "train_box":
                train_metrics["box"],

            "train_objectness":
                train_metrics["objectness"],

            "train_classification":
                train_metrics["classification"],

            "train_positive_obj_prob":
                train_metrics["positive_objectness"],

            "train_background_obj_prob":
                train_metrics["background_objectness"],


            "val_total":
                val_metrics["total"],

            "val_box":
                val_metrics["box"],

            "val_objectness":
                val_metrics["objectness"],

            "val_classification":
                val_metrics["classification"],

            "val_positive_obj_prob":
                val_metrics["positive_objectness"],

            "val_background_obj_prob":
                val_metrics["background_objectness"],


            "learning_rate":
                learning_rate,

            "epoch_minutes":
                epoch_minutes,
        }


        history.append(record)


        save_history(
            history,
            output_dir / "history.csv",
        )


        # ====================================================
        # PRINT EPOCH SUMMARY
        # ====================================================

        print(
            f"\nTrain loss: "
            f"{train_metrics['total']:.4f}"
        )

        print(
            f"Val loss:   "
            f"{val_metrics['total']:.4f}"
        )


        print(
            "\nObjectness probabilities:"
        )

        print(
            "  Train object cells:     "
            f"{train_metrics['positive_objectness']:.4f}"
        )

        print(
            "  Train background cells: "
            f"{train_metrics['background_objectness']:.4f}"
        )

        print(
            "  Val object cells:       "
            f"{val_metrics['positive_objectness']:.4f}"
        )

        print(
            "  Val background cells:   "
            f"{val_metrics['background_objectness']:.4f}"
        )


        print(
            f"\nTime: "
            f"{epoch_minutes:.2f} min"
        )

        print(
            f"LR:   "
            f"{learning_rate:.6f}"
        )


        # ====================================================
        # LAST CHECKPOINT
        # ====================================================

        save_checkpoint(
            weights_dir / "last.pt",
            model,
            optimizer,
            epoch,
            val_metrics["total"],
            config,
        )


        # ====================================================
        # BEST CHECKPOINT
        # ====================================================

        if (
            val_metrics["total"]
            <
            best_val_loss
        ):

            best_val_loss = (
                val_metrics["total"]
            )

            epochs_without_improvement = 0


            save_checkpoint(
                weights_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_val_loss,
                config,
            )


            print(
                "New best model saved."
            )


        else:

            epochs_without_improvement += 1


        # ====================================================
        # PERIODIC CHECKPOINT
        # ====================================================

        if epoch % 5 == 0:

            save_checkpoint(
                weights_dir /
                f"epoch_{epoch}.pt",

                model,
                optimizer,
                epoch,
                val_metrics["total"],
                config,
            )


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            not args.smoke
            and
            epochs_without_improvement
            >= args.patience
        ):

            print(
                "\nEarly stopping: "
                f"no validation improvement for "
                f"{args.patience} epochs."
            )

            break


    # ========================================================
    # FINISHED
    # ========================================================

    print(
        "\n" + "=" * 65
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 65
    )


    print(
        f"\nBest validation loss: "
        f"{best_val_loss:.4f}"
    )

    print(
        f"\nResults:\n"
        f"{output_dir.resolve()}"
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )


    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )


    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )


    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
    )


    parser.add_argument(
        "--grid",
        type=int,
        default=26,
    )


    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )


    parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )


    parser.add_argument(
        "--name",
        type=str,
        default="safetynet_v2",
    )


    parser.add_argument(
        "--smoke",
        action="store_true",
    )


    return parser.parse_args()


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":

    freeze_support()

    arguments = parse_args()

    train(arguments)