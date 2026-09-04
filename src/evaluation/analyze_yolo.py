from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

RUN_DIR = Path(
    "results/training/yolo11n_640_baseline"
)

RESULTS_CSV = RUN_DIR / "results.csv"

BEST_WEIGHTS = (
    RUN_DIR /
    "weights" /
    "best.pt"
)

DATA_YAML = Path(
    "data/processed/SH17_safety_1920/data.yaml"
)

OUTPUT_DIR = RUN_DIR / "analysis"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# EXPERIMENT INFORMATION
# ============================================================

MODEL_NAME = "YOLO11n"
INPUT_SIZE = 640

TRAIN_IMAGES = 6479
VAL_IMAGES = 1620

CLASSES = [
    "person",
    "tool",
    "helmet",
    "safety-vest",
    "gloves",
    "glasses",
    "face-mask",
]


# ============================================================
# PLOT STYLE
# ============================================================

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 220,

    "font.size": 11,
    "axes.titlesize": 16,
    "axes.labelsize": 12,

    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    "axes.spines.top": False,
    "axes.spines.right": False,

    "axes.grid": True,
    "grid.alpha": 0.20,
})


# ============================================================
# HELPERS
# ============================================================

def save_figure(
    filename
):

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        bbox_inches="tight"
    )

    plt.close()


def check_files():

    required = [
        RESULTS_CSV,
        BEST_WEIGHTS,
        DATA_YAML,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n"
                f"{path.resolve()}"
            )


# ============================================================
# LOAD TRAINING HISTORY
# ============================================================

def load_training_history():

    df = pd.read_csv(
        RESULTS_CSV
    )

    # Some Ultralytics versions may save
    # whitespace around column names.
    df.columns = (
        df.columns
        .str.strip()
    )

    required_columns = [
        "epoch",

        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",

        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",

        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing columns in results.csv:\n"
            + "\n".join(missing)
        )

    return df


# ============================================================
# FIGURE 1
# TRAINING AND VALIDATION LOSSES
# ============================================================

def create_loss_plot(
    df
):

    epochs = df["epoch"]

    plt.figure(
        figsize=(11, 7)
    )

    # Box loss
    plt.plot(
        epochs,
        df["train/box_loss"],
        label="Train box loss"
    )

    plt.plot(
        epochs,
        df["val/box_loss"],
        linestyle="--",
        label="Validation box loss"
    )

    # Classification loss
    plt.plot(
        epochs,
        df["train/cls_loss"],
        label="Train classification loss"
    )

    plt.plot(
        epochs,
        df["val/cls_loss"],
        linestyle="--",
        label="Validation classification loss"
    )

    # DFL loss
    plt.plot(
        epochs,
        df["train/dfl_loss"],
        label="Train DFL loss"
    )

    plt.plot(
        epochs,
        df["val/dfl_loss"],
        linestyle="--",
        label="Validation DFL loss"
    )

    plt.title(
        "YOLO11n Training and Validation Losses",
        fontweight="bold"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.legend(
        frameon=False,
        ncol=2
    )

    save_figure(
        "01_training_losses.png"
    )


# ============================================================
# FIGURE 2
# PERFORMANCE DURING TRAINING
# ============================================================

def create_metric_plot(
    df
):

    epochs = df["epoch"]

    plt.figure(
        figsize=(11, 7)
    )

    plt.plot(
        epochs,
        df["metrics/precision(B)"],
        label="Precision"
    )

    plt.plot(
        epochs,
        df["metrics/recall(B)"],
        label="Recall"
    )

    plt.plot(
        epochs,
        df["metrics/mAP50(B)"],
        label="mAP@0.5"
    )

    plt.plot(
        epochs,
        df["metrics/mAP50-95(B)"],
        label="mAP@0.5:0.95"
    )

    plt.title(
        "YOLO11n Validation Performance During Training",
        fontweight="bold"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Metric value"
    )

    plt.ylim(
        0,
        1
    )

    plt.legend(
        frameon=False
    )

    save_figure(
        "02_training_metrics.png"
    )


# ============================================================
# RUN FINAL VALIDATION
# ============================================================

def evaluate_best_model():

    print(
        "\nRunning final validation using best.pt..."
    )

    model = YOLO(
        str(BEST_WEIGHTS)
    )

    metrics = model.val(

        data=str(DATA_YAML),

        split="val",

        imgsz=INPUT_SIZE,
        batch=8,

        device=0,

        # Validation is short and we already know
        # this is the safest Windows configuration.
        workers=0,

        plots=False,
        verbose=True,
    )

    return metrics


# ============================================================
# CREATE PER-CLASS TABLE
# ============================================================

def create_per_class_table(
    metrics
):

    # Ultralytics returns a clean per-class
    # metric summary from validation.
    summary = metrics.summary(
        decimals=6
    )

    df = pd.DataFrame(
        summary
    )

    rename_map = {
        "Class": "class",
        "Images": "images",
        "Instances": "instances",

        "Box-P": "precision",
        "Box-R": "recall",
        "Box-F1": "f1",

        "mAP50": "mAP50",
        "mAP50-95": "mAP50-95",
    }

    df = df.rename(
        columns=rename_map
    )

    desired_order = [
        "class",
        "images",
        "instances",
        "precision",
        "recall",
        "f1",
        "mAP50",
        "mAP50-95",
    ]

    existing_columns = [
        column
        for column in desired_order
        if column in df.columns
    ]

    df = df[
        existing_columns
    ]

    df.to_csv(
        OUTPUT_DIR /
        "per_class_metrics.csv",

        index=False
    )

    return df


# ============================================================
# FIGURE 3
# PER-CLASS mAP
# ============================================================

def create_per_class_map_plot(
    df
):

    plot_df = (
        df
        .sort_values(
            "mAP50",
            ascending=True
        )
    )

    y = np.arange(
        len(plot_df)
    )

    height = 0.36

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        y - height / 2,
        plot_df["mAP50"],
        height=height,
        label="mAP@0.5"
    )

    plt.barh(
        y + height / 2,
        plot_df["mAP50-95"],
        height=height,
        label="mAP@0.5:0.95"
    )

    plt.yticks(
        y,
        plot_df["class"]
    )

    plt.xlim(
        0,
        1
    )

    plt.xlabel(
        "Average Precision"
    )

    plt.ylabel("")

    plt.title(
        "Detection Performance by Class",
        fontweight="bold"
    )

    plt.legend(
        frameon=False
    )

    # Numerical mAP50 labels
    for index, value in enumerate(
        plot_df["mAP50"]
    ):

        plt.text(
            value + 0.012,
            index - height / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9
        )

    save_figure(
        "03_per_class_map.png"
    )


# ============================================================
# FIGURE 4
# PRECISION / RECALL BY CLASS
# ============================================================

def create_precision_recall_plot(
    df
):

    plot_df = (
        df
        .sort_values(
            "precision",
            ascending=True
        )
    )

    y = np.arange(
        len(plot_df)
    )

    height = 0.36

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        y - height / 2,
        plot_df["precision"],
        height=height,
        label="Precision"
    )

    plt.barh(
        y + height / 2,
        plot_df["recall"],
        height=height,
        label="Recall"
    )

    plt.yticks(
        y,
        plot_df["class"]
    )

    plt.xlim(
        0,
        1
    )

    plt.xlabel(
        "Score"
    )

    plt.ylabel("")

    plt.title(
        "Precision and Recall by Class",
        fontweight="bold"
    )

    plt.legend(
        frameon=False
    )

    save_figure(
        "04_per_class_precision_recall.png"
    )


# ============================================================
# OVERALL METRICS
# ============================================================

def save_overall_metrics(
    metrics
):

    overall = {
        "precision":
            float(metrics.box.mp),

        "recall":
            float(metrics.box.mr),

        "mAP50":
            float(metrics.box.map50),

        "mAP50-95":
            float(metrics.box.map),

        "preprocess_ms_per_image":
            float(
                metrics.speed.get(
                    "preprocess",
                    0
                )
            ),

        "inference_ms_per_image":
            float(
                metrics.speed.get(
                    "inference",
                    0
                )
            ),

        "postprocess_ms_per_image":
            float(
                metrics.speed.get(
                    "postprocess",
                    0
                )
            ),
    }

    df = pd.DataFrame(
        [overall]
    )

    df.to_csv(
        OUTPUT_DIR /
        "overall_metrics.csv",

        index=False
    )

    return overall


# ============================================================
# EXPERIMENT SUMMARY
# ============================================================

def create_experiment_summary(
    history,
    overall,
    per_class
):

    # --------------------------------------------------------
    # Best epochs
    # --------------------------------------------------------

    map50_index = (
        history[
            "metrics/mAP50(B)"
        ].idxmax()
    )

    map5095_index = (
        history[
            "metrics/mAP50-95(B)"
        ].idxmax()
    )

    precision_index = (
        history[
            "metrics/precision(B)"
        ].idxmax()
    )

    recall_index = (
        history[
            "metrics/recall(B)"
        ].idxmax()
    )


    best_map50_epoch = int(
        history.loc[
            map50_index,
            "epoch"
        ]
    )

    best_map50 = float(
        history.loc[
            map50_index,
            "metrics/mAP50(B)"
        ]
    )


    best_map5095_epoch = int(
        history.loc[
            map5095_index,
            "epoch"
        ]
    )

    best_map5095 = float(
        history.loc[
            map5095_index,
            "metrics/mAP50-95(B)"
        ]
    )


    best_precision_epoch = int(
        history.loc[
            precision_index,
            "epoch"
        ]
    )

    best_precision = float(
        history.loc[
            precision_index,
            "metrics/precision(B)"
        ]
    )


    best_recall_epoch = int(
        history.loc[
            recall_index,
            "epoch"
        ]
    )

    best_recall = float(
        history.loc[
            recall_index,
            "metrics/recall(B)"
        ]
    )


    # --------------------------------------------------------
    # Training duration
    # --------------------------------------------------------

    if "time" in history.columns:

        training_seconds = float(
            history.iloc[-1]["time"]
        )

        training_hours = (
            training_seconds /
            3600
        )

    else:

        training_hours = float(
            "nan"
        )


    # --------------------------------------------------------
    # Strongest and weakest classes
    # --------------------------------------------------------

    best_class_row = (
        per_class.loc[
            per_class["mAP50"].idxmax()
        ]
    )

    weakest_class_row = (
        per_class.loc[
            per_class["mAP50"].idxmin()
        ]
    )


    # --------------------------------------------------------
    # Write summary
    # --------------------------------------------------------

    lines = [
        "YOLO11n BASELINE EXPERIMENT",
        "=" * 60,
        "",

        "Configuration",
        "-------------",
        f"Model:                  {MODEL_NAME}",
        f"Input resolution:       {INPUT_SIZE} x {INPUT_SIZE}",
        f"Training images:        {TRAIN_IMAGES}",
        f"Validation images:      {VAL_IMAGES}",
        f"Classes:                {len(CLASSES)}",
        f"Epochs completed:       {int(history['epoch'].max())}",
        f"Training time:          {training_hours:.2f} hours",
        "",

        "Final best.pt validation",
        "------------------------",
        f"Precision:              {overall['precision']:.4f}",
        f"Recall:                 {overall['recall']:.4f}",
        f"mAP@0.5:                {overall['mAP50']:.4f}",
        f"mAP@0.5:0.95:           {overall['mAP50-95']:.4f}",
        "",

        "Training-history best values",
        "----------------------------",
        (
            f"Best precision:         "
            f"{best_precision:.4f} "
            f"(epoch {best_precision_epoch})"
        ),
        (
            f"Best recall:            "
            f"{best_recall:.4f} "
            f"(epoch {best_recall_epoch})"
        ),
        (
            f"Best mAP@0.5:            "
            f"{best_map50:.4f} "
            f"(epoch {best_map50_epoch})"
        ),
        (
            f"Best mAP@0.5:0.95:       "
            f"{best_map5095:.4f} "
            f"(epoch {best_map5095_epoch})"
        ),
        "",

        "Per-class observations",
        "----------------------",
        (
            f"Strongest class:         "
            f"{best_class_row['class']} "
            f"(mAP@0.5 = "
            f"{best_class_row['mAP50']:.4f})"
        ),
        (
            f"Weakest class:           "
            f"{weakest_class_row['class']} "
            f"(mAP@0.5 = "
            f"{weakest_class_row['mAP50']:.4f})"
        ),
        "",

        "Inference performance",
        "---------------------",
        (
            f"Preprocess:              "
            f"{overall['preprocess_ms_per_image']:.2f} ms/image"
        ),
        (
            f"Inference:               "
            f"{overall['inference_ms_per_image']:.2f} ms/image"
        ),
        (
            f"Postprocess:             "
            f"{overall['postprocess_ms_per_image']:.2f} ms/image"
        ),
    ]

    text = "\n".join(
        lines
    )

    (
        OUTPUT_DIR /
        "experiment_summary.txt"
    ).write_text(
        text,
        encoding="utf-8"
    )

    return text


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("YOLO11n BASELINE ANALYSIS")
    print("=" * 65)


    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    check_files()


    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    print(
        "\nLoading training history..."
    )

    history = (
        load_training_history()
    )


    print(
        f"Epochs found: "
        f"{len(history)}"
    )


    # --------------------------------------------------------
    # Training plots
    # --------------------------------------------------------

    print(
        "Creating training-loss figure..."
    )

    create_loss_plot(
        history
    )


    print(
        "Creating training-metric figure..."
    )

    create_metric_plot(
        history
    )


    # --------------------------------------------------------
    # Best model evaluation
    # --------------------------------------------------------

    metrics = (
        evaluate_best_model()
    )


    # --------------------------------------------------------
    # Per-class analysis
    # --------------------------------------------------------

    print(
        "\nCreating per-class tables..."
    )

    per_class = (
        create_per_class_table(
            metrics
        )
    )


    print(
        "Creating per-class mAP figure..."
    )

    create_per_class_map_plot(
        per_class
    )


    print(
        "Creating precision/recall figure..."
    )

    create_precision_recall_plot(
        per_class
    )


    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    overall = (
        save_overall_metrics(
            metrics
        )
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = (
        create_experiment_summary(
            history,
            overall,
            per_class
        )
    )


    print(
        "\n" + summary
    )


    print(
        "\n" + "=" * 65
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print("=" * 65)

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()