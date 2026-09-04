from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO


RUN_DIR = Path("results/yolo/baseline")
RESULTS_CSV = RUN_DIR / "training" / "results.csv"
BEST_WEIGHTS = RUN_DIR / "weights" / "best.pt"
DATA_YAML = Path("data/processed/SH17_safety_1920/data.yaml")
OUTPUT_DIR = RUN_DIR / "analysis"

MODEL_NAME = "YOLO11n"
INPUT_SIZE = 640
TRAIN_IMAGES = 6479
VAL_IMAGES = 1620
CLASS_NAMES = [
    "person",
    "tool",
    "helmet",
    "safety-vest",
    "gloves",
    "glasses",
    "face-mask",
]


def configure_plot_style():
    plt.rcParams.update(
        {
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
        }
    )


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close()


def check_files():
    for path in (RESULTS_CSV, BEST_WEIGHTS, DATA_YAML):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found:\n{path.resolve()}")


def load_training_history():
    history = pd.read_csv(RESULTS_CSV)
    # Some Ultralytics versions include whitespace around column names.
    history.columns = history.columns.str.strip()
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
        column for column in required_columns if column not in history.columns
    ]
    if missing:
        raise RuntimeError("Missing columns in results.csv:\n" + "\n".join(missing))
    return history


def create_loss_plot(history):
    epochs = history["epoch"]
    plt.figure(figsize=(11, 7))
    plt.plot(epochs, history["train/box_loss"], label="Train box loss")
    plt.plot(
        epochs,
        history["val/box_loss"],
        linestyle="--",
        label="Validation box loss",
    )
    plt.plot(
        epochs,
        history["train/cls_loss"],
        label="Train classification loss",
    )
    plt.plot(
        epochs,
        history["val/cls_loss"],
        linestyle="--",
        label="Validation classification loss",
    )
    plt.plot(epochs, history["train/dfl_loss"], label="Train DFL loss")
    plt.plot(
        epochs,
        history["val/dfl_loss"],
        linestyle="--",
        label="Validation DFL loss",
    )
    plt.title("YOLO11n Training and Validation Losses", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(frameon=False, ncol=2)
    save_figure("01_training_losses.png")


def create_metric_plot(history):
    epochs = history["epoch"]
    plt.figure(figsize=(11, 7))
    plt.plot(epochs, history["metrics/precision(B)"], label="Precision")
    plt.plot(epochs, history["metrics/recall(B)"], label="Recall")
    plt.plot(epochs, history["metrics/mAP50(B)"], label="mAP@0.5")
    plt.plot(
        epochs,
        history["metrics/mAP50-95(B)"],
        label="mAP@0.5:0.95",
    )
    plt.title("YOLO11n Validation Performance During Training", fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.ylim(0, 1)
    plt.legend(frameon=False)
    save_figure("02_training_metrics.png")


def evaluate_best_model():
    print("\nRunning final validation using best.pt...")
    model = YOLO(str(BEST_WEIGHTS))
    return model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=INPUT_SIZE,
        batch=8,
        device=0,
        workers=0,
        plots=False,
        verbose=True,
    )


def create_per_class_table(metrics):
    per_class = pd.DataFrame(metrics.summary(decimals=6))
    per_class = per_class.rename(
        columns={
            "Class": "class",
            "Images": "images",
            "Instances": "instances",
            "Box-P": "precision",
            "Box-R": "recall",
            "Box-F1": "f1",
            "mAP50": "mAP50",
            "mAP50-95": "mAP50-95",
        }
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
    per_class = per_class[
        [column for column in desired_order if column in per_class.columns]
    ]
    per_class.to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False)
    return per_class


def create_per_class_map_plot(per_class):
    plot_data = per_class.sort_values("mAP50")
    y = np.arange(len(plot_data))
    height = 0.36
    plt.figure(figsize=(10, 6))
    plt.barh(
        y - height / 2,
        plot_data["mAP50"],
        height=height,
        label="mAP@0.5",
    )
    plt.barh(
        y + height / 2,
        plot_data["mAP50-95"],
        height=height,
        label="mAP@0.5:0.95",
    )
    plt.yticks(y, plot_data["class"])
    plt.xlim(0, 1)
    plt.xlabel("Average Precision")
    plt.ylabel("")
    plt.title("Detection Performance by Class", fontweight="bold")
    plt.legend(frameon=False)
    for index, value in enumerate(plot_data["mAP50"]):
        plt.text(
            value + 0.012,
            index - height / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )
    save_figure("03_per_class_map.png")


def create_precision_recall_plot(per_class):
    plot_data = per_class.sort_values("precision")
    y = np.arange(len(plot_data))
    height = 0.36
    plt.figure(figsize=(10, 6))
    plt.barh(
        y - height / 2,
        plot_data["precision"],
        height=height,
        label="Precision",
    )
    plt.barh(
        y + height / 2,
        plot_data["recall"],
        height=height,
        label="Recall",
    )
    plt.yticks(y, plot_data["class"])
    plt.xlim(0, 1)
    plt.xlabel("Score")
    plt.ylabel("")
    plt.title("Precision and Recall by Class", fontweight="bold")
    plt.legend(frameon=False)
    save_figure("04_per_class_precision_recall.png")


def save_overall_metrics(metrics):
    overall = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "preprocess_ms_per_image": float(metrics.speed.get("preprocess", 0)),
        "inference_ms_per_image": float(metrics.speed.get("inference", 0)),
        "postprocess_ms_per_image": float(metrics.speed.get("postprocess", 0)),
    }
    pd.DataFrame([overall]).to_csv(
        OUTPUT_DIR / "overall_metrics.csv",
        index=False,
    )
    return overall


def create_experiment_summary(history, overall, per_class):
    best_values = {}
    for name, column in {
        "map50": "metrics/mAP50(B)",
        "map5095": "metrics/mAP50-95(B)",
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
    }.items():
        index = history[column].idxmax()
        best_values[name] = (
            float(history.loc[index, column]),
            int(history.loc[index, "epoch"]),
        )

    training_hours = (
        float(history.iloc[-1]["time"]) / 3600
        if "time" in history.columns
        else float("nan")
    )
    strongest_class = per_class.loc[per_class["mAP50"].idxmax()]
    weakest_class = per_class.loc[per_class["mAP50"].idxmin()]
    best_map50, best_map50_epoch = best_values["map50"]
    best_map5095, best_map5095_epoch = best_values["map5095"]
    best_precision, best_precision_epoch = best_values["precision"]
    best_recall, best_recall_epoch = best_values["recall"]

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
        f"Classes:                {len(CLASS_NAMES)}",
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
        f"Best precision:         {best_precision:.4f} (epoch {best_precision_epoch})",
        f"Best recall:            {best_recall:.4f} (epoch {best_recall_epoch})",
        f"Best mAP@0.5:            {best_map50:.4f} (epoch {best_map50_epoch})",
        f"Best mAP@0.5:0.95:       {best_map5095:.4f} (epoch {best_map5095_epoch})",
        "",
        "Per-class observations",
        "----------------------",
        f"Strongest class:         {strongest_class['class']} "
        f"(mAP@0.5 = {strongest_class['mAP50']:.4f})",
        f"Weakest class:           {weakest_class['class']} "
        f"(mAP@0.5 = {weakest_class['mAP50']:.4f})",
        "",
        "Inference performance",
        "---------------------",
        f"Preprocess:              {overall['preprocess_ms_per_image']:.2f} ms/image",
        f"Inference:               {overall['inference_ms_per_image']:.2f} ms/image",
        f"Postprocess:             {overall['postprocess_ms_per_image']:.2f} ms/image",
    ]
    summary = "\n".join(lines)
    (OUTPUT_DIR / "experiment_summary.txt").write_text(summary, encoding="utf-8")
    return summary


def main():
    print("=" * 65)
    print("YOLO11n BASELINE ANALYSIS")
    print("=" * 65)
    check_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    print("\nLoading training history...")
    history = load_training_history()
    print(f"Epochs found: {len(history)}")
    print("Creating training-loss figure...")
    create_loss_plot(history)
    print("Creating training-metric figure...")
    create_metric_plot(history)

    metrics = evaluate_best_model()
    print("\nCreating per-class tables...")
    per_class = create_per_class_table(metrics)
    print("Creating per-class mAP figure...")
    create_per_class_map_plot(per_class)
    print("Creating precision/recall figure...")
    create_precision_recall_plot(per_class)
    overall = save_overall_metrics(metrics)
    summary = create_experiment_summary(history, overall, per_class)

    print("\n" + summary)
    print("\n" + "=" * 65)
    print("ANALYSIS COMPLETE")
    print("=" * 65)
    print(f"\nResults saved to:\n{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
