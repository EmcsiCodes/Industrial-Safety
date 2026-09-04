import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO


YOLO_RUN_DIR = Path("results/yolo/baseline")
YOLO_WEIGHTS = YOLO_RUN_DIR / "weights" / "best.pt"
YOLO_HISTORY = YOLO_RUN_DIR / "training" / "results.csv"
YOLO_OVERALL = YOLO_RUN_DIR / "analysis" / "overall_metrics.csv"
YOLO_PER_CLASS = YOLO_RUN_DIR / "analysis" / "per_class_metrics.csv"

SAFETYNET_RUN_DIR = Path("results/scratch/v2")
SAFETYNET_CONFIG = SAFETYNET_RUN_DIR / "training" / "config.json"
SAFETYNET_HISTORY = SAFETYNET_RUN_DIR / "training" / "history.csv"
SAFETYNET_CHECKPOINTS = (
    SAFETYNET_RUN_DIR / "evaluation" / "checkpoint_comparison.csv"
)
SAFETYNET_PER_CLASS = (
    SAFETYNET_RUN_DIR / "evaluation" / "best_per_class_metrics.csv"
)
DATASET_STATISTICS = Path("results/dataset_analysis/class_statistics.csv")

OUTPUT_ROOT = Path("results/final_figures")
OUTPUT_DIR = OUTPUT_ROOT / "model_comparison" / "quantitative"
QUALITATIVE_MANIFEST = (
    OUTPUT_ROOT / "model_comparison" / "qualitative" / "qualitative_manifest.csv"
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


def configure_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )


def load_csv(path, required_columns):
    if not path.exists():
        raise FileNotFoundError(f"Required experiment file not found: {path.resolve()}")
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    missing = sorted(set(required_columns) - set(data.columns))
    if missing:
        raise RuntimeError(f"Missing columns in {path}:\n" + "\n".join(missing))
    if data.empty:
        raise RuntimeError(f"Experiment file is empty: {path}")
    return data


def load_experiment_data():
    yolo_overall = load_csv(
        YOLO_OVERALL,
        [
            "precision",
            "recall",
            "mAP50",
            "mAP50-95",
            "inference_ms_per_image",
        ],
    )
    if len(yolo_overall) != 1:
        raise RuntimeError(f"Expected one row in {YOLO_OVERALL}, found {len(yolo_overall)}")
    yolo_history = load_csv(
        YOLO_HISTORY,
        ["epoch", "metrics/mAP50(B)", "metrics/mAP50-95(B)"],
    )
    yolo_per_class = load_csv(
        YOLO_PER_CLASS,
        ["class", "precision", "recall", "mAP50", "mAP50-95"],
    )
    safety_history = load_csv(
        SAFETYNET_HISTORY,
        ["epoch", "train_total", "val_total"],
    )
    safety_checkpoints = load_csv(
        SAFETYNET_CHECKPOINTS,
        [
            "checkpoint",
            "epoch",
            "val_loss",
            "precision",
            "recall",
            "mAP50",
            "mAP50-95",
            "inference_ms_per_image",
        ],
    )
    safety_per_class = load_csv(
        SAFETYNET_PER_CLASS,
        ["class", "precision", "recall", "mAP50", "mAP50-95"],
    )
    dataset_statistics = load_csv(
        DATASET_STATISTICS,
        ["class", "train_annotations", "median_bbox_area_percent"],
    )

    if not SAFETYNET_CONFIG.exists():
        raise FileNotFoundError(
            f"SafetyNet configuration not found: {SAFETYNET_CONFIG.resolve()}"
        )
    safety_config = json.loads(SAFETYNET_CONFIG.read_text(encoding="utf-8"))
    for key in ("image_size", "parameter_count", "pretrained"):
        if key not in safety_config:
            raise RuntimeError(f"Missing '{key}' in {SAFETYNET_CONFIG}")
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS.resolve()}")

    return {
        "yolo_overall": yolo_overall.iloc[0],
        "yolo_history": yolo_history,
        "yolo_per_class": yolo_per_class,
        "safety_history": safety_history,
        "safety_checkpoints": safety_checkpoints,
        "safety_per_class": safety_per_class,
        "safety_config": safety_config,
        "dataset_statistics": dataset_statistics,
    }


def save_figure(figure, filename):
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / filename, dpi=240, bbox_inches="tight")
    plt.close(figure)


def annotate_vertical_bars(axis, bars, values, formatter):
    for bar, value in zip(bars, values):
        axis.annotate(
            formatter(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def create_comparison_table(data, yolo_parameters):
    yolo = data["yolo_overall"]
    yolo_history = data["yolo_history"]
    yolo_best_epoch = int(
        yolo_history.loc[yolo_history["metrics/mAP50-95(B)"].idxmax(), "epoch"]
    )
    safety = data["safety_checkpoints"].loc[
        data["safety_checkpoints"]["mAP50-95"].idxmax()
    ]
    config = data["safety_config"]
    table = pd.DataFrame(
        [
            {
                "Model": "YOLO11n",
                "Pretrained": "Yes",
                "Input Size": "640 x 640",
                "Parameters": yolo_parameters,
                "Precision": float(yolo["precision"]),
                "Recall": float(yolo["recall"]),
                "mAP50": float(yolo["mAP50"]),
                "mAP50-95": float(yolo["mAP50-95"]),
                "Inference ms/image": float(yolo["inference_ms_per_image"]),
                "Best Checkpoint / Epoch": f"best.pt / epoch {yolo_best_epoch}",
                "Notes": "Ultralytics baseline; pretrained initialization",
            },
            {
                "Model": "SafetyNet v2",
                "Pretrained": "No",
                "Input Size": f"{config['image_size']} x {config['image_size']}",
                "Parameters": int(config["parameter_count"]),
                "Precision": float(safety["precision"]),
                "Recall": float(safety["recall"]),
                "mAP50": float(safety["mAP50"]),
                "mAP50-95": float(safety["mAP50-95"]),
                "Inference ms/image": float(safety["inference_ms_per_image"]),
                "Best Checkpoint / Epoch": (
                    f"{safety['checkpoint']} / epoch {int(safety['epoch'])} "
                    "(best_map.pt copy)"
                ),
                "Notes": "Custom single-scale detector; random initialization",
            },
        ]
    )
    table.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    return table


def plot_model_map(table):
    metrics = ["mAP50", "mAP50-95"]
    x = np.arange(len(metrics))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 6))
    yolo_bars = axis.bar(
        x - width / 2,
        table.loc[table["Model"] == "YOLO11n", metrics].iloc[0],
        width,
        label="YOLO11n",
    )
    safety_bars = axis.bar(
        x + width / 2,
        table.loc[table["Model"] == "SafetyNet v2", metrics].iloc[0],
        width,
        label="SafetyNet v2",
    )
    axis.set_title("Detection Accuracy: YOLO11n vs SafetyNet v2")
    axis.set_ylabel("Average Precision")
    axis.set_xticks(x, ["mAP@0.5", "mAP@0.5:0.95"])
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    annotate_vertical_bars(axis, yolo_bars, [bar.get_height() for bar in yolo_bars], lambda x: f"{x:.4f}")
    annotate_vertical_bars(axis, safety_bars, [bar.get_height() for bar in safety_bars], lambda x: f"{x:.4f}")
    save_figure(figure, "model_map_comparison.png")


def plot_model_parameters(table):
    values = table["Parameters"].to_numpy(dtype=float)
    labels = ["YOLO11n\n640 x 640", "SafetyNet v2\n416 x 416"]
    figure, axis = plt.subplots(figsize=(8, 6))
    bars = axis.bar(labels, values / 1_000_000)
    axis.set_title("Model Parameter Comparison")
    axis.set_ylabel("Parameters (millions)")
    annotate_vertical_bars(axis, bars, values, lambda x: f"{int(x):,}")
    save_figure(figure, "model_parameter_comparison.png")


def plot_inference_speed(table):
    values = table["Inference ms/image"].to_numpy(dtype=float)
    labels = ["YOLO11n\n640 x 640", "SafetyNet v2\n416 x 416"]
    figure, axis = plt.subplots(figsize=(8, 6))
    bars = axis.bar(labels, values)
    axis.set_title("Measured Inference Time at Different Input Resolutions")
    axis.set_ylabel("Inference time (ms/image)")
    annotate_vertical_bars(axis, bars, values, lambda x: f"{x:.2f} ms")
    figure.text(
        0.5,
        0.01,
        "Input resolution differs, so this is not an architecture-only speed comparison.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0.04, 1, 1])
    figure.savefig(
        OUTPUT_DIR / "model_inference_speed_comparison.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(figure)


def create_per_class_comparison(data):
    yolo = data["yolo_per_class"].set_index("class")
    safety = data["safety_per_class"].set_index("class")
    missing_yolo = sorted(set(CLASS_NAMES) - set(yolo.index))
    missing_safety = sorted(set(CLASS_NAMES) - set(safety.index))
    if missing_yolo or missing_safety:
        raise RuntimeError(
            "Missing per-class rows. "
            f"YOLO: {missing_yolo or 'none'}; SafetyNet: {missing_safety or 'none'}"
        )

    rows = []
    for class_name in CLASS_NAMES:
        rows.append(
            {
                "Class": class_name,
                "YOLO Precision": yolo.loc[class_name, "precision"],
                "YOLO Recall": yolo.loc[class_name, "recall"],
                "YOLO mAP50": yolo.loc[class_name, "mAP50"],
                "YOLO mAP50-95": yolo.loc[class_name, "mAP50-95"],
                "SafetyNet Precision": safety.loc[class_name, "precision"],
                "SafetyNet Recall": safety.loc[class_name, "recall"],
                "SafetyNet mAP50": safety.loc[class_name, "mAP50"],
                "SafetyNet mAP50-95": safety.loc[class_name, "mAP50-95"],
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT_DIR / "per_class_metrics_comparison.csv", index=False)

    y = np.arange(len(comparison))
    height = 0.36
    figure, axis = plt.subplots(figsize=(10, 7))
    yolo_bars = axis.barh(
        y - height / 2,
        comparison["YOLO mAP50"],
        height,
        label="YOLO11n",
    )
    safety_bars = axis.barh(
        y + height / 2,
        comparison["SafetyNet mAP50"],
        height,
        label="SafetyNet v2",
    )
    axis.set_yticks(y, comparison["Class"])
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("mAP@0.5")
    axis.set_title("Per-Class Detection Performance")
    axis.legend(frameon=False)
    for bars in (yolo_bars, safety_bars):
        for bar in bars:
            value = bar.get_width()
            axis.text(
                value + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=9,
            )
    save_figure(figure, "per_class_map50_comparison.png")
    return comparison


def plot_checkpoint_performance(checkpoints):
    plot_data = checkpoints.sort_values("epoch")
    best_detection = plot_data.loc[plot_data["mAP50-95"].idxmax()]
    minimum_loss = plot_data.loc[plot_data["val_loss"].idxmin()]
    maximum_map = float(plot_data["mAP50"].max())

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(plot_data["epoch"], plot_data["mAP50"], marker="o", label="mAP@0.5")
    axis.plot(
        plot_data["epoch"],
        plot_data["mAP50-95"],
        marker="o",
        label="mAP@0.5:0.95",
    )
    axis.scatter(
        best_detection["epoch"],
        best_detection["mAP50"],
        color="black",
        marker="*",
        s=150,
        zorder=3,
    )
    axis.axvline(
        minimum_loss["epoch"],
        linestyle="--",
        linewidth=1,
        label=f"Minimum validation loss (epoch {int(minimum_loss['epoch'])})",
    )
    axis.annotate(
        f"Best detection checkpoint\nepoch {int(best_detection['epoch'])}",
        xy=(best_detection["epoch"], best_detection["mAP50"]),
        xytext=(best_detection["epoch"] + 0.8, maximum_map * 1.16),
        arrowprops={"arrowstyle": "->"},
    )
    axis.set_title("SafetyNet v2 Checkpoint Detection Performance")
    axis.set_xlabel("Checkpoint epoch")
    axis.set_ylabel("Average Precision")
    axis.set_ylim(0, maximum_map * 1.35)
    axis.set_xticks(plot_data["epoch"].astype(int))
    axis.legend(frameon=True, framealpha=1, edgecolor="none", loc="upper left")
    save_figure(figure, "safetynet_checkpoint_performance.png")


def merge_dataset_performance(data):
    statistics = data["dataset_statistics"].set_index("class")
    yolo = data["yolo_per_class"].set_index("class")
    missing = sorted(set(CLASS_NAMES) - set(statistics.index))
    if missing:
        raise RuntimeError(f"Missing dataset statistics for: {', '.join(missing)}")
    return pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            "Training annotations": [
                int(statistics.loc[name, "train_annotations"]) for name in CLASS_NAMES
            ],
            "Median box area (%)": [
                float(statistics.loc[name, "median_bbox_area_percent"])
                for name in CLASS_NAMES
            ],
            "YOLO mAP50": [float(yolo.loc[name, "mAP50"]) for name in CLASS_NAMES],
        }
    )


def plot_relationship(data, x_column, filename, title, x_label):
    label_offsets = {
        "person": (6, 6),
        "tool": (6, 6),
        "helmet": (6, 16),
        "safety-vest": (6, -12),
        "gloves": (6, 12),
        "glasses": (6, 2),
        "face-mask": (6, -14),
    }
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(data[x_column], data["YOLO mAP50"], s=60)
    for _, row in data.iterrows():
        x_offset, y_offset = label_offsets[row["Class"]]
        axis.annotate(
            row["Class"],
            (row[x_column], row["YOLO mAP50"]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
        )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("YOLO11n mAP@0.5")
    axis.set_ylim(0, 1)
    figure.text(
        0.5,
        0.01,
        "Observed association only; class frequency or box size alone does not determine performance.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0.04, 1, 1])
    figure.savefig(OUTPUT_DIR / filename, dpi=240, bbox_inches="tight")
    plt.close(figure)


def write_notes_and_summaries(data, table):
    yolo = table.iloc[0]
    safety = table.iloc[1]
    history = data["safety_history"]
    minimum_loss = history.loc[history["val_total"].idxmin()]
    last_epoch = int(history["epoch"].max())
    notes = f"""MODEL COMPARISON NOTES
======================

Data sources
------------
YOLO overall metrics: {YOLO_OVERALL}
YOLO per-class metrics: {YOLO_PER_CLASS}
YOLO training history: {YOLO_HISTORY}
SafetyNet checkpoint metrics: {SAFETYNET_CHECKPOINTS}
SafetyNet per-class metrics: {SAFETYNET_PER_CLASS}
SafetyNet training history: {SAFETYNET_HISTORY}
Dataset statistics: {DATASET_STATISTICS}

Methodological limitations
--------------------------
- mAP@0.5 and mAP@0.5:0.95 are the primary comparison metrics, although they were produced by separate YOLO and custom evaluator implementations.
- Precision and recall are reported as stored, but their operating points are not confirmed to be identical and should not be used for a strong ranking claim.
- YOLO11n inference used 640 x 640 input; SafetyNet used 416 x 416 input. The speed measurements are therefore not a controlled architecture-only comparison.
- YOLO11n used pretrained initialization. SafetyNet was randomly initialized.
- Raw YOLO and SafetyNet loss values are not compared because their loss definitions differ.

Training behavior
-----------------
YOLO validation mAP continued improving through epoch 50. SafetyNet reached its minimum validation loss at epoch {int(minimum_loss['epoch'])} ({minimum_loss['val_total']:.6f}); training loss continued falling while validation loss later increased, and training stopped after epoch {last_epoch}. Its best measured detection checkpoint was epoch 10.

Skipped visualization
---------------------
The numerical SafetyNet v1-v2 objectness comparison was skipped because the v1 history does not contain object-cell and background-cell probability columns.
"""
    (OUTPUT_DIR / "comparison_notes.txt").write_text(notes, encoding="utf-8")

    model_summary = f"""MODEL COMPARISON SUMMARY
========================

YOLO11n achieved mAP@0.5 = {yolo['mAP50']:.6f} and mAP@0.5:0.95 = {yolo['mAP50-95']:.6f}. SafetyNet v2's best measured detection checkpoint, epoch 10, achieved mAP@0.5 = {safety['mAP50']:.6f} and mAP@0.5:0.95 = {safety['mAP50-95']:.6f}.

SafetyNet was smaller ({int(safety['Parameters']):,} parameters versus {int(yolo['Parameters']):,}) and had lower measured inference time ({safety['Inference ms/image']:.3f} versus {yolo['Inference ms/image']:.3f} ms/image), but it also used a lower input resolution (416 x 416 versus 640 x 640). The speed values are therefore not a controlled architecture-only comparison.

The custom SafetyNet detector remained substantially less accurate than the pretrained YOLO11n baseline. The results are consistent with the importance of modern feature extraction, multi-scale detection, transfer learning, target assignment, and localization strategies for difficult small-object PPE detection.
"""
    (OUTPUT_ROOT / "model_comparison_summary.txt").write_text(
        model_summary,
        encoding="utf-8",
    )

    descriptions = {
        "safetynet_training/01_safetynet_total_loss.png": "Shows training and validation total loss, including the minimum validation-loss epoch.",
        "safetynet_training/02_safetynet_loss_components.png": "Shows box, objectness, and classification loss evolution for training and validation.",
        "safetynet_training/03_safetynet_objectness.png": "Shows learned foreground/background objectness separation on training and validation data.",
        "safetynet_training/04_safetynet_learning_rate.png": "Documents the learning-rate schedule used during SafetyNet v2 training.",
        "model_comparison/quantitative/model_map_comparison.png": "Compares the stored mAP@0.5 and mAP@0.5:0.95 values for both detectors.",
        "model_comparison/quantitative/model_parameter_comparison.png": "Compares model parameter counts while identifying each input resolution.",
        "model_comparison/quantitative/model_inference_speed_comparison.png": "Compares measured inference time while noting the different input resolutions.",
        "model_comparison/quantitative/per_class_map50_comparison.png": "Shows the per-class mAP@0.5 gap on a common zero-to-one scale.",
        "model_comparison/quantitative/safetynet_checkpoint_performance.png": "Shows that minimum validation loss and best detection mAP occurred at different SafetyNet epochs.",
        "model_comparison/quantitative/class_frequency_vs_map.png": "Relates actual training annotation counts to YOLO per-class mAP without implying causation.",
        "model_comparison/quantitative/bbox_size_vs_map.png": "Relates median normalized box area to YOLO per-class mAP without implying causation.",
    }
    summary_lines = ["VISUALIZATION SUMMARY", "=" * 60, ""]
    for relative_path, description in descriptions.items():
        if (OUTPUT_ROOT / relative_path).exists():
            summary_lines.extend([relative_path, f"  {description}", ""])

    if QUALITATIVE_MANIFEST.exists():
        manifest = pd.read_csv(QUALITATIVE_MANIFEST)
        for _, row in manifest.iterrows():
            relative_path = f"model_comparison/qualitative/{row['Filename']}"
            summary_lines.extend(
                [
                    relative_path,
                    "  Same-image ground truth, YOLO11n, and SafetyNet "
                    f"comparison selected as {row['Selection Category']}.",
                    "",
                ]
            )
    (OUTPUT_ROOT / "visualization_summary.txt").write_text(
        "\n".join(summary_lines).rstrip() + "\n",
        encoding="utf-8",
    )


def main():
    data = load_experiment_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    yolo_model = YOLO(str(YOLO_WEIGHTS))
    yolo_parameters = sum(parameter.numel() for parameter in yolo_model.model.parameters())
    del yolo_model

    table = create_comparison_table(data, yolo_parameters)
    plot_model_map(table)
    plot_model_parameters(table)
    plot_inference_speed(table)
    create_per_class_comparison(data)
    plot_checkpoint_performance(data["safety_checkpoints"])

    relationship_data = merge_dataset_performance(data)
    relationship_data.to_csv(
        OUTPUT_DIR / "dataset_performance_relationship.csv",
        index=False,
    )
    plot_relationship(
        relationship_data,
        "Training annotations",
        "class_frequency_vs_map.png",
        "Training Annotation Count vs YOLO11n mAP@0.5",
        "Training annotations",
    )
    plot_relationship(
        relationship_data,
        "Median box area (%)",
        "bbox_size_vs_map.png",
        "Median Bounding-Box Size vs YOLO11n mAP@0.5",
        "Median bounding-box area (% of image)",
    )
    write_notes_and_summaries(data, table)

    safety = table.loc[table["Model"] == "SafetyNet v2"].iloc[0]
    yolo = table.loc[table["Model"] == "YOLO11n"].iloc[0]
    print(f"Created model-comparison outputs in {OUTPUT_DIR.resolve()}")
    print(f"YOLO11n: mAP50={yolo['mAP50']:.6f}, mAP50-95={yolo['mAP50-95']:.6f}")
    print(
        f"SafetyNet v2: mAP50={safety['mAP50']:.6f}, "
        f"mAP50-95={safety['mAP50-95']:.6f}"
    )


if __name__ == "__main__":
    main()
