from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HISTORY_CSV = Path("results/scratch/v2/training/history.csv")
OUTPUT_DIR = Path("results/final_figures/safetynet_training")
REQUIRED_COLUMNS = {
    "epoch",
    "train_total",
    "train_box",
    "train_objectness",
    "train_classification",
    "train_positive_obj_prob",
    "train_background_obj_prob",
    "val_total",
    "val_box",
    "val_objectness",
    "val_classification",
    "val_positive_obj_prob",
    "val_background_obj_prob",
    "learning_rate",
}


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


def load_history():
    if not HISTORY_CSV.exists():
        raise FileNotFoundError(f"SafetyNet history not found: {HISTORY_CSV.resolve()}")
    history = pd.read_csv(HISTORY_CSV)
    history.columns = history.columns.str.strip()
    missing = sorted(REQUIRED_COLUMNS - set(history.columns))
    if missing:
        raise RuntimeError(
            f"Missing columns in {HISTORY_CSV}:\n" + "\n".join(missing)
        )
    if history.empty:
        raise RuntimeError(f"SafetyNet history is empty: {HISTORY_CSV}")
    return history


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=240, bbox_inches="tight")
    plt.close()


def plot_total_loss(history):
    best_index = history["val_total"].idxmin()
    best_epoch = int(history.loc[best_index, "epoch"])
    best_loss = float(history.loc[best_index, "val_total"])

    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_total"], marker="o", label="Training")
    plt.plot(
        history["epoch"],
        history["val_total"],
        marker="o",
        label="Validation",
    )
    plt.scatter(best_epoch, best_loss, color="black", marker="*", s=150, zorder=3)
    plt.annotate(
        f"Minimum validation loss\nepoch {best_epoch}: {best_loss:.3f}",
        xy=(best_epoch, best_loss),
        xytext=(best_epoch + 0.8, best_loss + 0.25),
        arrowprops={"arrowstyle": "->"},
    )
    plt.title("SafetyNet v2 Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Total loss")
    plt.xticks(history["epoch"])
    plt.legend(frameon=False)
    save_figure("01_safetynet_total_loss.png")


def plot_loss_components(history):
    components = [
        ("box", "Box loss"),
        ("objectness", "Objectness loss"),
        ("classification", "Classification loss"),
    ]
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for axis, (column, title) in zip(axes, components):
        axis.plot(history["epoch"], history[f"train_{column}"], label="Training")
        axis.plot(history["epoch"], history[f"val_{column}"], label="Validation")
        axis.set_title(title)
        axis.set_ylabel("Loss")
        axis.legend(frameon=False)
    axes[-1].set_xlabel("Epoch")
    axes[-1].set_xticks(history["epoch"])
    figure.suptitle("SafetyNet v2 Loss Components", fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    figure.savefig(
        OUTPUT_DIR / "02_safetynet_loss_components.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_objectness(history):
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, prefix, title in (
        (axes[0], "train", "Training"),
        (axes[1], "val", "Validation"),
    ):
        axis.plot(
            history["epoch"],
            history[f"{prefix}_positive_obj_prob"],
            marker="o",
            label="Object cells",
        )
        axis.plot(
            history["epoch"],
            history[f"{prefix}_background_obj_prob"],
            marker="o",
            label="Background cells",
        )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylim(0, 1)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Mean objectness probability")
    figure.suptitle("SafetyNet v2 Foreground/Background Objectness Separation", fontsize=16)
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    figure.savefig(
        OUTPUT_DIR / "03_safetynet_objectness.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_learning_rate(history):
    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["learning_rate"], marker="o")
    plt.title("SafetyNet v2 Learning-Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.xticks(history["epoch"])
    save_figure("04_safetynet_learning_rate.png")


def main():
    history = load_history()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_total_loss(history)
    plot_loss_components(history)
    plot_objectness(history)
    plot_learning_rate(history)

    best_index = history["val_total"].idxmin()
    print(f"Created SafetyNet training figures in {OUTPUT_DIR.resolve()}")
    print(
        "Minimum validation loss: "
        f"epoch {int(history.loc[best_index, 'epoch'])}, "
        f"{history.loc[best_index, 'val_total']:.6f}"
    )
    print(
        "Skipped numerical SafetyNet v1-v2 objectness comparison: "
        "the v1 history has no object/background probability columns."
    )


if __name__ == "__main__":
    main()
