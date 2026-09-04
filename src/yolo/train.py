from multiprocessing import freeze_support
from pathlib import Path

from ultralytics import YOLO


DATA_YAML = Path("data/processed/SH17_safety_1920/data.yaml")
MODEL = Path("yolo11n.pt")
OUTPUT_DIR = Path("results/yolo").resolve()
EXPERIMENT_NAME = "baseline"


def main():
    print("=" * 60)
    print("EXPERIMENT 1 - YOLO11n 640 BASELINE")
    print("=" * 60)

    model = YOLO(str(MODEL))
    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        amp=False,
        patience=10,
        seed=42,
        deterministic=True,
        project=str(OUTPUT_DIR),
        name=EXPERIMENT_NAME,
        plots=True,
        save=True,
        save_period=5,
        verbose=True,
    )


if __name__ == "__main__":
    freeze_support()
    main()
