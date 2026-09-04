from multiprocessing import freeze_support
from ultralytics import YOLO


def main():

    model = YOLO("yolo11n.pt")

    model.train(
        data="data/processed/SH17_safety_1920/data.yaml",

        fraction=0.1,
        epochs=1,
        imgsz=640,

        batch=8,
        workers=4,

        device=0,
        amp=False,

        val=False,
        plots=False,

        project="results/training",
        name="speed_test_b8_w4",

        seed=42,
    )


if __name__ == "__main__":
    freeze_support()
    main()