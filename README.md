# Industrial Safety Object Detection

University computer-vision project for detecting personal protective equipment
and workplace-safety objects in the SH17 dataset. The project compares a
YOLO11n baseline with SafetyNet, a custom single-scale PyTorch detector.

The seven selected classes are `person`, `tool`, `helmet`, `safety-vest`,
`gloves`, `glasses`, and `face-mask`.

## Project structure

```text
src/
  app/           Minimal Streamlit video demo
  dataset/       Dataset inspection, preparation, and annotation visualization
  inference/     Reusable external-video inference backend
  yolo/          YOLO11n training and result analysis
  scratch/       Custom dataset, model, training, evaluation, and visualization
  visualization/ Presentation-ready training and model-comparison figures
archive/         Superseded experiments retained for reference
data/
  raw/SH17/                       Original SH17 dataset
  processed/SH17_safety_1920/     Active resized seven-class dataset
  processed/SH17_safety_unresized/ Legacy unresized processed dataset
  external/{images,videos}/       Inputs reserved for external testing
results/
  yolo/baseline/     Self-contained YOLO11n baseline experiment
  scratch/           Self-contained SafetyNet and smoke-test experiments
  dataset_analysis/  Dataset statistics and annotation samples
  final_figures/     Report-ready training and comparison figures
  external_tests/    Outputs reserved for future external tests
```

## Environment setup

Python 3.11 is recommended. Create and activate a virtual environment, then
install the PyTorch and torchvision builds appropriate for the machine's CPU or
CUDA setup. Install the remaining dependencies with:

```powershell
python -m pip install -r requirements.txt
```

Run all commands below from the project root.

## Main commands

Prepare the resized seven-class dataset (the script refuses to overwrite an
existing processed dataset):

```powershell
python src/dataset/prepare_dataset.py
```

Inspect or visualize the raw SH17 annotations:

```powershell
python src/dataset/inspect_dataset.py
python src/dataset/visualize_annotations.py
```

Train and analyze the YOLO11n baseline:

```powershell
python src/yolo/train.py
python src/yolo/analyze.py
```

Train and evaluate SafetyNet v2:

```powershell
python src/scratch/train.py
python src/scratch/evaluate.py
```

Create side-by-side SafetyNet prediction visualizations:

```powershell
python src/scratch/visualize.py --checkpoint results/scratch/v2/weights/best.pt
```

## Presentation visualizations

The visualization phase reads existing histories, metrics, and checkpoints. It
does not retrain either detector or modify the original experiment folders.
Generate the outputs in this order so the final summary includes the
qualitative manifest:

```powershell
python src/visualization/plot_safetynet_training.py
python src/visualization/qualitative_comparison.py --num-images 10 --yolo-conf 0.30 --safetynet-conf 0.60
python src/visualization/compare_models.py
```

All new figures, tables, and summaries are written under
`results/final_figures/`. The mAP values are the primary direct comparison;
reported precision and recall may use different operating points in the YOLO
and custom evaluators.

## Video detection demo

Process an external video from the command line:

```powershell
python src/inference/video.py --input data/external/videos/test.mp4 --output results/external_tests/test_annotated.mp4 --conf 0.30 --interval 5
```

Launch the minimal web interface with:

```powershell
streamlit run src/app/app.py
```

Both interfaces use `results/yolo/baseline/weights/best.pt`. YOLO inference is
performed every N frames. The most recent detections are reused on intermediate
frames to reduce processing cost. External source videos belong under
`data/external/videos/`, and generated videos belong under
`results/external_tests/`.

The demo prefers browser-compatible H.264 encoding. On systems where OpenCV
cannot provide H.264, it falls back to `mp4v`; that fallback may require using
the download button and a desktop video player.
