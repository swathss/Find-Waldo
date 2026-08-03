# Where Is Waldo? Classical Visual Search vs YOLO

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/-bKyY6qM)

This ACM40960 project trains and evaluates an AI system that locates Wally
(Waldo) in crowded illustrations. It compares classical multi-scale template
matching with a full-page YOLOv8n detector and an overlapping-tile YOLOv8n
detector designed for small objects.

Authors: Caixuan Du (25254693) and Swathi Ganesh (25204854).

## Research question

Can a small YOLO detector localise Wally on unseen crowded pages more accurately
than template matching, and does overlapping image tiling improve detection of
the smallest instances?

## Experimental design

| Method | Description | Role |
|---|---|---|
| Template matching | Bank of Wally crops from training images, matched at multiple scales | Classical baseline |
| YOLOv8n-full | COCO-pretrained YOLOv8n fine-tuned on complete 512 x 512 images | Neural baseline |
| YOLOv8n-tiles | YOLOv8n trained on 256 px tiles with 64 px overlap | Main model |

The main dataset is Wally-Finder v5 from Roboflow Universe. It provides 249
images and four character classes. Some labels are segmentation polygons, which
this project converts to tight detection boxes. The published split contains
source-page leakage, so all reported experiments use a reproducible grouped
split. Raw images are not stored in Git; see [data/README.md](data/README.md).

## Repository structure

```text
data/                 Local data instructions; raw data is ignored by Git
docs/                 Dataset card, annotation policy and experiment plan
notebooks/            Colab/Jupyter end-to-end workflow
scripts/              Auditing, tiling, training, prediction and evaluation CLIs
src/waldo_ai/         Shared implementation used by scripts and notebooks
tests/                 Unit tests for geometry and evaluation
REFERENCES.bib         Papers, datasets and software used by the project
environment.yml       Reproducible Conda environment
```

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate waldo-ai
pip install -e .
```

### Google Colab

Open `notebooks/waldo_pipeline_colab.ipynb`, enable a GPU runtime, and run the
cells in order. The notebook installs the package and calls the same scripts as
the command-line workflow.

## Data preparation

1. Download the YOLOv8 export of Wally-Finder v5.
2. Extract it to `data/roboflow/wally-finder-v5/`.
3. Audit the published split. This dataset contains multiple exports derived
   from the same source IDs, so the published split must not be used for final
   metrics:

```bash
python scripts/audit_dataset.py \
  --dataset data/roboflow/wally-finder-v5 \
  --output artifacts/audit_original
```

4. Create and audit a leakage-free group split:

```bash
python scripts/create_grouped_split.py \
  --dataset data/roboflow/wally-finder-v5 \
  --output data/processed/wally-finder-grouped --seed 42

python scripts/audit_dataset.py \
  --dataset data/processed/wally-finder-grouped \
  --output artifacts/audit_grouped
```

All experiments below use `data/processed/wally-finder-grouped`. Review its
`split_manifest.csv` plus the audit JSON and CSV files before training.

## Reproduce the experiments

### 1. Template-matching baseline

```bash
python scripts/run_baseline.py \
  --dataset data/processed/wally-finder-grouped \
  --split test \
  --output artifacts/baseline
```

### 2. Full-page YOLO

```bash
python scripts/train_yolo.py \
  --data data/processed/wally-finder-grouped/data.yaml \
  --name yolov8n_full \
  --epochs 80 --imgsz 512 --device 0
```

### 3. Image tiling and tiled YOLO

Tiles inherit the split of their source image. This is important: splitting
after slicing would leak nearly identical content across train and test sets.

```bash
python scripts/slice_dataset.py \
  --dataset data/processed/wally-finder-grouped \
  --output data/processed/wally-finder-tiles \
  --tile-size 256 --overlap 64 --negative-ratio 3

python scripts/train_yolo.py \
  --data data/processed/wally-finder-tiles/data.yaml \
  --name yolov8n_tiles \
  --epochs 100 --imgsz 512 --device 0
```

### 4. Page-level prediction and evaluation

```bash
python scripts/export_ground_truth.py \
  --dataset data/processed/wally-finder-grouped \
  --split test --output artifacts/test_ground_truth.csv

python scripts/predict_yolo.py \
  --weights runs/detect/yolov8n_full/weights/best.pt \
  --images data/processed/wally-finder-grouped/test/images \
  --output artifacts/yolov8n_full_predictions.csv --device 0

python scripts/predict_tiled.py \
  --weights runs/detect/yolov8n_tiles/weights/best.pt \
  --tiled-dataset data/processed/wally-finder-tiles \
  --output artifacts/yolov8n_tiles_predictions.csv --device 0

python scripts/evaluate_predictions.py \
  --dataset data/processed/wally-finder-grouped \
  --truth artifacts/test_ground_truth.csv \
  --predictions artifacts/yolov8n_tiles_predictions.csv \
  --output artifacts/evaluation/yolov8n_tiles
```

## Evaluation

The primary endpoint is Wally AP@0.50 on the untouched test split. Secondary
measures include mAP@0.50, mAP@0.50:0.95 from Ultralytics, precision, recall,
F1, page-level success rate, false positives per page, inference time and
performance by object size. Confidence thresholds are selected on validation
data, never on the test set.

## Current status

- [x] Research question and literature review
- [x] Reproducible repository structure
- [x] Dataset audit and leakage-check code
- [x] Template-matching baseline implementation
- [x] Image-slicing and page-level merge implementation
- [x] YOLO training and prediction scripts
- [x] Complete primary dataset audit and grouped re-split
- [x] Run and evaluate the template-matching baseline
- [x] Generate and audit the tiled dataset
- [ ] Train full-page YOLOv8n
- [ ] Train tiled YOLOv8n
- [ ] Run test-set comparison and error analysis
- [ ] Add final figures and poster

## Data, licensing and attribution

Project code is MIT licensed. Dataset and artwork rights are separate. The
Wally-Finder dataset page displays CC BY 4.0; the Hey-Waldo database uses ODbL
1.0 while warning that individual image rights may differ. Raw images and
generated tiles are therefore excluded from this repository.

All external papers, datasets, software and tutorials used by the project are
listed in [REFERENCES.bib](REFERENCES.bib). The implementation is original
project code built on documented APIs rather than copied tutorial repositories.
