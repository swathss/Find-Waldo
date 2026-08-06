# Find Waldo in *any* image

Detect Waldo in any "Where's Waldo?" picture, including scans, photos, and
brand-new or AI-generated illustrations, by training a detector on a
**synthetic data engine** instead of the few hundred real labelled images.

![example detection](results/detect_any.jpg)

## Motivation

"Where's Waldo" is a tiny-target search in dense, self-similar clutter: Waldo
is often ~10-40 px inside a 2000+ px page, surrounded by red-and-white
distractors that look just like him. The real bottleneck is not the model, it
is **labelled data**: the public datasets hold only a few hundred images, far
too few to generalize to unseen puzzles.

This project **manufactures** the data. It mines real Waldo cut-outs, then
composites them onto crowd backgrounds with heavy randomization and hard-
negative distractors to generate thousands of varied, labelled scenes. The
detector is trained mostly on synthetic images and **validated on real,
held-out ones**, so the score measures generalization rather than memorization.

Full method write-up: [SYNTH_APPROACH.md](SYNTH_APPROACH.md).

## Quick start

```bash
conda env create -f environment.yml && conda activate waldo-finder   # or: pip install -r requirements.txt

python src/download_dataset.py                              # clone licensed Git sources
python src/merge_datasets.py                                # add Wally-Finder; group-safe splits
python scripts/preview_dataset.py                           # inspect converted bounding boxes
python -m synth.foregrounds                                   # mine Waldo cut-outs
python -m synth.generate --train 3000 --val 500 --test 300    # build synthetic+real dataset
python scripts/train_synth.py --model yolov8s.pt --epochs 100 # train (Apple MPS / CUDA / CPU)
python scripts/eval_synth.py                                  # mAP + confusion matrix (real test split)
python scripts/detect_any.py --image path/to/puzzle.jpg       # find Waldo in ANY image
```

## Project layout

```
waldo-finder/
├── synth/                  # synthetic data engine
│   ├── foregrounds.py      #   mine real Waldo boxes -> alpha cut-outs
│   ├── backgrounds.py      #   sample Waldo-free crowd tiles
│   ├── compositor.py       #   paste + augment + distractors -> image + box
│   └── generate.py         #   build a YOLO dataset (synthetic + real)
├── scripts/
│   ├── train_synth.py      # train YOLOv8 (auto-selects MPS/CUDA/CPU)
│   ├── eval_synth.py       # mAP, precision/recall, confusion matrix
│   └── detect_any.py       # sliding-window tiled inference on any image
├── src/                    # earlier baseline scripts (template match, etc.)
├── web/                    # Flask demo (upload an image, get detections)
├── data/                   # datasets (git-ignored, regenerable)
├── DATASETS.md             # source inventory, licences, leakage controls
└── results/                # inference output images
```

## Data used

The local collection now combines three published sources:

| Dataset | Available data | How it is used |
|---|---:|---|
| Hey-Waldo | 317 colour 256 px patches | 286 hard negatives/backgrounds; positive classification patches are not promoted to inaccurate whole-image boxes |
| HereIsWally | 36 scenes, 43 Waldo boxes | Real labelled crops |
| Wally-Finder v5 | 249 images, 49 Wally boxes | Additional Wally examples and other-character hard negatives |

The merge is leakage-safe at the source-page level. For example,
`13.jpg`, `13_0_2.jpg`, and `13_jpg.rf.<hash>.jpg` are treated as the same
puzzle group and can only occur in one split. The generated
`data/processed_v2/manifest.csv` makes every assignment auditable. See
[DATASETS.md](DATASETS.md) for download instructions, licences, and caveats.

## How it finds Waldo in any image

Waldo is too small to survive downscaling a whole page, so
`scripts/detect_any.py` slides a 640 px window over the image at native
resolution, detects per tile, maps every box back to global coordinates, and
merges them with a global non-maximum suppression. One model then handles a
phone photo, a book scan, or a freshly generated illustration.

## Results

Produced by `scripts/eval_synth.py` (confusion matrix + PR curves under
`models/waldo_synth_eval/`), measured on the held-out test split.

| Metric (test split) | Value |
|---|---|
| mAP@0.5 | 0.796 |
| mAP@0.5:0.95 | 0.472 |
| Precision / Recall | 0.834 / 0.691 |

## Credits & prior art

- Hey-Waldo dataset, Valentino Constantinou (vc1492a): https://github.com/vc1492a/Hey-Waldo (ODbL 1.0 database licence)
- HereIsWally, Tadej Magajna: https://github.com/tadejmagajna/HereIsWally (MIT)
- Wally-Finder v5: https://universe.roboflow.com/wheres-wally/wally-finder/dataset/5 (CC BY 4.0)
- YOLOv8, Ultralytics: https://github.com/ultralytics/ultralytics
- Synthetic copy-paste augmentation is inspired by "Cut, Paste and Learn" (Dwibedi et al., 2017).

## License

See [LICENSE](LICENSE) if present; datasets retain their original licenses (above).
