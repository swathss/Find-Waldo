# Find Waldo

Detect Waldo in "Where's Waldo?" puzzle pages with a YOLOv8 detector. The model
is trained on native-scale page tiles, and the decoy characters Odlaw and Wizard
are added as hard negatives so it learns Waldo specifically rather than any
red-and-white pattern.

## Motivation

"Where's Waldo" is a tiny-target search in dense, self-similar clutter. Waldo is
often only 10 to 40 pixels inside a page that is 2000 pixels or more across, and
the background is full of red-and-white distractors that look like him. The main
difficulty is not the detector, it is the data: the public datasets hold only a
few hundred labelled images, which is far too few, and shrinking a whole page to
a small square destroys the few pixels that make up Waldo.

This project addresses both problems:

1. It slices full pages into fixed 640 pixel tiles at native resolution, so
   Waldo stays large enough to learn and the training scale matches inference.
2. It adds decoy characters as hard negatives, which cuts the confident false
   positives that are the main failure mode on these pages.

## Approach

1. Label Waldo on real book pages (12 books).
2. Split the data by book, so no book appears in more than one split. This tests
   cross-book generalization and removes leakage.
3. Tile each page into 640 pixel tiles. A tile that contains Waldo is a positive
   with the box re-computed inside the tile; other tiles are background
   negatives, capped at roughly three negatives per positive so they do not
   drown the positives.
4. Mine real Odlaw and Wizard crops from a separate 5-class set and paste them
   onto Waldo-free tiles with empty labels. This teaches the model that a
   partial match (wrong-colour stripes, a beard, a lone hat) is not Waldo,
   without ever labelling a real Waldo as negative.
5. Train YOLOv8s. At inference, slide a 640 pixel window across the full page and
   merge the per-tile detections with non-maximum suppression, so any page size
   works.

An earlier approach composited cut-out Waldos onto crowd backgrounds
(synthetic data). It did not transfer to real pages (it scored close to zero on
held-out pages), so the shipped model trains on real tiles. That code is kept
under `synth/` for reference and comparison.

## Quick start

```bash
conda env create -f environment.yml && conda activate waldo-finder   # or: pip install -r requirements.txt

python scripts/build_book_dataset.py                          # tile labelled pages, split by book
python scripts/build_decoys.py                                # add Odlaw/Wizard hard negatives
python scripts/train_synth.py --data data/book_decoy.yaml \
    --model yolov8s.pt --epochs 100 --name waldo_book_decoy --close-mosaic 10
python scripts/detect_any.py --image path/to/page.jpg \
    --weights models/waldo_book_decoy/weights/best.pt         # find Waldo in any page
```

## Project layout

```
waldo-finder/
  scripts/
    build_book_dataset.py   tile labelled pages into 640 tiles, split by book
    build_decoys.py         mine Odlaw/Wizard crops, build decoy-negative tiles
    train_synth.py          train YOLOv8 (auto-selects MPS, CUDA, or CPU)
    detect_any.py           sliding-window tiled inference on any image
    eval_synth.py           mAP, precision, recall on a held-out split
    compare_models.py       run several models on the same test set
  synth/                    earlier synthetic-compositing pipeline (reference)
  src/                      earlier baseline scripts (template match, prep)
  web/                      Flask demo (upload an image, get detections)
  data/                     datasets, kept local and git-ignored
  DATASETS.md               source inventory, licences, leakage controls
```

## Data

- Book pages: 12 books, 130 pages, 119 with Waldo and 9 without. The raw scans
  are copyrighted book pages and are kept local (git-ignored). The labels and
  the build scripts reproduce the tiled training set.
- 5-class set (Odlaw, Waldo, Wilma, Wizard, woof): used only as a source of real
  Odlaw and Wizard crops for the decoy negatives.
- Hey-Waldo and HereIsWally: used in earlier experiments. See DATASETS.md for
  sources, licences, and the leakage-safe splitting.

## Results

Single-class `yolov8s`, evaluated on a held-out test set split by book. Test
books `b04` and `b09` were never seen in training. Test set: 21 pages, 58 Waldo
instances (small, so treat as directional). Page hit-rate = top-1 box on the
real Waldo at IoU>=0.5. Confidence intervals are 95% Wilson score intervals.

### Headline: how the result was built up

| Stage | Page hit-rate | 95% CI |
|---|---|---|
| Synthetic-only | 0/21 = 0.00 | [0.00, 0.15] |
| Real page tiles | 10/21 = 0.48 | [0.28, 0.68] |
| + decoy hard-negatives (single-scale inference) | 13/21 = 0.62 | [0.41, 0.79] |
| + multi-scale WBF inference (locked result) | 16/21 = 0.76 | [0.55, 0.89] |

The large, defensible jump is synthetic (0.00) to real page tiles (0.48):
changing what the model trains on, from synthetic composites to real page tiles,
is the core result. Decoy negatives and multi-scale inference add further gains;
at n=21 the single-step increments have overlapping CIs, so the honest claim is
the cumulative 0.00 to 0.76, not each step in isolation.

### Effect of the decoy negatives (single-scale, same weights)

| Metric | Real tiles | + decoys |
|---|---|---|
| Precision | 0.54 | 0.79 |
| Recall | 0.63 | 0.57 |
| Tile mAP@0.5 | 0.60 | 0.66 |

Precision (0.54 to 0.79) and threshold-independent mAP@0.5 (0.60 to 0.66) are
the statistically credible gains from the decoys: precision is computed over
many detections and mAP@0.5 does not depend on a confidence threshold. Recall
and precision are at a fixed 0.25 threshold, so that split is partly threshold
placement.

### A larger training set did not help

A second model trained on multi-scale tiles plus copy-paste augmentation scored
15/21 = 0.71 on the same pages, a one-page regression versus 0.76. That is
inside the n=21 noise and the two CIs overlap, so it is a tie; the reported
model is the decoy model at 0.76. Book-fold cross-validation is the honest
tiebreaker and is future work.

On a completely unseen book ("Where's Waldo Now"), the model finds Waldo on a
minority of pages, which is the expected out-of-distribution gap and a target
for future work.

## How it finds Waldo in any image

Waldo is too small to survive shrinking a whole page, so `scripts/detect_any.py`
slides a 640 pixel window across the image at native resolution, runs the
detector on each tile, maps every box back to page coordinates, and merges them
with a global non-maximum suppression. One model then handles a phone photo, a
book scan, or a new illustration.

## Credits and prior art

- Hey-Waldo dataset, Valentino Constantinou (vc1492a): https://github.com/vc1492a/Hey-Waldo
- HereIsWally, Tadej Magajna: https://github.com/tadejmagajna/HereIsWally (MIT)
- YOLOv8, Ultralytics: https://github.com/ultralytics/ultralytics

## License

Code is released under the repository license. The datasets retain their
original licenses (see Credits and DATASETS.md). Raw book-page scans are
copyrighted by their publishers and are not redistributed here.
