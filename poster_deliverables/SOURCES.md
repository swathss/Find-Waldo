# Sources and provenance

Grouped by what each source was used for. Items marked "TO CONFIRM" need a
detail filled in by hand (exact edition or a Kaggle mirror URL).

---

## 1. Training positives - the book pages (real Waldo)

- What: 12 "Where's Waldo" / "Where's Wally" books, 128 pages labelled by us,
  120 Waldo bounding boxes. Used as the real positive training and evaluation
  data (tiled at native scale).
- Books used (folder slugs; confirm exact edition/publisher/year for citation):
  - b01 Where's Waldo Now?
  - b02 Where's Waldo? In Hollywood
  - b03 Where's Wally? The 25th Anniversary Annual
  - b04 The Great Waldo Search   (held-out test)
  - b05 Where's Waldo? The Magnificent Poster Book
  - b07 Where's Waldo? The Wonder Book
  - b08 Where's Waldo?
  - b09 Where's Waldo?           (held-out test)
  - b10 Where's Waldo?
  - b11 Where's Waldo? The Great Picture Hunt!   (validation)
  - b12 Where's Waldo? The Ultimate Fun Book
  - b16 Where's Wally?           (UK original)
- Rights and use: legally purchased Handford editions, used for
  non-commercial academic research only. The scans are NOT redistributed
  (kept local, git-ignored). Artwork and characters are
  (c) Martin Handford and the respective publisher.
- Our contribution: the bounding-box labels (single class, `waldo`), created
  by us. 

## 2. Training hard-negatives - the decoy characters (Odlaw, Wizard)

- What: real Odlaw and Wizard crops, pasted onto Waldo-free tiles as
  hard-negatives so the model learns Waldo specifically, not any red/white or
  striped pattern. Only Odlaw and Wizard were used (Wilma and woof excluded).
- Origin dataset (from `data/raw/kaggle_mohaneddz/data.yaml`):
  - Name: where-s-waldo-zu227 (version 1)
  - Author / workspace: mohaneds-workspace
  - Platform: Roboflow Universe
  - URL: https://universe.roboflow.com/mohaneds-workspace/where-s-waldo-zu227/dataset/1
  - Licence: CC BY 4.0
  - Obtained via a Kaggle mirror (local folder `kaggle_mohaneddz`).
- Annotations modified: yes. We did not use the dataset as shipped. We
  extracted only the Odlaw (class 0) and Wizard (class 3) crops from its
  bounding boxes and re-composited them as unlabelled negatives on new
  background tiles (`scripts/build_decoys.py`).
- Copyright-layer caveat: the CC BY 4.0 licence covers the Roboflow dataset
  and its annotation layer. The underlying artwork (the Odlaw and Wizard
  characters) is (c) Martin Handford and the publisher, so CC BY 4.0 does not
  grant rights to the original art. Used for non-commercial academic research.

## 3. Detection model - Ultralytics YOLOv8

- What: the detector architecture and training/inference framework. All models
  (`waldo_book_decoy`, `waldo_book_ms`, etc.) are YOLOv8s fine-tuned from
  COCO-pretrained weights.
- Package: ultralytics, version 8.4.60
- Licence: AGPL-3.0
- URL: https://github.com/ultralytics/ultralytics
- Note: AGPL-3.0 is a copyleft licence. It is fine for academic use; if code
  is ever deployed as a network service, AGPL obligations apply.

## 4. Inference and post-processing - tiling and box fusion

- Weighted Boxes Fusion (WBF): used to merge detections across the three
  inference scales (320/512/768) into one set of boxes.
  - Package: ensemble-boxes, version 1.0.9
  - Licence: MIT
  - URL: https://github.com/ZFTurbo/Weighted-Boxes-Fusion
- Sliding-window tiling: our own implementation
  (`scripts/detect_any.py`, `scripts/detect_multiscale.py`). We slice each page
  into overlapping tiles at native resolution, run the detector per tile, map
  boxes back to page coordinates, and fuse with WBF.
  - Honesty note: this is inspired by the SAHI (Slicing Aided Hyper Inference)
    method, but the SAHI library is NOT used and NOT a dependency
    (`import sahi` appears nowhere; the package is not installed). SAHI is
    cited below as a methodological reference only.

---

## Method references (papers, not code dependencies)

- Weighted Boxes Fusion: R. Solovyev, W. Wang, T. Gabruseva, "Weighted boxes
  fusion: Ensembling boxes from different object detection models," Image and
  Vision Computing, 2021.
- SAHI (tiling concept, not the library): F. C. Akyon, S. O. Altinuc, A. Temizel,
  "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection,"
  IEEE ICIP, 2022.
- Copy-paste augmentation (used to build extra positives in the `book_ms`
  experiment): D. Dwibedi, I. Misra, M. Hebert, "Cut, Paste and Learn: Surprisingly
  Easy Synthesis for Instance Detection," IEEE ICCV, 2017.
