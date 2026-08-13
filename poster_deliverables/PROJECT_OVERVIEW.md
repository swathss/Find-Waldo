# Project overview

Detect Waldo in "Where's Waldo" puzzle pages, including pages the model never
trained on. The detector is standard (YOLOv8s); the contribution is in the data
and the evaluation: training on native-scale page tiles, adding decoy
characters as hard negatives, multi-scale tiled inference, and honest
leakage-safe evaluation.

Reported model: **waldo_book_decoy**, page hit-rate **0.76 [0.55, 0.89]** on
held-out books. Full numbers in `METRICS.md`.

## Pipeline

1. **Label** Waldo on real book pages (12 books, 128 pages).
2. **Split by book** so no book appears in more than one split (tests
   cross-book generalization, no leakage).
3. **Tile** each page into fixed-size tiles at native resolution, so Waldo
   stays 15-40 px and training scale matches inference. Background tiles are
   negatives, capped ~3:1.
4. **Decoy hard-negatives**: paste real Odlaw and Wizard crops onto Waldo-free
   tiles with empty labels, so the model learns Waldo specifically, not any
   red/white or striped pattern.
5. **Train** YOLOv8s from COCO weights.
6. **Detect** with multi-scale sliding-window tiling (320/512/768) fused with
   Weighted Boxes Fusion, taking the top-1 box per page.

## What was built (components)

- **synth/** - an earlier synthetic-compositing pipeline (Track A). It did not
  transfer to real pages (~0.00), kept as a documented negative result.
- **scripts/build_book_dataset.py** - tile labelled pages, split by book.
- **scripts/build_decoys.py** - mine Odlaw/Wizard crops and build decoy
  negatives.
- **scripts/train_synth.py** - training (auto-selects MPS / CUDA / CPU;
  supports --resume).
- **scripts/detect_any.py** - single-scale tiled inference on any image.
- **scripts/detect_multiscale.py** - multi-scale tiled inference with Weighted
  Boxes Fusion (the reported 0.76 pipeline).
- **scripts/eval_synth.py, compare_models.py, book_fold_cv.py** - evaluation.
- **scripts/build_dataset_ms.py** - the multi-scale + copy-paste dataset builder
  for the waldo_book_ms experiment.
- **web/** - a Flask demo (see below).

## Models trained

| Model | Data | Purpose |
|---|---|---|
| waldo_yolov8n | old single-class set | earliest baseline (leaky, not comparable) |
| waldo_synth / waldo_synth_A | synthetic composites | Track A, negative result (~0.00 on real pages) |
| waldo_book_A | real book tiles | first model that finds real Waldo (0.48) |
| **waldo_book_decoy** | real tiles + decoys | **reported model (0.76)** |
| waldo_book_ms | multi-scale tiles + copy-paste | larger-data experiment, tied with decoy |

## Web demo (`web/app.py`)

A local Flask app to try the detector on any image.

- **Upload** a page.
- **Enhance (optional)** with one of three backends:
  - Sharpen + upscale (local, fast: denoise, Lanczos upscale, unsharp mask),
  - Real-ESRGAN (local super-resolution),
  - finegrain image enhancer (a diffusion upscaler via a Hugging Face Space,
    needs internet, ~30-60 s). Shows a before/after.
- **Confidence threshold** slider.
- **Find Waldo** runs the multi-scale WBF pipeline and shows results: a
  found / not-found badge, stats (detections, best confidence, size, time), and
  the detections list.
- Detection display: only the **top-1 box** is drawn by default; clicking any
  row in the list toggles that detection's box on the image (boxes are browser
  overlays, so they stay crisp and scale with the image).

Run: `python web/app.py`, then open http://localhost:8080.

## Evaluation

- Held-out test = books b04 + b09 (split by book).
- Metrics: page hit-rate (top-1 on real Waldo, IoU >= 0.5), tile mAP, precision,
  recall, all with 95% Wilson CIs. See `METRICS.md` and `eval_summary.md`.
- Ablation isolates the decoy effect; a second model isolates multi-scale
  training + copy-paste.

## Known limitations

- Small test set (n=21 pages), so single-step gains have overlapping CIs;
  book-fold cross-validation would firm up the numbers.
- Out-of-distribution gap: on a brand-new book's art style the model finds
  Waldo on a minority of pages with higher confidence.
- 5 test pages remain unsolved (tiny / occluded / camouflaged), and 2 of them
  are labelled on the look-alike Wenda rather than Waldo.

## Future work

- Book-fold cross-validation for a spread on the headline number.
- More books labelled to reduce the out-of-distribution gap.
- Multi-class detection (Waldo, Odlaw, Wizard, Wenda) for a confusion matrix.
- Proper upscaler can be introduced to avoid very low clarity images.

## Files in this folder

See `MANIFEST.md` for the full list and `SOURCES.md` for data/tool provenance.
