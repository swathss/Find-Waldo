# Metrics reference

Every metric that matters for the project, in one place. Reported model is
**waldo_book_decoy**. Test set throughout is the held-out books **b04 + b09**
(split by book, never seen in training): 21 pages, 58 Waldo instances.

Two metric levels are used:
- **Page hit-rate**: does the single top-1 box land on the real Waldo
  (IoU >= 0.5)? This is the task-relevant headline metric.
- **Tile mAP / val mAP**: standard object-detection average precision at the
  tile level (used during training and for the decoy ablation).

95% confidence intervals are Wilson score intervals.

---

## 1. Headline (held-out test, multi-scale WBF inference)

| Metric | Value |
|---|---|
| Page hit-rate | **16/21 = 0.76** |
| 95% CI | **[0.55, 0.89]** |
| Inference | multi-scale tiles 320/512/768, WBF, conf 0.10, top-1 per page |

Baseline for context: the earlier synthetic-only model scored ~0.00 on these
pages (never localized the real Waldo).

## 2. Lineage (page hit-rate on b04/b09, n=21)

| Stage | Hits | Hit-rate | 95% CI |
|---|---|---|---|
| Synthetic-only (Track A, documented negative result) | 0/21 | 0.00 | [0.00, 0.15] |
| Real page tiles | 10/21 | 0.48 | [0.28, 0.68] |
| + decoy hard-negatives (single-scale inference) | 13/21 | 0.62 | [0.41, 0.79] |
| + multi-scale WBF inference | 16/21 | 0.76 | [0.55, 0.89] |

The large, defensible jump is synthetic (0.00) to real page tiles (0.48). The
single-step increments have overlapping CIs at n=21, so the honest claim is the
cumulative 0.00 to 0.76.

## 3. Decoy hard-negatives ablation (tile level, book_yolo test)

Same recipe, with and without the decoy negatives (Odlaw + Wizard crops).
Single-scale inference, so this isolates the decoy effect.

| Metric | Real tiles (waldo_book_A) | + decoys (waldo_book_decoy) |
|---|---|---|
| Tile mAP@0.5 | 0.60 | 0.66 |
| Tile mAP@0.5:0.95 | 0.24 | 0.24 |
| Precision | 0.54 | 0.79 |
| Recall | 0.63 | 0.57 |

Precision (0.54 -> 0.79) and threshold-independent mAP@0.5 (0.60 -> 0.66) are
the credible gains. Precision/recall are at a fixed 0.25 threshold, so that
split is partly threshold placement.

## 4. Training (validation) metrics

Validation books are b11 + b16. Best-epoch values from each run's results.csv.

| Model | Epochs | Best epoch | Val mAP@0.5 | Val mAP@0.5:0.95 | Val precision | Val recall |
|---|---|---|---|---|---|---|
| waldo_book_decoy (reported) | 100 | 92 | 0.752 | 0.315 | 0.748 | 0.746 |
| waldo_book_ms (multi-scale + copy-paste) | 88 | 68 | 0.813 | 0.378 | 0.804 | 0.793 |

Note: waldo_book_ms has higher validation mAP but does not beat the reported
model on the task-relevant test page hit-rate (see section 5). Val mAP is
tile-level; test page hit-rate is top-1 on the real Waldo.

## 5. Model comparison on the test (page hit-rate, multi-scale WBF)

| Model | Hits | Hit-rate | 95% CI |
|---|---|---|---|
| waldo_book_decoy | 16/21 | 0.76 | [0.55, 0.89] |
| waldo_book_ms | 15/21 | 0.71 | [0.50, 0.86] |

Net difference is 1 page, inside the n=21 noise and with overlapping CIs, so it
is a statistical tie. Reported model is waldo_book_decoy. Per-page hit / IoU /
confidence for both models is in `eval_headtohead.csv`. Book-fold
cross-validation is the honest tiebreaker and is future work.

## 6. Out-of-distribution stress test (unseen book)

"Where's Waldo Now" (not in training), pages 9-31, 23 pages, no labels.
Reported model, detections per page:

| Inference | Pages with a box |
|---|---|
| Single-scale, conf 0.25 | 5/23 |
| Single-scale, conf 0.10 | 7/23 |
| Multi-scale WBF, conf 0.10 | 7/23 |
| Multi-scale, conf 0.001 (raw best guess) | 23/23, but ~16 at near-zero confidence |

Even at a permissive threshold ~15 pages produce no real detection: a genuine
recognition gap on a new art style, not just under-confidence. Multi-scale
inference does not help here because the model often cannot recognize Waldo at
any scale on this book.

## 7. Failure analysis (the 5 missed test pages)

| Page | Waldo size (px) | Note |
|---|---|---|
| b04_p010 | 26 x 52 | tiny, dark forest, back turned (occluded) |
| b09_p030 | 61 x 83 | camouflaged beside a red/white striped awning (9.1% red/white around him) |
| b09_p019 | 40 x 71 | small, partly occluded by a railing |
| b09_p008 | - | miss |
| b09_p031 | 66 x 95 | reasonably large and clear, a genuine miss (not tiny/camouflaged) |

Box sizes across all 120 labelled instances: min 28 px, max 356 px, mean 102 px
(see `figures/box_size_histogram.png`).

Label caveat: 2 of the missed test pages are labelled on the look-alike
character Wenda, not Waldo, so they are arguably label noise rather than true
misses.

## 8. Inference settings (reported pipeline)

| Setting | Value |
|---|---|
| Detector | YOLOv8s, fine-tuned from COCO |
| Tiling | sliding window at native resolution |
| Tile sizes | 320, 512, 768 |
| Box fusion | Weighted Boxes Fusion across scales |
| Confidence | 0.10 (demo default; sweepable) |
| Output | top-1 box per page |
| Runtime | ~10-20 s per full page on Apple M4 (MPS) |
