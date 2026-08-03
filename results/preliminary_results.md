# Preliminary results

This file records completed checks and results before GPU training. Generated
images, model weights and row-level prediction files remain in the ignored
`artifacts/` directory; the concise numerical record below is tracked in Git.

## Dataset audit and split

- Main data: Wally-Finder v5, 249 images, four classes.
- Label parser accepts both standard YOLO boxes and polygon-style YOLO labels;
  polygons are converted to tight axis-aligned boxes.
- The published split has 34 source filename groups spanning more than one
  split. It is therefore excluded from final performance reporting.
- A deterministic source-grouped split (seed 42) has 176 train, 34 validation
  and 39 test images across 57 source groups.
- Character boxes by split are 75 train, 17 validation and 15 test.
- SHA-256/source-group audit of the grouped split reports no cross-split issues.
- Tiling produces 636 images: 159 positive tiles and 477 sampled background
  tiles. The tiled audit reports no split or label issues.

## Classical baseline

The multi-scale template matcher used 34 Wally templates from the grouped
training split and returned at most five detections per test page. Evaluation is
page-level at IoU 0.50 on all 39 grouped test images.

| Metric | Result |
|---|---:|
| Wally true positives / false positives / false negatives | 1 / 194 / 6 |
| Wally precision | 0.0051 |
| Wally recall | 0.1429 |
| Wally F1 | 0.0099 |
| Wally AP50 | 0.0075 |
| Four-class mAP50 | 0.0019 |
| Mean runtime per page | 1.70 s |

This deliberately simple baseline confirms that raw template similarity is
not robust to scale, pose and dense red-and-white distractors. YOLO results are
pending GPU training; no neural-model metric is reported yet.
