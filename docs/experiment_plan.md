# Experiment plan

## Research question

Can a small YOLO detector localise Wally in unseen crowded illustrations more
accurately than classical template matching, and does overlapping image tiling
improve performance on the smallest instances?

## Comparisons

| ID | Method | Input | Purpose |
|---|---|---|---|
| TM | Multi-scale template matching | Full 512 x 512 page | Classical baseline |
| YF | YOLOv8n transfer learning | Full 512 x 512 page | Neural baseline |
| YT | YOLOv8n transfer learning | 256 px tiles, 25% overlap | Main small-object model |

Every comparison uses the same leakage-safe, source-grouped validation and test
images. The published split is not used for final metrics because 34 source
filename groups cross its train/validation/test boundaries. Generated tiles
inherit the split of their source page; no tile from one page may appear in a
different split.

## Model design

- Architecture: Ultralytics YOLOv8n.
- Initialisation: COCO-pretrained weights (`yolov8n.pt`).
- Full-page input: 512 px.
- Tiled input: 256 px crops resized by YOLO to 512 px.
- Tile overlap: 64 px (25%).
- Optimisation: Ultralytics defaults, early stopping patience 15.
- Maximum epochs: 80 for full-page; 100 for tiles.
- Batch size: 16, automatically reduced if Colab memory is insufficient.
- Seed: 42.

## Primary and secondary outcomes

Primary outcome:

- Wally AP@0.50 on the untouched test split.

Secondary outcomes:

- mAP@0.50 and mAP@0.50:0.95 across all four classes.
- Wally precision, recall and F1 at a fixed confidence threshold selected on
  the validation split.
- Page success rate: at least one Wally detection with IoU >= 0.50.
- False positives per page.
- Median inference time per page.
- AP and recall by bounding-box area (small, medium, large terciles determined
  from training data).

## Error categories

Each false result is assigned one or more categories:

1. missed tiny instance;
2. confusion with Wanda, Wizard or Yllaw;
3. confusion with red-and-white background pattern;
4. localisation error (correct character, IoU < 0.50);
5. tile-boundary truncation;
6. duplicate detections after tile merging;
7. domain shift on the external Hey-Waldo pages.

## Leakage controls

- Run SHA-256 and perceptual-hash duplicate checks across splits.
- Never split after tiling.
- Never select confidence thresholds on the test set.
- Keep the external Hey-Waldo pages outside model development.
- Report dataset limitations and any removed duplicates.
