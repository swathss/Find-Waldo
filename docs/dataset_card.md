# Dataset card: Wally-Finder v5

## Intended use

Academic comparison of classical visual search and small-object detection on
crowded cartoon illustrations. The project focuses on localisation and error
analysis, not commercial deployment.

## Dataset summary

The Roboflow Universe page reports 249 object-detection images, four character
classes, a 166/41/42 train/validation/test split, automatic orientation, and
stretch resizing to 512 x 512. The page displays a CC BY 4.0 license.

## Known limitations to audit

- Images may originate from copyrighted books or web sources even though the
  dataset page supplies an annotation/dataset license.
- Some images may be crops or transformed versions of the same original page.
- Stretch resizing changes character aspect ratios.
- Class counts may be strongly imbalanced.
- The published split may contain near duplicates.
- Small objects make annotation boxes sensitive to a few pixels of error.

## Required validation before modelling

1. Verify that every image opens and has a matching label file.
2. Verify label class IDs and normalized coordinates.
3. Count boxes and images per class and split.
4. Detect exact and perceptual duplicates across splits.
5. Visualise a stratified sample of labels.
6. Record box area relative to page area.
7. If leakage is found, group duplicates and regenerate a group-aware split.

## Audit outcome (3 August 2026)

The downloaded version contained 249 readable images and valid YOLO labels.
Some labels use YOLO segmentation polygons; these are converted to their tight
axis-aligned bounding boxes for the detection comparison. The original
Roboflow split placed 34 repeated source-filename groups in more than one split.
The project therefore uses a deterministic source-group split (seed 42): 176
train images, 34 validation images and 39 test images. A second audit found no
source groups crossing those splits.

## Distribution policy

Raw images and derived tiles remain local and are ignored by Git. Aggregate
statistics, code, evaluation tables, and low-resolution figures used under an
appropriate academic exception may be committed with attribution.
