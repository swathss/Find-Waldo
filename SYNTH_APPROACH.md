# Method

## Problem

Waldo is a small target (often 10-40 px) hidden in a dense page full of
red-and-white distractors. The main limitation is data: the public datasets
only have a few hundred labelled images, which isn't enough to train a detector
that generalizes to new puzzles.

## Approach

Rather than fine-tune on the ~250 real images, we generate training data:

1. Take real Waldo boxes from the labelled data and cut them out with a rough
   alpha matte (`synth/foregrounds.py`).
2. Paste those cut-outs onto Waldo-free crowd backgrounds, randomizing scale,
   rotation, flip, colour, occlusion, and adding red/white striped patches
   nearby as unlabelled distractors (`synth/compositor.py`).
3. Generate a few thousand labelled tiles (`synth/generate.py`).

Cut-outs are mined from the train split only. The val/test Waldo images are
never used to make training data, so they stay a clean benchmark.

## Training and evaluation

The model is YOLOv8s, transfer-learned from COCO weights, trained mostly on
synthetic tiles. Validation and test include the real held-out images, so the
reported numbers reflect real puzzles, not synthetic self-consistency.

## Inference on arbitrary images

`scripts/detect_any.py` slides a 640 px window over an image at full
resolution, runs the detector on each tile, maps the boxes back, and merges
them with non-maximum suppression. This handles pages of any size.

## Comparison

`scripts/compare_models.py` runs template matching, the old YOLOv8n, and the
synthetic YOLOv8s on the same real test pages and prints one table.

## Pipeline

    python -m synth.foregrounds
    python -m synth.generate --train 3000 --val 500 --test 300
    python scripts/train_synth.py --model yolov8s.pt --epochs 100
    python scripts/eval_synth.py
    python scripts/compare_models.py
    python scripts/detect_any.py --image some_page.jpg
