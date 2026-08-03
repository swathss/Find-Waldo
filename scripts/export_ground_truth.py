from __future__ import annotations

import argparse
from pathlib import Path

from waldo_ai.prediction_io import boxes_to_csv
from waldo_ai.yolo_io import image_files, read_yolo_labels, split_image_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO ground truth as one CSV for evaluation.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    boxes = []
    for image_path in image_files(split_image_dir(args.dataset, args.split)):
        boxes.extend(read_yolo_labels(image_path))
    boxes_to_csv(boxes, args.output)
    print(f"Saved {len(boxes)} boxes to {args.output}")


if __name__ == "__main__":
    main()

