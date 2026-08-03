from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from waldo_ai.baseline import build_template_bank, find_wally_class, match_template_bank
from waldo_ai.prediction_io import boxes_to_csv
from waldo_ai.yolo_io import image_files, split_image_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run the multi-scale template-matching baseline.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    template_dir = args.output / "templates"
    templates = build_template_bank(args.dataset, template_dir)
    class_id, _ = find_wally_class(args.dataset)
    predictions = []
    for image_path in tqdm(image_files(split_image_dir(args.dataset, args.split)), desc="Template matching"):
        predictions.extend(match_template_bank(image_path, templates, class_id, top_k=args.top_k))
    boxes_to_csv(predictions, args.output / f"predictions_{args.split}.csv")
    print(f"Saved {len(predictions)} predictions using {len(templates)} templates.")


if __name__ == "__main__":
    main()

