from __future__ import annotations

import argparse
import json
from pathlib import Path

from waldo_ai.metrics import evaluate
from waldo_ai.prediction_io import boxes_from_csv
from waldo_ai.yolo_io import class_names, load_dataset_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions with one metric implementation.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    _, config = load_dataset_yaml(args.dataset)
    names = class_names(config)
    per_class, errors, summary = evaluate(
        boxes_from_csv(args.truth), boxes_from_csv(args.predictions), names, args.iou
    )
    args.output.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(args.output / "per_class_metrics.csv", index=False)
    errors.to_csv(args.output / "matched_predictions_and_errors.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

