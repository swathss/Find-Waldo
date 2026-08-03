from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from waldo_ai.geometry import Box, non_max_suppression
from waldo_ai.prediction_io import boxes_to_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict on saved test tiles and merge boxes onto source pages.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--tiled-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--nms", type=float, default=0.4)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    from ultralytics import YOLO

    manifest = pd.read_csv(args.tiled_dataset / "tile_manifest.csv")
    manifest = manifest[manifest["split"] == "test"]
    model = YOLO(str(args.weights))
    predictions_by_source: dict[str, list[Box]] = {}
    for row in manifest.itertuples(index=False):
        tile_path = args.tiled_dataset / "images" / "test" / row.tile
        result = model.predict(
            source=str(tile_path), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
        )[0]
        source_predictions = predictions_by_source.setdefault(row.source, [])
        for coordinates, class_id, confidence in zip(
            result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()
        ):
            source_predictions.append(
                Box(
                    image_id=row.source,
                    class_id=int(class_id),
                    x1=float(coordinates[0] + row.x0),
                    y1=float(coordinates[1] + row.y0),
                    x2=float(coordinates[2] + row.x0),
                    y2=float(coordinates[3] + row.y0),
                    confidence=float(confidence),
                )
            )
    merged = [
        box
        for source_predictions in predictions_by_source.values()
        for box in non_max_suppression(source_predictions, args.nms)
    ]
    boxes_to_csv(merged, args.output)
    print(f"Saved {len(merged)} merged page-level predictions to {args.output}")


if __name__ == "__main__":
    main()

