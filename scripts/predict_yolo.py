from __future__ import annotations

import argparse
from pathlib import Path

from waldo_ai.geometry import Box
from waldo_ai.prediction_io import boxes_to_csv
from waldo_ai.yolo_io import image_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Ultralytics predictions in the common evaluation format.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    predictions = []
    for image_path in image_files(args.images):
        result = model.predict(
            source=str(image_path), imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
        )[0]
        for coordinates, class_id, confidence in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            predictions.append(
                Box(
                    image_id=image_path.name,
                    class_id=int(class_id),
                    x1=float(coordinates[0]),
                    y1=float(coordinates[1]),
                    x2=float(coordinates[2]),
                    y2=float(coordinates[3]),
                    confidence=float(confidence),
                )
            )
    boxes_to_csv(predictions, args.output)
    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()

