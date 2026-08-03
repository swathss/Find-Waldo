from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a reproducible YOLOv8n model.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="Use 0 on Colab GPU or cpu locally.")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        deterministic=True,
        patience=15,
        plots=True,
        save=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()

