"""
Step 3: Fine-tune YOLOv8-nano on the Waldo dataset.

YOLOv8 by Ultralytics: https://github.com/ultralytics/ultralytics
We start from the COCO-pretrained 'yolov8n.pt' weights (nano = fastest,
suitable for a small dataset).  Swap to 'yolov8s.pt' or 'yolov8m.pt'
for more capacity if you have GPU and more data.
"""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
DATA_YAML = ROOT / "data" / "processed" / "data.yaml"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"{DATA_YAML} not found. Run  python src/prepare_dataset.py  first."
        )

    model = YOLO("yolov8n.pt")   # downloads ~6 MB on first run

    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        name="waldo_yolov8n",
        project=str(MODELS_DIR),
        patience=15,          # early-stop if val loss stalls
        augment=True,         # built-in mosaic, flip, hsv, etc.
        exist_ok=True,
    )

    best = MODELS_DIR / "waldo_yolov8n" / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best}")
    print("Next: run  python src/evaluate.py")


if __name__ == "__main__":
    main()
