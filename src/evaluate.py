"""
Step 4 – Evaluate the trained model on the held-out test set.

Reports:
  • mAP@0.5    (mean Average Precision at IoU threshold 0.5)
  • Precision
  • Recall
  • F1 score

Reference: YOLO evaluation docs
  https://docs.ultralytics.com/modes/val/
"""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
DATA_YAML = ROOT / "data" / "processed" / "data.yaml"
BEST_PT   = ROOT / "models" / "waldo_yolov8n" / "weights" / "best.pt"


def main():
    if not BEST_PT.exists():
        raise FileNotFoundError(
            f"{BEST_PT} not found. Run  python src/train.py  first."
        )

    model = YOLO(str(BEST_PT))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        conf=0.25,
        iou=0.5,
        verbose=True,
    )

    p   = metrics.box.mp          # mean precision
    r   = metrics.box.mr          # mean recall
    map50 = metrics.box.map50     # mAP@0.5
    f1  = 2 * p * r / (p + r + 1e-9)

    print("\n─── Test-set results ───────────────────────────")
    print(f"  mAP@0.5   : {map50:.4f}")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1        : {f1:.4f}")
    print("────────────────────────────────────────────────")
    print("Next: run  python src/predict.py --image <path>")


if __name__ == "__main__":
    main()
