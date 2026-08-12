"""
Benchmarks model performance across the full test set and prints a summary table.

Metrics per confidence threshold (0.1 -> 0.9):
  Precision, Recall, F1, mAP@0.5

Run after training:
  python src/benchmark.py
"""

from pathlib import Path
from ultralytics import YOLO
import json

ROOT = Path(__file__).parent.parent
DATA_YAML = ROOT / "data" / "processed" / "data.yaml"
BEST_PT   = ROOT / "models" / "waldo_yolov8n" / "weights" / "best.pt"


def main():
    if not BEST_PT.exists():
        print(f"[ERROR] {BEST_PT} not found. Run train.py first.")
        return

    model = YOLO(str(BEST_PT))

    print(f"\n{'Conf':>6} {'Precision':>10} {'Recall':>8} {'F1':>8} {'mAP50':>8}")
    print("-" * 46)

    rows = []
    for conf in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]:
        m = model.val(data=str(DATA_YAML), split="test",
                      imgsz=640, conf=conf, iou=0.5, verbose=False)
        p     = m.box.mp
        r     = m.box.mr
        map50 = m.box.map50
        f1    = 2 * p * r / (p + r + 1e-9)
        rows.append(dict(conf=conf, precision=p, recall=r, f1=f1, map50=map50))
        print(f"{conf:>6.2f} {p:>10.4f} {r:>8.4f} {f1:>8.4f} {map50:>8.4f}")

    out = ROOT / "results" / "benchmark.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nFull results saved -> {out}")


if __name__ == "__main__":
    main()
