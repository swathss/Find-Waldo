"""
Alternative dataset prep: instead of feeding full scenes to YOLO,
extract 640x640 crops centered on Waldo's ground-truth location.

This gives the model a consistent, detectable Waldo size (~32-80px in 640px)
instead of the ~10px Waldo that appears when full 2048px scenes are resized.

Run this instead of prepare_dataset.py:
  python src/prepare_tiled_dataset.py
  python src/train.py
"""

import csv
import shutil
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed_tiled"

TILE = 640         # crop size fed to YOLO
CLASS_ID = 0
SEED = 42
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def yolo_line(cx, cy, w, h):
    return f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"


def collect():
    import cv2, numpy as np
    base = RAW / "HereIsWally"
    csv_path = base / "annotations" / "annotations.csv"
    img_dir  = base / "images"

    rows_by_file = {}
    for r in csv.DictReader(open(csv_path)):
        rows_by_file.setdefault(r["filename"], []).append(r)

    samples = []
    for filename, rows in rows_by_file.items():
        img_path = img_dir / filename
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        for row in rows:
            xmin = float(row["xmin"]); ymin = float(row["ymin"])
            xmax = float(row["xmax"]); ymax = float(row["ymax"])
            wcx  = (xmin + xmax) / 2;  wcy = (ymin + ymax) / 2

            # Center a TILExTILE crop on Waldo
            x1 = int(max(0, wcx - TILE / 2))
            y1 = int(max(0, wcy - TILE / 2))
            x2 = min(iw, x1 + TILE)
            y2 = min(ih, y1 + TILE)
            x1 = max(0, x2 - TILE)
            y1 = max(0, y2 - TILE)

            crop = img[y1:y2, x1:x2]
            cw, ch = x2 - x1, y2 - y1

            # Waldo coords in crop space
            cx = ((xmin + xmax) / 2 - x1) / cw
            cy = ((ymin + ymax) / 2 - y1) / ch
            bw = (xmax - xmin) / cw
            bh = (ymax - ymin) / ch
            cx = max(0.01, min(0.99, cx))
            cy = max(0.01, min(0.99, cy))

            stem = f"{Path(filename).stem}_crop"
            samples.append((crop, yolo_line(cx, cy, bw, bh), stem))

    print(f"  HereIsWally tile-crops: {len(samples)}")
    return samples


def write_split(samples, split_name, base):
    img_dir = base / split_name / "images"
    lbl_dir = base / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    import cv2
    for i, (crop, label, stem) in enumerate(samples):
        cv2.imwrite(str(img_dir / f"{split_name}_{i:04d}.jpg"), crop)
        (lbl_dir / f"{split_name}_{i:04d}.txt").write_text(label)
    print(f"  {split_name:5s}: {len(samples)} crops")


def main():
    samples = collect()
    if not samples:
        print("[ERROR] No samples. Run download_dataset.py first.")
        return

    random.seed(SEED)
    random.shuffle(samples)
    n = len(samples)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])

    splits = {
        "train": samples[:n_train],
        "val":   samples[n_train:n_train + n_val],
        "test":  samples[n_train + n_val:],
    }

    print(f"\nWriting {n} tile-crops...")
    for name, subset in splits.items():
        write_split(subset, name, PROC)

    yaml = PROC / "data.yaml"
    yaml.write_text(f"""path: {PROC.resolve()}
train: train/images
val:   val/images
test:  test/images
nc: 1
names: ['waldo']
""")
    print(f"\nWrote {yaml}")
    print("Now edit src/train.py: change DATA_YAML to point to data/processed_tiled/data.yaml")
    print("Then run: python src/train.py")


if __name__ == "__main__":
    main()
