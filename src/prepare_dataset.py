"""
Step 2 – Convert raw data → YOLO format dataset.

YOLO label format (one .txt per image, same stem):
  <class_id> <cx> <cy> <w> <h>   (all values 0-1, relative to image size)

We have two raw sources:
  A. HereIsWally: full scenes + CSV bboxes (annotations/annotations.csv)
  B. Hey-Waldo 64×64 patches → each positive patch is a full image,
       bbox = whole frame

The final split is 70 / 15 / 15  (train / val / test).
A data.yaml config file is written for YOLO training.
"""

import csv
import shutil
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
CLASS_ID = 0   # only one class: waldo
SEED = 42


# ── helpers ──────────────────────────────────────────────────────────────────

def yolo_line(cx, cy, w, h):
    return f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"


# ── collect samples ───────────────────────────────────────────────────────────

def collect_hereiswally():
    """
    Returns list of (img_path, label_lines) from HereIsWally repo.
    Annotations live in annotations/annotations.csv with columns:
      filename, width, height, class, xmin, ymin, xmax, ymax
    """
    base = RAW / "HereIsWally"
    csv_path = base / "annotations" / "annotations.csv"
    img_dir  = base / "images"

    if not csv_path.exists():
        print(f"  [warn] {csv_path} not found; skipping HereIsWally")
        return []

    # Group rows by filename
    rows_by_file = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows_by_file[row["filename"]].append(row)

    samples = []
    for filename, rows in rows_by_file.items():
        img_path = img_dir / filename
        if not img_path.exists():
            continue
        lines = []
        for row in rows:
            w_img = float(row["width"])
            h_img = float(row["height"])
            xmin  = float(row["xmin"])
            ymin  = float(row["ymin"])
            xmax  = float(row["xmax"])
            ymax  = float(row["ymax"])
            cx = ((xmin + xmax) / 2) / w_img
            cy = ((ymin + ymax) / 2) / h_img
            bw = (xmax - xmin) / w_img
            bh = (ymax - ymin) / h_img
            lines.append(yolo_line(cx, cy, bw, bh))
        if lines:
            samples.append((img_path, lines))

    print(f"  HereIsWally scenes with labels: {len(samples)}")
    return samples


def collect_heywaldo():
    """Returns list of (img_path, label_lines) from Hey-Waldo patches."""
    base = RAW / "Hey-Waldo" / "64" / "Waldo"
    if not base.exists():
        print("  [warn] Hey-Waldo/64/Waldo not found, skipping patches")
        return []
    samples = []
    for img_path in sorted(base.glob("*.jpg")):
        # Entire 64×64 patch is Waldo → bbox covers whole image
        lines = [yolo_line(0.5, 0.5, 1.0, 1.0)]
        samples.append((img_path, lines))
    print(f"  Hey-Waldo positive patches   : {len(samples)}")
    return samples


# ── split & write ─────────────────────────────────────────────────────────────

def write_split(samples, split_name):
    img_dir = PROC / split_name / "images"
    lbl_dir = PROC / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i, (src, lines) in enumerate(samples):
        stem = f"{split_name}_{i:04d}"
        shutil.copy(src, img_dir / (stem + src.suffix))
        with open(lbl_dir / (stem + ".txt"), "w") as f:
            f.writelines(lines)

    print(f"  {split_name:5s}: {len(samples)} images")


def write_yaml():
    yaml_path = PROC / "data.yaml"
    content = f"""# YOLO dataset config — Find Waldo
path: {PROC.resolve()}
train: train/images
val:   val/images
test:  test/images

nc: 1
names: ['waldo']
"""
    yaml_path.write_text(content)
    print(f"\nWrote {yaml_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Collecting samples…")
    all_samples = collect_hereiswally() + collect_heywaldo()

    if not all_samples:
        print("\n[ERROR] No labeled samples found.")
        print("Make sure you ran  python src/download_dataset.py  first.")
        return

    random.seed(SEED)
    random.shuffle(all_samples)
    n = len(all_samples)
    n_train = int(n * SPLITS["train"])
    n_val   = int(n * SPLITS["val"])

    splits = {
        "train": all_samples[:n_train],
        "val":   all_samples[n_train : n_train + n_val],
        "test":  all_samples[n_train + n_val :],
    }

    print(f"\nWriting {n} samples → train/val/test…")
    for name, subset in splits.items():
        write_split(subset, name)

    write_yaml()
    print("\nNext: run  python src/train.py")


if __name__ == "__main__":
    main()
