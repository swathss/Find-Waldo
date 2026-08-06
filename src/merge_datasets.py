"""
Merge all data sources into one unified YOLO dataset.

Sources:
  1. mohaneddz/wheres-waldo (Kaggle) — 542 YOLO-labeled tiles, class 1=Waldo
     Already split train/val/test — copy directly, remap class 1→0
  2. HereIsWally (GitHub) — 36 full scenes + CSV bboxes
     Convert to YOLO tiles (640×640 crops centered on Waldo)
  3. Hey-Waldo 256px patches (Kaggle residentmario) — 31 positive patches
     bbox = whole frame

Total expected: ~600 training samples (vs 52 before)
"""

import csv, shutil, random
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed_v2"

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def yolo_line(cls, cx, cy, w, h):
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"


# ── Source 1: mohaneddz (already YOLO, remap class 1→0) ──────────────────────

def collect_mohaneddz():
    base = RAW / "kaggle_mohaneddz" / "processed"
    samples = []
    for split in ["train", "val", "test"]:
        img_dir = base / split / "images"
        lbl_dir = base / split / "labels"
        for img in sorted(img_dir.glob("*.jpg")):
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            raw_lines = lbl.read_text().strip().splitlines()
            # Keep only Waldo (class 1), remap → 0
            new_lines = []
            for line in raw_lines:
                parts = line.split()
                if not parts:
                    continue
                if int(parts[0]) == 1:   # 1 = Waldo in 5-class scheme
                    new_lines.append(yolo_line(0, *map(float, parts[1:])))
            if new_lines:
                samples.append((img, new_lines))
    print(f"  mohaneddz (Waldo tiles)    : {len(samples)}")
    return samples


# ── Source 2: HereIsWally full scenes → 640px crops ──────────────────────────

def collect_hereiswally_crops():
    import cv2
    base = RAW / "HereIsWally"
    csv_path = base / "annotations" / "annotations.csv"
    img_dir  = base / "images"
    TILE = 640
    samples = []

    rows_by_file = {}
    for r in csv.DictReader(open(csv_path)):
        rows_by_file.setdefault(r["filename"], []).append(r)

    for filename, rows in rows_by_file.items():
        img_path = img_dir / filename
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        ih, iw = img.shape[:2]

        for row in rows:
            xmin,ymin = float(row["xmin"]), float(row["ymin"])
            xmax,ymax = float(row["xmax"]), float(row["ymax"])
            wcx, wcy  = (xmin+xmax)/2, (ymin+ymax)/2

            x1 = int(max(0, wcx - TILE/2))
            y1 = int(max(0, wcy - TILE/2))
            x2 = min(iw, x1 + TILE);  y2 = min(ih, y1 + TILE)
            x1 = max(0, x2 - TILE);   y1 = max(0, y2 - TILE)
            crop = img[y1:y2, x1:x2]
            cw, ch = x2-x1, y2-y1

            cx = ((xmin+xmax)/2 - x1) / cw
            cy = ((ymin+ymax)/2 - y1) / ch
            bw = (xmax-xmin) / cw
            bh = (ymax-ymin) / ch
            cx = max(0.01, min(0.99, cx))
            cy = max(0.01, min(0.99, cy))
            samples.append((crop, [yolo_line(0, cx, cy, bw, bh)],
                             f"wally_{Path(filename).stem}"))
    print(f"  HereIsWally crops          : {len(samples)}")
    return samples  # returns (ndarray, lines, stem) — handled separately


# ── Source 3: Hey-Waldo 256px patches ────────────────────────────────────────

def collect_256_patches():
    base = RAW / "kaggle_residentmario" / "wheres-waldo" / "Hey-Waldo" / "256" / "waldo"
    if not base.exists():
        print("  [skip] 256px patches not found")
        return []
    samples = [(p, [yolo_line(0, 0.5, 0.5, 1.0, 1.0)]) for p in sorted(base.glob("*.jpg"))]
    print(f"  Hey-Waldo 256px patches    : {len(samples)}")
    return samples


# ── Write split ───────────────────────────────────────────────────────────────

def write_split_paths(path_samples, split, base):
    import shutil
    img_dir = base / split / "images"
    lbl_dir = base / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i, (src_path, lines) in enumerate(path_samples):
        stem = f"{split}_{i:04d}"
        shutil.copy(src_path, img_dir / (stem + ".jpg"))
        (lbl_dir / (stem + ".txt")).write_text("".join(lines))


def write_split_arrays(array_samples, split, base, offset):
    import cv2
    img_dir = base / split / "images"
    lbl_dir = base / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i, (arr, lines, _) in enumerate(array_samples):
        stem = f"{split}_{offset+i:04d}"
        cv2.imwrite(str(img_dir / (stem + ".jpg")), arr)
        (lbl_dir / (stem + ".txt")).write_text("".join(lines))


def collect_yolo_folder(root: Path, waldo_class_id: int = 0, label: str = ""):
    """Generic collector for any YOLO dataset folder tree."""
    samples = []
    for img in sorted(root.rglob("*.jpg")):
        lbl = img.parent.parent / "labels" / (img.stem + ".txt")
        if not lbl.exists():
            lbl = img.with_suffix(".txt")
        if not lbl.exists():
            continue
        raw_lines = lbl.read_text().strip().splitlines()
        new_lines = []
        for line in raw_lines:
            parts = line.split()
            if not parts:
                continue
            if int(parts[0]) == waldo_class_id:
                new_lines.append(yolo_line(0, *map(float, parts[1:])))
        if new_lines:
            samples.append((img, new_lines))
    print(f"  {label:30s}: {len(samples)}")
    return samples


def main():
    print("Collecting from all sources…")
    moh_samples   = collect_mohaneddz()
    wall_samples  = collect_hereiswally_crops()
    patch_samples = collect_256_patches()

    RAW = Path(__file__).parent.parent / "data" / "raw"
    ffaraz_samples  = collect_yolo_folder(RAW / "kaggle_ffaraz",   0, "ffaraz/waldo-yolo")
    bakshi_samples  = collect_yolo_folder(RAW / "kaggle_bakshi15", 0, "bakshi15/waldo-dataset-v3")

    total_path = moh_samples + patch_samples + ffaraz_samples + bakshi_samples
    total_arr  = wall_samples

    random.seed(SEED)
    random.shuffle(total_path)
    random.shuffle(total_arr)

    n_p = len(total_path)
    n_a = len(total_arr)
    print(f"\nTotal: {n_p + n_a} samples")

    # Split path-based samples
    n_train_p = int(n_p * SPLITS["train"])
    n_val_p   = int(n_p * SPLITS["val"])
    splits_p = {
        "train": total_path[:n_train_p],
        "val":   total_path[n_train_p:n_train_p+n_val_p],
        "test":  total_path[n_train_p+n_val_p:],
    }

    # Split array-based samples
    n_train_a = int(n_a * SPLITS["train"])
    n_val_a   = int(n_a * SPLITS["val"])
    splits_a = {
        "train": total_arr[:n_train_a],
        "val":   total_arr[n_train_a:n_train_a+n_val_a],
        "test":  total_arr[n_train_a+n_val_a:],
    }

    print("\nWriting merged dataset…")
    for split in ["train", "val", "test"]:
        write_split_paths(splits_p[split], split, PROC)
        write_split_arrays(splits_a[split], split, PROC,
                           offset=len(splits_p[split]))
        total = len(splits_p[split]) + len(splits_a[split])
        print(f"  {split:5s}: {total} images")

    yaml = PROC / "data.yaml"
    yaml.write_text(f"""path: {PROC.resolve()}
train: train/images
val:   val/images
test:  test/images
nc: 1
names: ['waldo']
""")
    print(f"\nWrote {yaml}")
    print("Next: python src/train_v2.py")


if __name__ == "__main__":
    main()
