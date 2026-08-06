import argparse
import shutil
from pathlib import Path

import cv2

from .backgrounds import BackgroundBank
from .compositor import Compositor

ROOT = Path(__file__).resolve().parent.parent
FG_DIR = ROOT / "assets" / "foregrounds"
OUT = ROOT / "data" / "synth"
PROC = ROOT / "data" / "processed_v2"


def reset(out):
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)


def write_sample(split, stem, img, box):
    cv2.imwrite(str(OUT / split / "images" / f"{stem}.jpg"), img)
    cx, cy, bw, bh = box
    label = "" if bw <= 0 or bh <= 0 else f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
    (OUT / split / "labels" / f"{stem}.txt").write_text(label)


def gen_split(split, n, tile, seed):
    bank = BackgroundBank(seed=seed)
    comp = Compositor(FG_DIR, seed=seed + 1)
    for i in range(n):
        img, box = comp.compose(bank.sample_tile(tile))
        write_sample(split, f"syn_{split}_{i:05d}", img, box)
    print(f"  synthetic {split}: {n}")


def copy_real(split_src, split_dst, tile):
    # copy the real puzzle images into a split, resized to tile size
    img_dir = PROC / split_src / "images"
    lbl_dir = PROC / split_src / "labels"
    if not img_dir.exists():
        return 0
    count = 0
    for lbl in sorted(lbl_dir.glob("*.txt")):
        img_path = None
        for ext in (".jpg", ".png", ".jpeg"):
            cand = img_dir / (lbl.stem + ext)
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lines = []
        for ln in lbl.read_text().strip().splitlines():
            p = ln.split()
            if len(p) == 5 and not (float(p[3]) > 0.9 and float(p[4]) > 0.9):
                lines.append(ln)
        img = cv2.resize(img, (tile, tile))
        cv2.imwrite(str(OUT / split_dst / "images" / f"real_{img_path.stem}.jpg"), img)
        (OUT / split_dst / "labels" / f"real_{img_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        count += 1
    return count


def write_yaml():
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\n"
        "train: train/images\n"
        "val:   val/images\n"
        "test:  test/images\n"
        "nc: 1\n"
        "names: ['waldo']\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--val", type=int, default=600)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    reset(OUT)
    gen_split("train", args.train, args.tile, args.seed)
    gen_split("val", args.val, args.tile, args.seed + 100)
    gen_split("test", args.test, args.tile, args.seed + 200)

    r_train = copy_real("train", "train", args.tile)
    r_val = copy_real("val", "val", args.tile)
    r_test = copy_real("test", "test", args.tile)
    print(f"  real added -> train:{r_train} val:{r_val} test:{r_test}")

    write_yaml()
    print("done:", OUT)


if __name__ == "__main__":
    main()
