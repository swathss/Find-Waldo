"""Step 3: build decoy hard-negative tiles.

Mines real Odlaw / Wizard / woof crops from the 5-class mohaneddz set and pastes
them onto Waldo-free background tiles, saved with empty labels. These teach the
model that partial matches (yellow/black stripes, a beard, a lone hat) are NOT
Waldo, without ever labelling a real Waldo as negative.

Guardrails:
  * Wilma/Wenda is excluded on purpose - she shares almost all of Waldo's
    features, so using her as a negative would suppress the real Waldo.
  * Backgrounds are the book negative tiles (already Waldo-free), so no tile
    with a real Waldo is ever labelled negative.
"""
import argparse
import random
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
MOHA = ROOT / "data" / "raw" / "kaggle_mohaneddz"
BOOK_NEG = ROOT / "data" / "book_yolo" / "train" / "images"   # Waldo-free tiles
DECOY_CROPS = ROOT / "assets" / "decoys"
OUT = ROOT / "data" / "decoy_neg"

# mohaneddz classes: 0 Odlaw, 1 Waldo, 2 Wilma, 3 Wizard, 4 woof
# Only Odlaw (yellow/black) and Wizard (beard/robe) are used - both clearly lack
# Waldo's red/white+glasses+hat conjunction. Wilma is too Waldo-like, and woof
# (the dog) wears red/white stripes AND its labels are noisy (some crops are
# actually Waldo), so both are excluded to avoid teaching "Waldo = negative".
KEEP = {0: "odlaw", 3: "wizard"}


def mine_crops():
    DECOY_CROPS.mkdir(parents=True, exist_ok=True)
    for p in DECOY_CROPS.glob("*.png"):
        p.unlink()
    n = 0
    for sub in ("dataset", "processed/train", "processed/val", "processed/test"):
        img_dir, lbl_dir = MOHA / sub / "images", MOHA / sub / "labels"
        if not img_dir.exists():
            continue
        for lbl in sorted(lbl_dir.glob("*.txt")):
            img_path = next((img_dir / (lbl.stem + e) for e in (".jpg", ".png", ".jpeg")
                             if (img_dir / (lbl.stem + e)).exists()), None)
            if img_path is None:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            for line in lbl.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                if cls not in KEEP:
                    continue
                cx, cy, bw, bh = map(float, parts[1:])
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                if x2 - x1 < 12 or y2 - y1 < 12:
                    continue
                crop = img[max(0, y1):y2, max(0, x1):x2]
                if crop.size:
                    cv2.imwrite(str(DECOY_CROPS / f"{KEEP[cls]}_{n:04d}.png"), crop)
                    n += 1
    return n


def paste(dst, patch, x, y):
    ph, pw = patch.shape[:2]
    H, W = dst.shape[:2]
    if x + pw > W or y + ph > H:
        return
    dst[y:y + ph, x:x + pw] = patch


def build(n_tiles, per_tile_max, seed):
    rng = random.Random(seed)
    crops = sorted(DECOY_CROPS.glob("*.png"))
    backgrounds = sorted(BOOK_NEG.glob("*_n*.jpg"))
    if not crops or not backgrounds:
        raise RuntimeError("need decoy crops and book negative tiles first")
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    for p in list((OUT / "images").glob("*")) + list((OUT / "labels").glob("*")):
        p.unlink()

    for i in range(n_tiles):
        bg = cv2.imread(str(rng.choice(backgrounds)))
        if bg is None:
            continue
        H, W = bg.shape[:2]
        for _ in range(rng.randint(1, per_tile_max)):
            c = cv2.imread(str(rng.choice(crops)))
            if c is None:
                continue
            side = rng.randint(int(W * 0.06), int(W * 0.20))
            c = cv2.resize(c, (side, side))
            M = cv2.getRotationMatrix2D((side / 2, side / 2), rng.uniform(-15, 15), 1.0)
            c = cv2.warpAffine(c, M, (side, side), borderMode=cv2.BORDER_REFLECT)
            paste(bg, c, rng.randint(0, W - side), rng.randint(0, H - side))
        cv2.imwrite(str(OUT / "images" / f"decoy_{i:04d}.jpg"), bg)
        (OUT / "labels" / f"decoy_{i:04d}.txt").write_text("")   # empty = hard negative

    yaml = ROOT / "data" / "book_decoy.yaml"
    yaml.write_text(
        f"path: {ROOT}\n"
        f"train:\n  - data/book_yolo/train/images\n  - data/decoy_neg/images\n"
        "val: data/book_yolo/val/images\n"
        "test: data/book_yolo/test/images\n"
        "nc: 1\nnames: ['waldo']\n"
    )
    return n_tiles, yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=220, help="how many decoy-negative tiles")
    ap.add_argument("--per-tile-max", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    n = mine_crops()
    by_type = {t: len(list(DECOY_CROPS.glob(f"{t}_*.png"))) for t in set(KEEP.values())}
    print(f"mined {n} decoy crops: {by_type}")
    tiles, yaml = build(args.tiles, args.per_tile_max, args.seed)
    print(f"built {tiles} decoy-negative tiles -> {OUT}")
    print(f"wrote {yaml}")


if __name__ == "__main__":
    main()
