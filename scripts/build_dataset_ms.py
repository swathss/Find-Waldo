"""Rebuild the training set with multi-scale tiles plus copy-paste positives.

For each labelled page, tiles are cut at three sizes (320, 512, 768) so Waldo is
seen at a range of pixel scales. On top of the real tiles, extra positives are
made by pasting real Waldo crops into real background tiles at varied sizes,
positions, rotation, and brightness, with feathered edges. Decoy hard-negatives
(Odlaw and Wizard crops) stay in the negative pool.

The split is grouped by book, and the held-out books match the current model
(test b04 and b09, validation b11 and b16), so results stay comparable.

Copy-paste positives and decoys go into the training split only. Validation and
test are real tiles from the held-out books.
"""
import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from synth.compositor import load_rgba, rotate_scale, tight_bbox, color_jitter, paste

SRC = ROOT / "data" / "raw" / "book_pages"
FG_DIR = ROOT / "assets" / "foregrounds"       # real Waldo crops (train-split only)
DECOY_NEG = ROOT / "data" / "decoy_neg"         # prebuilt Odlaw/Wizard negative tiles
OUT = ROOT / "data" / "book_yolo_ms"

TILE_SIZES = [320, 512, 768]
CANVAS = 640                                    # copy-paste canvas size
MIN_VIS = 0.4
CP_MIN_PX, CP_MAX_PX = 28, 356                  # pasted Waldo size range


def book_id(stem):
    return stem.split("_", 1)[0]


def read_boxes(stem, w, h):
    lbl = SRC / "labels" / f"{stem}.txt"
    boxes = []
    if lbl.exists():
        for ln in lbl.read_text().strip().splitlines():
            p = ln.split()
            if len(p) != 5:
                continue
            _, cx, cy, bw, bh = map(float, p)
            if bw > 0.9 and bh > 0.9:
                continue
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def tile_origins(size, tile, step):
    if size <= tile:
        return [0]
    xs = list(range(0, size - tile + 1, step))
    if xs[-1] != size - tile:
        xs.append(size - tile)
    return xs


def boxes_in_tile(boxes, ox, oy, tile):
    kept = []
    for x1, y1, x2, y2 in boxes:
        ix1, iy1 = max(x1, ox), max(y1, oy)
        ix2, iy2 = min(x2, ox + tile), min(y2, oy + tile)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        area = (x2 - x1) * (y2 - y1)
        if area <= 0 or (iw * ih) / area < MIN_VIS:
            continue
        kept.append(((ix1 + ix2) / 2 - ox, (iy1 + iy2) / 2 - oy, ix2 - ix1, iy2 - iy1))
    return kept


def crop_tile(img, ox, oy, tile):
    crop = img[oy:oy + tile, ox:ox + tile]
    if crop.shape[:2] != (tile, tile):
        crop = cv2.copyMakeBorder(crop, 0, tile - crop.shape[0], 0, tile - crop.shape[1],
                                  cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return crop


def write_tile(split, stem, img, boxes_norm):
    cv2.imwrite(str(OUT / split / "images" / f"{stem}.jpg"), img)
    lines = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cx, cy, bw, bh in boxes_norm]
    (OUT / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def copy_paste(bg, fg_paths, rng):
    """Paste one real Waldo crop into a background tile; return image and box."""
    canvas = cv2.resize(bg, (CANVAS, CANVAS))
    fg = load_rgba(rng.choice(fg_paths))
    while fg is None:
        fg = load_rgba(rng.choice(fg_paths))
    if rng.random() < 0.5:
        fg = cv2.flip(fg, 1)
    fg[:, :, :3] = color_jitter(fg[:, :, :3], rng)          # brightness/colour jitter

    target = rng.randint(CP_MIN_PX, CP_MAX_PX)
    scale = target / max(fg.shape[0], fg.shape[1])
    warped = rotate_scale(fg, rng.uniform(-15, 15), scale)  # small rotation
    bbox = tight_bbox(warped[:, :, 3])
    if bbox is None:
        return None
    bx0, by0, bx1, by1 = bbox
    warped = warped[by0:by1, bx0:bx1]
    ph, pw = warped.shape[:2]
    px = rng.randint(0, max(0, CANVAS - pw))
    py = rng.randint(0, max(0, CANVAS - ph))
    paste(canvas, warped, px, py)                           # alpha blend = feathered edge
    cx = (px + pw / 2) / CANVAS
    cy = (py + ph / 2) / CANVAS
    return canvas, (cx, cy, pw / CANVAS, ph / CANVAS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-per-pos", type=float, default=3.0)
    ap.add_argument("--copy-paste", type=int, default=1500, help="extra copy-paste positives (train)")
    ap.add_argument("--val-books", default="b11,b16")
    ap.add_argument("--test-books", default="b04,b09")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ("train", "val", "test"):
        (OUT / s / "images").mkdir(parents=True, exist_ok=True)
        (OUT / s / "labels").mkdir(parents=True, exist_ok=True)

    val_books = set(args.val_books.split(","))
    test_books = set(args.test_books.split(","))
    counts = defaultdict(lambda: [0, 0])
    bg_pool = []           # Waldo-free real tiles, reused as copy-paste backgrounds
    train_bg_negs = []     # train background-negative names, trimmed later to hit the ratio

    for img_path in sorted((SRC / "images").glob("*.jpg")):
        stem = img_path.stem
        split = "val" if book_id(stem) in val_books else "test" if book_id(stem) in test_books else "train"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = read_boxes(stem, w, h)

        pos, neg = [], []
        for tile in TILE_SIZES:
            step = tile - tile // 4
            for oy in tile_origins(h, tile, step):
                for ox in tile_origins(w, tile, step):
                    crop = crop_tile(img, ox, oy, tile)
                    kept = boxes_in_tile(boxes, ox, oy, tile)
                    norm = [(cx / tile, cy / tile, bw / tile, bh / tile) for cx, cy, bw, bh in kept]
                    (pos if norm else neg).append((f"{stem}_t{tile}_{ox}_{oy}", crop, norm))
            # guarantee a Waldo-centred positive per box at each scale
            for x1, y1, x2, y2 in boxes:
                ox = int(min(max(0, (x1 + x2) / 2 - tile / 2), max(0, w - tile)))
                oy = int(min(max(0, (y1 + y2) / 2 - tile / 2), max(0, h - tile)))
                crop = crop_tile(img, ox, oy, tile)
                kept = boxes_in_tile(boxes, ox, oy, tile)
                if kept:
                    norm = [(cx / tile, cy / tile, bw / tile, bh / tile) for cx, cy, bw, bh in kept]
                    pos.append((f"{stem}_t{tile}c_{ox}_{oy}", crop, norm))

        rng.shuffle(neg)
        # train: keep all negatives now, trim to the target ratio at the end.
        # val/test: cap per page so the held-out sets keep a fixed 3:1 ratio.
        if split != "train":
            neg = neg[:max(2, int(round(len(pos) * args.neg_per_pos))) if pos else 4]
        for name, crop, norm in pos:
            write_tile(split, name, crop, norm)
            counts[split][0] += 1
        for name, crop, _ in neg:
            write_tile(split, name, crop, [])
            counts[split][1] += 1
            if split == "train":
                bg_pool.append(OUT / "train" / "images" / f"{name}.jpg")
                train_bg_negs.append(name)

    # copy-paste positives (train only)
    fg_paths = sorted(FG_DIR.glob("*.png"))
    for i in range(args.copy_paste):
        bg = cv2.imread(str(rng.choice(bg_pool)))
        if bg is None:
            continue
        out = copy_paste(bg, fg_paths, rng)
        if out is None:
            continue
        cimg, box = out
        write_tile("train", f"cp_{i:05d}", cimg, [box])
        counts["train"][0] += 1

    # decoy hard-negatives (train only) from the prebuilt pool
    dn = 0
    if (DECOY_NEG / "images").exists():
        for p in sorted((DECOY_NEG / "images").glob("*.jpg")):
            img = cv2.imread(str(p))
            if img is None:
                continue
            write_tile("train", f"decoy_{p.stem}", img, [])
            counts["train"][1] += 1
            dn += 1

    # trim train background negatives so total train neg:pos is about the target,
    # keeping every decoy negative
    target_neg = round(args.neg_per_pos * counts["train"][0])
    allowed_bg = max(0, target_neg - dn)
    if len(train_bg_negs) > allowed_bg:
        rng.shuffle(train_bg_negs)
        for name in train_bg_negs[allowed_bg:]:
            (OUT / "train" / "images" / f"{name}.jpg").unlink(missing_ok=True)
            (OUT / "train" / "labels" / f"{name}.txt").unlink(missing_ok=True)
            counts["train"][1] -= 1

    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 1\nnames: ['waldo']\n")

    print(f"tiles at {TILE_SIZES}, copy-paste {args.copy_paste} (size {CP_MIN_PX}-{CP_MAX_PX}px), decoys {dn}")
    print(f"val books {sorted(val_books)}  test books {sorted(test_books)}")
    for s in ("train", "val", "test"):
        p, n = counts[s]
        print(f"  {s:5s} positives={p:5d}  negatives={n:5d}  (neg:pos={n / max(p, 1):.1f})")
    print("done:", OUT)


if __name__ == "__main__":
    main()
