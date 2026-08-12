"""Tile the labelled book pages into 640 tiles for YOLO.

Splits by book (no book appears in more than one split), slices each page into
overlapping 640 tiles, and keeps a capped number of background tiles per page so
negatives don't drown the positives.
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "book_pages"
OUT = ROOT / "data" / "book_yolo"
TILE = 640
OVERLAP = 128
MIN_VIS = 0.4          # keep a box only if this fraction stays inside the tile


def book_id(stem):
    # filenames look like b08_wheres_waldo_p015 -> book id "b08"
    return stem.split("_", 1)[0]


def read_boxes(label_path, w, h):
    boxes = []
    if label_path.exists():
        for ln in label_path.read_text().strip().splitlines():
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


def boxes_in_tile(boxes, ox, oy):
    kept = []
    for x1, y1, x2, y2 in boxes:
        ix1, iy1 = max(x1, ox), max(y1, oy)
        ix2, iy2 = min(x2, ox + TILE), min(y2, oy + TILE)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        area = (x2 - x1) * (y2 - y1)
        if area <= 0 or (iw * ih) / area < MIN_VIS:
            continue
        lx1, ly1, lx2, ly2 = ix1 - ox, iy1 - oy, ix2 - ox, iy2 - oy
        kept.append(((lx1 + lx2) / 2 / TILE, (ly1 + ly2) / 2 / TILE,
                     (lx2 - lx1) / TILE, (ly2 - ly1) / TILE))
    return kept


def write_tile(split, stem, tile_img, boxes):
    cv2.imwrite(str(OUT / split / "images" / f"{stem}.jpg"), tile_img)
    lines = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cx, cy, bw, bh in boxes]
    (OUT / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neg-per-pos", type=float, default=3.0, help="max background tiles per positive tile, per page")
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
    step = TILE - OVERLAP
    counts = defaultdict(lambda: [0, 0])   # split -> [pos_tiles, neg_tiles]

    images = sorted((SRC / "images").glob("*.jpg"))
    for img_path in images:
        stem = img_path.stem
        b = book_id(stem)
        split = "val" if b in val_books else "test" if b in test_books else "train"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = read_boxes(SRC / "labels" / f"{stem}.txt", w, h)

        pos_tiles, neg_tiles = [], []
        # grid tiles
        for oy in tile_origins(h, TILE, step):
            for ox in tile_origins(w, TILE, step):
                crop = img[oy:oy + TILE, ox:ox + TILE]
                if crop.shape[:2] != (TILE, TILE):
                    crop = cv2.copyMakeBorder(crop, 0, TILE - crop.shape[0], 0,
                                              TILE - crop.shape[1], cv2.BORDER_CONSTANT, value=(114, 114, 114))
                kept = boxes_in_tile(boxes, ox, oy)
                (pos_tiles if kept else neg_tiles).append((ox, oy, crop, kept))

        # guarantee one clean, Waldo-centred positive per box
        for x1, y1, x2, y2 in boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            ox = int(min(max(0, cx - TILE / 2), max(0, w - TILE)))
            oy = int(min(max(0, cy - TILE / 2), max(0, h - TILE)))
            crop = img[oy:oy + TILE, ox:ox + TILE]
            if crop.shape[:2] != (TILE, TILE):
                crop = cv2.copyMakeBorder(crop, 0, TILE - crop.shape[0], 0,
                                          TILE - crop.shape[1], cv2.BORDER_CONSTANT, value=(114, 114, 114))
            kept = boxes_in_tile(boxes, ox, oy)
            if kept:
                pos_tiles.append((ox, oy, crop, kept))

        # cap negatives relative to positives on this page
        rng.shuffle(neg_tiles)
        cap = max(2, int(round(len(pos_tiles) * args.neg_per_pos))) if pos_tiles else 4
        neg_tiles = neg_tiles[:cap]

        for i, (ox, oy, crop, kept) in enumerate(pos_tiles):
            write_tile(split, f"{stem}_p{i}", crop, kept)
            counts[split][0] += 1
        for i, (ox, oy, crop, _) in enumerate(neg_tiles):
            write_tile(split, f"{stem}_n{i}", crop, [])
            counts[split][1] += 1

    (OUT / "data.yaml").write_text(
        f"path: {OUT}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 1\nnames: ['waldo']\n"
    )
    print(f"tile={TILE} overlap={OVERLAP} neg:pos cap={args.neg_per_pos}")
    print(f"val books={sorted(val_books)}  test books={sorted(test_books)}")
    for s in ("train", "val", "test"):
        p, n = counts[s]
        print(f"  {s:5s} positives={p:4d}  negatives={n:4d}  (neg:pos={n/max(p,1):.1f})")
    print("done:", OUT)


if __name__ == "__main__":
    main()
