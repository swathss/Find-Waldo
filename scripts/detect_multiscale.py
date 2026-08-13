"""Multi-scale tiled inference with Weighted Boxes Fusion.

Runs the detector over each page at three tile sizes (320, 512, 768), fuses the
per-scale boxes with Weighted Boxes Fusion, and keeps the single best box per
page. Running several tile sizes helps because Waldo appears at different pixel
sizes across pages, and fusion rewards a location that several scales agree on.

Usage:
    python scripts/detect_multiscale.py --weights models/waldo_book_decoy/weights/best.pt \
        --books b04,b09 --out results/multiscale_b04_b09
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

ROOT = Path(__file__).resolve().parent.parent
BOOK_DIR = ROOT / "data" / "raw" / "book_pages"


def pick_device():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tiled_boxes(model, img, tile, overlap, conf, device):
    """Slide one tile size over the page and return boxes in page pixels."""
    h, w = img.shape[:2]
    step = tile - overlap
    pad_w = (step - (w - tile) % step) % step if w > tile else tile - w
    pad_h = (step - (h - tile) % step) % step if h > tile else tile - h
    canvas = cv2.copyMakeBorder(img, 0, max(0, pad_h), 0, max(0, pad_w),
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    ch, cw = canvas.shape[:2]
    boxes, scores = [], []
    for y in range(0, ch - tile + 1, step):
        for x in range(0, cw - tile + 1, step):
            crop = canvas[y:y + tile, x:x + tile]
            res = model.predict(crop, conf=conf, device=device, verbose=False)[0]
            for b in res.boxes:
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                boxes.append([bx1 + x, by1 + y, bx2 + x, by2 + y])
                scores.append(float(b.conf[0]))
    return boxes, scores


def multiscale_detect(model, img, tiles=(320, 512, 768), conf=0.10, iou_thr=0.55, device="cpu"):
    """Return fused boxes sorted by score, as [x1, y1, x2, y2, score] in pixels."""
    h, w = img.shape[:2]
    boxes_list, scores_list, labels_list = [], [], []
    for t in tiles:
        b, s = tiled_boxes(model, img, t, t // 4, conf, device)
        norm = [[x1 / w, y1 / h, x2 / w, y2 / h] for x1, y1, x2, y2 in b]
        boxes_list.append(norm)
        scores_list.append(s)
        labels_list.append([0] * len(norm))

    if not any(boxes_list):
        return np.empty((0, 5), np.float32)

    fb, fs, _ = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list, iou_thr=iou_thr, skip_box_thr=0.0)
    out = np.array([[x1 * w, y1 * h, x2 * w, y2 * h, sc]
                    for (x1, y1, x2, y2), sc in zip(fb, fs)], np.float32)
    return out[out[:, 4].argsort()[::-1]] if len(out) else out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def gt_box(stem, w, h):
    lbl = BOOK_DIR / "labels" / f"{stem}.txt"
    if not lbl.exists() or not lbl.read_text().strip():
        return None
    cx, cy, bw, bh = map(float, lbl.read_text().split()[1:5])
    return [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "models" / "waldo_book_decoy" / "weights" / "best.pt"))
    ap.add_argument("--books", default="b04,b09")
    ap.add_argument("--tiles", default="320,512,768")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--out", default=str(ROOT / "results" / "multiscale_b04_b09"))
    args = ap.parse_args()

    books = set(args.books.split(","))
    tiles = tuple(int(t) for t in args.tiles.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    dev = pick_device()

    pages = []
    for f in sorted((BOOK_DIR / "images").glob("*.jpg")):
        stem = f.stem
        if stem.split("_")[0] in books and (BOOK_DIR / "labels" / f"{stem}.txt").read_text().strip():
            pages.append((stem, f))

    hits, best_ious, n = 0, [], len(pages)
    for stem, f in pages:
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        gt = gt_box(stem, w, h)
        dets = multiscale_detect(model, img, tiles, args.conf, device=dev)

        top = dets[0] if len(dets) else None
        best = max((iou(d[:4], gt) for d in dets), default=0.0) if gt is not None else 0.0
        best_ious.append(best)
        if top is not None and gt is not None and iou(top[:4], gt) >= 0.5:
            hits += 1

        vis = img.copy()
        if gt is not None:
            cv2.rectangle(vis, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (0, 0, 255), 3)
        if top is not None:
            x1, y1, x2, y2, s = top
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(vis, f"{s:.2f}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.imwrite(str(out / f"{stem}_multiscale.jpg"), vis)
        print(f"{stem}: top conf {top[4]:.2f} best-IoU {best:.2f}" if top is not None else f"{stem}: no box")

    print(f"\nMulti-scale ({'/'.join(map(str, tiles))}) + WBF, conf {args.conf}, top-1 per page")
    print(f"books {sorted(books)}: {n} pages")
    print(f"page hit-rate (top-1 IoU>=0.5): {hits}/{n} = {hits / max(n, 1):.3f}")
    print(f"mean best-IoU: {sum(best_ious) / max(n, 1):.3f}")
    print(f"annotated images -> {out}")


if __name__ == "__main__":
    main()
