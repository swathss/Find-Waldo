"""Fair head-to-head on the real held-out test pages.

Runs three detectors on data/processed_v2/test and reports the same metrics
for each:
  1. template matching (classical baseline)
  2. old YOLOv8n (single-class, trained on the real images only)
  3. new YOLOv8s (trained on the synthetic engine)

All three see the exact same images, so the numbers are comparable.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "processed_v2" / "test"
FG_DIR = ROOT / "assets" / "foregrounds"
IOU_THR = 0.5


def device():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(stem, w, h):
    lbl = TEST_DIR / "labels" / f"{stem}.txt"
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


def test_images():
    img_dir = TEST_DIR / "images"
    return sorted([p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg")])


# ---- detectors: each returns [(x1,y1,x2,y2,score), ...] for one image ----

def yolo_detector(weights):
    model = YOLO(str(weights))
    dev = device()

    def run(img):
        res = model.predict(img, conf=0.05, device=dev, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append((x1, y1, x2, y2, float(b.conf[0])))
        return out
    return run


def template_detector(n_templates=20):
    templates = []
    for p in sorted(FG_DIR.glob("fg_hiw_*.png"))[:n_templates]:
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        templates.append(cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY))

    def run(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        found = []
        for t in templates:
            for scale in (0.5, 0.75, 1.0, 1.5):
                th, tw = int(t.shape[0] * scale), int(t.shape[1] * scale)
                if th < 8 or tw < 8 or th >= H or tw >= W:
                    continue
                r = cv2.matchTemplate(gray, cv2.resize(t, (tw, th)), cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(r)
                found.append((loc[0], loc[1], loc[0] + tw, loc[1] + th, float(mx)))
        found.sort(key=lambda b: b[4], reverse=True)
        return found[:5]
    return run


def evaluate(detector, images):
    all_scores, all_tp = [], []
    n_gt = 0
    page_hits = 0
    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_gt(p.stem, w, h)
        n_gt += len(gt)
        preds = sorted(detector(img), key=lambda b: b[4], reverse=True)

        matched = set()
        for x1, y1, x2, y2, s in preds:
            best_j, best_iou = -1, 0.0
            for j, g in enumerate(gt):
                if j in matched:
                    continue
                v = iou((x1, y1, x2, y2), g)
                if v > best_iou:
                    best_iou, best_j = v, j
            hit = best_iou >= IOU_THR
            all_scores.append(s)
            all_tp.append(1 if hit else 0)
            if hit:
                matched.add(best_j)
        if gt and preds:
            top = preds[0]
            if any(iou(top[:4], g) >= IOU_THR for g in gt):
                page_hits += 1

    order = np.argsort(all_scores)[::-1]
    tp = np.array(all_tp)[order] if all_tp else np.array([])
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1 - tp)
    recall = cum_tp / n_gt if n_gt else np.zeros_like(cum_tp)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    ap = 0.0
    for r in np.linspace(0, 1, 11):
        p_at = precision[recall >= r].max() if np.any(recall >= r) else 0.0
        ap += p_at / 11

    total_tp = int(cum_tp[-1]) if len(cum_tp) else 0
    total_fp = int(cum_fp[-1]) if len(cum_fp) else 0
    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(n_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "AP50": ap, "precision": prec, "recall": rec, "f1": f1,
        "page_hit_rate": page_hits / max(len(images), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=str(ROOT / "models" / "waldo_yolov8n" / "weights" / "best.pt"))
    ap.add_argument("--new", default=str(ROOT / "models" / "waldo_synth" / "weights" / "best.pt"))
    args = ap.parse_args()

    images = test_images()
    print(f"benchmark: {len(images)} real test pages\n")

    rows = []
    rows.append(("template matching", evaluate(template_detector(), images)))
    if Path(args.old).exists():
        rows.append(("YOLOv8n (real only)", evaluate(yolo_detector(args.old), images)))
    if Path(args.new).exists():
        rows.append(("YOLOv8s (synthetic)", evaluate(yolo_detector(args.new), images)))

    hdr = f"{'model':22s} {'AP50':>7} {'prec':>7} {'recall':>7} {'F1':>7} {'page-hit':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in rows:
        print(f"{name:22s} {m['AP50']:7.3f} {m['precision']:7.3f} "
              f"{m['recall']:7.3f} {m['f1']:7.3f} {m['page_hit_rate']:9.3f}")

    out = ROOT / "results" / "comparison.md"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        f.write(f"# Model comparison ({len(images)} real held-out test pages, IoU 0.5)\n\n")
        f.write("| Model | AP@0.5 | Precision | Recall | F1 | Page hit rate |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, m in rows:
            f.write(f"| {name} | {m['AP50']:.3f} | {m['precision']:.3f} | "
                    f"{m['recall']:.3f} | {m['f1']:.3f} | {m['page_hit_rate']:.3f} |\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
