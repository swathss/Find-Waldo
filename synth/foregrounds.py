import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed_v2"
OUT = ROOT / "assets" / "foregrounds"

MIN_SIDE = 14
PAD_FRAC = 0.35


def grabcut_alpha(bgr):
    h, w = bgr.shape[:2]

    scale = 1
    if min(h, w) < 80:
        scale = int(np.ceil(80 / min(h, w)))
        bgr = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        h, w = bgr.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    mx, my = int(w * PAD_FRAC * 0.7), int(h * PAD_FRAC * 0.7)
    rect = (mx, my, max(1, w - 2 * mx), max(1, h - 2 * my))
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except cv2.error:
        alpha = np.zeros((h, w), np.uint8)

    frac = alpha.mean() / 255.0
    if frac < 0.06 or frac > 0.95:
        # grabcut gave up, use an ellipse over the middle instead
        alpha = np.zeros((h, w), np.uint8)
        cv2.ellipse(
            alpha,
            (w // 2, h // 2),
            (int(w * (0.5 - PAD_FRAC * 0.6)), int(h * (0.5 - PAD_FRAC * 0.4))),
            0, 0, 360, 255, -1,
        )

    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(1.0, min(h, w) / 40.0))
    if scale > 1:
        alpha = cv2.resize(alpha, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    return alpha


def save_rgba(bgr, alpha, path):
    if bgr.shape[:2] != alpha.shape[:2]:
        alpha = cv2.resize(alpha, (bgr.shape[1], bgr.shape[0]))
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    cv2.imwrite(str(path), rgba)


def crop_padded(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    if bw < MIN_SIDE or bh < MIN_SIDE:
        return None
    px, py = int(bw * PAD_FRAC), int(bh * PAD_FRAC)
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
    crop = img[cy1:cy2, cx1:cx2]
    return crop if crop.size else None


def from_hereiswally(idx):
    ann = RAW / "HereIsWally" / "annotations" / "annotations.csv"
    imgs = RAW / "HereIsWally" / "images"
    if not ann.exists():
        return idx
    for row in csv.DictReader(ann.open()):
        img = cv2.imread(str(imgs / row["filename"]))
        if img is None:
            continue
        crop = crop_padded(img, int(row["xmin"]), int(row["ymin"]),
                           int(row["xmax"]), int(row["ymax"]))
        if crop is None:
            continue
        save_rgba(crop, grabcut_alpha(crop), OUT / f"fg_hiw_{idx:04d}.png")
        idx += 1
    return idx


def from_processed(idx):
    # train split only - val/test Waldo appearances stay unseen so they can be
    # used as a clean benchmark
    for split in ("train",):
        img_dir = PROC / split / "images"
        lbl_dir = PROC / split / "labels"
        if not img_dir.exists():
            continue
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
            h, w = img.shape[:2]
            for line in lbl.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                if bw > 0.9 and bh > 0.9:      # whole-image box, skip
                    continue
                x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                crop = crop_padded(img, x1, y1, x2, y2)
                if crop is None:
                    continue
                save_rgba(crop, grabcut_alpha(crop), OUT / f"fg_proc_{idx:04d}.png")
                idx += 1
    return idx


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in OUT.glob("*.png"):
        p.unlink()
    n = from_hereiswally(0)
    print("hereiswally:", n)
    n = from_processed(n)
    print("+ processed:", n)
    # Hey-Waldo positive patches are whole crowd tiles, not isolated Waldo,
    # so they're not used here.
    print("total:", len(list(OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
