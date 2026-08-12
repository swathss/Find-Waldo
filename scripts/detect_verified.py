"""Detect Waldo, then verify each candidate with the Track C feature checker.

Pipeline:
  1. run the detector at low confidence to propose many candidates (high recall)
  2. score every candidate with the DINOv2 verifier (real Waldo vs decoys)
  3. keep the best-verified candidate and draw it

Run:
  python scripts/detect_verified.py --image page.jpg --weights models/waldo_synth_A/weights/best.pt
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from detect_any import detect, enhance_image
import verifier

ROOT = Path(__file__).resolve().parent.parent


def pick_device():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def crop_padded(img, x1, y1, x2, y2, pad_frac=0.3):
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    c = img[cy1:cy2, cx1:cx2]
    return c if c.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default=str(ROOT / "models" / "waldo_synth_A" / "weights" / "best.pt"))
    ap.add_argument("--conf", type=float, default=0.05, help="low, to propose many candidates")
    ap.add_argument("--candidates", type=int, default=30, help="max candidates to verify")
    ap.add_argument("--min-score", type=float, default=0.0, help="reject below this verifier score")
    ap.add_argument("--enhance", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "detect_verified.jpg"))
    ap.add_argument("--rebuild-banks", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"cannot read {args.image}")
    if args.enhance:
        img = enhance_image(img)

    # 1. propose candidates (high recall)
    model = YOLO(args.weights)
    dets = detect(model, img, conf=args.conf, device=pick_device())
    if len(dets) == 0:
        print("no candidates proposed")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, img)
        return
    dets = dets[dets[:, 4].argsort()[::-1][: args.candidates]]

    # 2. verify each candidate
    crops = [crop_padded(img, *d[:4]) for d in dets]
    keep = [i for i, c in enumerate(crops) if c is not None]
    dets, crops = dets[keep], [crops[i] for i in keep]
    pos_e, neg_e = verifier.build_banks(rebuild=args.rebuild_banks)
    vscores = verifier.score(crops, pos_e, neg_e)

    order = vscores.argsort()[::-1]
    print(f"{len(dets)} candidates verified (detector conf -> verifier score):")
    for i in order[:8]:
        x1, y1, x2, y2, dc = dets[i]
        print(f"  ({int(x1)},{int(y1)}) det={dc:.2f}  verify={vscores[i]:+.3f}")

    # 3. reject candidates the verifier dislikes, then rank survivors by the
    #    detector's confidence (the two signals together beat either alone)
    survivors = [i for i in range(len(dets)) if vscores[i] >= args.min_score]
    if survivors:
        best = max(survivors, key=lambda i: dets[i, 4])
        passed = True
    else:
        best = int(order[0])           # nothing passed; show the least-bad, flagged weak
        passed = False
    x1, y1, x2, y2, dc = dets[best]
    colour = (0, 200, 0) if passed else (0, 165, 255)
    label = f"waldo v={vscores[best]:+.2f} d={dc:.2f}" if passed else f"weak v={vscores[best]:+.2f}"
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colour, 3)
    cv2.putText(img, label, (int(x1), int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, img)
    verdict = "WALDO" if passed else "no confident waldo"
    print(f"-> {verdict}: best verifier score {vscores[best]:+.3f} at "
          f"({int(x1)},{int(y1)}) -> {args.out}")


if __name__ == "__main__":
    main()
