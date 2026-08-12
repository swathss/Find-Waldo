import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def pick_device():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def nms(boxes, scores, iou_thr):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return keep


def detect(model, img, tile=640, overlap=128, conf=0.25, device="cpu"):
    # Waldo is tiny, so slide a window over the full-res image and run each tile,
    # then map the boxes back and merge with nms
    H, W = img.shape[:2]
    step = tile - overlap
    all_boxes, all_scores = [], []

    pad_w = (step - (W - tile) % step) % step if W > tile else tile - W
    pad_h = (step - (H - tile) % step) % step if H > tile else tile - H
    canvas = cv2.copyMakeBorder(img, 0, max(0, pad_h), 0, max(0, pad_w),
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    CH, CW = canvas.shape[:2]

    for y in range(0, CH - tile + 1, step):
        for x in range(0, CW - tile + 1, step):
            crop = canvas[y:y + tile, x:x + tile]
            res = model.predict(crop, conf=conf, device=device, verbose=False)[0]
            for b in res.boxes:
                bx1, by1, bx2, by2 = b.xyxy[0].tolist()
                all_boxes.append([bx1 + x, by1 + y, bx2 + x, by2 + y])
                all_scores.append(float(b.conf[0]))

    if not all_boxes:
        return np.empty((0, 5), np.float32)
    boxes = np.array(all_boxes, np.float32)
    scores = np.array(all_scores, np.float32)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
    keep = nms(boxes, scores, iou_thr=0.4)
    return np.concatenate([boxes[keep], scores[keep, None]], axis=1)


def enhance_image(img, min_side=1500, max_side=4000):
    # clean up and enlarge the image so a small or soft waldo survives detection.
    # only runs when --enhance is passed.
    h, w = img.shape[:2]
    short = min(h, w)

    # denoise while the image is still small (faster), then upscale, then sharpen
    img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    if short < min_side:
        scale = min(min_side / short, max_side / max(h, w))
        if scale > 1.01:
            img = cv2.resize(img, (round(w * scale), round(h * scale)),
                             interpolation=cv2.INTER_LANCZOS4)
    soft = cv2.GaussianBlur(img, (0, 0), 1.2)
    img = cv2.addWeighted(img, 1.6, soft, -0.6, 0)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default=str(ROOT / "models" / "waldo_synth" / "weights" / "best.pt"))
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=str(ROOT / "results" / "detect_any.jpg"))
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--enhance", action="store_true", help="upscale and sharpen the image before detection")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found - train first")

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"cannot read {args.image}")

    if args.enhance:
        img = enhance_image(img)
        print(f"enhanced image -> {img.shape[1]}x{img.shape[0]}")

    device = pick_device()
    model = YOLO(str(weights))
    dets = detect(model, img, args.tile, args.overlap, args.conf, device)

    if len(dets) > args.topk:
        dets = dets[dets[:, 4].argsort()[::-1][: args.topk]]

    for x1, y1, x2, y2, s in dets:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.putText(img, f"waldo {s:.2f}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, img)
    best = dets[:, 4].max() if len(dets) else 0
    print(f"{len(dets)} detection(s), best conf {best:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
