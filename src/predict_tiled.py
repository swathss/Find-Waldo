"""
Tiled inference for large Where's Waldo scenes.

The problem: Waldo is ~0.01–0.07% of the scene image — after YOLO resizes
to 640px, Waldo shrinks to just 3–15 pixels, making reliable detection
nearly impossible.

The fix: Slice the scene into overlapping 640×640 tiles, run YOLO on each,
then merge the detections back into scene coordinates using NMS.

Credit: tiling strategy inspired by SAHI (Slicing Aided Hyper Inference)
  https://github.com/obss/sahi

Usage:
  python src/predict_tiled.py --image path/to/scene.jpg
  python src/predict_tiled.py --image path/to/scene.jpg --tile-size 640 --overlap 0.2
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
BEST_PT = ROOT / "models" / "waldo_yolov8n" / "weights" / "best.pt"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def tile_image(img: np.ndarray, tile_size: int, overlap: float):
    """Yield (tile, x_offset, y_offset) for each tile."""
    h, w = img.shape[:2]
    step = int(tile_size * (1 - overlap))
    for y in range(0, h, step):
        for x in range(0, w, step):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            yield img[y1:y2, x1:x2], x1, y1


def nms(boxes, scores, iou_threshold=0.4):
    """Simple NMS; returns kept indices."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou < iou_threshold]
    return keep


def predict_tiled(image_path: Path, tile_size: int = 640,
                  overlap: float = 0.2, conf: float = 0.25):
    model = YOLO(str(BEST_PT))
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Cannot read {image_path}")
        return

    h, w = img.shape[:2]
    print(f"  Scene size : {w}×{h}")

    all_boxes, all_scores = [], []

    tiles = list(tile_image(img, tile_size, overlap))
    print(f"  Tiles      : {len(tiles)} ({tile_size}px, {int(overlap*100)}% overlap)")

    for tile, ox, oy in tiles:
        results = model.predict(source=tile, conf=conf, iou=0.45,
                                imgsz=tile_size, verbose=False)
        for r in results:
            for box in r.boxes:
                tx1, ty1, tx2, ty2 = map(int, box.xyxy[0])
                all_boxes.append([ox + tx1, oy + ty1, ox + tx2, oy + ty2])
                all_scores.append(float(box.conf[0]))

    if not all_boxes:
        print("  Waldo not found in any tile.")
        return

    boxes  = np.array(all_boxes, dtype=float)
    scores = np.array(all_scores)
    keep   = nms(boxes, scores, iou_threshold=0.4)

    print(f"  Detections : {len(all_boxes)} raw → {len(keep)} after NMS")

    vis = img.copy()
    for i in keep:
        x1, y1, x2, y2 = map(int, boxes[i])
        sc = scores[i]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(vis, f"Waldo {sc:.2f}", (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        print(f"  [FOUND] ({x1},{y1})–({x2},{y2})  conf={sc:.2f}")

    out = RESULTS / f"tiled_{image_path.name}"
    cv2.imwrite(str(out), vis)
    print(f"  Saved → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",     required=True)
    parser.add_argument("--tile-size", type=int,   default=640)
    parser.add_argument("--overlap",   type=float, default=0.2)
    parser.add_argument("--conf",      type=float, default=0.25)
    args = parser.parse_args()

    predict_tiled(Path(args.image),
                  tile_size=args.tile_size,
                  overlap=args.overlap,
                  conf=args.conf)


if __name__ == "__main__":
    main()
