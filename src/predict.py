"""
Step 5: Run inference on a single image and save the annotated result.

Usage:
  python src/predict.py --image path/to/waldo_scene.jpg
  python src/predict.py --image path/to/waldo_scene.jpg --conf 0.3

Also includes a classic OpenCV template-matching baseline as a sanity check.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
BEST_PT  = ROOT / "models" / "waldo_yolov8n" / "weights" / "best.pt"
RESULTS  = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# YOLO inference

def predict_yolo(image_path: Path, conf: float = 0.25):
    if not BEST_PT.exists():
        print(f"[ERROR] Model not found at {BEST_PT}. Train first.")
        return

    model = YOLO(str(BEST_PT))
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=0.45,
        imgsz=640,
        save=False,
        verbose=False,
    )

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    found = 0
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = f"Waldo {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"  [YOLO] Waldo found at ({x1},{y1})-({x2},{y2})  conf={confidence:.2f}")
            found += 1

    if found == 0:
        print("  [YOLO] Waldo not found (try lowering --conf)")

    out_path = RESULTS / f"yolo_{image_path.name}"
    cv2.imwrite(str(out_path), img)
    print(f"  Saved -> {out_path}")


# Template-matching baseline (no ML)

def predict_template(image_path: Path, template_path: Path = None):
    """
    Classic sliding-window template match using the Hey-Waldo patches as
    a single averaged template.  This is the non-ML baseline.
    """
    waldo_patches_dir = ROOT / "data" / "raw" / "Hey-Waldo" / "64" / "Waldo"
    if template_path is None and not waldo_patches_dir.exists():
        print("  [template] Waldo patches not found; skipping baseline.")
        return

    # Build mean template from all 64x64 patches
    patches = list(waldo_patches_dir.glob("*.jpg"))[:30]  # use up to 30
    templates = []
    for p in patches:
        t = cv2.imread(str(p))
        if t is not None:
            templates.append(t.astype(np.float32))
    if not templates:
        print("  [template] No patches loaded.")
        return
    mean_template = np.mean(templates, axis=0).astype(np.uint8)

    scene = cv2.imread(str(image_path))
    if scene is None:
        return

    result = cv2.matchTemplate(scene, mean_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    h, w = mean_template.shape[:2]
    x1, y1 = max_loc
    x2, y2 = x1 + w, y1 + h

    cv2.rectangle(scene, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(scene, f"Template {max_val:.2f}", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print(f"  [template] Best match at ({x1},{y1})-({x2},{y2})  score={max_val:.2f}")

    out_path = RESULTS / f"template_{image_path.name}"
    cv2.imwrite(str(out_path), scene)
    print(f"  Saved -> {out_path}")


# CLI

def main():
    parser = argparse.ArgumentParser(description="Find Waldo in an image")
    parser.add_argument("--image",    required=True,  help="Path to scene image")
    parser.add_argument("--conf",     type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--baseline", action="store_true", help="Also run template-matching baseline")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return

    print(f"\nRunning on: {image_path.name}")
    print("-" * 50)

    print("\n[1/2] YOLO inference...")
    predict_yolo(image_path, conf=args.conf)

    if args.baseline:
        print("\n[2/2] Template-matching baseline...")
        predict_template(image_path)


if __name__ == "__main__":
    main()
