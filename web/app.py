"""
Find Waldo: Flask web app
Run: python web/app.py
Then open: http://localhost:5000
"""

import io
import base64
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB max upload

ROOT    = Path(__file__).parent.parent
MODEL   = ROOT / "models/waldo_synth/weights/best.pt"
FALLBACK= ROOT / "runs/detect/models/waldo_v2/weights/best.pt"

# Load model once at startup
model_path = MODEL if MODEL.exists() else FALLBACK
print(f"Loading model: {model_path}")
yolo = YOLO(str(model_path))
print("Model ready ✓")


def pil_to_b64(img: Image.Image, fmt="JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def cv2_to_b64(img_bgr: np.ndarray) -> str:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return pil_to_b64(pil)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    conf_thresh = float(request.form.get("conf", 0.25))

    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not read image"}), 400

    h, w = img.shape[:2]

    # Run YOLO
    results = yolo.predict(
        source=img,
        conf=conf_thresh,
        iou=0.45,
        imgsz=640,
        verbose=False,
    )

    # Draw detections
    annotated = img.copy()
    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])

            # Draw red box + label
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 220), 3)
            label = f"Waldo  {confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 8, y1), (0, 0, 220), -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": round(confidence, 3),
                "cx": (x1 + x2) // 2,
                "cy": (y1 + y2) // 2,
            })

    return jsonify({
        "found": len(detections) > 0,
        "count": len(detections),
        "detections": detections,
        "image_w": w,
        "image_h": h,
        "result_image": cv2_to_b64(annotated),
        "original_image": cv2_to_b64(img),
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
