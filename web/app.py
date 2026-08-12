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
MODEL   = ROOT / "models/waldo_synth_A/weights/best.pt"
FALLBACK= ROOT / "models/waldo_synth/weights/best.pt"

# Load model once at startup
model_path = MODEL if MODEL.exists() else FALLBACK
print(f"Loading model: {model_path}")
yolo = YOLO(str(model_path))
print("Model ready ok")


def _nms(dets, iou_thresh=0.45):
    if len(dets) == 0:
        return np.empty((0, 5), dtype=np.float32)

    dets = np.asarray(dets, dtype=np.float32)
    x1, y1, x2, y2, scores = dets.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= iou_thresh]

    return dets[keep]


def detect_tiled(model, img, tile=640, overlap=128, conf=0.25):
    # waldo is tiny, so slide a window over the full image instead of shrinking it
    h, w = img.shape[:2]
    step = tile - overlap
    x_starts = list(range(0, max(w - tile, 0) + 1, step))
    y_starts = list(range(0, max(h - tile, 0) + 1, step))
    last_x = max(w - tile, 0)
    last_y = max(h - tile, 0)
    if not x_starts or x_starts[-1] != last_x:
        x_starts.append(last_x)
    if not y_starts or y_starts[-1] != last_y:
        y_starts.append(last_y)

    dets = []
    for y in y_starts:
        for x in x_starts:
            crop = img[y:min(y + tile, h), x:min(x + tile, w)]
            results = model.predict(
                source=crop,
                conf=conf,
                iou=0.45,
                imgsz=tile,
                verbose=False,
            )
            for result in results:
                for box in result.boxes:
                    bx1, by1, bx2, by2 = map(float, box.xyxy[0])
                    confidence = float(box.conf[0])
                    dets.append([bx1 + x, by1 + y, bx2 + x, by2 + y, confidence])

    return _nms(dets)


def enhance_image(img, min_side=1500, max_side=4000):
    # clean up and enlarge the image so a small or soft waldo survives detection.
    # only runs when the user turns on the enhance toggle.
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


# Real-ESRGAN super-resolution, loaded lazily the first time it is used.
SR_WEIGHTS = ROOT / "models/sr/RealESRGAN_x4plus_anime_6B.pth"
_sr_model = None


def _load_sr():
    global _sr_model
    if _sr_model is None:
        import torch
        from spandrel import ModelLoader, ImageModelDescriptor
        dev = "mps" if torch.backends.mps.is_available() else \
              ("cuda" if torch.cuda.is_available() else "cpu")
        m = ModelLoader().load_from_file(str(SR_WEIGHTS))
        if not isinstance(m, ImageModelDescriptor):
            raise RuntimeError("unexpected model type")
        _sr_model = (m.to(dev).eval(), dev)
    return _sr_model


def super_resolve(img, max_long=3000):
    # run Real-ESRGAN, then cap the long side so detection stays reasonably fast
    import torch
    model, dev = _load_sr()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(dev)
    with torch.no_grad():
        out = model(t).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    sr = cv2.cvtColor((out * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    long_side = max(sr.shape[:2])
    if long_side > max_long:
        s = max_long / long_side
        sr = cv2.resize(sr, (round(sr.shape[1] * s), round(sr.shape[0] * s)),
                        interpolation=cv2.INTER_AREA)
    return sr


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
    do_enhance = request.form.get("enhance", "false") == "true"
    method = request.form.get("method", "builtin")

    # Read image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not read image"}), 400

    # keep the true upload for the "before" image, enhance only if asked to
    original = img.copy()
    was_enhanced = False
    enhance_method = None
    if do_enhance:
        if method == "ai" and SR_WEIGHTS.exists():
            try:
                img = super_resolve(img)
                enhance_method = "ai"
            except Exception:
                img = enhance_image(img)      # fall back if the SR model fails
                enhance_method = "builtin (AI unavailable)"
        else:
            img = enhance_image(img)
            enhance_method = "builtin"
        was_enhanced = True
    h, w = img.shape[:2]

    # Draw detections
    annotated = img.copy()
    detections = []

    dets = detect_tiled(yolo, img, tile=640, overlap=128, conf=conf_thresh)
    if len(dets) > 10:
        dets = dets[dets[:, 4].argsort()[::-1][:10]]
    for x1, y1, x2, y2, confidence in dets:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        confidence = float(confidence)

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
        "enhanced": was_enhanced,
        "enhance_method": enhance_method,
        "result_image": cv2_to_b64(annotated),
        "original_image": cv2_to_b64(original),
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
