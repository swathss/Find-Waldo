"""Find Waldo web app.

Flow: upload -> optional enhance (local sharpen, Real-ESRGAN, or the finegrain
diffusion enhancer) -> pick a confidence threshold -> detect with multi-scale
WBF inference on the reported model.

Run: python web/app.py   then open http://localhost:8080
"""
import sys
import base64
import time
import tempfile

import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024

ROOT = Path(__file__).parent.parent
MODEL = ROOT / "models/waldo_book_decoy/weights/best.pt"
FALLBACK = ROOT / "models/waldo_book_ms/weights/best.pt"

sys.path.insert(0, str(ROOT / "scripts"))
from detect_multiscale import multiscale_detect, pick_device

yolo = YOLO(str(MODEL if MODEL.exists() else FALLBACK))
DEV = pick_device()
print(f"Model ready on {DEV}: {MODEL.name if MODEL.exists() else FALLBACK.name}")


# ---------- image <-> base64 ----------
def b64_to_cv2(data):
    if "," in data:
        data = data.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(data), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def cv2_to_b64(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ---------- enhancement backends ----------
def enhance_sharpen(img, min_side=1500, max_side=4000):
    """Local, fast: denoise, upscale small images, then unsharp mask."""
    h, w = img.shape[:2]
    img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    if min(h, w) < min_side:
        s = min(min_side / min(h, w), max_side / max(h, w))
        if s > 1.01:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_LANCZOS4)
    soft = cv2.GaussianBlur(img, (0, 0), 1.2)
    return cv2.addWeighted(img, 1.6, soft, -0.6, 0)


_sr = None


def enhance_realesrgan(img, max_long=3000):
    global _sr
    import torch
    if _sr is None:
        from spandrel import ModelLoader, ImageModelDescriptor
        m = ModelLoader().load_from_file(str(ROOT / "models/sr/RealESRGAN_x4plus_anime_6B.pth"))
        if not isinstance(m, ImageModelDescriptor):
            raise RuntimeError("unexpected SR model type")
        _sr = (m.to(DEV).eval(), DEV)
    model, dev = _sr
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(dev)
    with torch.no_grad():
        out = model(t).clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    sr = cv2.cvtColor((out * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    if max(sr.shape[:2]) > max_long:
        s = max_long / max(sr.shape[:2])
        sr = cv2.resize(sr, (round(sr.shape[1] * s), round(sr.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return sr


_fg_client = None


def enhance_finegrain(img, upscale=2):
    """Diffusion enhancer via the finegrain Hugging Face Space (needs internet)."""
    global _fg_client
    from gradio_client import Client, handle_file
    if _fg_client is None:
        _fg_client = Client("finegrain/finegrain-image-enhancer")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        cv2.imwrite(f.name, img)
        path = f.name
    res = _fg_client.predict(
        input_image=handle_file(path),
        prompt="a detailed Where's Waldo illustration, clean sharp linework, high detail",
        negative_prompt="blurry, low quality, artifacts",
        seed=42, upscale_factor=upscale, controlnet_scale=0.6, controlnet_decay=1.0,
        condition_scale=6, tile_width=112, tile_height=144, denoise_strength=0.35,
        num_inference_steps=18, solver="DDIM", api_name="/process",
    )
    after = res[1] if isinstance(res, (list, tuple)) else res
    out = cv2.imread(after)
    if out is None:
        raise RuntimeError("enhancer returned no image")
    return out


ENHANCERS = {"sharpen": enhance_sharpen, "realesrgan": enhance_realesrgan, "finegrain": enhance_finegrain}


# ---------- routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/enhance", methods=["POST"])
def enhance():
    data = request.get_json(silent=True) or {}
    method = data.get("method", "sharpen")
    if "image" not in data:
        return jsonify({"error": "No image provided."}), 400
    if method not in ENHANCERS:
        return jsonify({"error": f"Unknown method: {method}"}), 400
    img = b64_to_cv2(data["image"])
    if img is None:
        return jsonify({"error": "Could not read the image."}), 400
    t0 = time.time()
    try:
        out = ENHANCERS[method](img)
    except Exception as e:
        return jsonify({"error": f"Enhancement failed ({method}): {e}"}), 502
    h, w = out.shape[:2]
    return jsonify({"image": cv2_to_b64(out), "width": w, "height": h,
                    "method": method, "seconds": round(time.time() - t0, 1)})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    if "image" not in data:
        return jsonify({"error": "No image provided."}), 400
    conf = float(data.get("conf", 0.15))
    img = b64_to_cv2(data["image"])
    if img is None:
        return jsonify({"error": "Could not read the image."}), 400

    # cap very large (enhanced) images so detection stays responsive
    if max(img.shape[:2]) > 2600:
        s = 2600 / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)))
    h, w = img.shape[:2]

    t0 = time.time()
    dets = multiscale_detect(yolo, img, tiles=(320, 512, 768), conf=conf, device=DEV)
    if len(dets) > 10:
        dets = dets[dets[:, 4].argsort()[::-1][:10]]

    # return a clean image plus coordinates; the browser overlays the boxes so the
    # user can show one detection at a time
    out = []
    for i, (x1, y1, x2, y2, c) in enumerate(dets, 1):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        out.append({"rank": i, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2, "confidence": round(float(c), 3)})

    return jsonify({
        "found": len(out) > 0,
        "count": len(out),
        "detections": out,
        "width": w, "height": h,
        "seconds": round(time.time() - t0, 1),
        "result_image": cv2_to_b64(img),
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8080)
