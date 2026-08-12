"""Track C: a feature verifier that checks a candidate crop really is Waldo.

The detector is good at proposing candidates but fires on anything with a bit of
red and white. This module scores each candidate by comparing its DINOv2
embedding against real Waldo crops and against decoy crops, so partial matches
(a red jacket, a striped shirt with no glasses/hat) get pushed down.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
# real Waldo boxes cropped in-context from the training pages (train split only,
# so val/test pages stay unseen). These match the busy-scene style of the
# candidate crops, which white-background cut-outs did not.
TRAIN_IMG = ROOT / "data" / "processed_v2" / "train" / "images"
TRAIN_LBL = ROOT / "data" / "processed_v2" / "train" / "labels"
CACHE = ROOT / "models" / "verifier"

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

_model = None
_dev = None


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_encoder():
    global _model, _dev
    if _model is None:
        _dev = _device()
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
        _model = m.to(_dev).eval()
    return _model, _dev


def _to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    x = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(x).permute(2, 0, 1)


def embed(crops, batch=32):
    # crops: list of BGR images -> L2-normalised embeddings (N, 384)
    model, dev = load_encoder()
    out = []
    for i in range(0, len(crops), batch):
        t = torch.stack([_to_tensor(c) for c in crops[i:i + batch]]).to(dev)
        with torch.no_grad():
            e = model(t).float().cpu().numpy()
        out.append(e)
    e = np.concatenate(out, axis=0) if out else np.zeros((0, 384), np.float32)
    n = np.linalg.norm(e, axis=1, keepdims=True)
    return e / np.clip(n, 1e-8, None)


def _crop_padded(img, x1, y1, x2, y2, pad_frac=0.3):
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
    c = img[cy1:cy2, cx1:cx2]
    return c if c.size else None


def _gather_crops(neg_per_page=6, seed=0):
    # positives: real Waldo boxes cropped in-context; negatives: random same-size
    # crops from the same pages that do not overlap the Waldo box
    rng = np.random.default_rng(seed)
    pos, neg = [], []
    for lbl in sorted(TRAIN_LBL.glob("*.txt")):
        img_path = next((TRAIN_IMG / (lbl.stem + e) for e in (".jpg", ".png", ".jpeg")
                         if (TRAIN_IMG / (lbl.stem + e)).exists()), None)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = []
        for ln in lbl.read_text().strip().splitlines():
            p = ln.split()
            if len(p) != 5:
                continue
            _, cx, cy, bw, bh = map(float, p)
            if bw > 0.9 and bh > 0.9:              # whole-image box, skip
                continue
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            boxes.append((x1, y1, x2, y2))
            c = _crop_padded(img, x1, y1, x2, y2)
            if c is not None:
                pos.append(c)
        # sample negatives of a plausible Waldo size away from the real boxes
        bw0 = int(np.mean([b[2] - b[0] for b in boxes])) if boxes else max(16, w // 20)
        bh0 = int(np.mean([b[3] - b[1] for b in boxes])) if boxes else max(24, h // 15)
        for _ in range(neg_per_page):
            if w - bw0 <= 1 or h - bh0 <= 1:
                break
            rx, ry = int(rng.integers(0, w - bw0)), int(rng.integers(0, h - bh0))
            if any(rx < b[2] and rx + bw0 > b[0] and ry < b[3] and ry + bh0 > b[1] for b in boxes):
                continue
            c = _crop_padded(img, rx, ry, rx + bw0, ry + bh0)
            if c is not None:
                neg.append(c)
    return pos, neg


def build_banks(rebuild=False):
    # cache positive (Waldo) and negative (scene clutter) reference embeddings
    CACHE.mkdir(parents=True, exist_ok=True)
    pos_f, neg_f = CACHE / "pos.npy", CACHE / "neg.npy"
    if not rebuild and pos_f.exists() and neg_f.exists():
        return np.load(pos_f), np.load(neg_f)

    pos, neg = _gather_crops()
    if not pos:
        raise RuntimeError(f"no Waldo boxes found under {TRAIN_LBL}")

    pos_e, neg_e = embed(pos), embed(neg)
    np.save(pos_f, pos_e)
    np.save(neg_f, neg_e)
    print(f"verifier banks: {len(pos_e)} Waldo refs, {len(neg_e)} decoy refs")
    return pos_e, neg_e


def score(crops, pos_e, neg_e):
    # waldo-ness = closeness to real Waldo minus closeness to decoys, in [-2, 2]
    if not crops:
        return np.zeros((0,), np.float32)
    e = embed(crops)
    pos_sim = (e @ pos_e.T).max(axis=1) if len(pos_e) else np.zeros(len(e), np.float32)
    neg_sim = (e @ neg_e.T).max(axis=1) if len(neg_e) else np.zeros(len(e), np.float32)
    return pos_sim - neg_sim
