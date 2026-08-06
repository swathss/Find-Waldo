import math
import random
from pathlib import Path

import cv2
import numpy as np


def load_rgba(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def rotate_scale(rgba, angle, scale):
    h, w = rgba.shape[:2]
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    rgba = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
    diag = int(math.hypot(new_w, new_h)) + 2
    canvas = np.zeros((diag, diag, 4), np.uint8)
    ox, oy = (diag - new_w) // 2, (diag - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = rgba
    M = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle, 1.0)
    return cv2.warpAffine(canvas, M, (diag, diag), flags=cv2.INTER_LINEAR)


def tight_bbox(alpha):
    ys, xs = np.where(alpha > 20)
    if xs.size == 0:
        return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def color_jitter(bgr, rng):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-8, 8)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.75, 1.25), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.7, 1.3), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def paste(dst, patch_rgba, x, y):
    ph, pw = patch_rgba.shape[:2]
    H, W = dst.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + pw), min(H, y + ph)
    if x1 <= x0 or y1 <= y0:
        return
    sx0, sy0 = x0 - x, y0 - y
    patch = patch_rgba[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    a = patch[:, :, 3:4].astype(np.float32) / 255.0
    dst[y0:y1, x0:x1] = (
        patch[:, :, :3].astype(np.float32) * a
        + dst[y0:y1, x0:x1].astype(np.float32) * (1 - a)
    ).astype(np.uint8)


def stripe_distractor(size, rng):
    # red/white striped patch - the usual thing that fools a Waldo detector
    patch = np.zeros((size, size, 4), np.uint8)
    sh = max(2, size // rng.randint(4, 8))
    for i in range(0, size, sh * 2):
        patch[i:i + sh, :, :3] = (60, 60, 210)
        patch[i + sh:i + 2 * sh, :, :3] = (245, 245, 245)
    patch[:, :, 3] = 255
    M = cv2.getRotationMatrix2D((size / 2, size / 2), rng.uniform(0, 360), 1.0)
    return cv2.warpAffine(patch, M, (size, size))


class Compositor:
    def __init__(self, fg_dir, seed=42, waldo_min_frac=0.05, waldo_max_frac=0.22):
        self.fgs = self._curate(sorted(Path(fg_dir).glob("*.png")))
        if not self.fgs:
            raise RuntimeError(f"no usable foregrounds in {fg_dir}")
        self.rng = random.Random(seed)
        self.waldo_min_frac = waldo_min_frac
        self.waldo_max_frac = waldo_max_frac

    @staticmethod
    def _curate(paths):
        # drop bad mattes (slivers, near-empty/near-full alpha); the hereiswally
        # crops are the cleanest so count them twice to sample them more often
        good = []
        for p in paths:
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if im is None or im.ndim != 3 or im.shape[2] != 4:
                continue
            h, w = im.shape[:2]
            if min(h, w) < 10:
                continue
            if max(h, w) / min(h, w) > 3.2:
                continue
            frac = (im[:, :, 3] > 20).mean()
            if frac < 0.12 or frac > 0.9:
                continue
            good.append(p)
            if p.name.startswith("fg_hiw_"):
                good.append(p)
        return good

    def compose(self, bg):
        rng = self.rng
        H, W = bg.shape[:2]
        tile = bg.copy()

        for _ in range(rng.randint(0, 4)):
            ds = rng.randint(int(W * 0.05), int(W * 0.18))
            paste(tile, stripe_distractor(ds, rng),
                  rng.randint(0, W - ds), rng.randint(0, H - ds))

        fg = load_rgba(rng.choice(self.fgs))
        while fg is None:
            fg = load_rgba(rng.choice(self.fgs))
        if rng.random() < 0.5:
            fg = cv2.flip(fg, 1)
        fg[:, :, :3] = color_jitter(fg[:, :, :3], rng)

        target = rng.uniform(self.waldo_min_frac, self.waldo_max_frac) * W
        scale = target / max(fg.shape[0], fg.shape[1])
        warped = rotate_scale(fg, rng.uniform(-18, 18), scale)

        bbox = tight_bbox(warped[:, :, 3])
        if bbox is None:
            return tile, (0.5, 0.5, 0.0, 0.0)
        bx0, by0, bx1, by1 = bbox
        warped = warped[by0:by1, bx0:bx1]
        ph, pw = warped.shape[:2]

        if rng.random() < 0.2:                 # sometimes clip an edge (occlusion)
            cut = rng.uniform(0.1, 0.3)
            if rng.random() < 0.5:
                warped[: int(ph * cut), :, 3] = 0
            else:
                warped[:, : int(pw * cut), 3] = 0

        px = rng.randint(0, max(0, W - pw))
        py = rng.randint(0, max(0, H - ph))
        paste(tile, warped, px, py)

        if rng.random() < 0.3:
            tile = cv2.GaussianBlur(tile, (3, 3), 0)
        if rng.random() < 0.3:
            noise = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 5, tile.shape)
            tile = np.clip(tile.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        vis = tight_bbox(warped[:, :, 3])
        if vis is None:
            return tile, (0.5, 0.5, 0.0, 0.0)
        vx0, vy0, vx1, vy1 = vis
        cx = (px + (vx0 + vx1) / 2) / W
        cy = (py + (vy0 + vy1) / 2) / H
        bw = (vx1 - vx0) / W
        bh = (vy1 - vy0) / H
        return tile, (cx, cy, bw, bh)
