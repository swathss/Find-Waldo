import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


def collect_paths():
    paths = []
    for size in (256, 128):
        d = RAW / "Hey-Waldo" / str(size) / "notwaldo"
        if d.exists():
            paths += sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
    for extra in ("Hey-Waldo/original-images", "HereIsWally/images"):
        d = RAW / extra
        if d.exists():
            paths += sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
    return paths


class BackgroundBank:
    def __init__(self, seed=42):
        self.paths = collect_paths()
        if not self.paths:
            raise RuntimeError("no backgrounds found under data/raw/")
        self.rng = random.Random(seed)

    def sample_tile(self, size):
        for _ in range(8):
            img = cv2.imread(str(self.rng.choice(self.paths)))
            if img is None:
                continue
            h, w = img.shape[:2]
            if h < size or w < size:
                img = cv2.resize(img, (max(size, w), max(size, h)))
                h, w = img.shape[:2]
            x = self.rng.randint(0, w - size)
            y = self.rng.randint(0, h - size)
            return img[y:y + size, x:x + size].copy()
        return np.random.default_rng(0).integers(0, 255, (size, size, 3)).astype(np.uint8)
