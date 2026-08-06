import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed_v2"


def first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def collect_paths():
    paths = []
    hey_waldo = first_existing(RAW / "Hey-Waldo", RAW / "hey-waldo")
    for size in (256, 128):
        d = hey_waldo / str(size) / "notwaldo"
        if d.exists():
            paths += sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))

    # Empty-label samples from the leakage-safe training split are also valid
    # hard-negative backgrounds. Do not sample full puzzle pages: they contain
    # an unlabelled Waldo and may belong to validation or test groups.
    image_dir = PROC / "train" / "images"
    label_dir = PROC / "train" / "labels"
    if image_dir.exists():
        for image in sorted(image_dir.iterdir()):
            label = label_dir / f"{image.stem}.txt"
            if image.suffix.lower() in {".jpg", ".jpeg", ".png"} and label.exists() and not label.read_text().strip():
                paths.append(image)
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
