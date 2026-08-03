from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image

from .geometry import Box

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_dataset_yaml(dataset_root: Path) -> tuple[Path, dict]:
    candidates = sorted(dataset_root.glob("*.yaml")) + sorted(dataset_root.glob("*.yml"))
    if not candidates:
        raise FileNotFoundError(f"No YAML file found in {dataset_root}")
    yaml_path = candidates[0]
    return yaml_path, yaml.safe_load(yaml_path.read_text(encoding="utf-8"))


def class_names(config: dict) -> dict[int, str]:
    names = config.get("names", {})
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(key): str(value) for key, value in names.items()}


def split_image_dir(dataset_root: Path, split: str) -> Path:
    aliases = {"val": ["valid", "val"], "valid": ["valid", "val"]}
    for name in aliases.get(split, [split]):
        for candidate in (dataset_root / name / "images", dataset_root / "images" / name):
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Cannot find image directory for split '{split}' in {dataset_root}")


def image_files(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def matching_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" not in parts:
        raise ValueError(f"Expected an images directory in {image_path}")
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_yolo_labels(image_path: Path) -> list[Box]:
    width, height = Image.open(image_path).size
    label_path = matching_label_path(image_path)
    if not label_path.exists():
        return []
    boxes: list[Box] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Malformed label at {label_path}:{line_number}")
        class_id = int(float(fields[0]))
        coordinates = list(map(float, fields[1:]))
        if len(coordinates) == 4:
            xc, yc, bw, bh = coordinates
            x1, y1 = xc - bw / 2, yc - bh / 2
            x2, y2 = xc + bw / 2, yc + bh / 2
        elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
            # YOLO segmentation labels contain normalized polygon x,y pairs.
            xs = coordinates[0::2]
            ys = coordinates[1::2]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            raise ValueError(f"Unsupported YOLO label shape at {label_path}:{line_number}")
        boxes.append(
            Box(
                image_id=image_path.name,
                class_id=class_id,
                x1=x1 * width,
                y1=y1 * height,
                x2=x2 * width,
                y2=y2 * height,
            ).clipped(width, height)
        )
    return boxes


def write_yolo_labels(path: Path, boxes: Iterable[Box], width: int, height: int) -> None:
    lines = []
    for box in boxes:
        clipped = box.clipped(width, height)
        xc = min(1.0, max(0.0, ((clipped.x1 + clipped.x2) / 2) / width))
        yc = min(1.0, max(0.0, ((clipped.y1 + clipped.y2) / 2) / height))
        bw = min(1.0, max(0.0, clipped.width / width))
        bh = min(1.0, max(0.0, clipped.height / height))
        lines.append(f"{box.class_id} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
