from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .geometry import Box, non_max_suppression
from .yolo_io import class_names, image_files, load_dataset_yaml, read_yolo_labels, split_image_dir


def find_wally_class(dataset_root: Path) -> tuple[int, dict[int, str]]:
    _, config = load_dataset_yaml(dataset_root)
    names = class_names(config)
    matches = [class_id for class_id, name in names.items() if name.strip().lower() in {"wally", "waldo"}]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Wally/Waldo class, found {names}")
    return matches[0], names


def build_template_bank(dataset_root: Path, output_dir: Path, padding: float = 0.10) -> list[Path]:
    class_id, _ = find_wally_class(dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    templates: list[Path] = []
    for image_path in image_files(split_image_dir(dataset_root, "train")):
        image = Image.open(image_path).convert("RGB")
        for index, box in enumerate(read_yolo_labels(image_path)):
            if box.class_id != class_id:
                continue
            pad_x, pad_y = box.width * padding, box.height * padding
            crop_box = (
                max(0, int(box.x1 - pad_x)),
                max(0, int(box.y1 - pad_y)),
                min(image.width, int(box.x2 + pad_x)),
                min(image.height, int(box.y2 + pad_y)),
            )
            if crop_box[2] - crop_box[0] < 4 or crop_box[3] - crop_box[1] < 4:
                continue
            destination = output_dir / f"{image_path.stem}_{index}.png"
            image.crop(crop_box).save(destination)
            templates.append(destination)
    (output_dir / "metadata.json").write_text(
        json.dumps({"source": str(dataset_root), "class_id": class_id, "templates": len(templates)}, indent=2),
        encoding="utf-8",
    )
    return templates


def match_template_bank(
    image_path: Path,
    template_paths: list[Path],
    class_id: int,
    scales: tuple[float, ...] = (0.75, 1.0, 1.25),
    top_k: int = 5,
    nms_threshold: float = 0.3,
) -> list[Box]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    candidates: list[Box] = []
    for template_path in template_paths:
        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue
        for scale in scales:
            width = max(4, int(round(template.shape[1] * scale)))
            height = max(4, int(round(template.shape[0] * scale)))
            if width >= gray.shape[1] or height >= gray.shape[0]:
                continue
            resized = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
            response = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
            flat = response.ravel()
            count = min(top_k, flat.size)
            indices = np.argpartition(flat, -count)[-count:]
            for flat_index in indices:
                y, x = np.unravel_index(flat_index, response.shape)
                candidates.append(
                    Box(
                        image_id=image_path.name,
                        class_id=class_id,
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + width),
                        y2=float(y + height),
                        confidence=float(response[y, x]),
                    )
                )
    return non_max_suppression(candidates, nms_threshold)[:top_k]

