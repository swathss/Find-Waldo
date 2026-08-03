from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

from .yolo_io import class_names, image_files, load_dataset_yaml, matching_label_path, read_yolo_labels, split_image_dir


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difference_hash(image: Image.Image, size: int = 8) -> str:
    gray = image.convert("L").resize((size + 1, size))
    pixels = list(gray.getdata())
    bits = []
    for row in range(size):
        offset = row * (size + 1)
        bits.extend(pixels[offset + col] > pixels[offset + col + 1] for col in range(size))
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:0{size * size // 4}x}"


def audit_dataset(dataset_root: Path, output_dir: Path) -> dict:
    _, config = load_dataset_yaml(dataset_root)
    names = class_names(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    issues: list[dict] = []
    class_counter: Counter[tuple[str, int]] = Counter()

    for split in ("train", "valid", "test"):
        try:
            image_dir = split_image_dir(dataset_root, split)
        except FileNotFoundError as exc:
            issues.append({"type": "missing_split", "split": split, "detail": str(exc)})
            continue
        for image_path in image_files(image_dir):
            try:
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
                    dhash = _difference_hash(image)
            except Exception as exc:  # damaged inputs should be reported, not hidden
                issues.append({"type": "unreadable_image", "path": str(image_path), "detail": str(exc)})
                continue
            label_path = matching_label_path(image_path)
            if not label_path.exists():
                issues.append({"type": "missing_label", "path": str(image_path)})
            try:
                boxes = read_yolo_labels(image_path)
            except Exception as exc:
                issues.append({"type": "invalid_label", "path": str(label_path), "detail": str(exc)})
                boxes = []
            for box in boxes:
                class_counter[(split, box.class_id)] += 1
                if box.class_id not in names:
                    issues.append({"type": "unknown_class", "path": str(label_path), "class_id": box.class_id})
                if box.x1 < 0 or box.y1 < 0 or box.x2 > width or box.y2 > height or box.area <= 0:
                    issues.append({"type": "invalid_box", "path": str(label_path), "box": box.__dict__})
            rows.append(
                {
                    "split": split,
                    "image": image_path.name,
                    "source_key": image_path.name.split(".rf.")[0],
                    "path": str(image_path),
                    "width": width,
                    "height": height,
                    "boxes": len(boxes),
                    "sha256": _sha256(image_path),
                    "dhash": dhash,
                    "min_relative_area": min((box.area / (width * height) for box in boxes), default=0.0),
                }
            )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        for source_key, group in frame.groupby("source_key"):
            if len(group) > 1 and group["split"].nunique() > 1:
                issues.append(
                    {
                        "type": "cross_split_source_duplicate",
                        "source_key": source_key,
                        "items": group[["split", "image"]].to_dict("records"),
                    }
                )
        for key in ("sha256", "dhash"):
            for value, group in frame.groupby(key):
                if len(group) > 1 and group["split"].nunique() > 1:
                    issues.append(
                        {
                            "type": f"cross_split_{key}_duplicate",
                            "hash": value,
                            "items": group[["split", "image"]].to_dict("records"),
                        }
                    )
        frame.to_csv(output_dir / "dataset_inventory.csv", index=False)

    counts = [
        {"split": split, "class_id": class_id, "class_name": names.get(class_id, "unknown"), "boxes": count}
        for (split, class_id), count in sorted(class_counter.items())
    ]
    pd.DataFrame(counts).to_csv(output_dir / "class_counts.csv", index=False)
    summary = {
        "dataset_root": str(dataset_root.resolve()),
        "images": len(rows),
        "classes": names,
        "issues": issues,
        "issue_count": len(issues),
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
