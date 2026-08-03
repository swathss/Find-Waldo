from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image

from .geometry import Box, tile_origins
from .yolo_io import class_names, image_files, load_dataset_yaml, read_yolo_labels, split_image_dir, write_yolo_labels


def _box_in_tile(box: Box, x0: int, y0: int, tile_size: int, min_visibility: float) -> Box | None:
    centre_x, centre_y = box.centre
    if not (x0 <= centre_x < x0 + tile_size and y0 <= centre_y < y0 + tile_size):
        return None
    local = box.translated(-x0, -y0).clipped(tile_size, tile_size)
    if box.area <= 0 or local.area / box.area < min_visibility:
        return None
    return local


def slice_dataset(
    dataset_root: Path,
    output_root: Path,
    tile_size: int = 256,
    overlap: int = 64,
    negative_ratio: float = 3.0,
    min_visibility: float = 0.4,
    seed: int = 42,
) -> dict:
    _, config = load_dataset_yaml(dataset_root)
    names = class_names(config)
    random_generator = random.Random(seed)
    if output_root.exists():
        shutil.rmtree(output_root)
    manifest: list[dict] = []
    totals = {"positive_tiles": 0, "negative_tiles": 0}

    for split in ("train", "valid", "test"):
        source_dir = split_image_dir(dataset_root, split)
        destination_split = "val" if split == "valid" else split
        image_out = output_root / "images" / destination_split
        label_out = output_root / "labels" / destination_split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        pending_negative: list[tuple[Image.Image, str, int, int, str]] = []

        for image_path in image_files(source_dir):
            image = Image.open(image_path).convert("RGB")
            boxes = read_yolo_labels(image_path)
            for y0 in tile_origins(image.height, tile_size, overlap):
                for x0 in tile_origins(image.width, tile_size, overlap):
                    tile_boxes = [
                        local
                        for box in boxes
                        if (local := _box_in_tile(box, x0, y0, tile_size, min_visibility)) is not None
                    ]
                    stem = f"{image_path.stem}__x{x0}_y{y0}"
                    tile = image.crop((x0, y0, x0 + tile_size, y0 + tile_size))
                    if tile_boxes:
                        tile.save(image_out / f"{stem}.jpg", quality=95)
                        write_yolo_labels(label_out / f"{stem}.txt", tile_boxes, tile_size, tile_size)
                        totals["positive_tiles"] += 1
                        manifest.append(
                            {"split": destination_split, "tile": f"{stem}.jpg", "source": image_path.name, "x0": x0, "y0": y0, "objects": len(tile_boxes)}
                        )
                    else:
                        pending_negative.append((tile, stem, x0, y0, image_path.name))

        positive_in_split = sum(1 for row in manifest if row["split"] == destination_split and row["objects"] > 0)
        keep_negatives = min(len(pending_negative), int(round(positive_in_split * negative_ratio)))
        for tile, stem, x0, y0, source_name in random_generator.sample(pending_negative, keep_negatives):
            tile.save(image_out / f"{stem}.jpg", quality=95)
            write_yolo_labels(label_out / f"{stem}.txt", [], tile_size, tile_size)
            totals["negative_tiles"] += 1
            manifest.append(
                {"split": destination_split, "tile": f"{stem}.jpg", "source": source_name, "x0": x0, "y0": y0, "objects": 0}
            )

    with (output_root / "tile_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "tile", "source", "x0", "y0", "objects"])
        writer.writeheader()
        writer.writerows(manifest)
    yaml_config = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    (output_root / "data.yaml").write_text(yaml.safe_dump(yaml_config, sort_keys=False), encoding="utf-8")
    return {**totals, "tiles": len(manifest), "output": str(output_root)}

