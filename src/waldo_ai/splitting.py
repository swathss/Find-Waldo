from __future__ import annotations

import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from .yolo_io import class_names, image_files, load_dataset_yaml, matching_label_path, read_yolo_labels, split_image_dir


def _source_key(filename: str) -> str:
    return filename.split(".rf.")[0]


def _choose_assignment(groups: dict[str, list[Path]], seed: int) -> dict[str, str]:
    keys = sorted(groups)
    target = {"train": 0.70, "valid": 0.15, "test": 0.15}
    best: tuple[float, dict[str, str]] | None = None
    total_images = sum(len(images) for images in groups.values())
    group_classes = {
        key: Counter(box.class_id for image in images for box in read_yolo_labels(image))
        for key, images in groups.items()
    }
    total_classes = Counter()
    for counts in group_classes.values():
        total_classes.update(counts)
    split_group_counts = {
        "train": round(len(keys) * target["train"]),
        "valid": round(len(keys) * target["valid"]),
    }
    split_group_counts["test"] = len(keys) - split_group_counts["train"] - split_group_counts["valid"]
    for attempt in range(5000):
        rng = random.Random(seed + attempt)
        shuffled = keys.copy()
        rng.shuffle(shuffled)
        boundaries = (
            split_group_counts["train"],
            split_group_counts["train"] + split_group_counts["valid"],
        )
        assignment = {
            key: "train" if index < boundaries[0] else "valid" if index < boundaries[1] else "test"
            for index, key in enumerate(shuffled)
        }
        image_counts = Counter()
        class_counts: Counter[tuple[str, int]] = Counter()
        for key, split in assignment.items():
            image_counts[split] += len(groups[key])
            for class_id, count in group_classes[key].items():
                class_counts[(split, class_id)] += count
        if any(class_counts[(split, class_id)] == 0 for split in target for class_id in total_classes):
            continue
        image_score = sum(abs(image_counts[name] / total_images - target[name]) for name in target)
        class_score = sum(
            abs(class_counts[(split, class_id)] / total - target[split])
            for class_id, total in total_classes.items()
            for split in target
        )
        score = 2 * image_score + class_score
        if best is None or score < best[0]:
            best = (score, assignment)
    assert best is not None
    return best[1]


def create_grouped_split(dataset_root: Path, output_root: Path, seed: int = 42) -> dict:
    _, config = load_dataset_yaml(dataset_root)
    names = class_names(config)
    groups: defaultdict[str, list[Path]] = defaultdict(list)
    for old_split in ("train", "valid", "test"):
        for image_path in image_files(split_image_dir(dataset_root, old_split)):
            groups[_source_key(image_path.name)].append(image_path)

    assignment = _choose_assignment(groups, seed)
    resolved_output = output_root.resolve()
    if resolved_output == dataset_root.resolve() or output_root.parent == output_root:
        raise ValueError("Output must be a separate, non-root directory")
    if output_root.exists():
        shutil.rmtree(output_root)
    rows = []
    class_counts: Counter[tuple[str, int]] = Counter()
    image_counts = Counter()

    for source_key, images in groups.items():
        new_split = assignment[source_key]
        image_directory = output_root / new_split / "images"
        label_directory = output_root / new_split / "labels"
        image_directory.mkdir(parents=True, exist_ok=True)
        label_directory.mkdir(parents=True, exist_ok=True)
        for image_path in images:
            label_path = matching_label_path(image_path)
            shutil.copy2(image_path, image_directory / image_path.name)
            if label_path.exists():
                shutil.copy2(label_path, label_directory / label_path.name)
            else:
                (label_directory / f"{image_path.stem}.txt").write_text("", encoding="utf-8")
            boxes = read_yolo_labels(image_path)
            image_counts[new_split] += 1
            for box in boxes:
                class_counts[(new_split, box.class_id)] += 1
            rows.append(
                {
                    "source_key": source_key,
                    "image": image_path.name,
                    "original_split": image_path.parent.parent.name,
                    "new_split": new_split,
                    "boxes": len(boxes),
                }
            )

    with (output_root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    yaml_config = {
        "path": str(output_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": names,
    }
    (output_root / "data.yaml").write_text(yaml.safe_dump(yaml_config, sort_keys=False), encoding="utf-8")
    return {
        "groups": len(groups),
        "images": dict(image_counts),
        "class_boxes": {
            f"{split}:{names.get(class_id, class_id)}": count
            for (split, class_id), count in sorted(class_counts.items())
        },
        "output": str(output_root),
    }
