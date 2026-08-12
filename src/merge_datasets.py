"""Build a leakage-safe, single-class YOLO dataset from public sources.

The same puzzle page appears in several repositories and often has many
Roboflow-exported variants.  Samples are therefore split by a normalized
source-page group, never by individual file.  This keeps near-duplicates out
of validation and test when they already occur in training.

Supported sources:
  * HereIsWally: full scenes with CSV bounding boxes (MIT repository).
  * Wally-Finder v5: 249 YOLO images, four character classes (CC BY 4.0).
    Class 0 (Wally) is retained; other-character-only images become useful
    hard negatives with empty YOLO label files.
  * Hey-Waldo: 256 px classification patches (ODbL).  Only the negative
    patches are used here because positive patches do not locate Wally.  The
    positive patches remain available to the synthetic/background pipeline.

The generated data/processed_v2 directory is reproducible and git-ignored.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
ROBOFLOW = ROOT / "data" / "roboflow" / "wally-finder-v5"
PROC = ROOT / "data" / "processed_v2"

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class Sample:
    image: Path
    labels: tuple[str, ...]
    source: str
    group: str
    crop: tuple[int, int, int, int] | None = None

    @property
    def positive(self) -> bool:
        return bool(self.labels)


def find_existing(*paths: Path) -> Path:
    """Return the first existing path (Windows and Linux layouts supported)."""
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def yolo_line(cls: int, cx: float, cy: float, width: float, height: float) -> str:
    values = (cx, cy, width, height)
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"YOLO coordinates outside [0, 1]: {values}")
    return f"{cls} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"


def normalized_group(filename: str, source: str) -> str:
    """Map exported/cropped filenames back to a stable source-page group."""
    stem = Path(filename).stem
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_(?:jpe?g|png)$", "", stem, flags=re.IGNORECASE)

    # Hey-Waldo patches use <page>_<row>_<column>.jpg.  Numeric filenames in
    # the other datasets use the same page numbering, including leading zeroes.
    match = re.match(r"^(\d+)(?:_\d+_\d+)?$", stem)
    if match:
        return f"page:{int(match.group(1))}"

    slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-") or "unknown"
    return f"{source.lower()}:{slug}"


def remap_yolo_labels(label_path: Path, waldo_class_id: int = 0) -> tuple[str, ...]:
    """Keep one class and convert YOLO boxes or polygons to class-zero boxes."""
    if not label_path.exists():
        return ()

    labels: list[str] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = raw_line.split()
        if not parts:
            continue
        try:
            class_id = int(parts[0])
            coords = tuple(map(float, parts[1:]))
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number}: invalid YOLO row") from exc
        if class_id == waldo_class_id:
            if len(coords) == 4:
                cx, cy, width, height = coords
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                # Roboflow YOLOv8 segmentation export: x1 y1 x2 y2 ...
                xs, ys = coords[0::2], coords[1::2]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
                width, height = xmax - xmin, ymax - ymin
            else:
                raise ValueError(
                    f"{label_path}:{line_number}: expected a YOLO box or polygon"
                )
            labels.append(yolo_line(0, cx, cy, width, height))
    return tuple(labels)


def collect_hereiswally() -> list[Sample]:
    base = find_existing(RAW / "HereIsWally", RAW / "here-is-wally")
    csv_path = base / "annotations" / "annotations.csv"
    image_dir = base / "images"
    if not csv_path.exists():
        print(f"  [skip] HereIsWally not found at {base}")
        return []

    rows_by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows_by_file[row["filename"]].append(row)

    samples: list[Sample] = []
    tile = 640
    for filename, rows in sorted(rows_by_file.items()):
        image_path = image_dir / filename
        if not image_path.exists():
            continue

        image_width = int(float(rows[0]["width"]))
        image_height = int(float(rows[0]["height"]))
        # One crop per annotated object.  The group remains the page filename,
        # so pages with multiple boxes can never cross split boundaries.
        for annotation_index, row in enumerate(rows):
            xmin, ymin = float(row["xmin"]), float(row["ymin"])
            xmax, ymax = float(row["xmax"]), float(row["ymax"])
            object_cx, object_cy = (xmin + xmax) / 2, (ymin + ymax) / 2

            x1 = int(max(0, object_cx - tile / 2))
            y1 = int(max(0, object_cy - tile / 2))
            x2, y2 = min(image_width, x1 + tile), min(image_height, y1 + tile)
            x1, y1 = max(0, x2 - tile), max(0, y2 - tile)
            crop_width, crop_height = x2 - x1, y2 - y1
            if crop_width <= 0 or crop_height <= 0:
                continue

            label = yolo_line(
                0,
                (object_cx - x1) / crop_width,
                (object_cy - y1) / crop_height,
                (xmax - xmin) / crop_width,
                (ymax - ymin) / crop_height,
            )
            samples.append(
                Sample(
                    image=image_path,
                    labels=(label,),
                    source="HereIsWally",
                    group=normalized_group(filename, "HereIsWally"),
                    crop=(x1, y1, x2, y2),
                )
            )

    print(f"  HereIsWally crops          : {len(samples)}")
    return samples


def matching_label(image: Path) -> Path:
    # Roboflow layout: <split>/images/x.jpg and <split>/labels/x.txt
    if image.parent.name == "images":
        return image.parent.parent / "labels" / f"{image.stem}.txt"
    return image.with_suffix(".txt")


def collect_wallyfinder() -> list[Sample]:
    base = find_existing(ROBOFLOW, RAW / "wally-finder-v5")
    if not base.exists():
        print(f"  [skip] Wally-Finder v5 not found at {base}")
        return []

    samples: list[Sample] = []
    for split_name in ("train", "valid", "val", "test"):
        image_dir = base / split_name / "images"
        if not image_dir.exists():
            continue
        for image in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            samples.append(
                Sample(
                    image=image,
                    labels=remap_yolo_labels(matching_label(image), waldo_class_id=0),
                    source="Wally-Finder-v5",
                    group=normalized_group(image.name, "Wally-Finder-v5"),
                )
            )

    positive = sum(sample.positive for sample in samples)
    print(f"  Wally-Finder v5           : {len(samples)} ({positive} positive)")
    return samples


def collect_heywaldo_negatives() -> list[Sample]:
    base = find_existing(RAW / "Hey-Waldo", RAW / "hey-waldo") / "256" / "notwaldo"
    if not base.exists():
        print(f"  [skip] Hey-Waldo negatives not found at {base}")
        return []

    samples = [
        Sample(
            image=image,
            labels=(),
            source="Hey-Waldo-negative",
            group=normalized_group(image.name, "Hey-Waldo"),
        )
        for image in sorted(base.iterdir())
        if image.suffix.lower() in IMAGE_SUFFIXES
    ]
    print(f"  Hey-Waldo hard negatives  : {len(samples)}")
    return samples


def split_group_names(
    groups: dict[str, list[Sample]], seed: int = SEED
) -> dict[str, str]:
    """Stratify positive and negative page groups, then assign whole groups."""
    positive_groups = sorted(
        group for group, samples in groups.items() if any(sample.positive for sample in samples)
    )
    negative_groups = sorted(set(groups) - set(positive_groups))
    rng = random.Random(seed)
    rng.shuffle(positive_groups)
    rng.shuffle(negative_groups)

    assignment: dict[str, str] = {}
    for names in (positive_groups, negative_groups):
        count = len(names)
        if not count:
            continue
        train_end = int(count * SPLIT_RATIOS["train"])
        val_end = train_end + int(count * SPLIT_RATIOS["val"])
        if count >= 3:
            train_end = min(max(train_end, 1), count - 2)
            val_end = min(max(val_end, train_end + 1), count - 1)
        for index, group in enumerate(names):
            assignment[group] = (
                "train" if index < train_end else "val" if index < val_end else "test"
            )
    return assignment


def sample_fingerprint(sample: Sample) -> str:
    """Hash decoded pixels so identical images with different JPEG bytes match."""
    from PIL import Image

    with Image.open(sample.image) as opened:
        image = opened.crop(sample.crop) if sample.crop is not None else opened.copy()
        image = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:RGB".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


def link_duplicate_groups(samples: Iterable[Sample]) -> dict[str, str]:
    """Union filename-derived groups when any decoded sample is identical."""
    groups = sorted({sample.group for sample in samples})
    parent = {group: group for group in groups}

    def find(group: str) -> str:
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parent[second] = first

    first_group_by_hash: dict[str, str] = {}
    for sample in samples:
        fingerprint = sample_fingerprint(sample)
        previous = first_group_by_hash.setdefault(fingerprint, sample.group)
        union(previous, sample.group)
    return {group: find(group) for group in groups}


def safe_stem(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_") or "sample"


def reset_output(output: Path) -> None:
    if output.exists():
        # This directory is created by this script and is git-ignored.
        shutil.rmtree(output)
    for split in SPLIT_RATIOS:
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)


def write_samples(
    samples: Iterable[Sample],
    assignment: dict[str, str],
    duplicate_clusters: dict[str, str],
    output: Path,
) -> None:
    from PIL import Image

    manifest_rows: list[dict[str, object]] = []
    counters: dict[tuple[str, str], int] = defaultdict(int)
    for sample in samples:
        split = assignment[sample.group]
        key = (split, sample.source)
        index = counters[key]
        counters[key] += 1
        stem = f"{safe_stem(sample.source)}_{index:04d}"

        destination = output / split / "images" / f"{stem}.jpg"
        try:
            with Image.open(sample.image) as opened:
                image = opened.crop(sample.crop) if sample.crop is not None else opened.copy()
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(destination, format="JPEG", quality=95)
        except (OSError, ValueError) as exc:
            raise ValueError(f"could not convert {sample.image}") from exc

        (output / split / "labels" / f"{stem}.txt").write_text(
            "\n".join(sample.labels) + ("\n" if sample.labels else ""),
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "split": split,
                "source": sample.source,
                "group": sample.group,
                "duplicate_cluster": duplicate_clusters[sample.group],
                "source_image": sample.image.as_posix(),
                "output_image": destination.relative_to(output).as_posix(),
                "waldo_boxes": len(sample.labels),
            }
        )

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)


def write_yaml(output: Path) -> None:
    (output / "data.yaml").write_text(
        f"path: {output.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names: ['waldo']\n",
        encoding="utf-8",
    )


def print_inventory(
    samples: list[Sample],
    assignment: dict[str, str],
    duplicate_clusters: dict[str, str],
) -> None:
    print("\nMerged inventory:")
    for split in SPLIT_RATIOS:
        subset = [sample for sample in samples if assignment[sample.group] == split]
        boxes = sum(len(sample.labels) for sample in subset)
        groups = {sample.group for sample in subset}
        print(f"  {split:5s}: {len(subset):4d} images, {boxes:3d} boxes, {len(groups):2d} groups")

    split_groups = {
        split: {sample.group for sample in samples if assignment[sample.group] == split}
        for split in SPLIT_RATIOS
    }
    assert not (split_groups["train"] & split_groups["val"])
    assert not (split_groups["train"] & split_groups["test"])
    assert not (split_groups["val"] & split_groups["test"])
    cluster_splits: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        cluster_splits[duplicate_clusters[sample.group]].add(assignment[sample.group])
    assert all(len(splits) == 1 for splits in cluster_splits.values())
    linked = len(set(duplicate_clusters)) - len(set(duplicate_clusters.values()))
    print(
        "  leakage check: PASS "
        f"(no page/content group crosses splits; {linked} filename groups linked by pixels)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROC)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print("Collecting licensed data sources...")
    samples = collect_hereiswally() + collect_wallyfinder() + collect_heywaldo_negatives()
    if not samples or not any(sample.positive for sample in samples):
        raise SystemExit("No positive Waldo samples found. Run src/download_dataset.py first.")

    duplicate_clusters = link_duplicate_groups(samples)
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[duplicate_clusters[sample.group]].append(sample)
    cluster_assignment = split_group_names(groups, seed=args.seed)
    assignment = {
        group: cluster_assignment[cluster]
        for group, cluster in duplicate_clusters.items()
    }

    reset_output(args.output)
    write_samples(samples, assignment, duplicate_clusters, args.output)
    write_yaml(args.output)
    print_inventory(samples, assignment, duplicate_clusters)
    print(f"\nWrote {args.output / 'data.yaml'} and {args.output / 'manifest.csv'}")


if __name__ == "__main__":
    main()
