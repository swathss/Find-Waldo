"""Download the reproducible Git-hosted datasets and report local inventory.

Wally-Finder v5 is distributed through Roboflow Universe.  Download its YOLOv8
export from the URL printed by this script and extract it to either
data/roboflow/wally-finder-v5 or data/raw/wally-finder-v5.  An existing export
is detected automatically.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
ROBOFLOW = ROOT / "data" / "roboflow" / "wally-finder-v5"

GIT_SOURCES = (
    (
        "Hey-Waldo",
        "https://github.com/vc1492a/Hey-Waldo.git",
        RAW / "Hey-Waldo",
    ),
    (
        "HereIsWally",
        "https://github.com/tadejmagajna/HereIsWally.git",
        RAW / "HereIsWally",
    ),
)
WALLY_FINDER_URL = "https://universe.roboflow.com/wheres-wally/wally-finder/dataset/5"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def find_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def clone_repo(name: str, url: str, destination: Path) -> None:
    # Accommodate the lowercase folder created by earlier versions on Linux.
    existing = find_existing(destination, destination.with_name(destination.name.lower()))
    if existing.exists():
        print(f"[skip] {name}: {existing}")
        return
    print(f"[clone] {url} -> {destination}")
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(destination)],
        check=True,
    )


def image_count(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in directory.rglob("*")
    )


def inventory() -> None:
    hey = find_existing(RAW / "Hey-Waldo", RAW / "hey-waldo")
    here = find_existing(RAW / "HereIsWally", RAW / "here-is-wally")
    roboflow = find_existing(ROBOFLOW, RAW / "wally-finder-v5")

    hey_positive = image_count(hey / "256" / "waldo")
    hey_negative = image_count(hey / "256" / "notwaldo")

    annotation_file = here / "annotations" / "annotations.csv"
    here_rows = 0
    here_scenes = 0
    if annotation_file.exists():
        with annotation_file.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        here_rows = len(rows)
        here_scenes = len({row["filename"] for row in rows})

    wally_finder_images = sum(
        image_count(roboflow / split / "images")
        for split in ("train", "valid", "val", "test")
    )

    print("\nDataset inventory:")
    print(f"  Hey-Waldo 256px       : {hey_positive} positive, {hey_negative} negative")
    print(f"  HereIsWally           : {here_scenes} scenes, {here_rows} boxes")
    print(f"  Wally-Finder v5       : {wally_finder_images} images")
    if not wally_finder_images:
        print(f"\nOptional third source (YOLOv8 export, CC BY 4.0):\n  {WALLY_FINDER_URL}")
        print(f"Extract to:\n  {ROBOFLOW}")
    print("\nNext: python src/merge_datasets.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="do not access the network; only report data already present",
    )
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if not args.inventory_only:
        for name, url, destination in GIT_SOURCES:
            clone_repo(name, url, destination)
    inventory()


if __name__ == "__main__":
    main()
