"""Draw a small contact sheet of positive samples from each dataset split."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent


def positive_pairs(data: Path, split: str) -> list[tuple[Path, Path]]:
    label_dir = data / split / "labels"
    image_dir = data / split / "images"
    pairs = []
    for label in sorted(label_dir.glob("*.txt")):
        if not label.read_text(encoding="utf-8").strip():
            continue
        image = image_dir / f"{label.stem}.jpg"
        if image.exists():
            pairs.append((image, label))
    return pairs


def render_tile(image_path: Path, label_path: Path, split: str, size: int) -> Image.Image:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    image.thumbnail((size, size - 28))
    tile = Image.new("RGB", (size, size), "white")
    offset_x = (size - image.width) // 2
    offset_y = 24 + (size - 28 - image.height) // 2
    tile.paste(image, (offset_x, offset_y))
    draw = ImageDraw.Draw(tile)
    draw.text((6, 5), f"{split}: {image_path.stem}", fill="black")
    for line in label_path.read_text(encoding="utf-8").splitlines():
        _, cx, cy, width, height = map(float, line.split())
        x1 = offset_x + (cx - width / 2) * image.width
        y1 = offset_y + (cy - height / 2) * image.height
        x2 = offset_x + (cx + width / 2) * image.width
        y2 = offset_y + (cy + height / 2) * image.height
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=3)
    return tile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed_v2")
    parser.add_argument("--per-split", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.data / "preview.jpg"

    rng = random.Random(args.seed)
    size = 320
    splits = ("train", "val", "test")
    selected: dict[str, list[tuple[Path, Path]]] = {}
    for split in splits:
        pairs = positive_pairs(args.data, split)
        selected[split] = rng.sample(pairs, min(args.per_split, len(pairs)))

    rows = max(len(pairs) for pairs in selected.values())
    sheet = Image.new("RGB", (size * len(splits), size * rows), (230, 230, 230))
    for column, split in enumerate(splits):
        for row, (image, label) in enumerate(selected[split]):
            sheet.paste(render_tile(image, label, split, size), (column * size, row * size))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    print(output)


if __name__ == "__main__":
    main()
