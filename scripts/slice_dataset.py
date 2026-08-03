from __future__ import annotations

import argparse
import json
from pathlib import Path

from waldo_ai.tiling import slice_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Create overlapping YOLO tiles without crossing source splits.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/wally-finder-tiles"))
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--negative-ratio", type=float, default=3.0)
    parser.add_argument("--min-visibility", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = slice_dataset(
        args.dataset,
        args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        negative_ratio=args.negative_ratio,
        min_visibility=args.min_visibility,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

