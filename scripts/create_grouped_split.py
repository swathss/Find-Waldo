from __future__ import annotations

import argparse
import json
from pathlib import Path

from waldo_ai.splitting import create_grouped_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-split Roboflow data by source ID to prevent leakage.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/wally-finder-grouped"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(create_grouped_split(args.dataset, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()

