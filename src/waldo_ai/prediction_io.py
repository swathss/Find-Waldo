from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .geometry import Box


COLUMNS = ["image_id", "class_id", "x1", "y1", "x2", "y2", "confidence"]


def boxes_to_csv(boxes: Iterable[Box], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([box.__dict__ for box in boxes], columns=COLUMNS).to_csv(path, index=False)


def boxes_from_csv(path: Path) -> list[Box]:
    frame = pd.read_csv(path)
    return [Box(**row) for row in frame[COLUMNS].to_dict("records")]

