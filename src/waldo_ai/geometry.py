from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable


@dataclass(frozen=True)
class Box:
    image_id: str
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def clipped(self, width: float, height: float) -> "Box":
        return replace(
            self,
            x1=min(max(self.x1, 0.0), width),
            y1=min(max(self.y1, 0.0), height),
            x2=min(max(self.x2, 0.0), width),
            y2=min(max(self.y2, 0.0), height),
        )

    def translated(self, dx: float, dy: float, image_id: str | None = None) -> "Box":
        return replace(
            self,
            image_id=self.image_id if image_id is None else image_id,
            x1=self.x1 + dx,
            y1=self.y1 + dy,
            x2=self.x2 + dx,
            y2=self.y2 + dy,
        )


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def non_max_suppression(boxes: Iterable[Box], threshold: float = 0.5) -> list[Box]:
    ordered = sorted(boxes, key=lambda box: box.confidence, reverse=True)
    kept: list[Box] = []
    while ordered:
        current = ordered.pop(0)
        kept.append(current)
        ordered = [
            candidate
            for candidate in ordered
            if candidate.class_id != current.class_id or iou(current, candidate) < threshold
        ]
    return kept


def tile_origins(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Require tile_size > 0 and 0 <= overlap < tile_size")
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    origins = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if origins[-1] != last:
        origins.append(last)
    return origins

