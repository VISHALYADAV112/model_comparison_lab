from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


PERSON_SYNONYMS = {"person", "pedestrian", "human", "people"}


def canonical_label(label: str) -> str:
    normalized = label.strip().lower().rstrip(".")
    if normalized in PERSON_SYNONYMS or any(word in PERSON_SYNONYMS for word in normalized.split()):
        return "person"
    return normalized


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Invalid box coordinates: {self}")

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def translate(self, dx: float, dy: float) -> "Box":
        return Box(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def clip(self, width: float, height: float) -> "Box":
        return Box(
            max(0.0, min(width, self.x1)),
            max(0.0, min(height, self.y1)),
            max(0.0, min(width, self.x2)),
            max(0.0, min(height, self.y2)),
        )

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass
class Detection:
    box: Box
    score: float
    label: str
    model: str
    tile_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "box_xyxy": self.box.to_list(),
            "score": float(self.score),
            "label": self.label,
            "model": self.model,
            "tile_id": self.tile_id,
            "object_height_px": self.box.height,
            "object_width_px": self.box.width,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Tile:
    tile_id: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)
