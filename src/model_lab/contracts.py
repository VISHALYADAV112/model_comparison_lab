from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    score: float
    label: str
    class_id: int | None = None
    mask_path: str | None = None
    instance_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])


@dataclass
class ModelResult:
    model: str
    source: str
    width: int
    height: int
    elapsed_seconds: float
    detections: list[Detection]
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        sizes = [min(item.width, item.height) for item in self.detections]
        return {
            "model": self.model,
            "count": len(self.detections),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "smallest_box_side_px": round(min(sizes), 2) if sizes else None,
            "median_box_side_px": round(median(sizes), 2) if sizes else None,
            "error": self.error,
        }

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["summary"] = self.summary()
        return value


def resolve_mask_path(manifest_path: Path, mask_path: str | None) -> Path | None:
    if not mask_path:
        return None
    candidate = Path(mask_path)
    return candidate if candidate.is_absolute() else manifest_path.parent / candidate

