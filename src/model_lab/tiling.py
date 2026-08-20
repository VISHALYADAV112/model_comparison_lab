from __future__ import annotations

from dataclasses import dataclass

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Tile:
    tile_id: str
    x1: int
    y1: int
    x2: int
    y2: int


def _axis_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, round(tile_size * (1.0 - overlap)))
    starts = list(range(0, max(1, length - tile_size + 1), stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def generate_tiles(width: int, height: int, tile_size: int, overlap: float = 0.2) -> list[Tile]:
    """Overlapping source-resolution tiles, ported from long_range_vision.tiling.

    Keeping full source pixels per tile is what lets a proposal detector see a
    small/distant object that a single whole-image resize would erase.
    """
    if width <= 0 or height <= 0 or tile_size <= 0:
        raise ValueError("Image dimensions and tile size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("Overlap must be in [0, 1)")

    actual_w = min(width, tile_size)
    actual_h = min(height, tile_size)
    tiles: list[Tile] = []
    for row, y1 in enumerate(_axis_starts(height, actual_h, overlap)):
        for col, x1 in enumerate(_axis_starts(width, actual_w, overlap)):
            tiles.append(Tile(f"r{row:03d}_c{col:03d}", x1, y1, x1 + actual_w, y1 + actual_h))
    return tiles


def translate_box(box: Box, dx: float, dy: float) -> Box:
    x1, y1, x2, y2 = box
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def clip_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )


def padded_crop_box(box: Box, width: int, height: int, padding: float = 1.0) -> tuple[int, int, int, int]:
    """A padded integer crop box for handing a candidate region to a verifier model."""
    if padding < 0:
        raise ValueError("Padding must be non-negative")
    x1, y1, x2, y2 = box
    dx = (x2 - x1) * padding
    dy = (y2 - y1) * padding
    cx1, cy1, cx2, cy2 = clip_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), width, height)
    return (round(cx1), round(cy1), round(cx2), round(cy2))
