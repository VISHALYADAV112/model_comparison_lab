from __future__ import annotations

from .types import Box, Detection, Tile


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


def map_detection_to_image(detection: Detection, tile: Tile, image_width: int, image_height: int) -> Detection:
    return Detection(
        box=detection.box.translate(tile.x1, tile.y1).clip(image_width, image_height),
        score=detection.score,
        label=detection.label,
        model=detection.model,
        tile_id=tile.tile_id,
        metadata=dict(detection.metadata),
    )


def padded_crop_box(box: Box, image_width: int, image_height: int, padding: float = 1.0) -> Box:
    if padding < 0:
        raise ValueError("Padding must be non-negative")
    dx = box.width * padding
    dy = box.height * padding
    return Box(box.x1 - dx, box.y1 - dy, box.x2 + dx, box.y2 + dy).clip(image_width, image_height)

