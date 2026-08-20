import pytest

from model_lab.tiling import clip_box, generate_tiles, padded_crop_box, translate_box


def test_generate_tiles_returns_one_tile_when_image_fits() -> None:
    tiles = generate_tiles(640, 480, tile_size=1008)

    assert len(tiles) == 1
    tile = tiles[0]
    assert (tile.x1, tile.y1, tile.x2, tile.y2) == (0, 0, 640, 480)


def test_generate_tiles_covers_a_larger_image_with_overlap() -> None:
    tiles = generate_tiles(2000, 1000, tile_size=1008, overlap=0.2)

    assert len(tiles) > 1
    xs = {tile.x1 for tile in tiles} | {tile.x2 for tile in tiles}
    assert min(xs) == 0
    assert max(xs) == 2000
    for tile in tiles:
        assert tile.x2 - tile.x1 == 1008
        assert tile.y2 - tile.y1 == 1000
        assert 0 <= tile.x1 <= tile.x2 <= 2000
        assert 0 <= tile.y1 <= tile.y2 <= 1000


def test_generate_tiles_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        generate_tiles(0, 100, tile_size=100)
    with pytest.raises(ValueError):
        generate_tiles(100, 100, tile_size=0)
    with pytest.raises(ValueError):
        generate_tiles(100, 100, tile_size=50, overlap=1.0)


def test_translate_and_clip_box_map_tile_local_to_image_coordinates() -> None:
    box = (10.0, 20.0, 30.0, 40.0)

    translated = translate_box(box, dx=500, dy=200)
    assert translated == (510.0, 220.0, 530.0, 240.0)

    clipped = clip_box((-10.0, -5.0, 650.0, 300.0), width=640, height=250)
    assert clipped == (0.0, 0.0, 640.0, 250.0)


def test_padded_crop_box_grows_and_clips_to_image_bounds() -> None:
    box = (100.0, 100.0, 120.0, 140.0)

    roi = padded_crop_box(box, width=640, height=480, padding=1.0)

    # padding=1.0 doubles width/height of the pad on each side (dx = width * 1.0)
    assert roi == (80, 60, 140, 180)


def test_padded_crop_box_clips_near_image_edge() -> None:
    box = (0.0, 0.0, 20.0, 20.0)

    roi = padded_crop_box(box, width=640, height=480, padding=2.0)

    assert roi[0] == 0
    assert roi[1] == 0
    assert roi[2] <= 640
    assert roi[3] <= 480


def test_padded_crop_box_rejects_negative_padding() -> None:
    with pytest.raises(ValueError):
        padded_crop_box((0.0, 0.0, 10.0, 10.0), width=100, height=100, padding=-0.1)
