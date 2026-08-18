import pytest

from model_lab.adapters.meta_sam3 import parse_prompt_spec
from model_lab.adapters.sam3_cpp import parse_boxes, parse_points


def test_point_and_box_text_formats() -> None:
    assert parse_points("10,20; 30.5,40") == [(10.0, 20.0), (30.5, 40.0)]
    assert parse_boxes("1,2,3,4;5,6,7,8") == [
        (1.0, 2.0, 3.0, 4.0),
        (5.0, 6.0, 7.0, 8.0),
    ]


def test_video_prompt_spec_supports_objects_and_refinement() -> None:
    prompt = parse_prompt_spec("frame:25;id:3;p:10,20;p:11,21;n:30,40;b:1,2,50,60")
    assert prompt["frame"] == 25
    assert prompt["id"] == 3
    assert prompt["positive"] == [(10.0, 20.0), (11.0, 21.0)]
    assert prompt["negative"] == [(30.0, 40.0)]
    assert prompt["box"] == (1.0, 2.0, 50.0, 60.0)


def test_bad_prompt_fails_early() -> None:
    with pytest.raises(ValueError):
        parse_prompt_spec("point-without-key")
    with pytest.raises(ValueError):
        parse_points("1,2,3")

