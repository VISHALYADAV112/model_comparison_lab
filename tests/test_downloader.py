from model_lab.config import load_config
from model_lab.downloader import model_status


def test_status_lists_both_sam_backends() -> None:
    names = {item["component"] for item in model_status(load_config())}
    assert names == {
        "YOLO",
        "RF-DETR",
        "SAM 3 official image",
        "SAM 3.1 official video",
        "SAM 3 public Q8_0",
        "SAM 3 Q8 bridge",
    }

