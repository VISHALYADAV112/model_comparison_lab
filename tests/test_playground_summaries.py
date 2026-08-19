from model_lab.playground.service import _records, comparison_summary, video_summary


def test_empty_optional_video_records_accept_none() -> None:
    assert _records(None) == []
    assert _records("") == []


def test_comparison_summary_explains_results_and_errors() -> None:
    summary = comparison_summary(
        {
            "detector_target_filter": "person",
            "results": [
                {
                    "model": "YOLO/test.pt",
                    "summary": {
                        "model": "YOLO/test.pt",
                        "count": 4,
                        "elapsed_seconds": 0.25,
                        "smallest_box_side_px": 7.5,
                    },
                }
            ],
            "errors": {"sam3": "RuntimeError: unavailable"},
        }
    )
    assert "YOLO/test.pt" in summary
    assert "Target:" in summary
    assert "person" in summary
    assert "4" in summary
    assert "7.5 px" in summary
    assert "Some models failed" in summary
    assert "SAM 3" in summary
    assert "RuntimeError: unavailable" in summary


def test_video_summary_counts_frames_objects_and_masks() -> None:
    summary = video_summary(
        {
            "frames": [
                {"detections": [{"instance_id": 1}, {"instance_id": 2}]},
                {"detections": [{"instance_id": 1}]},
            ],
            "elapsed_seconds": 2.5,
        }
    )
    assert "2 frames" in summary
    assert "2 unique objects" in summary
    assert "3 frame-level masks" in summary
    assert "2.50 seconds" in summary
