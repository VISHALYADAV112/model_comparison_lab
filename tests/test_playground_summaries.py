from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from model_lab.playground.service import (
    BoundedVideoController,
    PlaygroundService,
    _records,
    comparison_summary,
    quick_video_frame_limit,
    video_summary,
)


def test_bounded_dashboard_outputs_include_live_preview(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "frames.jsonl").write_text("{}\n")
    payload = {
        "status": "running",
        "live_preview": str(output / "live_preview.jpg"),
        "latest_segment": None,
    }

    values = BoundedVideoController._outputs(output, payload)

    assert len(values) == 6
    assert values[1] == payload["live_preview"]
    assert values[2] is None
    assert values[-1] is payload


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


@pytest.mark.parametrize(
    ("selection", "expected"),
    [("all", 0), ("first_60", 60), ("first_300", 300), (42, 42)],
)
def test_quick_video_frame_limit(selection: str | int, expected: int) -> None:
    assert quick_video_frame_limit(selection) == expected


def test_video_summary_explains_a_short_frame_limited_result() -> None:
    payload = {
        "fps": 30,
        "source_frame_count": 780,
        "requested_max_frames": 60,
        "frames": [{"detections": []} for _ in range(60)],
    }
    summary = video_summary(payload)
    assert "2.0 seconds" in summary
    assert "shorter than the source" in summary
    assert "Whole uploaded video" in summary


@pytest.mark.parametrize(
    ("engine", "expected_backend", "expected_offload", "expected_batch"),
    [
        ("official_balanced", "official", True, 4),
        ("official_low_vram", "official", True, 1),
        ("official_fast", "official", False, 16),
        ("q8", "q8", False, 0),
    ],
)
def test_quick_video_maps_memory_profiles(
    engine: str,
    expected_backend: str,
    expected_offload: bool,
    expected_batch: int,
) -> None:
    service = PlaygroundService(SimpleNamespace())
    service.run_video = Mock(
        return_value=(
            "annotated.mp4",
            "manifest.json",
            "results.zip",
            {"frames": []},
        )
    )

    service.quick_video("video.mp4", "vehicle", engine, 60, 0.5)

    arguments = service.run_video.call_args.args
    assert arguments[0] == expected_backend
    assert arguments[10] is expected_offload
    assert arguments[13] == expected_batch


def test_compare_reports_json_name_matches_mode(tmp_path, monkeypatch) -> None:
    config = SimpleNamespace(outputs_dir=tmp_path)
    service = PlaygroundService(config)

    monkeypatch.setattr("model_lab.playground.service.compare_image", lambda *args, **kwargs: {})
    _, report, _ = service.compare("img.jpg", ["yolo"], "object", "official")
    assert report.endswith("comparison.json")

    monkeypatch.setattr(
        "model_lab.playground.service.compare_image_cascade", lambda *args, **kwargs: {}
    )
    _, report, _ = service.compare("img.jpg", ["yolo"], "object", "official", cascade=True)
    assert report.endswith("cascade.json")
