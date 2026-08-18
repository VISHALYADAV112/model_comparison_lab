from model_lab.contracts import Detection, ModelResult


def test_result_summary_reports_smallest_object_side() -> None:
    result = ModelResult(
        model="test",
        source="image.jpg",
        width=100,
        height=100,
        elapsed_seconds=1.23456,
        detections=[
            Detection(box=(0, 0, 10, 20), score=0.9, label="a"),
            Detection(box=(0, 0, 4, 8), score=0.8, label="b"),
        ],
    )
    summary = result.summary()
    assert summary["count"] == 2
    assert summary["smallest_box_side_px"] == 4
    assert summary["median_box_side_px"] == 7
    assert summary["elapsed_seconds"] == 1.2346

