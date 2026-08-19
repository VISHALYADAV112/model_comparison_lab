from model_lab.compare import detector_classes_for_target, filter_detector_result
from model_lab.contracts import Detection, ModelResult


def _result() -> ModelResult:
    return ModelResult(
        model="detector",
        source="image.jpg",
        width=100,
        height=100,
        elapsed_seconds=0.1,
        detections=[
            Detection((0, 0, 10, 20), 0.9, "person"),
            Detection((10, 0, 30, 20), 0.8, "chair"),
            Detection((30, 0, 50, 20), 0.7, "person"),
        ],
    )


def test_person_target_keeps_only_person_detections() -> None:
    result = filter_detector_result(_result(), "person")

    assert [item.label for item in result.detections] == ["person", "person"]
    assert result.metadata["target_filter"] == {
        "query": "person",
        "matched_closed_set_classes": ["person"],
        "detections_before_filter": 3,
        "detections_after_filter": 2,
    }


def test_open_concepts_map_to_relevant_coco_classes() -> None:
    assert detector_classes_for_target("all distant people") == {"person"}
    assert detector_classes_for_target("aircraft") == {"airplane"}
    assert detector_classes_for_target("animal") == {
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    }
    assert "truck" in detector_classes_for_target("vehicles")


def test_generic_object_target_preserves_all_detector_classes() -> None:
    result = _result()

    assert filter_detector_result(result, "object") is result
    assert len(result.detections) == 3
    assert "target_filter" not in result.metadata
