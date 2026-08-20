from model_lab.contracts import Detection
from model_lab.fusion import iou, non_max_suppression, weighted_box_fusion


def _det(box, score, label="person", source_model="yolo") -> Detection:
    return Detection(box=box, score=score, label=label, metadata={"source_model": source_model})


def test_iou_matches_expected_overlap() -> None:
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 5.0, 15.0, 15.0)
    assert iou(a, b) == 25.0 / 175.0
    assert iou(a, a) == 1.0
    assert iou((0.0, 0.0, 1.0, 1.0), (5.0, 5.0, 6.0, 6.0)) == 0.0


def test_non_max_suppression_keeps_the_higher_scoring_overlapping_box() -> None:
    detections = [
        _det((0, 0, 10, 10), 0.9),
        _det((1, 1, 11, 11), 0.8),
        _det((50, 50, 60, 60), 0.7),
    ]

    kept = non_max_suppression(detections, iou_threshold=0.3)

    assert len(kept) == 2
    assert kept[0].score == 0.9
    assert kept[1].box == (50, 50, 60, 60)


def test_non_max_suppression_does_not_suppress_across_labels() -> None:
    detections = [
        _det((0, 0, 10, 10), 0.9, label="person"),
        _det((0, 0, 10, 10), 0.8, label="chair"),
    ]

    kept = non_max_suppression(detections, iou_threshold=0.3)

    assert {item.label for item in kept} == {"person", "chair"}


def test_weighted_box_fusion_merges_agreeing_models_and_keeps_provenance() -> None:
    detections = [
        _det((0, 0, 10, 10), 0.9, source_model="yolo"),
        _det((2, 2, 12, 12), 0.7, source_model="rfdetr"),
    ]

    fused = weighted_box_fusion(detections, iou_threshold=0.3)

    assert len(fused) == 1
    result = fused[0]
    assert result.metadata["agreement_count"] == 2
    assert result.metadata["source_models"] == ["rfdetr", "yolo"]
    assert result.metadata["source_model"] == "ensemble"
    # fused box is the score-weighted average of the two source boxes
    assert 0.0 < result.box[0] < 2.0


def test_weighted_box_fusion_keeps_single_model_detections_unmerged() -> None:
    detections = [
        _det((0, 0, 10, 10), 0.9, source_model="yolo"),
        _det((100, 100, 110, 110), 0.6, source_model="rfdetr"),
    ]

    fused = weighted_box_fusion(detections, iou_threshold=0.3)

    assert len(fused) == 2
    assert all(item.metadata["agreement_count"] == 1 for item in fused)


def test_weighted_box_fusion_respects_model_weights() -> None:
    detections = [
        _det((0, 0, 10, 10), 0.5, source_model="low_weight"),
        _det((10, 10, 20, 20), 0.5, source_model="high_weight"),
    ]

    fused = weighted_box_fusion(
        detections,
        model_weights={"low_weight": 0.01, "high_weight": 100.0},
        iou_threshold=0.0,
    )

    # both boxes are far apart (iou below threshold check is moot here since
    # threshold 0.0 still requires iou >= 0.0, but non-overlapping boxes have
    # iou 0.0, so they still merge into one cluster at threshold 0.0)
    assert len(fused) == 1
    heavy = fused[0]
    # the box should be pulled strongly toward the heavily weighted detection
    assert heavy.box[0] > 8.0
