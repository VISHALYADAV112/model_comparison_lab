from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from .fusion import iou
from .types import Box, Detection, canonical_label


SIZE_BINS: tuple[tuple[str, float, float], ...] = (
    ("0-2px", 0.0, 2.0),
    ("2-8px", 2.0, 8.0),
    ("8-16px", 8.0, 16.0),
    ("16-32px", 16.0, 32.0),
    ("32-64px", 32.0, 64.0),
    ("64px+", 64.0, float("inf")),
)


@dataclass
class MatchSummary:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    matched_iou_sum: float = 0.0

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    @property
    def mean_matched_iou(self) -> float:
        return self.matched_iou_sum / self.true_positive if self.true_positive else 0.0

    def to_dict(self) -> dict[str, float | int]:
        result = asdict(self)
        result.update(
            precision=self.precision,
            recall=self.recall,
            f1=self.f1,
            mean_matched_iou=self.mean_matched_iou,
        )
        return result


def _bin_name(height: float) -> str:
    for name, lower, upper in SIZE_BINS:
        if lower <= height < upper:
            return name
    return "unknown"


def _detection_from_item(item: dict[str, Any], model: str) -> Detection:
    if "box_xyxy" in item:
        x1, y1, x2, y2 = (float(value) for value in item["box_xyxy"])
    elif "bbox" in item:
        x1, y1, width, height = (float(value) for value in item["bbox"])
        x2, y2 = x1 + width, y1 + height
    else:
        raise ValueError("Every annotation needs box_xyxy or COCO bbox coordinates")
    metadata = dict(item.get("metadata", {}))
    if item.get("id") is not None and "annotation_id" not in metadata:
        metadata["annotation_id"] = item["id"]
    return Detection(
        box=Box(x1, y1, x2, y2),
        score=float(item.get("score", 1.0)),
        label=canonical_label(str(item.get("label", "person"))),
        model=str(item.get("model", model)),
        tile_id=item.get("tile_id"),
        metadata=metadata,
    )


def load_prediction_report(path: str | Path) -> list[Detection]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload.get("detections", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError("Prediction JSON must be a list or contain a detections list")
    return detections_from_items(items)


def detections_from_items(items: list[dict[str, Any]], model: str = "prediction") -> list[Detection]:
    return [_detection_from_item(item, model) for item in items]


def load_ground_truth(
    path: str | Path,
    *,
    image_name: str | None = None,
    image_id: int | None = None,
    person_only: bool = True,
) -> list[Detection]:
    """Load either compact box JSON or a COCO instances file.

    Compact format: {"annotations": [{"box_xyxy": [...], "label": "person"}]}.
    For COCO, select one image with image_name or image_id; if the file contains
    exactly one image, selection is automatic.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [_detection_from_item(item, "ground_truth") for item in payload]
    if not isinstance(payload, dict) or "annotations" not in payload:
        raise ValueError("Ground-truth JSON must be a list or contain annotations")

    annotations = payload["annotations"]
    if not payload.get("images"):
        return [_detection_from_item(item, "ground_truth") for item in annotations]

    images = payload["images"]
    selected_id = image_id
    if selected_id is None and image_name is not None:
        matches = [item["id"] for item in images if Path(str(item.get("file_name", ""))).name == Path(image_name).name]
        if len(matches) != 1:
            raise ValueError(f"Expected one COCO image named {image_name!r}, found {len(matches)}")
        selected_id = int(matches[0])
    if selected_id is None and len(images) == 1:
        selected_id = int(images[0]["id"])
    if selected_id is None:
        raise ValueError("COCO ground truth has multiple images; pass --image-name or --image-id")

    category_labels = {
        int(item["id"]): canonical_label(str(item.get("name", item["id"])))
        for item in payload.get("categories", [])
    }
    results: list[Detection] = []
    for item in annotations:
        if int(item.get("image_id", -1)) != selected_id or bool(item.get("iscrowd", False)):
            continue
        label = category_labels.get(int(item.get("category_id", -1)), "person")
        if person_only and label != "person":
            continue
        annotated = dict(item)
        annotated["label"] = label
        results.append(_detection_from_item(annotated, "ground_truth"))
    return results


def match_detections(
    predictions: list[Detection],
    ground_truth: list[Detection],
    iou_threshold: float = 0.5,
) -> dict[str, dict[str, float | int]]:
    overall = MatchSummary()
    bins: dict[str, MatchSummary] = defaultdict(MatchSummary)
    unmatched_gt = set(range(len(ground_truth)))

    for prediction in sorted(predictions, key=lambda item: item.score, reverse=True):
        best_index: int | None = None
        best_iou = 0.0
        for index in unmatched_gt:
            truth = ground_truth[index]
            if truth.label.lower() != prediction.label.lower():
                continue
            overlap = iou(prediction.box, truth.box)
            if overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_index is not None and best_iou >= iou_threshold:
            truth = ground_truth[best_index]
            unmatched_gt.remove(best_index)
            overall.true_positive += 1
            overall.matched_iou_sum += best_iou
            bucket = bins[_bin_name(truth.box.height)]
            bucket.true_positive += 1
            bucket.matched_iou_sum += best_iou
        else:
            overall.false_positive += 1

    overall.false_negative = len(unmatched_gt)
    for index in unmatched_gt:
        bins[_bin_name(ground_truth[index].box.height)].false_negative += 1
    for summary in bins.values():
        # False positives have no true size. Keep them in overall metrics instead
        # of assigning a predicted size to a ground-truth size bucket.
        summary.false_positive = 0

    return {
        "overall": overall.to_dict(),
        "by_ground_truth_height": {name: bins[name].to_dict() for name, _, _ in SIZE_BINS},
    }
