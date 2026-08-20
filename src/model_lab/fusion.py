from __future__ import annotations

from collections import defaultdict

from .contracts import Detection
from .tiling import Box


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def non_max_suppression(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]:
    kept: list[Detection] = []
    by_label: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        by_label[detection.label.lower()].append(detection)
    for label_detections in by_label.values():
        for candidate in sorted(label_detections, key=lambda item: item.score, reverse=True):
            if all(
                iou(candidate.box, accepted.box) < iou_threshold
                for accepted in kept
                if accepted.label.lower() == candidate.label.lower()
            ):
                kept.append(candidate)
    return sorted(kept, key=lambda item: item.score, reverse=True)


def _source_model(detection: Detection) -> str:
    return str(detection.metadata.get("source_model", "unknown"))


def weighted_box_fusion(
    detections: list[Detection],
    model_weights: dict[str, float] | None = None,
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Fuse matching same-label boxes from different source models while retaining provenance.

    Ported from long_range_vision.fusion. Deliberately conservative: only
    same-label boxes with sufficient IoU join a cluster. A one-model cluster
    remains a valid output, marked with agreement_count=1. Provenance is read
    from detection.metadata["source_model"] since model_lab's Detection has no
    dedicated model field.
    """
    model_weights = model_weights or {}
    remaining = sorted(detections, key=lambda item: item.score, reverse=True)
    fused: list[Detection] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        rest: list[Detection] = []
        for candidate in remaining:
            if candidate.label.lower() == seed.label.lower() and iou(seed.box, candidate.box) >= iou_threshold:
                cluster.append(candidate)
            else:
                rest.append(candidate)
        remaining = rest

        weights = [max(1e-9, det.score * model_weights.get(_source_model(det), 1.0)) for det in cluster]
        total = sum(weights)
        box = (
            sum(det.box[0] * weight for det, weight in zip(cluster, weights)) / total,
            sum(det.box[1] * weight for det, weight in zip(cluster, weights)) / total,
            sum(det.box[2] * weight for det, weight in zip(cluster, weights)) / total,
            sum(det.box[3] * weight for det, weight in zip(cluster, weights)) / total,
        )
        sources = sorted({_source_model(det) for det in cluster})
        fused.append(
            Detection(
                box=box,
                score=max(det.score for det in cluster),
                label=seed.label,
                metadata={
                    "source_model": "ensemble" if len(sources) > 1 else sources[0],
                    "agreement_count": len(sources),
                    "source_models": sources,
                    "source_scores": [det.score for det in cluster],
                },
            )
        )
    return sorted(fused, key=lambda item: item.score, reverse=True)
