from __future__ import annotations

from collections import defaultdict

from .types import Box, Detection


def iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def non_max_suppression(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]:
    kept: list[Detection] = []
    by_label: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        by_label[detection.label.lower()].append(detection)
    for label_detections in by_label.values():
        for candidate in sorted(label_detections, key=lambda item: item.score, reverse=True):
            if all(iou(candidate.box, accepted.box) < iou_threshold for accepted in kept if accepted.label.lower() == candidate.label.lower()):
                kept.append(candidate)
    return sorted(kept, key=lambda item: item.score, reverse=True)


def weighted_box_fusion(
    detections: list[Detection],
    model_weights: dict[str, float] | None = None,
    iou_threshold: float = 0.5,
) -> list[Detection]:
    """Fuse matching detections while retaining provenance.

    This is deliberately conservative: only same-label boxes with sufficient IoU
    join a cluster. A one-model cluster remains a valid output and is marked with
    agreement_count=1.
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

        weights = [max(1e-9, det.score * model_weights.get(det.model, 1.0)) for det in cluster]
        total = sum(weights)
        box = Box(
            sum(det.box.x1 * weight for det, weight in zip(cluster, weights)) / total,
            sum(det.box.y1 * weight for det, weight in zip(cluster, weights)) / total,
            sum(det.box.x2 * weight for det, weight in zip(cluster, weights)) / total,
            sum(det.box.y2 * weight for det, weight in zip(cluster, weights)) / total,
        )
        sources = sorted({det.model for det in cluster})
        fused.append(
            Detection(
                box=box,
                score=max(det.score for det in cluster),
                label=seed.label,
                model="ensemble" if len(sources) > 1 else sources[0],
                metadata={
                    "agreement_count": len(sources),
                    "source_models": sources,
                    "source_scores": [det.score for det in cluster],
                },
            )
        )
    return sorted(fused, key=lambda item: item.score, reverse=True)

