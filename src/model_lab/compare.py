from __future__ import annotations

import gc
import json
import re
from collections.abc import Iterable
from pathlib import Path

from .adapters import MetaSam3Adapter, RFDetrAdapter, Sam3CppAdapter, YoloAdapter
from .config import LabConfig
from .contracts import ModelResult
from .rendering import render_result

_TARGET_CLASS_ALIASES = {
    "person": {"person"},
    "people": {"person"},
    "human": {"person"},
    "pedestrian": {"person"},
    "vehicle": {
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
    },
    "aircraft": {"airplane"},
    "plane": {"airplane"},
    "ship": {"boat"},
    "animal": {
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
    },
    "bike": {"bicycle"},
    "motorbike": {"motorcycle"},
}
_SHOW_ALL_TARGETS = {"all", "any", "anything", "object", "objects", "everything"}


def _normalized_label(value: str) -> str:
    return " ".join(value.lower().replace("_", " ").replace("-", " ").split())


def detector_classes_for_target(target: str) -> set[str] | None:
    """Map an open prompt to comparable closed-set COCO detector labels."""
    normalized = _normalized_label(target)
    if normalized in _SHOW_ALL_TARGETS:
        return None
    classes: set[str] = set()
    for alias, labels in _TARGET_CLASS_ALIASES.items():
        plural = f"{alias}s"
        if re.search(rf"\b(?:{re.escape(alias)}|{re.escape(plural)})\b", normalized):
            classes.update(labels)
    if classes:
        return classes
    if normalized.endswith("s") and not normalized.endswith("ss"):
        normalized = normalized[:-1]
    return {normalized}


def filter_detector_result(result: ModelResult, target: str) -> ModelResult:
    allowed = detector_classes_for_target(target)
    if allowed is None:
        return result
    before = len(result.detections)
    result.detections = [item for item in result.detections if _normalized_label(item.label) in allowed]
    result.metadata["target_filter"] = {
        "query": target,
        "matched_closed_set_classes": sorted(allowed),
        "detections_before_filter": before,
        "detections_after_filter": len(result.detections),
    }
    return result


def compare_image(
    config: LabConfig,
    image: Path,
    output_dir: Path,
    *,
    models: Iterable[str] = ("yolo", "rfdetr", "sam3"),
    sam_text: str = "object",
    sam_backend: str | None = None,
    detector_target: str | None = None,
) -> dict:
    image = image.expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = list(dict.fromkeys(models))
    results: list[ModelResult] = []
    errors: dict[str, str] = {}
    for name in requested:
        try:
            if name == "yolo":
                adapter = YoloAdapter(config)
                result = adapter.predict_image(image, output_dir / name)
            elif name == "rfdetr":
                adapter = RFDetrAdapter(config)
                result = adapter.predict_image(image, output_dir / name)
            elif name == "sam3":
                selected_backend = sam_backend or config.raw["sam3"].get("backend", "official")
                if selected_backend == "official":
                    adapter = MetaSam3Adapter(config)
                elif selected_backend == "q8":
                    adapter = Sam3CppAdapter(config)
                else:
                    raise ValueError(f"Unknown SAM backend: {selected_backend}")
                result = adapter.predict_text_image(image, output_dir / name, sam_text)
            else:
                raise ValueError(f"Unknown model: {name}")
            if detector_target and name in {"yolo", "rfdetr"}:
                result = filter_detector_result(result, detector_target)
            render_result(image, result, output_dir / f"{name}_annotated.jpg")
            results.append(result)
            del adapter
            gc.collect()
        except Exception as exc:  # noqa: BLE001 - preserve other model results in a benchmark run
            errors[name] = f"{type(exc).__name__}: {exc}"
    payload = {
        "schema_version": 1,
        "source": str(image),
        "sam_text_prompt": sam_text,
        "detector_target_filter": detector_target,
        "sam_backend": sam_backend or config.raw["sam3"].get("backend", "official"),
        "warning": (
            "This is an inference report, not an accuracy ranking. Add ground-truth annotations and run the "
            "evaluation command before drawing quality conclusions."
        ),
        "results": [result.to_dict() for result in results],
        "errors": errors,
    }
    (output_dir / "comparison.json").write_text(json.dumps(payload, indent=2))
    return payload
