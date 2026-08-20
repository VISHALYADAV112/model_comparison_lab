from __future__ import annotations

from typing import Any

from ..types import Box, Detection, canonical_label
from .base import ModelAdapter


class RFDETRAdapter(ModelAdapter):
    def __init__(self, name: str, config: dict[str, Any], device: str = "auto") -> None:
        super().__init__(name, config, device)
        try:
            from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
            from rfdetr.assets.coco_classes import COCO_CLASSES
        except ImportError as exc:
            raise RuntimeError("RF-DETR is not installed. Install the project with the [rfdetr] extra.") from exc
        variants = {
            "nano": RFDETRNano,
            "small": RFDETRSmall,
            "medium": RFDETRMedium,
            "large": RFDETRLarge,
        }
        variant = str(config.get("variant", "large")).lower()
        if variant not in variants:
            raise ValueError(f"Unsupported Apache RF-DETR variant: {variant}")
        self.model = variants[variant]()
        self.coco_classes = COCO_CLASSES

    def detect(self, image: Any, prompts: list[str]) -> list[Detection]:
        threshold = float(self.config.get("threshold", 0.2))
        result = self.model.predict(image, threshold=threshold)
        allowed = {prompt.lower() for prompt in prompts}
        detections: list[Detection] = []
        class_names = result.data.get("class_name") if hasattr(result, "data") else None
        for index, (coords, score, class_id) in enumerate(zip(result.xyxy, result.confidence, result.class_id)):
            label = str(class_names[index]) if class_names is not None else str(self.coco_classes[int(class_id)])
            if allowed and label.lower() not in allowed and not (label.lower() == "person" and "human" in allowed):
                continue
            metadata: dict[str, object] = {"class_id": int(class_id)}
            masks = getattr(result, "mask", None)
            if masks is not None and len(masks) > index:
                metadata["mask_area_px"] = int(masks[index].sum())
            detections.append(
                Detection(
                    box=Box(*[float(value) for value in coords]),
                    score=float(score),
                    label=canonical_label(label),
                    model=self.name,
                    metadata=metadata,
                )
            )
        return detections
