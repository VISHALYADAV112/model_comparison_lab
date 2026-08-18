from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

from PIL import Image

from ..config import LabConfig
from ..contracts import Detection, ModelResult
from .base import DetectorAdapter


class RFDetrAdapter(DetectorAdapter):
    def __init__(self, config: LabConfig) -> None:
        self.config = config
        os.environ["RF_HOME"] = str(config.models_dir / "rfdetr")
        try:
            from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
        except ImportError as exc:
            raise RuntimeError("RF-DETR is not installed. Run: pip install -e '.[rfdetr]'") from exc
        variants = {
            "nano": RFDETRNano,
            "small": RFDETRSmall,
            "medium": RFDETRMedium,
            "large": RFDETRLarge,
        }
        name = str(config.raw["rfdetr"]["variant"]).lower()
        if name not in variants:
            raise ValueError(f"Unknown RF-DETR variant: {name}")
        if not config.rfdetr_model.exists():
            raise FileNotFoundError(
                f"Missing {config.rfdetr_model}. Run: model-lab models download --model rfdetr"
            )
        self.model = variants[name](pretrain_weights=str(config.rfdetr_model))

    def predict_image(self, image: Path, output_dir: Path) -> ModelResult:
        settings = self.config.raw["rfdetr"]
        start = perf_counter()
        prediction = self.model.predict(str(image), threshold=float(settings["confidence"]))
        elapsed = perf_counter() - start
        width, height = Image.open(image).size
        names = prediction.data.get("class_name") if hasattr(prediction, "data") else None
        detections: list[Detection] = []
        for index, (box, score, class_value) in enumerate(
            zip(prediction.xyxy, prediction.confidence, prediction.class_id)
        ):
            class_id = int(class_value)
            label = str(names[index]) if names is not None else str(class_id)
            detections.append(
                Detection(
                    box=tuple(float(value) for value in box),
                    score=float(score),
                    label=label,
                    class_id=class_id,
                )
            )
        return ModelResult(
            model=f"RF-DETR/{settings['checkpoint']}",
            source=str(image),
            width=width,
            height=height,
            elapsed_seconds=elapsed,
            detections=detections,
            metadata={"task": "closed-set object detection", "variant": settings["variant"]},
        )

