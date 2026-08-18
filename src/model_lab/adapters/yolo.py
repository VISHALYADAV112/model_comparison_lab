from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from ..config import LabConfig
from ..contracts import Detection, ModelResult
from .base import DetectorAdapter


class YoloAdapter(DetectorAdapter):
    def __init__(self, config: LabConfig) -> None:
        self.config = config
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("YOLO is not installed. Run: pip install -e '.[yolo]'") from exc
        if not config.yolo_model.exists():
            raise FileNotFoundError(
                f"Missing {config.yolo_model}. Run: model-lab models download --model yolo"
            )
        self.model = YOLO(str(config.yolo_model))

    def predict_image(self, image: Path, output_dir: Path) -> ModelResult:
        settings = self.config.raw["yolo"]
        kwargs: dict[str, Any] = {
            "source": str(image),
            "conf": float(settings["confidence"]),
            "imgsz": int(settings["image_size"]),
            "verbose": False,
        }
        if settings.get("device", "auto") != "auto":
            kwargs["device"] = settings["device"]
        start = perf_counter()
        prediction = self.model.predict(**kwargs)[0]
        elapsed = perf_counter() - start
        width, height = Image.open(image).size
        detections: list[Detection] = []
        if prediction.boxes is not None:
            boxes = prediction.boxes.xyxy.detach().cpu().tolist()
            scores = prediction.boxes.conf.detach().cpu().tolist()
            classes = prediction.boxes.cls.detach().cpu().tolist()
            names = prediction.names
            for box, score, class_value in zip(boxes, scores, classes):
                class_id = int(class_value)
                detections.append(
                    Detection(
                        box=tuple(float(value) for value in box),
                        score=float(score),
                        label=str(names[class_id]),
                        class_id=class_id,
                    )
                )
        return ModelResult(
            model=f"YOLO/{settings['checkpoint']}",
            source=str(image),
            width=width,
            height=height,
            elapsed_seconds=elapsed,
            detections=detections,
            metadata={"task": "closed-set object detection", "image_size": settings["image_size"]},
        )

