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
        self._optimize_for_inference()

    def _optimize_for_inference(self) -> None:
        """Switch RF-DETR to its fast float16 path when CUDA is available.

        The library logs this hint itself: the default float32 eager path is
        several times slower on GPU tensor cores. Traced compilation is
        avoided so any batch size stays valid.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return
            self.model.inference(compile=False, dtype="float16")
        except Exception as exc:  # noqa: BLE001 - optimization is optional, never break inference for it
            print(f"[rf-detr] float16 inference optimization unavailable: {exc}")

    def predict_image(self, image: Path, output_dir: Path) -> ModelResult:
        return self.predict_images([image], output_dir)[0]

    def predict_images(self, images: list[Path], output_dir: Path) -> list[ModelResult]:
        settings = self.config.raw["rfdetr"]
        start = perf_counter()
        predictions = self.model.predict(
            [str(image) for image in images], threshold=float(settings["confidence"])
        )
        elapsed = perf_counter() - start
        results: list[ModelResult] = []
        for image, prediction in zip(images, predictions):
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
            results.append(
                ModelResult(
                    model=f"RF-DETR/{settings['checkpoint']}",
                    source=str(image),
                    width=width,
                    height=height,
                    elapsed_seconds=elapsed / max(1, len(images)),
                    detections=detections,
                    metadata={"task": "closed-set object detection", "variant": settings["variant"]},
                )
            )
        return results

