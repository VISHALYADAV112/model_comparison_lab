from __future__ import annotations

from typing import Any

from ..types import Box, Detection, canonical_label
from .base import ModelAdapter, choose_torch_device


class GroundingDINOAdapter(ModelAdapter):
    def __init__(self, name: str, config: dict[str, Any], device: str = "auto") -> None:
        super().__init__(name, config, device)
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Grounding DINO dependencies are missing. Install the [grounding] extra.") from exc
        self.torch = torch
        self.device = choose_torch_device(torch, device)
        model_id = str(config.get("model_id", "IDEA-Research/grounding-dino-tiny"))
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def detect(self, image: Any, prompts: list[str]) -> list[Detection]:
        labels = prompts or ["person"]
        inputs = self.processor(images=image, text=[labels], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=float(self.config.get("threshold", 0.2)),
            text_threshold=float(self.config.get("text_threshold", 0.2)),
            target_sizes=[(image.height, image.width)],
            text_labels=[labels],
        )[0]
        text_labels = result.get("text_labels", result.get("labels", []))
        return [
            Detection(
                box=Box(*[float(value) for value in box.detach().cpu().tolist()]),
                score=float(score.detach().cpu().item()),
                label=canonical_label(str(label)),
                model=self.name,
                metadata={"text_label": str(label)},
            )
            for box, score, label in zip(result["boxes"], result["scores"], text_labels)
        ]
