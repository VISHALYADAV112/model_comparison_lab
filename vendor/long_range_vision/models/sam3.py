from __future__ import annotations

from typing import Any

from ..fusion import non_max_suppression
from ..types import Box, Detection, canonical_label
from .base import ModelAdapter, choose_torch_device


class SAM3Adapter(ModelAdapter):
    def __init__(self, name: str, config: dict[str, Any], device: str = "auto") -> None:
        super().__init__(name, config, device)
        try:
            import torch
            from transformers import Sam3Model, Sam3Processor
        except ImportError as exc:
            raise RuntimeError("SAM 3 dependencies are missing. Install the [sam3] extra.") from exc
        self.torch = torch
        self.device = choose_torch_device(torch, device)
        model_id = str(config.get("model_id", "facebook/sam3"))
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def detect(self, image: Any, prompts: list[str]) -> list[Detection]:
        detections: list[Detection] = []
        for prompt in prompts or ["person"]:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            result = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=float(self.config.get("threshold", 0.2)),
                mask_threshold=float(self.config.get("mask_threshold", 0.5)),
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]
            for box, score, mask in zip(result["boxes"], result["scores"], result["masks"]):
                detections.append(
                    Detection(
                        box=Box(*[float(value) for value in box.detach().cpu().tolist()]),
                        score=float(score.detach().cpu().item()),
                        label=canonical_label(prompt),
                        model=self.name,
                        metadata={
                            "mask_area_px": int(mask.detach().cpu().sum().item()),
                            "text_prompt": prompt,
                        },
                    )
                )
        return non_max_suppression(detections, iou_threshold=0.6)
