from __future__ import annotations

from typing import Any

from ..types import Box, Detection
from .base import ModelAdapter


class MockAdapter(ModelAdapter):
    def detect(self, image: Any, prompts: list[str]) -> list[Detection]:
        width, height = image.size
        return [
            Detection(
                box=Box(width * 0.45, height * 0.40, width * 0.55, height * 0.75),
                score=0.9,
                label=prompts[0] if prompts else "person",
                model=self.name,
            )
        ]

