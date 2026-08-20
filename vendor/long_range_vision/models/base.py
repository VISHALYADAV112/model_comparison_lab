from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import Detection


class ModelAdapter(ABC):
    def __init__(self, name: str, config: dict[str, Any], device: str = "auto") -> None:
        self.name = name
        self.config = config
        self.device = device

    @abstractmethod
    def detect(self, image: Any, prompts: list[str]) -> list[Detection]:
        raise NotImplementedError


def choose_torch_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"

