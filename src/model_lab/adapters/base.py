from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..contracts import ModelResult


class DetectorAdapter(ABC):
    @abstractmethod
    def predict_image(self, image: Path, output_dir: Path) -> ModelResult:
        raise NotImplementedError

