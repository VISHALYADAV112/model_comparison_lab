from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..contracts import ModelResult


class DetectorAdapter(ABC):
    @abstractmethod
    def predict_image(self, image: Path, output_dir: Path) -> ModelResult:
        raise NotImplementedError

    def predict_images(self, images: list[Path], output_dir: Path) -> list[ModelResult]:
        """Run detection on several images, one ModelResult per input.

        The default implementation loops over predict_image; adapters whose
        backends support batched inference override this for fewer, larger
        forward passes (the cascade tile phase's main speed lever).
        """
        return [self.predict_image(image, output_dir) for image in images]

