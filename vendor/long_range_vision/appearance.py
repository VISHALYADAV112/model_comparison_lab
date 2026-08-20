from __future__ import annotations

import math
from typing import Any, Protocol

from .tiling import padded_crop_box
from .types import Detection


class AppearanceEncoder(Protocol):
    """Encode source-resolution object crops into L2-normalized tokens."""

    name: str
    dimension: int
    pretrained: bool

    def encode(self, crops: list[Any]) -> list[list[float]]: ...


def _normalize(values: Any) -> list[float]:
    import numpy as np

    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        vector /= norm
    return [float(value) for value in vector]


class HistogramAppearanceEncoder:
    """Portable non-learned control for appearance-aware association.

    It combines color, luminance, gradient-orientation and coarse spatial
    statistics. This is deliberately identified as a control rather than a
    pretrained video representation.
    """

    name = "histogram-v1"
    dimension = 64
    pretrained = False

    def encode(self, crops: list[Any]) -> list[list[float]]:
        import numpy as np
        from PIL import Image

        results: list[list[float]] = []
        for crop in crops:
            rgb = np.asarray(
                crop.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            features: list[float] = []
            for channel in range(3):
                histogram, _ = np.histogram(rgb[..., channel], bins=8, range=(0.0, 256.0))
                features.extend((histogram / max(1, histogram.sum())).tolist())
            luminance, _ = np.histogram(gray, bins=16, range=(0.0, 256.0))
            features.extend((luminance / max(1, luminance.sum())).tolist())

            dx = np.zeros_like(gray)
            dy = np.zeros_like(gray)
            dx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
            dy[1:-1, :] = gray[2:, :] - gray[:-2, :]
            magnitude = np.hypot(dx, dy)
            orientation = (np.arctan2(dy, dx) + math.pi) % math.pi
            gradient, _ = np.histogram(
                orientation,
                bins=8,
                range=(0.0, math.pi),
                weights=magnitude,
            )
            features.extend((gradient / max(1e-6, float(gradient.sum()))).tolist())

            coarse = gray.reshape(4, 16, 4, 16).mean(axis=(1, 3)) / 255.0
            features.extend(coarse.reshape(-1).tolist())
            results.append(_normalize(features))
        return results


class MobileNetAppearanceEncoder:
    """Small pretrained visual encoder used only on proposed object crops."""

    name = "mobilenet-v3-small-imagenet"
    dimension = 576
    pretrained = True

    def __init__(self, device: str = "auto", batch_size: int = 64) -> None:
        try:
            import torch
            from torch import nn
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        except ImportError as exc:
            raise RuntimeError(
                "The MobileNet appearance encoder requires torch and torchvision."
            ) from exc

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.batch_size = max(1, batch_size)
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        model.classifier = nn.Identity()
        self.model = model.eval().to(self.device)
        self.transform = weights.transforms()

    def encode(self, crops: list[Any]) -> list[list[float]]:
        if not crops:
            return []
        import torch

        results: list[list[float]] = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                batch_crops = crops[start : start + self.batch_size]
                batch = torch.stack([self.transform(crop.convert("RGB")) for crop in batch_crops])
                features = self.model(batch.to(self.device))
                features = torch.nn.functional.normalize(features.float(), dim=1)
                results.extend(features.cpu().tolist())
        return results


def create_appearance_encoder(
    name: str,
    *,
    device: str = "auto",
    batch_size: int = 64,
) -> AppearanceEncoder | None:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"", "none", "off", "disabled"}:
        return None
    if normalized in {"histogram", "histogram_v1", "portable"}:
        return HistogramAppearanceEncoder()
    if normalized in {"mobilenet", "mobilenet_v3_small", "pretrained"}:
        return MobileNetAppearanceEncoder(device=device, batch_size=batch_size)
    raise ValueError(
        f"Unknown appearance encoder {name!r}; choose none, histogram, or mobilenet_v3_small"
    )


def attach_appearance_tokens(
    image: Any,
    detections: list[Detection],
    encoder: AppearanceEncoder | None,
    *,
    padding: float = 0.35,
) -> None:
    """Attach transient embeddings to detections without writing them to reports."""
    if encoder is None or not detections:
        return
    crops: list[Any] = []
    reliabilities: list[float] = []
    for detection in detections:
        roi = padded_crop_box(detection.box, image.width, image.height, padding=padding)
        coordinates = (round(roi.x1), round(roi.y1), round(roi.x2), round(roi.y2))
        crops.append(image.crop(coordinates).convert("RGB"))
        # Tiny crops contain less measured appearance information. The token is
        # retained, but its contribution to matching is reduced accordingly.
        scale = math.sqrt(max(0.0, detection.box.area))
        reliabilities.append(min(1.0, scale / 24.0))

    embeddings = encoder.encode(crops)
    if len(embeddings) != len(detections):
        raise RuntimeError("Appearance encoder returned a different number of tokens than crops")
    for detection, embedding, reliability in zip(detections, embeddings, reliabilities, strict=True):
        detection.metadata["appearance_embedding"] = embedding
        detection.metadata["appearance_reliability"] = reliability
        detection.metadata["appearance_encoder"] = encoder.name
