from __future__ import annotations

from typing import Any

from .base import ModelAdapter


def create_adapter(name: str, config: dict[str, Any], device: str = "auto") -> ModelAdapter:
    adapter = str(config.get("adapter", "")).lower()
    if adapter == "mock":
        from .mock import MockAdapter

        return MockAdapter(name, config, device)
    if adapter == "rfdetr":
        from .rfdetr import RFDETRAdapter

        return RFDETRAdapter(name, config, device)
    if adapter == "grounding_dino":
        from .grounding_dino import GroundingDINOAdapter

        return GroundingDINOAdapter(name, config, device)
    if adapter == "sam3":
        from .sam3 import SAM3Adapter

        return SAM3Adapter(name, config, device)
    raise ValueError(f"Unknown model adapter: {adapter!r}")

