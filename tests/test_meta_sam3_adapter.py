from __future__ import annotations

import sys
from contextlib import AbstractContextManager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image

from model_lab.adapters.meta_sam3 import MetaSam3Adapter


def test_official_image_context_enables_meta_precision_settings(monkeypatch) -> None:
    autocast = Mock(return_value=SimpleNamespace())
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
            cudnn=SimpleNamespace(allow_tf32=False),
        ),
        bfloat16="bf16",
        autocast=autocast,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    context = MetaSam3Adapter._image_inference_context()

    assert context is autocast.return_value
    assert fake_torch.backends.cuda.matmul.allow_tf32 is True
    assert fake_torch.backends.cudnn.allow_tf32 is True
    autocast.assert_called_once_with(device_type="cuda", dtype="bf16")


def test_text_inference_runs_inside_scoped_autocast(monkeypatch, tmp_path: Path) -> None:
    class TrackingContext(AbstractContextManager):
        active = False

        def __enter__(self):
            self.active = True
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.active = False
            return False

    context = TrackingContext()

    class FakeProcessor:
        def __init__(self, model, *, device: str, confidence_threshold: float) -> None:
            assert device == "cuda"
            assert confidence_threshold == 0.35

        def set_image(self, image):
            assert context.active
            return {}

        def set_text_prompt(self, text: str, state: dict):
            assert context.active
            assert text == "vehicle"
            return {
                "masks": np.ones((1, 1, 8, 8), dtype=bool),
                "boxes": np.asarray([[1, 1, 7, 7]], dtype=np.float32),
                "scores": np.asarray([0.9], dtype=np.float32),
            }

    sam3_package = ModuleType("sam3")
    sam3_package.__path__ = []
    model_package = ModuleType("sam3.model")
    model_package.__path__ = []
    processor_module = ModuleType("sam3.model.sam3_image_processor")
    processor_module.Sam3Processor = FakeProcessor
    builder_module = ModuleType("sam3.model_builder")
    builder_module.build_sam3_image_model = lambda **kwargs: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "sam3", sam3_package)
    monkeypatch.setitem(sys.modules, "sam3.model", model_package)
    monkeypatch.setitem(sys.modules, "sam3.model.sam3_image_processor", processor_module)
    monkeypatch.setitem(sys.modules, "sam3.model_builder", builder_module)

    checkpoint = tmp_path / "sam3.pt"
    checkpoint.touch()
    source = tmp_path / "source.png"
    Image.new("RGB", (8, 8), "white").save(source)
    config = SimpleNamespace(
        sam3_official_image_model=checkpoint,
        raw={"sam3": {"score_threshold": 0.35}},
    )
    adapter = MetaSam3Adapter(config)
    monkeypatch.setattr(adapter, "_require_cuda", lambda: None)
    monkeypatch.setattr(adapter, "_release_cuda", lambda *args: None)
    monkeypatch.setattr(adapter, "_image_inference_context", lambda: context)

    manifest, payload = adapter.run_image(source, tmp_path / "result", mode="text", text="vehicle")

    assert manifest.exists()
    assert payload["frames"][0]["detections"][0]["score"] == np.float32(0.9)
    assert (manifest.parent / "mask_000.png").exists()
    assert context.active is False
