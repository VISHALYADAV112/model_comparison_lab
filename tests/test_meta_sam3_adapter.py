from __future__ import annotations

import sys
from contextlib import AbstractContextManager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from model_lab.adapters.meta_sam3 import (
    MetaSam3Adapter,
    _numpy,
    close_video_session,
    configure_video_predictor,
    start_video_session,
    stream_video_responses,
)


def test_close_video_session_clears_nested_tensor_references() -> None:
    state = {"feature_cache": {"tensor": object()}}
    session = {"state": state, "session_id": "session-1"}

    class FakePredictor:
        def __init__(self) -> None:
            self._all_inference_states = {"session-1": session}

        def handle_request(self, request):
            assert request == {"type": "close_session", "session_id": "session-1"}
            return {"is_success": True}

    predictor = FakePredictor()

    close_video_session(predictor, "session-1")

    assert predictor._all_inference_states == {}
    assert state == {}
    assert session == {}


def test_video_predictor_uses_requested_grounding_batch_size() -> None:
    predictor = SimpleNamespace(model=SimpleNamespace(batched_grounding_batch_size=16))

    result = configure_video_predictor(predictor, 4)

    assert result is predictor
    assert predictor.model.batched_grounding_batch_size == 4


@pytest.mark.parametrize("batch_size", [0, 17])
def test_video_predictor_rejects_invalid_grounding_batch_size(batch_size: int) -> None:
    predictor = SimpleNamespace(model=SimpleNamespace(batched_grounding_batch_size=16))

    with pytest.raises(ValueError, match="between 1 and 16"):
        configure_video_predictor(predictor, batch_size)


def test_multiplex_session_filters_unsupported_false_offload_option() -> None:
    calls = []

    class FakeModel:
        def init_state(self, resource_path, offload_video_to_cpu=False, async_loading_frames=False):
            calls.append(
                {
                    "resource_path": resource_path,
                    "offload_video_to_cpu": offload_video_to_cpu,
                    "async_loading_frames": async_loading_frames,
                }
            )
            return {"frames": []}

    predictor = SimpleNamespace(model=FakeModel(), async_loading_frames=True, _all_inference_states={})

    response = start_video_session(predictor, "/tmp/video.mp4", offload_state_to_cpu=False)

    assert calls == [
        {
            "resource_path": "/tmp/video.mp4",
            "offload_video_to_cpu": False,
            "async_loading_frames": True,
        }
    ]
    assert response["session_id"] in predictor._all_inference_states


def test_multiplex_session_rejects_requested_unsupported_state_offload() -> None:
    class FakeModel:
        def init_state(self, resource_path, offload_video_to_cpu=False):
            raise AssertionError("init_state should not run")

    predictor = SimpleNamespace(model=FakeModel(), _all_inference_states={})

    with pytest.raises(ValueError, match="does not support state offload"):
        start_video_session(predictor, "/tmp/video.mp4", offload_state_to_cpu=True)


def test_video_stream_aligns_meta_finite_bound_and_enforces_exact_limit() -> None:
    requests = []
    detector_limits = []
    stream_closed = False

    class FakeDetector:
        def forward_video_grounding_batched_multigpu(self, **kwargs):
            detector_limits.append(kwargs["max_frame_num_to_track"])

        def forward_video_grounding_multigpu(self, **kwargs):
            detector_limits.append(kwargs["max_frame_num_to_track"])

    class FakePredictor:
        model = SimpleNamespace(detector=FakeDetector())

        def handle_stream_request(self, request):
            requests.append(request)
            self.model.detector.forward_video_grounding_batched_multigpu(
                max_frame_num_to_track=request["max_frame_num_to_track"]
            )

            def responses():
                nonlocal stream_closed
                try:
                    for frame_index in range(10):
                        yield {"frame_index": frame_index, "outputs": {}}
                finally:
                    stream_closed = True

            return responses()

    request = {
        "type": "propagate_in_video",
        "session_id": "session-1",
        "max_frame_num_to_track": 3,
    }
    responses = list(stream_video_responses(FakePredictor(), request, max_frames=3))

    assert [response["frame_index"] for response in responses] == [0, 1, 2]
    assert requests[0]["max_frame_num_to_track"] == 2
    assert detector_limits == [3]
    assert request["max_frame_num_to_track"] == 3
    assert stream_closed is True


def test_video_stream_rejects_negative_frame_limit() -> None:
    predictor = SimpleNamespace(handle_stream_request=Mock())

    with pytest.raises(ValueError, match="Maximum frames"):
        list(stream_video_responses(predictor, {}, max_frames=-1))

    predictor.handle_stream_request.assert_not_called()


def test_bfloat_tensor_is_cast_to_float32_before_numpy() -> None:
    expected = np.asarray([0.25, 0.75], dtype=np.float32)

    class FakeBFloatTensor:
        converted = False

        def detach(self):
            return self

        def cpu(self):
            return self

        def is_floating_point(self) -> bool:
            return True

        def float(self):
            self.converted = True
            return self

        def numpy(self):
            if not self.converted:
                raise TypeError("Got unsupported ScalarType BFloat16")
            return expected

    tensor = FakeBFloatTensor()

    result = _numpy(tensor)

    assert tensor.converted is True
    np.testing.assert_array_equal(result, expected)


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
