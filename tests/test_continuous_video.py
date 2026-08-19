from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from model_lab.cli import build_parser
from model_lab.continuous_video import (
    ContinuousResultWriter,
    ContinuousSamRunner,
    ContinuousVideoSettings,
    FramePacket,
    RollingFrameLoader,
    SparseFrameStore,
    TrackArchive,
    prune_continuous_state,
)


def test_continuous_settings_keep_native_history() -> None:
    assert ContinuousVideoSettings().validate().state_history_frames == 32
    with pytest.raises(ValueError, match="at least 16"):
        ContinuousVideoSettings(state_history_frames=15).validate()
    with pytest.raises(ValueError, match="preview interval"):
        ContinuousVideoSettings(preview_interval_seconds=0).validate()


def test_long_video_cli_defaults_to_continuous_native_engine() -> None:
    args = build_parser().parse_args(
        ["long-video", "--input", "video.mp4", "--text", "person"]
    )

    assert args.engine == "continuous"


def test_continuous_state_initializes_text_backbone_without_full_stream_loop() -> None:
    torch = pytest.importorskip("torch")

    class FakeModel:
        TEXT_ID_FOR_TEXT = 0
        tracker = SimpleNamespace(model=SimpleNamespace())

        @torch.inference_mode()
        def init_state(self, **_):
            self.find_input = SimpleNamespace(text_ids=torch.tensor([-1]))
            self.input_batch = SimpleNamespace(
                img_batch=SimpleNamespace(tensors=None),
                find_text_batch=["placeholder", "visual", "geometric"],
                find_inputs=[self.find_input],
            )
            return {"input_batch": self.input_batch}

        def _init_backbone_out(self, state):
            assert state["input_batch"].find_text_batch[0] == "person"
            return {"language_features": "initialized"}

    class FakeLoader:
        def __len__(self):
            return 10_000_000

        @staticmethod
        def first_pil_image():
            return Image.new("RGB", (4, 4))

    model = FakeModel()
    state = ContinuousSamRunner(config=None)._initialize_state(
        SimpleNamespace(model=model), FakeLoader(), "person"
    )

    assert state["num_frames"] == 10_000_000
    assert state["backbone_out"] == {"language_features": "initialized"}
    assert torch.is_inference(model.find_input.text_ids)
    assert model.find_input.text_ids.tolist() == [0]


def test_continuous_writer_publishes_atomic_live_preview(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    writer = ContinuousResultWriter(
        tmp_path,
        width=32,
        height=24,
        fps=10,
        source_path=None,
    )
    packet = FramePacket(
        frame_index=0,
        capture_sequence=0,
        bgr=np.full((24, 32, 3), 90, dtype=np.uint8),
        captured_at_utc=None,
    )
    try:
        writer.write(
            0,
            {
                "out_binary_masks": [],
                "out_obj_ids": [],
                "out_probs": [],
                "out_boxes_xywh": [],
            },
            packet,
        )
        preview = cv2.imread(str(writer.live_preview))
        assert preview is not None
        assert preview.shape[:2] == (24, 32)
        assert not writer.live_preview.with_suffix(".jpg.tmp").exists()
    finally:
        writer.video_writer.release()
        writer.archive.close()


def test_sparse_frame_store_does_not_scale_with_frame_number() -> None:
    store = SparseFrameStore(None)
    store[10_000_000] = "tracked"
    store[5] = "old"
    store.prune_before(100)

    assert len(store) == 1
    assert store[10_000_000] == "tracked"
    assert store[5] is None


def test_rolling_file_loader_decodes_lazily_and_releases(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24)
    )
    assert writer.isOpened()
    for index in range(8):
        writer.write(np.full((24, 32, 3), index * 20, dtype=np.uint8))
    writer.release()

    loader = RollingFrameLoader(
        str(source),
        stop_event=threading.Event(),
        frame_buffer_frames=48,
        tensor_factory=lambda frame: int(frame.mean()),
    )
    try:
        assert loader.next_frame_index == 0
        available, ended = loader.prepare(0, 3)
        assert (available, ended) == (3, False)
        assert loader.next_frame_index == 3
        assert len(loader.cache) == 3
        loader.release_through(1)
        assert list(loader.cache) == [2]
    finally:
        loader.close()


def test_pruner_keeps_first_and_latest_conditioning_frames() -> None:
    tracker_state = {
        "obj_ids": [4],
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {
            0: {index: f"mask {index}" for index in range(0, 100, 10)}
        },
        "output_dict": {
            "cond_frame_outputs": {
                index: {"value": index} for index in range(0, 100, 10)
            },
            "non_cond_frame_outputs": {index: {"value": index} for index in range(100)},
        },
        "output_dict_per_obj": {
            0: {
                "cond_frame_outputs": {
                    index: {"value": index} for index in range(0, 100, 10)
                },
                "non_cond_frame_outputs": {
                    index: {"value": index} for index in range(100)
                },
            }
        },
        "temp_output_dict_per_obj": {},
        "consolidated_frame_inds": {
            "cond_frame_outputs": set(range(0, 100, 10)),
            "non_cond_frame_outputs": set(range(100)),
        },
        "first_ann_frame_idx": 0,
        "frames_already_tracked": {index: {} for index in range(100)},
    }
    state = {
        "cached_frame_outputs": {index: {} for index in range(100)},
        "sam2_inference_states": [tracker_state],
        "tracker_metadata": {
            "obj_ids_all_gpu": np.asarray([4]),
            "obj_id_to_score": {4: 0.9, 9: -10_000},
            "rank0_metadata": {
                "suppressed_obj_ids": {index: set() for index in range(100)},
                "removed_obj_ids": {9},
            },
        },
        "generator_state": {
            "hotstart_buffer": [],
            "postprocess_yield_list": [],
            "unconfirmed_obj_ids_per_frame": {},
        },
    }

    telemetry = prune_continuous_state(state, processed_frame=99, history_frames=32)

    assert min(tracker_state["output_dict"]["non_cond_frame_outputs"]) == 68
    assert set(tracker_state["output_dict"]["cond_frame_outputs"]) == {60, 70, 80, 90}
    assert set(tracker_state["mask_inputs_per_obj"][0]) == {60, 70, 80, 90}
    assert tracker_state["first_ann_frame_idx"] == 60
    assert state["tracker_metadata"]["obj_id_to_score"] == {4: 0.9}
    assert telemetry["active_objects"] == 1


def test_track_archive_keeps_best_crop_and_manual_identity_columns(
    tmp_path: Path,
) -> None:
    pytest.importorskip("cv2")
    archive = TrackArchive(tmp_path)
    frame = np.full((40, 50, 3), 127, dtype=np.uint8)
    archive.update(
        2,
        [{"instance_id": 7, "score": 0.7, "box": [5, 6, 25, 30]}],
        frame,
    )
    archive.update(
        9,
        [{"instance_id": 7, "score": 0.9, "box": [8, 8, 30, 35]}],
        frame,
    )
    archive.close()

    connection = sqlite3.connect(tmp_path / "track_identities.sqlite3")
    row = connection.execute(
        "SELECT first_frame, last_frame, best_score, best_crop, verified_identity, embedding "
        "FROM tracks WHERE sam_track_id = 7"
    ).fetchone()
    connection.close()
    assert row[:3] == (2, 9, 0.9)
    assert (tmp_path / row[3]).is_file()
    assert row[4:] == (None, None)
