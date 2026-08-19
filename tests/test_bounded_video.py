from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from model_lab.bounded_video import (
    BoundedSamRunner,
    BoundedVideoSettings,
    GlobalIdentityStitcher,
    OpenCVChunkSource,
    RtspChunkQueue,
    VideoChunk,
    box_iou,
    redact_rtsp_url,
    validate_rtsp_url,
)


def _chunk(
    root: Path,
    index: int,
    *,
    start: int,
    frame_count: int,
    overlap: int,
    unique: int,
) -> VideoChunk:
    inference = root / f"inference_{index}.mp4"
    source = root / f"unique_{index}.mp4"
    inference.touch()
    source.touch()
    return VideoChunk(
        index=index,
        inference_path=inference,
        unique_path=source,
        global_start_frame=start,
        frame_count=frame_count,
        overlap_prefix=overlap,
        unique_frame_count=unique,
        fps=10.0,
        width=32,
        height=24,
    )


def _frames(count: int, local_id: int, box: list[float] | None = None) -> list[dict]:
    detection_box = box or [2.0, 2.0, 12.0, 12.0]
    return [
        {
            "frame_index": frame_index,
            "detections": [
                {
                    "instance_id": local_id,
                    "box": detection_box,
                    "score": 0.9,
                    "mask": f"mask_{frame_index}.png",
                }
            ],
        }
        for frame_index in range(count)
    ]


def test_bounded_settings_reject_overlap_as_large_as_chunk() -> None:
    with pytest.raises(ValueError, match="Overlap"):
        BoundedVideoSettings(chunk_frames=30, overlap_frames=30).validate()


def test_rtsp_url_is_validated_and_credentials_are_redacted() -> None:
    source = "rtsp://camera-user:secret@192.0.2.10:8554/private/token?auth=secret"

    assert validate_rtsp_url(source) == source
    assert redact_rtsp_url(source) == "rtsp://192.0.2.10:8554"
    with pytest.raises(ValueError, match="rtsp"):
        validate_rtsp_url("https://example.com/video.mp4")


def test_box_iou_handles_overlap_and_empty_boxes() -> None:
    assert box_iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)
    assert box_iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0


def test_identity_stitcher_keeps_id_across_chunk_overlap(tmp_path: Path) -> None:
    stitcher = GlobalIdentityStitcher(iou_threshold=0.3, ttl_frames=20, overlap_frames=2)
    first = _chunk(tmp_path, 0, start=0, frame_count=4, overlap=0, unique=4)
    second = _chunk(tmp_path, 1, start=2, frame_count=4, overlap=2, unique=2)

    first_frames = stitcher.apply(first, _frames(4, local_id=7))
    second_frames = stitcher.apply(second, _frames(4, local_id=99))

    assert first_frames[0]["detections"][0]["instance_id"] == 0
    assert second_frames[0]["detections"][0]["instance_id"] == 0
    assert second_frames[0]["detections"][0]["chunk_instance_id"] == 99
    assert stitcher.next_global_id == 1


def test_file_chunk_source_decodes_once_with_fixed_overlap(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24)
    )
    assert writer.isOpened()
    for index in range(10):
        writer.write(np.full((24, 32, 3), index * 10, dtype=np.uint8))
    writer.release()
    settings = BoundedVideoSettings(chunk_frames=5, overlap_frames=2)

    chunks = list(
        OpenCVChunkSource(
            str(source), tmp_path / "chunks", settings, threading.Event()
        )
    )

    assert [item.global_start_frame for item in chunks] == [0, 3, 6]
    assert [item.frame_count for item in chunks] == [5, 5, 4]
    assert [item.overlap_prefix for item in chunks] == [0, 2, 2]
    assert [item.unique_frame_count for item in chunks] == [5, 3, 2]


def test_rtsp_pending_queue_drops_the_oldest_chunk(tmp_path: Path) -> None:
    source = SimpleNamespace(stop_event=threading.Event())
    pending = RtspChunkQueue(source, capacity=1)
    first = _chunk(tmp_path, 0, start=0, frame_count=4, overlap=0, unique=4)
    second = _chunk(tmp_path, 1, start=3, frame_count=4, overlap=1, unique=3)

    pending._put_bounded(first)
    pending._put_bounded(second)

    assert pending.dropped_chunks == 1
    assert not first.inference_path.exists()
    assert not first.unique_path.exists()
    assert pending.queue.get_nowait() == second


def test_rtsp_queue_preserves_capture_error_after_pending_chunks(tmp_path: Path) -> None:
    failure = RuntimeError("camera disconnected")

    class FailingSource:
        def __init__(self) -> None:
            self.stop_event = threading.Event()

        def __iter__(self):
            yield _chunk(tmp_path, 0, start=0, frame_count=4, overlap=0, unique=4)
            raise failure

    pending = RtspChunkQueue(FailingSource(), capacity=1)
    iterator = iter(pending)

    assert next(iterator).index == 0
    with pytest.raises(RuntimeError, match="camera disconnected"):
        next(iterator)
    assert pending.dropped_chunks == 0


def test_runner_discards_prefetched_chunk_at_max_chunks(tmp_path: Path) -> None:
    class FakeAdapter:
        def __init__(self, config) -> None:
            self.config = config

        def run_video(self, video, output, **kwargs):
            output.mkdir(parents=True)
            payload = {
                "width": 32,
                "height": 24,
                "fps": 10,
                "frames": _frames(kwargs["max_frames"], local_id=12),
            }
            manifest = output / "manifest.json"
            manifest.write_text(json.dumps(payload))
            return manifest, payload

    def fake_renderer(source: Path, manifest: Path, output: Path) -> Path:
        output.touch()
        return output

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    first = _chunk(inputs, 0, start=0, frame_count=4, overlap=0, unique=4)
    prefetched = _chunk(inputs, 1, start=3, frame_count=4, overlap=1, unique=3)
    settings = BoundedVideoSettings(
        chunk_frames=4,
        overlap_frames=1,
        identity_ttl_frames=4,
    )
    runner = BoundedSamRunner(
        object(), adapter_factory=FakeAdapter, renderer=fake_renderer
    )

    updates = list(
        runner.run(
            [first, prefetched],
            tmp_path / "output",
            source_kind="long_video",
            source_label="source.mp4",
            target="vehicle",
            settings=settings,
            stop_event=threading.Event(),
            max_chunks=1,
        )
    )

    assert updates[-1]["processed_chunks"] == 1
    assert not prefetched.inference_path.exists()
    assert not prefetched.unique_path.exists()


def test_runner_commits_unique_frames_incrementally(tmp_path: Path) -> None:
    class FakeAdapter:
        def __init__(self, config) -> None:
            self.config = config

        def run_video(self, video, output, **kwargs):
            output.mkdir(parents=True)
            payload = {
                "width": 32,
                "height": 24,
                "fps": 10,
                "elapsed_seconds": 0.1,
                "frames": _frames(kwargs["max_frames"], local_id=12),
            }
            manifest = output / "manifest.json"
            manifest.write_text(json.dumps(payload))
            return manifest, payload

    def fake_renderer(source: Path, manifest: Path, output: Path) -> Path:
        assert source.exists()
        assert manifest.exists()
        output.touch()
        return output

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    chunks = [
        _chunk(inputs, 0, start=0, frame_count=4, overlap=0, unique=4),
        _chunk(inputs, 1, start=2, frame_count=4, overlap=2, unique=2),
    ]
    settings = BoundedVideoSettings(
        chunk_frames=4,
        overlap_frames=2,
        grounding_batch_size=1,
        max_active_objects=8,
        identity_ttl_frames=4,
    )
    runner = BoundedSamRunner(
        object(), adapter_factory=FakeAdapter, renderer=fake_renderer
    )

    updates = list(
        runner.run(
            chunks,
            tmp_path / "output",
            source_kind="long_video",
            source_label="source.mp4",
            target="vehicle",
            settings=settings,
            stop_event=threading.Event(),
        )
    )

    assert updates[-1]["status"] == "complete"
    assert updates[-1]["processed_chunks"] == 2
    assert updates[-1]["processed_frames"] == 6
    assert updates[-1]["unique_objects"] == 1
    records = [json.loads(line) for line in (tmp_path / "output" / "frames.jsonl").read_text().splitlines()]
    assert [record["frame_index"] for record in records] == list(range(6))
    assert records[-1]["detections"][0]["mask"].startswith("chunks/chunk_000001/")


def test_production_runner_uses_a_finite_isolated_cuda_worker(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []

    def fake_subprocess_run(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index("--output") + 1])
        frame_count = int(command[command.index("--max-frames") + 1])
        output.mkdir(parents=True)
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "width": 32,
                    "height": 24,
                    "fps": 10,
                    "frames": _frames(frame_count, local_id=4),
                }
            )
        )
        return SimpleNamespace(returncode=0)

    def fake_renderer(source: Path, manifest: Path, output: Path) -> Path:
        output.touch()
        return output

    monkeypatch.setattr("model_lab.bounded_video.subprocess.run", fake_subprocess_run)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    chunk = _chunk(inputs, 0, start=0, frame_count=4, overlap=0, unique=4)
    config = SimpleNamespace(root=tmp_path, raw={"sam3": {}})
    runner = BoundedSamRunner(config, renderer=fake_renderer, isolated_workers=True)
    settings = BoundedVideoSettings(
        chunk_frames=4,
        overlap_frames=1,
        grounding_batch_size=1,
        max_active_objects=8,
        identity_ttl_frames=4,
        worker_timeout_seconds=123,
    )

    updates = list(
        runner.run(
            [chunk],
            tmp_path / "output",
            source_kind="long_video",
            source_label="source.mp4",
            target="vehicle",
            settings=settings,
            stop_event=threading.Event(),
        )
    )

    command, kwargs = calls[0]
    assert command[1:3] == ["-m", "model_lab.bounded_worker"]
    assert command[command.index("--grounding-batch-size") + 1] == "1"
    assert command[command.index("--max-num-objects") + 1] == "8"
    assert kwargs["timeout"] == 123
    assert updates[-1]["status"] == "complete"
