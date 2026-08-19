from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .adapters.meta_sam3 import MetaSam3Adapter
from .config import LabConfig
from .rendering import render_video_manifest


@dataclass(frozen=True)
class BoundedVideoSettings:
    """Hard limits that keep one SAM 3.1 session independent of stream length."""

    chunk_frames: int = 60
    overlap_frames: int = 8
    grounding_batch_size: int = 1
    max_active_objects: int = 16
    threshold: float = 0.5
    identity_iou_threshold: float = 0.3
    identity_ttl_frames: int = 60
    worker_timeout_seconds: float = 1800

    def validate(self) -> BoundedVideoSettings:
        if self.chunk_frames < 2:
            raise ValueError("Chunk size must be at least 2 frames")
        if not 0 <= self.overlap_frames < self.chunk_frames:
            raise ValueError("Overlap must be non-negative and smaller than the chunk size")
        if not 1 <= self.grounding_batch_size <= 16:
            raise ValueError("Grounding batch size must be between 1 and 16")
        if not 1 <= self.max_active_objects <= 64:
            raise ValueError("Maximum active objects must be between 1 and 64")
        if not 0 < self.threshold < 1:
            raise ValueError("Output threshold must be between 0 and 1")
        if not 0 <= self.identity_iou_threshold <= 1:
            raise ValueError("Identity IoU threshold must be between 0 and 1")
        if self.identity_ttl_frames < self.overlap_frames:
            raise ValueError("Identity TTL must be at least as large as the overlap")
        if self.worker_timeout_seconds <= 0:
            raise ValueError("Worker timeout must be positive")
        return self


@dataclass(frozen=True)
class VideoChunk:
    index: int
    inference_path: Path
    unique_path: Path
    global_start_frame: int
    frame_count: int
    overlap_prefix: int
    unique_frame_count: int
    fps: float
    width: int
    height: int
    unique_started_at_utc: str | None = None


@dataclass
class _TrackState:
    global_id: int
    last_frame: int
    box: tuple[float, float, float, float]


def redact_rtsp_url(value: str) -> str:
    """Return a non-secret source label suitable for manifests and errors."""
    parts = urlsplit(value)
    hostname = parts.hostname or "unknown-host"
    host = f"[{hostname}]" if ":" in hostname else hostname
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is not None:
        host = f"{host}:{port}"
    return urlunsplit((parts.scheme.lower(), host, "", "", ""))


def validate_rtsp_url(value: str) -> str:
    candidate = (value or "").strip()
    parts = urlsplit(candidate)
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("Enter a valid RTSP port") from exc
    if (
        parts.scheme.lower() not in {"rtsp", "rtsps"}
        or not parts.hostname
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise ValueError("Enter a valid rtsp:// or rtsps:// URL")
    return candidate


def box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    ax0, ay0, ax1, ay1 = (float(value) for value in left)
    bx0, by0, bx1, by1 = (float(value) for value in right)
    intersection_width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    intersection_height = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    right_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


class GlobalIdentityStitcher:
    """Map chunk-local SAM IDs onto small, CPU-resident global track records."""

    def __init__(self, *, iou_threshold: float, ttl_frames: int, overlap_frames: int) -> None:
        self.iou_threshold = float(iou_threshold)
        self.ttl_frames = int(ttl_frames)
        self.overlap_frames = int(overlap_frames)
        self.next_global_id = 0
        self.previous_frames: dict[int, list[dict[str, Any]]] = {}
        self.tracks: dict[int, _TrackState] = {}

    def _new_global_id(self) -> int:
        value = self.next_global_id
        self.next_global_id += 1
        return value

    @staticmethod
    def _detections_by_local_id(frames: list[dict[str, Any]]) -> dict[int, list[tuple[int, dict[str, Any]]]]:
        result: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        for frame in frames:
            frame_index = int(frame["frame_index"])
            for detection in frame.get("detections", []):
                local_id = int(detection.get("instance_id", -1))
                result.setdefault(local_id, []).append((frame_index, detection))
        return result

    def _overlap_scores(
        self,
        chunk: VideoChunk,
        local_tracks: dict[int, list[tuple[int, dict[str, Any]]]],
    ) -> list[tuple[float, int, int]]:
        scores: list[tuple[float, int, int]] = []
        for local_id, observations in local_tracks.items():
            sums: dict[int, float] = {}
            counts: dict[int, int] = {}
            for local_frame, detection in observations:
                if local_frame >= chunk.overlap_prefix:
                    continue
                global_frame = chunk.global_start_frame + local_frame
                for previous in self.previous_frames.get(global_frame, []):
                    global_id = int(previous["instance_id"])
                    sums[global_id] = sums.get(global_id, 0.0) + box_iou(
                        detection["box"], previous["box"]
                    )
                    counts[global_id] = counts.get(global_id, 0) + 1
            for global_id, total in sums.items():
                score = total / counts[global_id]
                if score >= self.iou_threshold:
                    scores.append((score, local_id, global_id))
        return sorted(scores, reverse=True)

    def _registry_match(
        self,
        observations: list[tuple[int, dict[str, Any]]],
        chunk: VideoChunk,
        used_global_ids: set[int],
    ) -> int | None:
        if not observations:
            return None
        local_frame, detection = observations[0]
        global_frame = chunk.global_start_frame + local_frame
        best: tuple[float, int] | None = None
        for global_id, state in self.tracks.items():
            if global_id in used_global_ids or global_frame - state.last_frame > self.ttl_frames:
                continue
            score = box_iou(detection["box"], state.box)
            if score >= self.iou_threshold and (best is None or score > best[0]):
                best = (score, global_id)
        return best[1] if best is not None else None

    def apply(self, chunk: VideoChunk, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
        local_tracks = self._detections_by_local_id(frames)
        mapping: dict[int, int] = {}
        used_global_ids: set[int] = set()
        for _, local_id, global_id in self._overlap_scores(chunk, local_tracks):
            if local_id not in mapping and global_id not in used_global_ids:
                mapping[local_id] = global_id
                used_global_ids.add(global_id)
        for local_id, observations in local_tracks.items():
            if local_id in mapping:
                continue
            global_id = self._registry_match(observations, chunk, used_global_ids)
            if global_id is None:
                global_id = self._new_global_id()
            mapping[local_id] = global_id
            used_global_ids.add(global_id)

        mapped_frames: list[dict[str, Any]] = []
        for frame in frames:
            local_frame = int(frame["frame_index"])
            global_frame = chunk.global_start_frame + local_frame
            detections: list[dict[str, Any]] = []
            for raw in frame.get("detections", []):
                detection = dict(raw)
                local_id = int(detection.get("instance_id", -1))
                global_id = mapping[local_id]
                detection["chunk_instance_id"] = local_id
                detection["instance_id"] = global_id
                detections.append(detection)
                self.tracks[global_id] = _TrackState(
                    global_id=global_id,
                    last_frame=global_frame,
                    box=tuple(float(value) for value in detection["box"]),
                )
            mapped_frames.append({"frame_index": local_frame, "detections": detections})

        final_global_frame = chunk.global_start_frame + max(0, chunk.frame_count - 1)
        self.tracks = {
            global_id: state
            for global_id, state in self.tracks.items()
            if final_global_frame - state.last_frame <= self.ttl_frames
        }
        tail_start = max(0, chunk.frame_count - self.overlap_frames)
        self.previous_frames = {
            chunk.global_start_frame + int(frame["frame_index"]): [dict(item) for item in frame["detections"]]
            for frame in mapped_frames
            if int(frame["frame_index"]) >= tail_start
        }
        return mapped_frames


class OpenCVChunkSource:
    """Decode a file once and retain only the overlap frames in CPU memory."""

    def __init__(
        self,
        source: str,
        output_dir: Path,
        settings: BoundedVideoSettings,
        stop_event: threading.Event,
        *,
        rtsp: bool = False,
        maximum_minutes: float = 0,
        reconnect_attempts: int = 5,
        reconnect_delay_seconds: float = 2,
    ) -> None:
        self.source = source
        self.output_dir = output_dir
        self.settings = settings.validate()
        self.stop_event = stop_event
        self.rtsp = rtsp
        self.maximum_minutes = float(maximum_minutes)
        self.reconnect_attempts = max(0, int(reconnect_attempts))
        self.reconnect_delay_seconds = max(0.0, float(reconnect_delay_seconds))
        self.reconnect_count = 0

    @staticmethod
    def _writer(path: Path, fps: float, width: int, height: int):
        import cv2

        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV could not create temporary chunk {path.name}")
        return writer

    def _open_capture(self):
        import cv2

        capture = cv2.VideoCapture()
        parameters: list[int] = []
        for name, milliseconds in (
            ("CAP_PROP_OPEN_TIMEOUT_MSEC", 10_000),
            ("CAP_PROP_READ_TIMEOUT_MSEC", 10_000),
        ):
            option = getattr(cv2, name, None)
            if option is not None:
                parameters.extend([option, milliseconds])
        api = cv2.CAP_FFMPEG if self.rtsp else cv2.CAP_ANY
        try:
            capture.open(self.source, api, parameters)
        except TypeError:
            capture.open(self.source, api)
        if not capture.isOpened():
            label = redact_rtsp_url(self.source) if self.rtsp else Path(self.source).name
            raise RuntimeError(f"Cannot open video source {label}")
        return capture

    def _reconnect(self):
        last_error: Exception | None = None
        for _ in range(self.reconnect_attempts):
            if self.stop_event.wait(self.reconnect_delay_seconds):
                return None
            try:
                capture = self._open_capture()
                self.reconnect_count += 1
                return capture
            except RuntimeError as exc:
                last_error = exc
        if last_error is not None:
            raise RuntimeError(
                f"RTSP reconnect failed for {redact_rtsp_url(self.source)}"
            ) from last_error
        return None

    def __iter__(self) -> Iterator[VideoChunk]:
        import cv2

        self.output_dir.mkdir(parents=True, exist_ok=True)
        capture = self._open_capture()
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        if not math.isfinite(fps) or not 1 <= fps <= 240:
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("Video source did not report a valid frame size")
        max_unique_frames = (
            max(1, int(self.maximum_minutes * 60 * fps)) if self.maximum_minutes > 0 else 0
        )
        overlap: deque[Any] = deque(maxlen=self.settings.overlap_frames or None)
        total_unique_frames = 0
        chunk_index = 0
        try:
            while not self.stop_event.is_set():
                overlap_prefix = len(overlap)
                frames_to_read = self.settings.chunk_frames - overlap_prefix
                if max_unique_frames:
                    frames_to_read = min(frames_to_read, max_unique_frames - total_unique_frames)
                if frames_to_read <= 0:
                    return
                inference_path = self.output_dir / f"chunk_{chunk_index:06d}_inference.mp4"
                unique_path = self.output_dir / f"chunk_{chunk_index:06d}_unique.mp4"
                inference_writer = self._writer(inference_path, fps, width, height)
                unique_writer = self._writer(unique_path, fps, width, height)
                for frame in overlap:
                    inference_writer.write(frame)
                new_frames = 0
                read_failed = False
                unique_started_at_utc = (
                    datetime.now(timezone.utc).isoformat() if self.rtsp else None
                )
                try:
                    while new_frames < frames_to_read and not self.stop_event.is_set():
                        ok, frame = capture.read()
                        if not ok:
                            read_failed = True
                            break
                        if frame.shape[1] != width or frame.shape[0] != height:
                            frame = cv2.resize(frame, (width, height))
                        inference_writer.write(frame)
                        unique_writer.write(frame)
                        if self.settings.overlap_frames:
                            overlap.append(frame.copy())
                        new_frames += 1
                finally:
                    inference_writer.release()
                    unique_writer.release()
                if not new_frames:
                    inference_path.unlink(missing_ok=True)
                    unique_path.unlink(missing_ok=True)
                    if self.rtsp and read_failed:
                        capture.release()
                        replacement = self._reconnect()
                        if replacement is None:
                            return
                        capture = replacement
                        continue
                    return
                global_start = max(0, total_unique_frames - overlap_prefix)
                total_unique_frames += new_frames
                yield VideoChunk(
                    index=chunk_index,
                    inference_path=inference_path,
                    unique_path=unique_path,
                    global_start_frame=global_start,
                    frame_count=overlap_prefix + new_frames,
                    overlap_prefix=overlap_prefix,
                    unique_frame_count=new_frames,
                    fps=fps,
                    width=width,
                    height=height,
                    unique_started_at_utc=unique_started_at_utc,
                )
                chunk_index += 1
                if max_unique_frames and total_unique_frames >= max_unique_frames:
                    return
                if read_failed:
                    if not self.rtsp:
                        return
                    capture.release()
                    replacement = self._reconnect()
                    if replacement is None:
                        return
                    capture = replacement
        finally:
            capture.release()


class RtspChunkQueue:
    """Capture RTSP continuously while keeping at most a fixed number of pending clips."""

    _END = object()

    def __init__(self, source: OpenCVChunkSource, *, capacity: int = 2) -> None:
        self.source = source
        self.queue: queue.Queue[VideoChunk | Exception | object] = queue.Queue(
            maxsize=max(1, capacity)
        )
        self.dropped_chunks = 0
        self.producer_error: Exception | None = None
        self.thread = threading.Thread(target=self._produce, name="rtsp-chunk-capture", daemon=True)

    @staticmethod
    def _discard(chunk: VideoChunk) -> None:
        chunk.inference_path.unlink(missing_ok=True)
        chunk.unique_path.unlink(missing_ok=True)

    def _put_bounded(self, item: VideoChunk) -> None:
        while True:
            try:
                self.queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    previous = self.queue.get_nowait()
                except queue.Empty:
                    continue
                if isinstance(previous, VideoChunk):
                    self._discard(previous)
                    self.dropped_chunks += 1

    def _produce(self) -> None:
        try:
            for chunk in self.source:
                self._put_bounded(chunk)
        except Exception as exc:  # noqa: BLE001
            # Preserve capture-thread failures so the dashboard worker can report them.
            self.producer_error = exc
        finally:
            # Waiting here preserves all accepted chunks. close() drains the queue
            # before joining, so an early consumer exit cannot strand this thread.
            self.queue.put(self._END)

    def close(self) -> None:
        self.source.stop_event.set()
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, VideoChunk):
                self._discard(item)
        if self.thread.is_alive():
            self.thread.join(timeout=12)

    def __iter__(self) -> Iterator[VideoChunk]:
        self.thread.start()
        try:
            while True:
                item = self.queue.get()
                if item is self._END:
                    if self.producer_error is not None:
                        raise self.producer_error
                    return
                if isinstance(item, Exception):
                    raise item
                if isinstance(item, VideoChunk):
                    yield item
        finally:
            self.close()


class BoundedSamRunner:
    """Run one finite SAM session per clip and commit progress incrementally."""

    def __init__(
        self,
        config: LabConfig,
        *,
        adapter_factory: Callable[[LabConfig], MetaSam3Adapter] = MetaSam3Adapter,
        renderer: Callable[[Path, Path, Path], Path] = render_video_manifest,
        isolated_workers: bool | None = None,
    ) -> None:
        self.config = config
        self.adapter_factory = adapter_factory
        self.renderer = renderer
        self.isolated_workers = (
            adapter_factory is MetaSam3Adapter if isolated_workers is None else isolated_workers
        )

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(path)

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def _run_chunk(
        self,
        chunk: VideoChunk,
        chunk_output: Path,
        worker_config: Path,
        target: str,
        settings: BoundedVideoSettings,
    ) -> tuple[Path, dict[str, Any]]:
        if not self.isolated_workers:
            adapter = self.adapter_factory(self.config)
            return adapter.run_video(
                chunk.inference_path,
                chunk_output,
                mode="text",
                text=target.strip(),
                start_frame=0,
                max_frames=chunk.frame_count,
                propagation_direction="forward",
                output_prob_threshold=settings.threshold,
                offload_video_to_cpu=True,
                offload_state_to_cpu=False,
                grounding_batch_size=settings.grounding_batch_size,
                max_num_objects=settings.max_active_objects,
            )

        command = [
            sys.executable,
            "-m",
            "model_lab.bounded_worker",
            "--worker-config",
            str(worker_config),
            "--input",
            str(chunk.inference_path),
            "--output",
            str(chunk_output),
            "--text",
            target.strip(),
            "--max-frames",
            str(chunk.frame_count),
            "--threshold",
            str(settings.threshold),
            "--grounding-batch-size",
            str(settings.grounding_batch_size),
            "--max-num-objects",
            str(settings.max_active_objects),
        ]
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        subprocess.run(
            command,
            check=True,
            env=environment,
            timeout=settings.worker_timeout_seconds,
        )
        manifest = chunk_output / "manifest.json"
        if not manifest.is_file():
            raise RuntimeError("The isolated SAM 3.1 worker produced no manifest")
        return manifest, json.loads(manifest.read_text())

    def run(
        self,
        chunks: Iterable[VideoChunk],
        output_dir: Path,
        *,
        source_kind: str,
        source_label: str,
        target: str,
        settings: BoundedVideoSettings,
        stop_event: threading.Event,
        max_chunks: int = 0,
        dropped_chunks: Callable[[], int] | None = None,
    ) -> Iterator[dict[str, Any]]:
        settings.validate()
        if not target.strip():
            raise ValueError("Describe the object or concept to track")
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "index.json").exists():
            raise FileExistsError(f"Bounded-video output already exists: {output_dir}")
        results_dir = output_dir / "chunks"
        segments_dir = output_dir / "segments"
        results_dir.mkdir()
        segments_dir.mkdir()
        frames_jsonl = output_dir / "frames.jsonl"
        chunks_jsonl = output_dir / "chunks.jsonl"
        index_path = output_dir / "index.json"
        worker_config = output_dir / "worker_config.json"
        if self.isolated_workers:
            self._write_json(
                worker_config,
                {"root": str(self.config.root), "raw": self.config.raw},
            )
        stitcher = GlobalIdentityStitcher(
            iou_threshold=settings.identity_iou_threshold,
            ttl_frames=settings.identity_ttl_frames,
            overlap_frames=settings.overlap_frames,
        )
        started = time.perf_counter()
        processed_chunks = 0
        processed_frames = 0
        mask_count = 0
        latest_segment: Path | None = None
        status = "running"

        def snapshot() -> dict[str, Any]:
            return {
                "schema_version": 1,
                "mode": source_kind,
                "model": "SAM 3.1 Object Multiplex",
                "source": source_label,
                "target": target.strip(),
                "status": status,
                "settings": asdict(settings),
                "processed_chunks": processed_chunks,
                "processed_frames": processed_frames,
                "unique_objects": stitcher.next_global_id,
                "frame_level_masks": mask_count,
                "dropped_rtsp_chunks": dropped_chunks() if dropped_chunks else 0,
                "elapsed_seconds": time.perf_counter() - started,
                "latest_segment": str(latest_segment) if latest_segment else None,
                "frames_jsonl": str(frames_jsonl),
                "chunks_jsonl": str(chunks_jsonl),
            }

        self._write_json(index_path, snapshot())
        chunk_iterator = iter(chunks)
        try:
            for chunk in chunk_iterator:
                if stop_event.is_set() or (max_chunks and processed_chunks >= max_chunks):
                    chunk.inference_path.unlink(missing_ok=True)
                    chunk.unique_path.unlink(missing_ok=True)
                    break
                chunk_output = results_dir / f"chunk_{chunk.index:06d}"
                _, payload = self._run_chunk(
                    chunk,
                    chunk_output,
                    worker_config,
                    target,
                    settings,
                )
                mapped_frames = stitcher.apply(chunk, list(payload.get("frames", [])))
                unique_frames: list[dict[str, Any]] = []
                for frame in mapped_frames:
                    local_frame = int(frame["frame_index"])
                    if local_frame < chunk.overlap_prefix:
                        continue
                    global_frame = chunk.global_start_frame + local_frame
                    jsonl_detections = []
                    for raw_detection in frame["detections"]:
                        detection = dict(raw_detection)
                        mask = detection.get("mask")
                        if mask and not Path(mask).is_absolute():
                            detection["mask"] = str(
                                Path("chunks") / chunk_output.name / str(mask)
                            )
                        jsonl_detections.append(detection)
                    record = {
                        "frame_index": global_frame,
                        "chunk_index": chunk.index,
                        "source_time_seconds": round(global_frame / chunk.fps, 6),
                        "detections": jsonl_detections,
                    }
                    if chunk.unique_started_at_utc:
                        started_at = datetime.fromisoformat(chunk.unique_started_at_utc)
                        offset = (local_frame - chunk.overlap_prefix) / chunk.fps
                        record["captured_at_utc"] = (
                            started_at + timedelta(seconds=offset)
                        ).isoformat()
                    self._append_jsonl(frames_jsonl, record)
                    mask_count += len(frame["detections"])
                    unique_frames.append(
                        {
                            "frame_index": local_frame - chunk.overlap_prefix,
                            "global_frame_index": global_frame,
                            "detections": frame["detections"],
                        }
                    )
                stitched_payload = {
                    **{key: value for key, value in payload.items() if key != "frames"},
                    "source": source_label,
                    "chunk_index": chunk.index,
                    "global_start_frame": chunk.global_start_frame,
                    "overlap_prefix": chunk.overlap_prefix,
                    "frames": unique_frames,
                }
                stitched_manifest = chunk_output / "stitched_manifest.json"
                stitched_manifest.write_text(json.dumps(stitched_payload, indent=2))
                latest_segment = self.renderer(
                    chunk.unique_path,
                    stitched_manifest,
                    segments_dir / f"segment_{chunk.index:06d}.mp4",
                )
                processed_chunks += 1
                processed_frames += chunk.unique_frame_count
                chunk_record = {
                    "chunk_index": chunk.index,
                    "global_start_frame": chunk.global_start_frame,
                    "overlap_prefix": chunk.overlap_prefix,
                    "unique_frames": chunk.unique_frame_count,
                    "unique_started_at_utc": chunk.unique_started_at_utc,
                    "manifest": str(stitched_manifest),
                    "segment": str(latest_segment),
                    "elapsed_seconds": payload.get("elapsed_seconds"),
                    "cuda_peak_allocated_mb": payload.get("cuda_peak_allocated_mb"),
                    "cuda_peak_reserved_mb": payload.get("cuda_peak_reserved_mb"),
                }
                self._append_jsonl(chunks_jsonl, chunk_record)
                chunk.inference_path.unlink(missing_ok=True)
                chunk.unique_path.unlink(missing_ok=True)
                self._write_json(index_path, snapshot())
                yield snapshot()
            status = "stopped" if stop_event.is_set() else "complete"
        except Exception:
            status = "failed"
            self._write_json(index_path, snapshot())
            raise
        finally:
            close_chunks = getattr(chunk_iterator, "close", None)
            if close_chunks is not None:
                close_chunks()
            self._write_json(index_path, snapshot())
        yield snapshot()
