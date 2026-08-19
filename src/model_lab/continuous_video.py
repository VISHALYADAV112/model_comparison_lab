from __future__ import annotations

import gc
import inspect
import json
import math
import queue
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
from PIL import Image

from .adapters.meta_sam3 import (
    MetaSam3Adapter,
    _aligned_meta_tracking_bounds,
    configure_video_predictor,
)
from .bounded_video import redact_rtsp_url
from .config import LabConfig
from .rendering import COLORS, _encode_browser_video


class EndOfVideo(RuntimeError):
    """Raised internally when a finite file or stopped live source is exhausted."""


@dataclass(frozen=True)
class ContinuousVideoSettings:
    """Limits for one native SAM session whose state survives every rolling window."""

    window_frames: int = 60
    grounding_batch_size: int = 1
    max_active_objects: int = 16
    threshold: float = 0.5
    state_history_frames: int = 32
    frame_buffer_frames: int = 96
    rtsp_queue_frames: int = 64
    progress_every_frames: int = 25
    preview_interval_seconds: float = 1.0

    def validate(self) -> ContinuousVideoSettings:
        if self.window_frames < 1:
            raise ValueError("Rolling window size must be positive")
        if not 1 <= self.grounding_batch_size <= 16:
            raise ValueError("Grounding batch size must be between 1 and 16")
        if not 1 <= self.max_active_objects <= 64:
            raise ValueError("Maximum active objects must be between 1 and 64")
        if not 0 < self.threshold < 1:
            raise ValueError("Output threshold must be between 0 and 1")
        if self.state_history_frames < 16:
            raise ValueError("Native SAM state history must retain at least 16 frames")
        if self.frame_buffer_frames < self.state_history_frames + 16:
            raise ValueError(
                "Frame buffer is too small for SAM hot-start and post-processing"
            )
        if self.rtsp_queue_frames < 1:
            raise ValueError("RTSP frame queue must be positive")
        if not 0.1 <= self.preview_interval_seconds <= 10:
            raise ValueError(
                "Dashboard preview interval must be between 0.1 and 10 seconds"
            )
        return self


@dataclass
class FramePacket:
    frame_index: int
    capture_sequence: int
    bgr: np.ndarray
    captured_at_utc: str | None
    tensor: Any = None


class SparseFrameStore:
    """List-like sparse storage for SAM's global per-frame bookkeeping.

    The official runtime allocates several Python lists with one element per
    source frame. A sparse store preserves the integer-indexed interface while
    keeping only non-default entries, which is essential for an unknown-length
    RTSP session.
    """

    def __init__(self, default: Any = None) -> None:
        self.default = default
        self.values: dict[int, Any] = {}

    def __getitem__(self, index: int) -> Any:
        return self.values.get(int(index), self.default)

    def __setitem__(self, index: int, value: Any) -> None:
        index = int(index)
        if value is self.default or (
            isinstance(value, (str, int, float, bool, type(None)))
            and value == self.default
        ):
            self.values.pop(index, None)
        else:
            self.values[index] = value

    def __iter__(self):
        return iter(self.values.values())

    def __len__(self) -> int:
        return len(self.values)

    def clear(self) -> None:
        self.values.clear()

    def prune_before(self, frame_index: int) -> None:
        for key in [key for key in self.values if key < frame_index]:
            self.values.pop(key, None)


class CompactIdMapping:
    """Map global frame IDs to positions in one detector batch without a huge tensor."""

    def __init__(self, global_ids: list[int], device: Any) -> None:
        self.positions = {int(value): index for index, value in enumerate(global_ids)}
        self.device = device

    def __getitem__(self, indices: Any) -> Any:
        import torch

        if hasattr(indices, "detach"):
            raw = indices.detach().cpu().reshape(-1).tolist()
        elif isinstance(indices, (list, tuple)):
            raw = list(indices)
        else:
            raw = [indices]
        mapped = [self.positions[int(value)] for value in raw]
        return torch.tensor(mapped, dtype=torch.long, device=self.device)


class RollingFrameLoader:
    """Sequential file/RTSP decoder with a bounded cache and lazy SAM preprocessing."""

    _UNBOUNDED_LENGTH = 2_147_483_647

    def __init__(
        self,
        source: str,
        *,
        stop_event: threading.Event,
        image_size: int = 1008,
        frame_buffer_frames: int = 96,
        hard_frame_limit: int = 0,
        rtsp: bool = False,
        maximum_minutes: float = 0,
        rtsp_queue_frames: int = 64,
        reconnect_attempts: int = 5,
        reconnect_delay_seconds: float = 2,
        tensor_factory: Callable[[np.ndarray], Any] | None = None,
    ) -> None:
        self.source = source
        self.stop_event = stop_event
        self.image_size = int(image_size)
        self.frame_buffer_frames = int(frame_buffer_frames)
        self.hard_frame_limit = max(0, int(hard_frame_limit))
        self.rtsp = bool(rtsp)
        self.maximum_minutes = max(0.0, float(maximum_minutes))
        self.reconnect_attempts = max(0, int(reconnect_attempts))
        self.reconnect_delay_seconds = max(0.0, float(reconnect_delay_seconds))
        self.tensor_factory = tensor_factory or self._preprocess
        self.cache: OrderedDict[int, FramePacket] = OrderedDict()
        self.next_frame_index = 0
        self.capture_sequence = 0
        self.dropped_capture_frames = 0
        self.reconnect_count = 0
        self._closed = threading.Event()
        self._producer_done = threading.Event()
        self._producer_error: Exception | None = None
        self._queue: queue.Queue[tuple[int, np.ndarray, str] | None] | None = None
        self._thread: threading.Thread | None = None
        self._capture = self._open_capture()

        import cv2

        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 25.0)
        if not math.isfinite(self.fps) or not 1 <= self.fps <= 240:
            self.fps = 25.0
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.width <= 0 or self.height <= 0:
            self._capture.release()
            raise RuntimeError("Video source did not report a valid frame size")
        reported_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.rtsp:
            self.total_frames = self._UNBOUNDED_LENGTH
        elif self.hard_frame_limit:
            self.total_frames = (
                min(reported_frames, self.hard_frame_limit)
                if reported_frames
                else self.hard_frame_limit
            )
        else:
            self.total_frames = reported_frames or self._UNBOUNDED_LENGTH

        if self.rtsp:
            self._queue = queue.Queue(maxsize=max(1, int(rtsp_queue_frames)))
            self._thread = threading.Thread(
                target=self._capture_rtsp,
                name="sam31-rolling-rtsp-capture",
                daemon=True,
            )
            self._thread.start()

    @property
    def device(self):
        import torch

        return torch.device("cpu")

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (len(self), 3, self.image_size, self.image_size)

    def __len__(self) -> int:
        return self.total_frames

    def _open_capture(self):
        import cv2

        capture = cv2.VideoCapture()
        parameters: list[int] = []
        if self.rtsp:
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
            label = (
                redact_rtsp_url(self.source) if self.rtsp else Path(self.source).name
            )
            raise RuntimeError(f"Cannot open video source {label}")
        return capture

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _put_rtsp(self, item: tuple[int, np.ndarray, str]) -> None:
        assert self._queue is not None
        while not self._closed.is_set():
            try:
                self._queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self.dropped_capture_frames += 1
                except queue.Empty:
                    continue

    def _reconnect(self):
        last_error: Exception | None = None
        for _ in range(self.reconnect_attempts):
            if (
                self._closed.wait(self.reconnect_delay_seconds)
                or self.stop_event.is_set()
            ):
                return None
            try:
                capture = self._open_capture()
                self.reconnect_count += 1
                return capture
            except RuntimeError as exc:
                last_error = exc
        if last_error is not None:
            self._producer_error = RuntimeError(
                f"RTSP reconnect failed for {redact_rtsp_url(self.source)}"
            )
        return None

    def _capture_rtsp(self) -> None:
        capture = self._capture
        started = time.monotonic()
        try:
            while not self._closed.is_set() and not self.stop_event.is_set():
                if (
                    self.maximum_minutes
                    and time.monotonic() - started >= self.maximum_minutes * 60
                ):
                    break
                ok, frame = capture.read()
                if not ok:
                    capture.release()
                    replacement = self._reconnect()
                    if replacement is None:
                        break
                    capture = replacement
                    self._capture = capture
                    continue
                sequence = self.capture_sequence
                self.capture_sequence += 1
                self._put_rtsp((sequence, frame, self._utc_now()))
        except Exception as exc:  # noqa: BLE001
            self._producer_error = exc
        finally:
            capture.release()
            self._producer_done.set()

    def _next_raw_frame(self) -> tuple[int, np.ndarray, str | None]:
        if self.hard_frame_limit and self.next_frame_index >= self.hard_frame_limit:
            raise EndOfVideo("Requested frame limit reached")
        if self.rtsp:
            assert self._queue is not None
            while True:
                try:
                    item = self._queue.get(timeout=0.25)
                    if item is not None:
                        sequence, frame, timestamp = item
                        return sequence, frame, timestamp
                except queue.Empty:
                    if self._producer_done.is_set():
                        if self._producer_error is not None:
                            raise RuntimeError(
                                str(self._producer_error)
                            ) from self._producer_error
                        raise EndOfVideo("RTSP capture ended")
                    if self._closed.is_set():
                        raise EndOfVideo("RTSP capture stopped")
        ok, frame = self._capture.read()
        if not ok:
            self.total_frames = self.next_frame_index
            raise EndOfVideo("Video file ended")
        sequence = self.capture_sequence
        self.capture_sequence += 1
        return sequence, frame, None

    def _preprocess(self, bgr: np.ndarray) -> Any:
        """Match the pinned SAM 3.1 OpenCV video loader one frame at a time."""
        import cv2
        import torch

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_CUBIC,
        )
        # Keep these operations identical to Meta's pinned
        # load_video_frames_from_video_file_using_cv2 implementation so a
        # rolling run does not silently change model inputs versus its ordinary
        # complete-file path.
        values = resized.astype(np.float32)
        tensor = torch.from_numpy(values).permute(2, 0, 1)
        mean = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float16)[:, None, None]
        std = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float16)[:, None, None]
        tensor.sub_(mean)
        tensor.div_(std)
        return tensor

    def __getitem__(self, index: int) -> Any:
        index = int(index)
        if index < 0:
            raise IndexError(index)
        cached = self.cache.get(index)
        if cached is not None:
            if cached.tensor is None:
                cached.tensor = self.tensor_factory(cached.bgr)
            return cached.tensor
        if index < self.next_frame_index:
            raise RuntimeError(f"SAM requested released frame {index}")
        while self.next_frame_index <= index:
            sequence, frame, timestamp = self._next_raw_frame()
            packet = FramePacket(
                frame_index=self.next_frame_index,
                capture_sequence=sequence,
                bgr=frame,
                captured_at_utc=timestamp,
            )
            self.cache[self.next_frame_index] = packet
            self.next_frame_index += 1
            if len(self.cache) > self.frame_buffer_frames:
                raise RuntimeError(
                    "SAM output fell farther behind than the fixed rolling frame buffer. "
                    "Increase bounded_video.frame_buffer_frames after checking CPU RAM."
                )
        packet = self.cache[index]
        packet.tensor = self.tensor_factory(packet.bgr)
        return packet.tensor

    def prepare(
        self, start: int, count: int, *, lookahead: bool = False
    ) -> tuple[int, bool]:
        requested = count + (1 if lookahead else 0)
        available = 0
        ended = False
        for index in range(start, start + requested):
            try:
                self[index]
            except EndOfVideo:
                ended = True
                break
            available += 1
        processable = min(count, available)
        if lookahead and available <= count:
            ended = True
        return processable, ended

    def packet(self, frame_index: int) -> FramePacket:
        try:
            return self.cache[int(frame_index)]
        except KeyError as exc:
            raise RuntimeError(
                f"Original frame {frame_index} is no longer buffered"
            ) from exc

    def first_pil_image(self) -> Image.Image:
        self[0]
        import cv2

        rgb = cv2.cvtColor(self.packet(0).bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def release_through(self, frame_index: int) -> None:
        for index in [index for index in self.cache if index <= frame_index]:
            self.cache.pop(index, None)

    def release_tensors_through(self, frame_index: int) -> None:
        for index, packet in self.cache.items():
            if index <= frame_index:
                packet.tensor = None

    def close(self) -> None:
        self._closed.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=12)
        try:
            self._capture.release()
        except Exception as exc:  # noqa: BLE001
            print(f"[continuous-sam31] capture close warning: {exc}", flush=True)
        self.cache.clear()


def _rolling_get_img_feats(self: Any, backbone_out: dict[str, Any], img_ids: Any):
    """SAM image-backbone adapter using a compact mapping for global stream IDs."""
    import torch

    container = backbone_out["img_batch_all_stages"]
    loader = getattr(container, "tensors", None)
    if not isinstance(loader, RollingFrameLoader):
        original = self._model_lab_original_get_img_feats
        return original(backbone_out, img_ids)

    if "backbone_fpn" in backbone_out:
        local_ids = backbone_out["id_mapping"][img_ids]
        vis_feats = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vis_pos_enc = backbone_out["vision_pos_enc"][-self.num_feature_levels :]
        sizes = [value.shape[-2:] for value in vis_pos_enc]
        features = [value[local_ids].flatten(2).permute(2, 0, 1) for value in vis_feats]
        positions = [
            value[local_ids].flatten(2).permute(2, 0, 1) for value in vis_pos_enc
        ]
        return backbone_out, features, positions, sizes

    global_ids = [int(value) for value in img_ids.detach().cpu().reshape(-1).tolist()]
    unique_ids = list(dict.fromkeys(global_ids))
    images = torch.stack([loader[index] for index in unique_ids]).to(
        dtype=torch.float32,
        device=self.device,
    )
    mapping = CompactIdMapping(unique_ids, self.device)
    local_ids = mapping[img_ids]
    computed = {
        **backbone_out,
        **self.backbone.forward_image(images),
        "id_mapping": mapping,
    }
    vis_feats = computed["backbone_fpn"][-self.num_feature_levels :]
    vis_pos_enc = computed["vision_pos_enc"][-self.num_feature_levels :]
    sizes = [value.shape[-2:] for value in vis_pos_enc]
    features = [value[local_ids].flatten(2).permute(2, 0, 1) for value in vis_feats]
    positions = [value[local_ids].flatten(2).permute(2, 0, 1) for value in vis_pos_enc]
    return computed, features, positions, sizes


@contextmanager
def rolling_backbone_loader(model: Any) -> Iterator[None]:
    detector = model.detector
    original = detector._get_img_feats
    detector._model_lab_original_get_img_feats = original
    detector._get_img_feats = MethodType(_rolling_get_img_feats, detector)
    try:
        yield
    finally:
        detector._get_img_feats = original
        try:
            del detector._model_lab_original_get_img_feats
        except AttributeError:
            pass


def _prune_mapping_before(value: Any, cutoff: int) -> None:
    if isinstance(value, SparseFrameStore):
        value.prune_before(cutoff)
    elif isinstance(value, dict):
        for key in [key for key in value if isinstance(key, int) and key < cutoff]:
            value.pop(key, None)


def _prune_tracker_state(tracker_state: dict[str, Any], cutoff: int) -> None:
    output_dict = tracker_state.get("output_dict", {})
    point_inputs = tracker_state.get("point_inputs_per_obj", {})
    mask_inputs = tracker_state.get("mask_inputs_per_obj", {})
    per_object_outputs = tracker_state.get("output_dict_per_obj", {})
    retained_input_frames: set[int] = set()

    # Periodic detector reconditioning adds another GPU mask input every 16
    # frames. The pinned tracker attends to at most four conditioning frames,
    # so retaining all older detector inputs only leaks memory. Explicit point
    # prompts remain protected in case this helper is reused for interactivity.
    object_indices = set(point_inputs) | set(mask_inputs) | set(per_object_outputs)
    retained_by_object: dict[int, set[int]] = {}
    for object_index in object_indices:
        point_frames = {int(frame) for frame in point_inputs.get(object_index, {})}
        object_masks = mask_inputs.get(object_index, {})
        mask_frames = sorted(int(frame) for frame in object_masks)
        retained = point_frames | set(mask_frames[-4:])
        retained_by_object[int(object_index)] = retained
        retained_input_frames.update(retained)
        if isinstance(object_masks, dict):
            for frame in list(object_masks):
                if int(frame) not in retained:
                    object_masks.pop(frame, None)

    non_cond = output_dict.get("non_cond_frame_outputs", {})
    if isinstance(non_cond, dict):
        for frame in list(non_cond):
            if (
                isinstance(frame, int)
                and frame < cutoff
                and frame not in retained_input_frames
            ):
                non_cond.pop(frame, None)

    cond = output_dict.get("cond_frame_outputs", {})
    if isinstance(cond, dict):
        keys = sorted(key for key in cond if isinstance(key, int))
        keep = retained_input_frames | set(keys[-4:])
        for key in keys:
            if key not in keep:
                cond.pop(key, None)

    for per_object_key in ("output_dict_per_obj", "temp_output_dict_per_obj"):
        for object_index, per_object in tracker_state.get(per_object_key, {}).items():
            object_non_cond = per_object.get("non_cond_frame_outputs", {})
            if isinstance(object_non_cond, dict):
                for frame in list(object_non_cond):
                    if (
                        isinstance(frame, int)
                        and frame < cutoff
                        and frame
                        not in retained_by_object.get(int(object_index), set())
                    ):
                        object_non_cond.pop(frame, None)
            object_cond = per_object.get("cond_frame_outputs", {})
            if isinstance(object_cond, dict):
                keys = sorted(key for key in object_cond if isinstance(key, int))
                keep = retained_by_object.get(int(object_index), set()) | set(keys[-4:])
                for key in keys:
                    if key not in keep:
                        object_cond.pop(key, None)

    remaining_inputs: set[int] = set()
    for collection in (point_inputs, mask_inputs):
        for per_object in collection.values():
            remaining_inputs.update(int(frame) for frame in per_object)

    # Meta asserts before every propagation that consolidated prompt-frame
    # indices exactly equal the union of point and mask input indices. Detector
    # reconditioning prompts are commonly stored as non-conditioning outputs,
    # so both consolidated sets must follow the retained inputs rather than a
    # generic age cutoff.
    consolidated = tracker_state.get("consolidated_frame_inds", {})
    cond_consolidated = consolidated.get("cond_frame_outputs", set())
    non_cond_consolidated = consolidated.get("non_cond_frame_outputs", set())
    if isinstance(cond_consolidated, set) and isinstance(non_cond_consolidated, set):
        cond_consolidated.intersection_update(remaining_inputs)
        non_cond_consolidated.intersection_update(remaining_inputs)
        known = cond_consolidated | non_cond_consolidated
        for frame in remaining_inputs - known:
            if frame in cond:
                cond_consolidated.add(frame)
            elif frame in non_cond:
                non_cond_consolidated.add(frame)

    if "first_ann_frame_idx" in tracker_state:
        tracker_state["first_ann_frame_idx"] = min(remaining_inputs, default=None)

    _prune_mapping_before(tracker_state.get("frames_already_tracked", {}), cutoff)


def _int_set(value: Any) -> set[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return {int(item) for item in np.asarray(value).reshape(-1).tolist()}


def prune_continuous_state(
    inference_state: dict[str, Any],
    *,
    processed_frame: int,
    history_frames: int,
) -> dict[str, int]:
    """Evict only history older than SAM 3.1's native attention/pointer windows."""
    cutoff = max(0, int(processed_frame) - int(history_frames) + 1)
    for key in (
        "previous_stages_out",
        "per_frame_raw_point_input",
        "per_frame_raw_box_input",
        "per_frame_visual_prompt",
        "per_frame_geometric_prompt",
        "per_frame_cur_step",
    ):
        _prune_mapping_before(inference_state.get(key), cutoff)

    cached_outputs = inference_state.get("cached_frame_outputs", {})
    _prune_mapping_before(cached_outputs, cutoff)
    tracker_states = inference_state.get("sam2_inference_states", [])
    for tracker_state in tracker_states:
        _prune_tracker_state(tracker_state, cutoff)

    # The tracker keeps the maximum assigned ID separately, so retired object
    # records are unnecessary after they have also left the hot-start buffers.
    generator_state = inference_state.get("generator_state", {})
    buffered_ids: set[int] = set()
    for _, output in generator_state.get("hotstart_buffer", []):
        buffered_ids.update(int(value) for value in output.get("obj_id_to_mask", {}))
    for _, output, _ in generator_state.get("postprocess_yield_list", []):
        buffered_ids.update(int(value) for value in output.get("obj_id_to_mask", {}))

    metadata = inference_state.get("tracker_metadata", {})
    for key, value in metadata.items():
        if "frame_wise" in str(key):
            _prune_mapping_before(value, cutoff)
    rank0 = metadata.get("rank0_metadata", {}) if isinstance(metadata, dict) else {}
    _prune_mapping_before(rank0.get("suppressed_obj_ids", {}), cutoff)
    _prune_mapping_before(
        generator_state.get("unconfirmed_obj_ids_per_frame", {}), cutoff
    )
    active_ids = _int_set(metadata.get("obj_ids_all_gpu", []))
    retained_ids = active_ids | buffered_ids
    for key in ("obj_id_to_score", "obj_id_to_last_occluded"):
        values = metadata.get(key)
        if isinstance(values, dict):
            for object_id in list(values):
                if int(object_id) not in retained_ids:
                    values.pop(object_id, None)
    for key in ("obj_first_frame_idx", "unmatched_frame_inds", "trk_keep_alive"):
        values = rank0.get(key)
        if isinstance(values, dict):
            for object_id in list(values):
                if int(object_id) not in retained_ids:
                    values.pop(object_id, None)
    removed = rank0.get("removed_obj_ids")
    if isinstance(removed, set):
        removed.intersection_update(retained_ids)
    hotstart_removed = generator_state.get("hotstart_removed_obj_ids")
    if isinstance(hotstart_removed, set):
        hotstart_removed.intersection_update(retained_ids)
    overlap_history = rank0.get("overlap_pair_to_frame_inds")
    if isinstance(overlap_history, dict):
        for pair in list(overlap_history):
            if not {int(value) for value in pair}.issubset(retained_ids):
                overlap_history.pop(pair, None)

    if isinstance(tracker_states, list):
        tracker_states[:] = [state for state in tracker_states if state.get("obj_ids")]
    return {
        "cutoff": cutoff,
        "cached_output_frames": len(cached_outputs)
        if isinstance(cached_outputs, dict)
        else 0,
        "tracker_states": len(tracker_states),
        "active_objects": len(active_ids),
    }


def _disable_meta_vos_autotrim(model: Any) -> None:
    """Keep Meta's first-prompt-only VOS optimization out of text tracking."""
    tracker_model = getattr(model.tracker, "model", model.tracker)
    if hasattr(tracker_model, "trim_past_non_cond_mem_for_eval"):
        tracker_model.trim_past_non_cond_mem_for_eval = False


class TrackArchive:
    """Disk-backed SAM track catalogue and best crop for later human/ReID matching."""

    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / "track_identities.sqlite3"
        self.crop_dir = output_dir / "identity_candidates"
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """
            CREATE TABLE tracks (
                sam_track_id INTEGER PRIMARY KEY,
                first_frame INTEGER NOT NULL,
                last_frame INTEGER NOT NULL,
                best_score REAL NOT NULL,
                best_crop TEXT,
                verified_identity TEXT,
                embedding_model TEXT,
                embedding BLOB
            )
            """
        )
        self.connection.commit()
        self.pending_updates = 0

    def update(
        self,
        frame_index: int,
        detections: list[dict[str, Any]],
        frame_bgr: np.ndarray,
    ) -> None:
        import cv2

        for detection in detections:
            track_id = int(detection["instance_id"])
            score = float(detection["score"])
            row = self.connection.execute(
                "SELECT best_score, best_crop FROM tracks WHERE sam_track_id = ?",
                (track_id,),
            ).fetchone()
            crop_path = row[1] if row else None
            if row is None or score > float(row[0]):
                x0, y0, x1, y1 = (round(value) for value in detection["box"])
                height, width = frame_bgr.shape[:2]
                x0, x1 = max(0, x0), min(width, x1)
                y0, y1 = max(0, y0), min(height, y1)
                if x1 > x0 and y1 > y0:
                    candidate = self.crop_dir / f"sam_track_{track_id:08d}.jpg"
                    cv2.imwrite(str(candidate), frame_bgr[y0:y1, x0:x1])
                    crop_path = str(candidate.relative_to(self.path.parent))
            if row is None:
                self.connection.execute(
                    """
                    INSERT INTO tracks (
                        sam_track_id, first_frame, last_frame, best_score, best_crop
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (track_id, frame_index, frame_index, score, crop_path),
                )
            else:
                best_score = max(score, float(row[0]))
                self.connection.execute(
                    """
                    UPDATE tracks SET last_frame = ?, best_score = ?, best_crop = ?
                    WHERE sam_track_id = ?
                    """,
                    (frame_index, best_score, crop_path, track_id),
                )
        self.pending_updates += 1
        if self.pending_updates >= 25:
            self.connection.commit()
            self.pending_updates = 0

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


class ContinuousResultWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        width: int,
        height: int,
        fps: float,
        source_path: Path | None,
    ) -> None:
        import cv2

        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.source_path = source_path
        self.masks_dir = output_dir / "masks"
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.frames_path = output_dir / "frames.jsonl"
        self.live_preview = output_dir / "live_preview.jpg"
        self.raw_video = output_dir / "annotated.opencv.mp4"
        self.final_video = output_dir / "annotated.mp4"
        self.written_frames = 0
        self.video_writer = cv2.VideoWriter(
            str(self.raw_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not self.video_writer.isOpened():
            raise RuntimeError("OpenCV could not create the continuous annotated video")
        self.archive = TrackArchive(output_dir)

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def write(
        self, frame_index: int, outputs: dict[str, Any], packet: FramePacket
    ) -> int:
        import cv2

        frame = packet.bgr.copy()
        detections: list[dict[str, Any]] = []
        masks = np.asarray(outputs.get("out_binary_masks", []))
        for object_id, score, box_xywh, mask in zip(
            np.asarray(outputs.get("out_obj_ids", [])),
            np.asarray(outputs.get("out_probs", [])),
            np.asarray(outputs.get("out_boxes_xywh", [])),
            masks,
        ):
            track_id = int(object_id)
            x, y, box_width, box_height = [float(value) for value in box_xywh]
            box = [
                x * self.width,
                y * self.height,
                (x + box_width) * self.width,
                (y + box_height) * self.height,
            ]
            binary = np.asarray(mask).squeeze().astype(bool)
            mask_path = (
                self.masks_dir / f"frame_{frame_index:09d}_object_{track_id:08d}.png"
            )
            Image.fromarray(binary.astype(np.uint8) * 255, mode="L").save(mask_path)
            detections.append(
                {
                    "box": box,
                    "score": float(score),
                    "iou_score": None,
                    "instance_id": track_id,
                    "mask": str(mask_path.relative_to(self.output_dir)),
                }
            )

            color = COLORS[track_id % len(COLORS)]
            color_bgr = (color[2], color[1], color[0])
            if binary.shape != frame.shape[:2]:
                binary = cv2.resize(
                    binary.astype(np.uint8),
                    (self.width, self.height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            overlay = np.empty_like(frame)
            overlay[:] = color_bgr
            frame[binary] = (0.58 * frame[binary] + 0.42 * overlay[binary]).astype(
                np.uint8
            )
            x0, y0, x1, y1 = (int(value) for value in box)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color_bgr, 2)
            cv2.putText(
                frame,
                f"id={track_id} {float(score):.2f}",
                (x0, max(15, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_bgr,
                1,
                cv2.LINE_AA,
            )

        record: dict[str, Any] = {
            "frame_index": frame_index,
            "capture_sequence": packet.capture_sequence,
            "source_time_seconds": round(packet.capture_sequence / self.fps, 6),
            "detections": detections,
        }
        if packet.captured_at_utc:
            record["captured_at_utc"] = packet.captured_at_utc
        self._append_jsonl(self.frames_path, record)
        self.archive.update(frame_index, detections, packet.bgr)
        self.video_writer.write(frame)
        self.written_frames += 1
        encoded, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not encoded:
            raise RuntimeError("OpenCV could not encode the live annotated preview")
        temporary_preview = self.live_preview.with_suffix(".jpg.tmp")
        temporary_preview.write_bytes(jpeg.tobytes())
        temporary_preview.replace(self.live_preview)
        return len(detections)

    def close(self) -> Path | None:
        self.video_writer.release()
        self.archive.close()
        if self.written_frames == 0:
            self.raw_video.unlink(missing_ok=True)
            return None
        _encode_browser_video(self.raw_video, self.source_path, self.final_video)
        return self.final_video


class ContinuousSamRunner:
    """One SAM 3.1 model/session for an entire file or RTSP lifetime."""

    def __init__(self, config: LabConfig) -> None:
        self.config = config

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(path)

    @staticmethod
    def _cuda_memory() -> dict[str, float | None]:
        try:
            import torch

            megabyte = 1024 * 1024
            return {
                "cuda_allocated_mb": round(torch.cuda.memory_allocated() / megabyte, 2),
                "cuda_reserved_mb": round(torch.cuda.memory_reserved() / megabyte, 2),
                "cuda_peak_allocated_mb": round(
                    torch.cuda.max_memory_allocated() / megabyte, 2
                ),
                "cuda_peak_reserved_mb": round(
                    torch.cuda.max_memory_reserved() / megabyte, 2
                ),
            }
        except (ImportError, RuntimeError):
            return {
                "cuda_allocated_mb": None,
                "cuda_reserved_mb": None,
                "cuda_peak_allocated_mb": None,
                "cuda_peak_reserved_mb": None,
            }

    @staticmethod
    def _sparse_state(inference_state: dict[str, Any]) -> None:
        for key in (
            "previous_stages_out",
            "per_frame_raw_point_input",
            "per_frame_raw_box_input",
            "per_frame_visual_prompt",
            "per_frame_geometric_prompt",
        ):
            inference_state[key] = SparseFrameStore(None)
        inference_state["per_frame_cur_step"] = SparseFrameStore(0)

    def _initialize_state(
        self,
        predictor: Any,
        loader: RollingFrameLoader,
        target: str,
    ) -> dict[str, Any]:
        import torch

        model = predictor.model
        with torch.inference_mode():
            inference_state = model.init_state(
                resource_path=[loader.first_pil_image()],
                offload_video_to_cpu=True,
                async_loading_frames=False,
            )
            inference_state["num_frames"] = len(loader)
            inference_state["is_image_only"] = False
            inference_state["input_batch"].img_batch.tensors = loader
            inference_state["input_batch"].find_text_batch[0] = target
            for find_input in inference_state["input_batch"].find_inputs:
                find_input.text_ids[...] = model.TEXT_ID_FOR_TEXT
            inference_state["text_prompt"] = target
            if not hasattr(model, "_init_backbone_out"):
                raise RuntimeError(
                    "The installed SAM 3.1 runtime lacks the expected language-cache API. "
                    "Re-run scripts/install_meta_sam3.sh to install the pinned commit."
                )
            # Calling add_prompt would loop over the declared (potentially endless)
            # stream length. Initialize the same text-only backbone cache directly;
            # image features remain lazy and are supplied by RollingFrameLoader.
            inference_state["backbone_out"] = model._init_backbone_out(inference_state)
            self._sparse_state(inference_state)
            inference_state["generator_state"] = {
                "hotstart_buffer": [],
                "hotstart_removed_obj_ids": set(),
                "unconfirmed_obj_ids_per_frame": {},
                "postprocess_yield_list": [],
            }
            # Meta documents this flag only for first-frame-prompted VOS. The
            # text-grounding tracker periodically adds detector mask prompts,
            # and its direct-mask outputs omit fields that the auto-trimmer
            # assumes are present. Our window-boundary pruner provides the
            # bounded-memory behavior without invoking that incompatible path.
            _disable_meta_vos_autotrim(model)
        return inference_state

    def run(
        self,
        loader: RollingFrameLoader,
        output_dir: Path,
        *,
        source_kind: str,
        source_label: str,
        source_path: Path | None,
        target: str,
        settings: ContinuousVideoSettings,
        stop_event: threading.Event,
        maximum_windows: int = 0,
    ) -> Iterator[dict[str, Any]]:
        settings.validate()
        if not target.strip():
            raise ValueError("Describe the object or concept to track")
        MetaSam3Adapter._require_cuda()
        if not self.config.sam3_official_video_model.exists():
            raise FileNotFoundError(
                f"Missing {self.config.sam3_official_video_model}. "
                "Run: model-lab models download --model sam3-official"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        index_path = output_dir / "index.json"
        if index_path.exists():
            raise FileExistsError(
                f"Continuous-video output already exists: {output_dir}"
            )

        import torch
        from sam3.model.sam3_multiplex_tracking import Sam3MultiplexTrackingProd
        from sam3.model_builder import build_sam3_predictor

        if (
            "is_last_batch"
            not in inspect.signature(
                Sam3MultiplexTrackingProd.propagate_in_video
            ).parameters
        ):
            raise RuntimeError(
                "The installed SAM 3.1 runtime lacks Meta's persistent batched-video API. "
                "Re-run scripts/install_meta_sam3.sh to install the pinned commit."
            )

        model_settings = self.config.raw["sam3"]
        torch.cuda.reset_peak_memory_stats()
        predictor = build_sam3_predictor(
            checkpoint_path=str(self.config.sam3_official_video_model),
            version="sam3.1",
            compile=bool(model_settings.get("compile", False)),
            max_num_objects=settings.max_active_objects,
            multiplex_count=int(model_settings.get("multiplex_count", 16)),
            use_fa3=bool(model_settings.get("use_flash_attention_3", False)),
        )
        configure_video_predictor(predictor, settings.grounding_batch_size)
        inference_state: dict[str, Any] | None = None
        writer: ContinuousResultWriter | None = None
        started = time.perf_counter()
        last_progress = started
        last_preview_update = 0.0
        processed_frames = 0
        processed_windows = 0
        mask_count = 0
        highest_object_id = -1
        latest_video: Path | None = None
        latest_preview: Path | None = None
        status = "running"
        last_pruning = {"cutoff": 0, "cached_output_frames": 0, "tracker_states": 0}

        def snapshot() -> dict[str, Any]:
            return {
                "schema_version": 2,
                "engine": "continuous_native_sam31",
                "mode": source_kind,
                "model": "SAM 3.1 Object Multiplex",
                "source": source_label,
                "target": target.strip(),
                "status": status,
                "settings": asdict(settings),
                "processed_chunks": processed_windows,
                "processed_windows": processed_windows,
                "processed_frames": processed_frames,
                "unique_objects": highest_object_id + 1,
                "frame_level_masks": mask_count,
                "dropped_rtsp_frames": loader.dropped_capture_frames,
                "dropped_rtsp_chunks": 0,
                "rtsp_reconnects": loader.reconnect_count,
                "elapsed_seconds": time.perf_counter() - started,
                "latest_segment": str(latest_video) if latest_video else None,
                "live_preview": str(latest_preview) if latest_preview else None,
                "continuous_video": str(output_dir / "annotated.mp4"),
                "frames_jsonl": str(output_dir / "frames.jsonl"),
                "identity_database": str(output_dir / "track_identities.sqlite3"),
                "identity_database_scope": (
                    "SAM track IDs and best crops for human review; automatic cross-visit "
                    "identity requires a face/person-ReID embedding model"
                ),
                "rolling_state": last_pruning,
                **self._cuda_memory(),
            }

        def consume_responses(
            responses: Iterator[tuple[Any, dict[str, Any]]],
        ) -> Iterator[dict[str, Any]]:
            nonlocal highest_object_id, last_preview_update, last_progress
            nonlocal latest_preview, mask_count, processed_frames
            assert writer is not None
            for frame_index, outputs in responses:
                frame_index = int(frame_index)
                packet = loader.packet(frame_index)
                mask_count += writer.write(frame_index, outputs, packet)
                latest_preview = writer.live_preview
                ids = np.asarray(outputs.get("out_obj_ids", []))
                if ids.size:
                    highest_object_id = max(highest_object_id, int(ids.max()))
                processed_frames += 1
                loader.release_through(frame_index)
                now = time.perf_counter()
                if (
                    processed_frames % settings.progress_every_frames == 0
                    or now - last_progress >= 5
                ):
                    memory = self._cuda_memory()
                    print(
                        f"[continuous-sam31] wrote frame {frame_index}; "
                        f"{processed_frames} total, {mask_count} masks, "
                        f"CUDA {memory['cuda_allocated_mb']} MB allocated / "
                        f"{memory['cuda_peak_allocated_mb']} MB peak",
                        flush=True,
                    )
                    last_progress = now
                if now - last_preview_update >= settings.preview_interval_seconds:
                    self._write_json(index_path, snapshot())
                    last_preview_update = now
                    yield snapshot()

        self._write_json(index_path, snapshot())
        try:
            inference_state = self._initialize_state(predictor, loader, target.strip())
            writer = ContinuousResultWriter(
                output_dir,
                width=loader.width,
                height=loader.height,
                fps=loader.fps,
                source_path=source_path,
            )
            model = predictor.model
            frame_start = 0
            file_limit = len(loader)
            if maximum_windows:
                file_limit = min(file_limit, maximum_windows * settings.window_frames)

            with rolling_backbone_loader(model):
                while frame_start < file_limit and not stop_event.is_set():
                    if source_kind == "rtsp":
                        requested = settings.grounding_batch_size
                        count, source_ended = loader.prepare(
                            frame_start,
                            requested,
                            lookahead=True,
                        )
                    else:
                        requested = min(
                            settings.window_frames, file_limit - frame_start
                        )
                        # File length is known from the container. Do not pre-decode
                        # a whole progress window; SAM pulls only its current detector
                        # microbatch through RollingFrameLoader.__getitem__.
                        count = requested
                        source_ended = frame_start + count >= file_limit
                    if count <= 0:
                        break

                    batch_end = frame_start + count
                    is_last_batch = source_ended or batch_end >= file_limit
                    if is_last_batch:
                        inference_state["num_frames"] = batch_end
                        for tracker_state in inference_state.get(
                            "sam2_inference_states", []
                        ):
                            tracker_state["num_frames"] = batch_end

                    print(
                        f"[continuous-sam31] window {processed_windows + 1}: "
                        f"frames {frame_start}-{batch_end - 1}, one persistent session",
                        flush=True,
                    )
                    with _aligned_meta_tracking_bounds(predictor, enabled=True):
                        responses = Sam3MultiplexTrackingProd.propagate_in_video(
                            model,
                            inference_state=inference_state,
                            start_frame_idx=frame_start,
                            max_frame_num_to_track=count - 1,
                            reverse=False,
                            output_prob_thresh=settings.threshold,
                            is_last_batch=is_last_batch,
                        )
                        yield from consume_responses(responses)

                    processed_windows += 1
                    last_pruning = prune_continuous_state(
                        inference_state,
                        processed_frame=batch_end - 1,
                        history_frames=settings.state_history_frames,
                    )
                    loader.release_tensors_through(batch_end - 1)
                    self._write_json(index_path, snapshot())
                    yield snapshot()
                    frame_start = batch_end
                    if is_last_batch:
                        break

                # Stop can arrive while a progress window is running. Meta's
                # production generator intentionally keeps its last hot-start
                # frames for the next call, so finish that buffer with an empty
                # final batch rather than silently dropping those output frames.
                generator_state = inference_state.get("generator_state", {})
                if stop_event.is_set() and generator_state.get("hotstart_buffer"):
                    inference_state["num_frames"] = frame_start
                    for tracker_state in inference_state.get(
                        "sam2_inference_states", []
                    ):
                        tracker_state["num_frames"] = frame_start
                    flush_responses = Sam3MultiplexTrackingProd.propagate_in_video(
                        model,
                        inference_state=inference_state,
                        start_frame_idx=frame_start,
                        max_frame_num_to_track=-1,
                        reverse=False,
                        output_prob_thresh=settings.threshold,
                        is_last_batch=True,
                    )
                    yield from consume_responses(flush_responses)
                    last_pruning = prune_continuous_state(
                        inference_state,
                        processed_frame=max(0, frame_start - 1),
                        history_frames=settings.state_history_frames,
                    )
                    self._write_json(index_path, snapshot())

            status = "stopped" if stop_event.is_set() else "complete"
            latest_video = writer.close()
            writer = None
        except Exception:
            status = "failed"
            self._write_json(index_path, snapshot())
            raise
        except KeyboardInterrupt:
            status = "stopped"
            self._write_json(index_path, snapshot())
            raise
        finally:
            loader.close()
            if writer is not None:
                try:
                    latest_video = writer.close()
                    writer = None
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[continuous-sam31] partial output finalization warning: {exc}",
                        flush=True,
                    )
            if inference_state is not None:
                inference_state.clear()
            del predictor
            gc.collect()
            torch.cuda.empty_cache()
            self._write_json(index_path, snapshot())
        yield snapshot()
