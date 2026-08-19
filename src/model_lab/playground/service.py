from __future__ import annotations

import json
import shutil
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..adapters.meta_sam3 import MetaSam3Adapter
from ..adapters.sam3_cpp import Sam3CppAdapter, parse_boxes, parse_points
from ..bounded_video import (
    BoundedSamRunner,
    BoundedVideoSettings,
    OpenCVChunkSource,
    RtspChunkQueue,
    redact_rtsp_url,
    validate_rtsp_url,
)
from ..compare import compare_image
from ..config import LabConfig
from ..doctor import doctor_report
from ..downloader import download_models, model_status
from ..rendering import manifest_mask_files, render_sam_manifest, render_video_manifest
from .sessions import MetaVideoSessionController


def _path(value: str | Path | None) -> Path:
    if not value:
        raise ValueError("Upload a file first")
    return Path(value).expanduser().resolve()


def _records(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.splitlines() if item.strip()]


def comparison_summary(payload: dict) -> str:
    display_names = {"yolo": "YOLO26-L", "rfdetr": "RF-DETR Large", "sam3": "SAM 3"}
    lines = [
        "### Comparison finished",
        "| Model | Objects / masks | Time | Smallest detected side |",
        "|---|---:|---:|---:|",
    ]
    target_filter = payload.get("detector_target_filter")
    if target_filter:
        lines[1:1] = [
            (
                f"**Target:** `{target_filter}` — YOLO and RF-DETR boxes are filtered to matching closed-set "
                "classes; SAM 3 uses the text directly."
            ),
            "",
        ]
    for result in payload.get("results", []):
        summary = result.get("summary", {})
        smallest = summary.get("smallest_box_side_px")
        lines.append(
            "| {model} | {count} | {elapsed:.2f} s | {smallest} |".format(
                model=summary.get("model", result.get("model", "unknown")),
                count=summary.get("count", len(result.get("detections", []))),
                elapsed=float(summary.get("elapsed_seconds", result.get("elapsed_seconds", 0))),
                smallest=f"{smallest} px" if smallest is not None else "—",
            )
        )
    if not payload.get("results"):
        lines.append("| No model completed | 0 | — | — |")
    errors = payload.get("errors", {})
    if errors:
        lines.extend(["", "### ⚠️ Some models failed"])
        for name, message in errors.items():
            safe_message = str(message).replace("`", "'").replace("\n", " ")
            lines.append(f"- **{display_names.get(name, name)}:** `{safe_message}`")
        lines.append("Successful models are still shown below. The exact failure is also saved in the JSON report.")
    lines.extend(
        [
            "",
            "The count is an inference result, not an accuracy score. Use labeled ground truth before ranking models.",
        ]
    )
    return "\n".join(lines)


def video_summary(payload: dict) -> str:
    frames = payload.get("frames", [])
    object_ids = {
        detection.get("instance_id")
        for frame in frames
        for detection in frame.get("detections", [])
        if detection.get("instance_id") is not None
    }
    masks = sum(len(frame.get("detections", [])) for frame in frames)
    elapsed = payload.get("elapsed_seconds")
    elapsed_text = f" in {float(elapsed):.2f} seconds" if elapsed is not None else ""
    fps = float(payload.get("fps") or 0)
    duration = len(frames) / fps if fps > 0 else 0
    source_frames = int(payload.get("source_frame_count") or 0)
    frame_limit = int(payload.get("requested_max_frames") or 0)
    coverage = f" The annotated result is about **{duration:.1f} seconds** long." if duration else ""
    if frame_limit and source_frames > len(frames):
        coverage += (
            f" It is shorter than the source because the selected limit was **{frame_limit} frames**; "
            "choose **Whole uploaded video** to process all frames."
        )
    elif source_frames and len(frames) >= source_frames:
        coverage += " The whole uploaded video was processed."
    return (
        "### Video tracking finished\n"
        f"Processed **{len(frames)} frames**, tracked **{len(object_ids)} unique objects**, "
        f"and produced **{masks} frame-level masks**{elapsed_text}.{coverage}\n\n"
        "The H.264 video below should play in the browser. The ZIP contains the masks and exact JSON manifest."
    )


def quick_video_frame_limit(selection: str | float) -> int:
    choices = {"all": 0, "first_60": 60, "first_300": 300}
    if isinstance(selection, str) and selection in choices:
        return choices[selection]
    value = int(selection)
    if value < 0:
        raise ValueError("Maximum frames cannot be negative")
    return value


def bounded_video_summary(payload: dict[str, Any]) -> str:
    status = str(payload.get("status", "running"))
    status_label = {
        "running": "Processing",
        "complete": "Completed",
        "stopped": "Stopped safely",
        "failed": "Failed",
        "starting": "Starting",
    }.get(status, status.title())
    dropped = int(payload.get("dropped_rtsp_chunks") or 0)
    dropped_text = (
        f" RTSP capture dropped **{dropped} pending chunks** to prevent an unbounded queue."
        if dropped
        else ""
    )
    return (
        f"### {status_label}\n"
        f"Processed **{int(payload.get('processed_frames') or 0)} frames** in "
        f"**{int(payload.get('processed_chunks') or 0)} bounded chunks**, assigned "
        f"**{int(payload.get('unique_objects') or 0)} global object IDs**, and wrote "
        f"**{int(payload.get('frame_level_masks') or 0)} masks**.{dropped_text}\n\n"
        "Only one finite SAM 3.1 session is resident at a time. Results are committed after each chunk."
    )


class BoundedVideoController:
    def __init__(self, config: LabConfig, job_factory, active_session_check=None) -> None:
        self.config = config
        self.job_factory = job_factory
        self.active_session_check = active_session_check or (lambda: False)
        self.runner = BoundedSamRunner(config)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def _settings(
        self,
        chunk_frames: int,
        overlap_frames: int,
        grounding_batch_size: int,
        max_active_objects: int,
        threshold: float,
    ) -> BoundedVideoSettings:
        chunk_size = int(chunk_frames)
        return BoundedVideoSettings(
            chunk_frames=chunk_size,
            overlap_frames=int(overlap_frames),
            grounding_batch_size=int(grounding_batch_size),
            max_active_objects=int(max_active_objects),
            threshold=float(threshold),
            identity_ttl_frames=max(60, chunk_size),
            worker_timeout_seconds=float(
                self.config.raw.get("bounded_video", {}).get(
                    "worker_timeout_seconds", 1800
                )
            ),
        ).validate()

    @staticmethod
    def _outputs(output: Path, payload: dict[str, Any]) -> tuple[str, str | None, str, str | None, dict]:
        frames = output / "frames.jsonl"
        return (
            bounded_video_summary(payload),
            payload.get("latest_segment"),
            str(output / "index.json"),
            str(frames) if frames.exists() else None,
            payload,
        )

    def stop(self) -> tuple[str, dict[str, Any]]:
        self.stop_event.set()
        payload = {
            "status": "stopping",
            "message": "Stop requested. The active finite SAM chunk will finish and release its session.",
        }
        return "### Stop requested\nThe current bounded chunk will finish, then capture will stop.", payload

    def _run(
        self,
        chunks,
        output: Path,
        *,
        source_kind: str,
        source_label: str,
        target: str,
        settings: BoundedVideoSettings,
        max_chunks: int = 0,
        dropped_chunks=None,
    ) -> Iterator[tuple[str, str | None, str, str | None, dict]]:
        if self.active_session_check():
            raise RuntimeError(
                "Close the persistent SAM 3.1 session in tab 6 before starting bounded processing"
            )
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Another long-video or RTSP job is already using the GPU")
        self.stop_event.clear()
        try:
            starting = {
                "status": "starting",
                "processed_frames": 0,
                "processed_chunks": 0,
                "unique_objects": 0,
                "frame_level_masks": 0,
                "dropped_rtsp_chunks": 0,
            }
            yield (
                "### Starting bounded SAM 3.1 processing\nPreparing the first finite video chunk.",
                None,
                None,
                None,
                starting,
            )
            for payload in self.runner.run(
                chunks,
                output,
                source_kind=source_kind,
                source_label=source_label,
                target=target,
                settings=settings,
                stop_event=self.stop_event,
                max_chunks=max_chunks,
                dropped_chunks=dropped_chunks,
            ):
                yield self._outputs(output, payload)
        finally:
            self.stop_event.set()
            self.lock.release()

    def run_file(
        self,
        video: str,
        target: str,
        chunk_frames: int,
        overlap_frames: int,
        grounding_batch_size: int,
        max_active_objects: int,
        threshold: float,
        max_chunks: int,
    ) -> Iterator[tuple[str, str | None, str, str | None, dict]]:
        video_path = _path(video)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        if int(max_chunks) < 0:
            raise ValueError("Maximum chunks cannot be negative")
        settings = self._settings(
            chunk_frames,
            overlap_frames,
            grounding_batch_size,
            max_active_objects,
            threshold,
        )
        output = self.job_factory("long_video")
        chunks = OpenCVChunkSource(
            str(video_path), output / "inputs", settings, self.stop_event
        )
        yield from self._run(
            chunks,
            output,
            source_kind="long_video",
            source_label=video_path.name,
            target=target,
            settings=settings,
            max_chunks=int(max_chunks),
        )

    def run_rtsp(
        self,
        rtsp_url: str,
        target: str,
        chunk_frames: int,
        overlap_frames: int,
        grounding_batch_size: int,
        max_active_objects: int,
        threshold: float,
        maximum_minutes: float,
    ) -> Iterator[tuple[str, str | None, str, str | None, dict]]:
        source = validate_rtsp_url(rtsp_url)
        if float(maximum_minutes) < 0:
            raise ValueError("Maximum RTSP duration cannot be negative")
        settings = self._settings(
            chunk_frames,
            overlap_frames,
            grounding_batch_size,
            max_active_objects,
            threshold,
        )
        output = self.job_factory("rtsp")
        capture = OpenCVChunkSource(
            source,
            output / "inputs",
            settings,
            self.stop_event,
            rtsp=True,
            maximum_minutes=float(maximum_minutes),
            reconnect_attempts=int(
                self.config.raw.get("bounded_video", {}).get("rtsp_reconnect_attempts", 5)
            ),
            reconnect_delay_seconds=float(
                self.config.raw.get("bounded_video", {}).get("rtsp_reconnect_delay_seconds", 2)
            ),
        )
        queued = RtspChunkQueue(
            capture,
            capacity=int(
                self.config.raw.get("bounded_video", {}).get("rtsp_queue_capacity", 2)
            ),
        )
        yield from self._run(
            queued,
            output,
            source_kind="rtsp",
            source_label=redact_rtsp_url(source),
            target=target,
            settings=settings,
            dropped_chunks=lambda: queued.dropped_chunks,
        )


class PlaygroundService:
    def __init__(self, config: LabConfig) -> None:
        self.config = config
        self.sessions = MetaVideoSessionController(config)
        self.bounded = BoundedVideoController(
            config, self._job, active_session_check=lambda: bool(self.sessions.sessions)
        )

    def _job(self, kind: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.config.outputs_dir / "playground" / f"{stamp}_{kind}_{uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def health(self) -> str:
        report = doctor_report(self.config)
        models = report["models"]
        ready = sum(bool(item["ready"]) for item in models)
        packages_ready = all(report["packages"].values())
        gpu = report.get("gpu") or "No NVIDIA GPU detected"
        icon = "✅" if ready == len(models) and packages_ready and report.get("gpu") else "⚠️"
        return (
            f"{icon} **System status:** {ready}/{len(models)} model components ready · "
            f"GPU: {gpu} · Python {report['python']}"
        )

    def run_image(
        self,
        backend: str,
        image: str,
        mode: str,
        text: str,
        positive: str,
        negative: str,
        box: str,
        positive_exemplars: str,
        negative_exemplars: str,
        multimask: bool,
        mask_input: str | None,
        threshold: float,
    ) -> tuple[str, list[str], str, str | None, dict]:
        image_path = _path(image)
        output = self._job("sam_image")
        boxes = parse_boxes(box)
        common: dict[str, Any] = {
            "mode": mode,
            "text": text,
            "positive_points": parse_points(positive),
            "negative_points": parse_points(negative),
            "box": boxes[0] if boxes else None,
            "positive_exemplars": parse_boxes(positive_exemplars),
            "negative_exemplars": parse_boxes(negative_exemplars),
            "multimask": multimask,
        }
        if backend == "official":
            adapter = MetaSam3Adapter(self.config)
            common.update(mask_input=_path(mask_input) if mask_input else None, confidence_threshold=threshold)
        elif backend == "q8":
            adapter = Sam3CppAdapter(self.config)
            common.update(score_threshold=threshold)
        else:
            raise ValueError(f"Unknown backend {backend!r}")
        manifest, payload = adapter.run_image(image_path, output, **common)
        annotated = render_sam_manifest(image_path, manifest, output / "annotated.jpg")
        masks = [str(path) for path in manifest_mask_files(manifest)]
        logits = payload.get("low_res_logits")
        logits_path = str(output / logits) if logits else None
        return str(annotated), [str(annotated), *masks], str(manifest), logits_path, payload

    def run_video(
        self,
        backend: str,
        video: str,
        mode: str,
        text: str,
        objects: str,
        refinements: str,
        removals: str,
        start_frame: int,
        max_frames: int,
        direction: str,
        offload_video: bool,
        offload_state: bool,
        threshold: float,
        grounding_batch_size: int = 0,
    ) -> tuple[str, str, str, dict]:
        video_path = _path(video)
        output = self._job("sam_video")
        common: dict[str, Any] = {
            "mode": mode,
            "text": text,
            "objects": _records(objects),
            "refinements": _records(refinements),
            "start_frame": int(start_frame),
            "max_frames": int(max_frames),
        }
        if backend == "official":
            adapter = MetaSam3Adapter(self.config)
            common.update(
                removals=_records(removals),
                propagation_direction=direction,
                offload_video_to_cpu=offload_video,
                offload_state_to_cpu=offload_state,
                output_prob_threshold=threshold,
                grounding_batch_size=int(grounding_batch_size) or None,
            )
        elif backend == "q8":
            if (removals or "").strip() or direction != "forward":
                raise ValueError("Removal and backward/both propagation require the official backend")
            adapter = Sam3CppAdapter(self.config)
            common.update(score_threshold=threshold)
        else:
            raise ValueError(f"Unknown backend {backend!r}")
        manifest, payload = adapter.run_video(video_path, output, **common)
        try:
            import cv2

            capture = cv2.VideoCapture(str(video_path))
            if capture.isOpened():
                payload["source_frame_count"] = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
        except ImportError:
            pass
        payload["requested_max_frames"] = int(max_frames)
        manifest.write_text(json.dumps(payload, indent=2))
        annotated = render_video_manifest(video_path, manifest, output / "annotated.mp4")
        archive = shutil.make_archive(str(output) + "_results", "zip", root_dir=output)
        return str(annotated), str(manifest), archive, payload

    def compare(
        self,
        image: str,
        models: list[str],
        sam_text: str,
        sam_backend: str,
        detector_target: str | None = None,
    ) -> tuple[list[str], str, dict]:
        image_path = _path(image)
        output = self._job("comparison")
        payload = compare_image(
            self.config,
            image_path,
            output,
            models=models,
            sam_text=sam_text,
            sam_backend=sam_backend,
            detector_target=detector_target,
        )
        images = [str(path) for path in sorted(output.glob("*_annotated.jpg"))]
        return images, str(output / "comparison.json"), payload

    def quick_compare(
        self,
        image: str,
        target: str,
        models: list[str],
        sam_backend: str,
        filter_detectors: bool = True,
    ) -> tuple[str, list[tuple[str, str]], str, dict]:
        if not models:
            raise ValueError("Select at least one model")
        if "sam3" in models and not target.strip():
            raise ValueError("Describe what SAM 3 should find, for example: vehicle")
        target = target.strip() or "object"
        images, report, payload = self.compare(
            image,
            models,
            target,
            sam_backend,
            detector_target=target if filter_detectors else None,
        )
        captions = {
            "yolo": f"YOLO26-L — {target} only" if filter_detectors else "YOLO26-L — all detected classes",
            "rfdetr": (
                f"RF-DETR Large — {target} only" if filter_detectors else "RF-DETR Large — all detected classes"
            ),
            "sam3": f"SAM 3 segmentation: {target}",
        }
        gallery = [
            (
                path,
                captions.get(Path(path).stem.removesuffix("_annotated"), Path(path).stem),
            )
            for path in images
        ]
        return comparison_summary(payload), gallery, report, payload

    def quick_video(
        self,
        video: str,
        target: str,
        engine: str,
        frame_range: str | float,
        threshold: float,
    ) -> tuple[str, str, str, str, dict]:
        if not target.strip():
            raise ValueError("Describe the object to track, for example: vehicle")
        profiles = {
            "official_balanced": ("official", True, 4),
            "official_low_vram": ("official", True, 1),
            "official_fast": ("official", False, 16),
            "q8": ("q8", False, 0),
        }
        if engine not in profiles:
            raise ValueError(f"Unknown video engine/profile {engine!r}")
        backend, offload_video, grounding_batch_size = profiles[engine]
        max_frames = quick_video_frame_limit(frame_range)
        annotated, manifest, archive, payload = self.run_video(
            backend,
            video,
            "text",
            target.strip(),
            "",
            "",
            "",
            0,
            int(max_frames),
            "forward",
            offload_video,
            False,
            float(threshold),
            grounding_batch_size,
        )
        return video_summary(payload), annotated, manifest, archive, payload

    def status(self) -> tuple[list[list[Any]], list[dict[str, object]]]:
        status = model_status(self.config)
        rows = [[item["component"], item["ready"], item["size_mb"], item["path"]] for item in status]
        return rows, status

    def download(self, model: str) -> tuple[list[list[Any]], list[dict[str, object]]]:
        download_models(self.config, model)
        return self.status()
