from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path
import re
import time
from typing import Any

from .appearance import attach_appearance_tokens, create_appearance_encoder
from .metrics import detections_from_items
from .pipeline import LongRangePipeline, save_report
from .tiling import padded_crop_box
from .tracking import GlobalMotionEstimator, TemporalTracker, TrackState


@dataclass(frozen=True)
class VideoSettings:
    detection_interval: int = 5
    min_hits: int = 2
    max_missed_keyframes: int = 2
    association_iou: float = 0.2
    start_frame: int = 0
    max_frames: int | None = None
    save_keyframe_reports: bool = True
    appearance_encoder: str = "none"
    appearance_device: str = "auto"
    appearance_batch_size: int = 64
    appearance_roi_padding: float = 0.35
    appearance_weight: float = 0.35
    appearance_min_similarity: float = 0.25
    appearance_cross_label_similarity: float = 0.90
    appearance_momentum: float = 0.85
    replay_keyframe_dir: str | None = None

    def validate(self) -> None:
        if self.detection_interval < 1:
            raise ValueError("detection_interval must be at least 1")
        if self.start_frame < 0:
            raise ValueError("start_frame cannot be negative")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be positive when supplied")
        if not 0.0 <= self.association_iou <= 1.0:
            raise ValueError("association_iou must be between 0 and 1")
        if self.appearance_batch_size < 1:
            raise ValueError("appearance_batch_size must be positive")
        if self.appearance_roi_padding < 0.0:
            raise ValueError("appearance_roi_padding cannot be negative")
        if self.appearance_weight < 0.0:
            raise ValueError("appearance_weight cannot be negative")
        if not -1.0 <= self.appearance_min_similarity <= 1.0:
            raise ValueError("appearance_min_similarity must be between -1 and 1")
        if not -1.0 <= self.appearance_cross_label_similarity <= 1.0:
            raise ValueError("appearance_cross_label_similarity must be between -1 and 1")
        if not 0.0 <= self.appearance_momentum < 1.0:
            raise ValueError("appearance_momentum must be in [0, 1)")


def _gray_array(image: Any) -> Any:
    import numpy as np

    return np.asarray(image.convert("L"), dtype=np.uint8)


def _crop_quality(crop: Any, score: float) -> float:
    import numpy as np

    gray = np.asarray(crop.convert("L"), dtype=np.float32)
    if gray.size < 16 or min(gray.shape) < 2:
        return 0.0
    horizontal = np.diff(gray, axis=1)
    vertical = np.diff(gray, axis=0)
    sharpness = float(np.var(horizontal) + np.var(vertical))
    contrast = float(np.std(gray))
    return (sharpness + 0.25 * contrast) * float(np.log1p(gray.size)) * max(0.1, score)


def _update_best_crops(image: Any, tracks: list[TrackState], frame_index: int) -> None:
    for track in tracks:
        roi = padded_crop_box(track.box, image.width, image.height, padding=0.35)
        coordinates = (round(roi.x1), round(roi.y1), round(roi.x2), round(roi.y2))
        if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
            continue
        crop = image.crop(coordinates)
        quality = _crop_quality(crop, track.score_ema)
        if quality > track.best_quality:
            track.best_quality = quality
            track.best_frame = frame_index
            track.best_crop = crop.copy()


def _track_color(track: TrackState) -> str:
    if not track.confirmed:
        return "#ffcc00"
    if len(track.source_models) > 1:
        return "#00ff6a"
    return "#00c8ff"


def _draw_tracks(image: Any, tracks: list[TrackState], min_hits: int, detection_frame: bool) -> Any:
    from PIL import ImageDraw

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    width = max(2, image.width // 1000)
    for track in tracks:
        color = _track_color(track)
        draw.rectangle(track.box.to_list(), outline=color, width=width)
        state = "C" if track.confirmed else "T"
        text = (
            f"{track.label} #{track.track_id} {state} "
            f"{track.temporal_confidence(min_hits):.2f} h{track.hits}"
        )
        draw.text(
            (track.box.x1, max(0, track.box.y1 - 13)),
            text,
            fill=color,
            stroke_width=2,
            stroke_fill="black",
        )
    marker = "DETECT" if detection_frame else "TRACK"
    draw.rectangle((0, 0, 150, 24), fill="black")
    draw.text((6, 5), marker, fill="#ffffff")
    return annotated


def _safe_label(label: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", label.strip())
    return cleaned.strip("_") or "object"


def _save_best_crops(tracks: list[TrackState], crops_dir: Path) -> None:
    crops_dir.mkdir(parents=True, exist_ok=True)
    for track in tracks:
        if track.best_crop is None:
            continue
        path = crops_dir / f"track_{track.track_id:05d}_{_safe_label(track.label)}.jpg"
        track.best_crop.save(path, quality=95)
        track.best_crop_path = str(path)


def run_video(
    detector: LongRangePipeline | None,
    input_path: str | Path,
    output_dir: str | Path,
    settings: VideoSettings,
) -> dict[str, Any]:
    """Run sparse transformer detections plus recurrent temporal object memory."""
    settings.validate()
    if detector is None and settings.replay_keyframe_dir is None:
        raise ValueError("A detector is required unless replay_keyframe_dir is supplied")
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is required. Install the project with the [video] extra.") from exc

    source_path = Path(input_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    keyframe_dir = destination / "keyframes"
    if settings.save_keyframe_reports:
        keyframe_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = destination / "annotated.mp4"
    frames_path = destination / "frames.jsonl"
    summary_path = destination / "tracks.json"
    crops_dir = destination / "best_crops"

    tracker = TemporalTracker(
        min_hits=settings.min_hits,
        max_missed_keyframes=settings.max_missed_keyframes,
        association_iou=settings.association_iou,
        appearance_weight=settings.appearance_weight,
        appearance_min_similarity=settings.appearance_min_similarity,
        appearance_cross_label_similarity=settings.appearance_cross_label_similarity,
        appearance_momentum=settings.appearance_momentum,
    )
    appearance_encoder = create_appearance_encoder(
        settings.appearance_encoder,
        device=settings.appearance_device,
        batch_size=settings.appearance_batch_size,
    )
    # The portable NumPy backend avoids loading OpenCV's bundled FFmpeg beside
    # PyAV's FFmpeg, a combination that can conflict on macOS.
    motion = GlobalMotionEstimator(prefer_opencv=False)
    model_timing: dict[str, float] = defaultdict(float)
    keyframe_summaries: list[dict[str, Any]] = []
    observed_models: set[str] = set(detector.adapters if detector is not None else [])
    processed_frames = 0
    decoded_frame_index = -1
    previous_gray: Any = None
    started = time.perf_counter()

    input_container = av.open(str(source_path))
    input_stream = next((stream for stream in input_container.streams if stream.type == "video"), None)
    if input_stream is None:
        input_container.close()
        raise ValueError(f"No video stream found in {source_path}")
    fps_fraction = input_stream.average_rate or Fraction(30, 1)
    fps = float(fps_fraction)
    source_width = int(input_stream.codec_context.width)
    source_height = int(input_stream.codec_context.height)

    output_container = av.open(str(annotated_path), mode="w")
    output_stream = output_container.add_stream("libx264", rate=fps_fraction)
    output_stream.width = source_width
    output_stream.height = source_height
    output_stream.pix_fmt = "yuv420p"
    output_stream.options = {"crf": "23", "preset": "veryfast"}

    try:
        with frames_path.open("w", encoding="utf-8") as frame_log:
            for decoded_frame_index, frame in enumerate(input_container.decode(input_stream)):
                if decoded_frame_index < settings.start_frame:
                    continue
                if settings.max_frames is not None and processed_frames >= settings.max_frames:
                    break

                image = frame.to_image().convert("RGB")
                gray = _gray_array(image)
                affine = motion.estimate(previous_gray, gray)
                detection_frame = processed_frames % settings.detection_interval == 0
                detections = []
                detector_report: dict[str, Any] | None = None
                if detection_frame:
                    replayed = settings.replay_keyframe_dir is not None
                    if replayed:
                        replay_path = (
                            Path(settings.replay_keyframe_dir)
                            / f"frame_{decoded_frame_index:08d}.json"
                        )
                        if not replay_path.is_file():
                            raise FileNotFoundError(
                                f"Missing replay report for detection frame {decoded_frame_index}: {replay_path}"
                            )
                        replay_started = time.perf_counter()
                        detector_report = json.loads(replay_path.read_text(encoding="utf-8"))
                        model_timing["replay_load"] += time.perf_counter() - replay_started
                    else:
                        assert detector is not None
                        detector_report = detector.run_pil_image(
                            image,
                            input_id=f"{source_path}#frame={decoded_frame_index}",
                        )
                    detections = detections_from_items(detector_report["detections"])
                    observed_models.update(str(value) for value in detector_report.get("models", []))
                    if not replayed:
                        for model_name, elapsed in detector_report["timing_seconds"].items():
                            model_timing[model_name] += float(elapsed)
                    appearance_started = time.perf_counter()
                    attach_appearance_tokens(
                        image,
                        detections,
                        appearance_encoder,
                        padding=settings.appearance_roi_padding,
                    )
                    appearance_elapsed = time.perf_counter() - appearance_started
                    model_timing["appearance_encoder"] += appearance_elapsed
                    keyframe_summaries.append(
                        {
                            "frame_index": decoded_frame_index,
                            "time_seconds": float(frame.time) if frame.time is not None else decoded_frame_index / fps,
                            "detection_count": len(detections),
                            "timing_seconds": detector_report["timing_seconds"],
                            "appearance_seconds": appearance_elapsed,
                            "replayed": replayed,
                        }
                    )
                    if settings.save_keyframe_reports:
                        save_report(detector_report, keyframe_dir / f"frame_{decoded_frame_index:08d}.json")
                    print(
                        f"[video] detection frame {decoded_frame_index}: "
                        f"{len(detections)} candidates, {len(tracker.active)} active tracks, "
                        f"appearance={appearance_encoder.name if appearance_encoder else 'off'}",
                        flush=True,
                    )

                tracks, detected_track_ids = tracker.update(
                    frame_index=decoded_frame_index,
                    width=image.width,
                    height=image.height,
                    affine=affine,
                    detections=detections,
                    detection_frame=detection_frame,
                )
                _update_best_crops(image, tracks, decoded_frame_index)
                annotated = _draw_tracks(image, tracks, settings.min_hits, detection_frame)
                annotated_frame = av.VideoFrame.from_image(annotated)
                annotated_frame.pts = processed_frames
                annotated_frame.time_base = Fraction(fps_fraction.denominator, fps_fraction.numerator)
                for packet in output_stream.encode(annotated_frame):
                    output_container.mux(packet)

                frame_record = {
                    "frame_index": decoded_frame_index,
                    "processed_index": processed_frames,
                    "time_seconds": float(frame.time) if frame.time is not None else decoded_frame_index / fps,
                    "detection_frame": detection_frame,
                    "candidate_count": len(detections),
                    "camera_affine_2x3": [list(affine[0]), list(affine[1])],
                    "tracks": [
                        track.to_frame_dict(track.track_id in detected_track_ids, settings.min_hits)
                        for track in tracks
                    ],
                }
                frame_log.write(json.dumps(frame_record, separators=(",", ":")) + "\n")
                previous_gray = gray
                processed_frames += 1
    finally:
        for packet in output_stream.encode():
            output_container.mux(packet)
        output_container.close()
        input_container.close()

    all_tracks = tracker.all_tracks()
    _save_best_crops(all_tracks, crops_dir)
    track_summaries = [track.to_summary_dict(fps, settings.min_hits) for track in all_tracks]
    confirmed = [track for track in all_tracks if track.confirmed]
    elapsed_total = time.perf_counter() - started
    summary = {
        "schema_version": "1.1-video",
        "input": str(source_path),
        "source": {
            "width": source_width,
            "height": source_height,
            "fps": fps,
            "processed_frames": processed_frames,
            "start_frame": settings.start_frame,
            "last_decoded_frame": decoded_frame_index,
        },
        "architecture": (
            "Transformer detections on source-resolution keyframes; ROI appearance tokens; recurrent "
            "per-object memory combining appearance, camera motion, residual velocity and evidence; "
            "selective source-pixel best-crop retention."
        ),
        "settings": asdict(settings),
        "motion_estimator": motion.description,
        "models": sorted(observed_models),
        "appearance_memory": {
            "enabled": appearance_encoder is not None,
            "encoder": appearance_encoder.name if appearance_encoder is not None else None,
            "dimension": appearance_encoder.dimension if appearance_encoder is not None else 0,
            "pretrained": appearance_encoder.pretrained if appearance_encoder is not None else False,
            "update": "L2-normalized exponential moving average per track",
            "video_pretrained": False,
        },
        "keyframe_count": len(keyframe_summaries),
        "keyframes": keyframe_summaries,
        "timing_seconds": {
            "total_wall": elapsed_total,
            "model_total": dict(model_timing),
            "model_mean_per_keyframe": {
                name: value / max(1, len(keyframe_summaries)) for name, value in model_timing.items()
            },
            "effective_processing_fps": processed_frames / elapsed_total if elapsed_total else 0.0,
        },
        "track_count": len(all_tracks),
        "confirmed_track_count": len(confirmed),
        "tentative_track_count": len(all_tracks) - len(confirmed),
        "confirmed_tracks_by_label": dict(Counter(track.label for track in confirmed)),
        "tracks": track_summaries,
        "outputs": {
            "annotated_video": str(annotated_path),
            "frame_jsonl": str(frames_path),
            "best_crops_directory": str(crops_dir),
            "keyframe_reports_directory": str(keyframe_dir) if settings.save_keyframe_reports else None,
        },
        "limitations": [
            "Confirmation is temporal consistency, not ground-truth correctness.",
            (
                "The recurrent association memory is deterministic; its ROI token may be pretrained, but "
                "V-JEPA-style native-video pretraining is not active in this run."
            ),
            "Audio is not copied to the annotated output.",
            "Absolute distance and speed require camera calibration or an external ranging source.",
        ],
    }
    save_report(summary, summary_path)
    return summary
