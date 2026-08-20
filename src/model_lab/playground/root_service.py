from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..config import LabConfig

try:
    from long_range_vision.pipeline import LongRangePipeline, draw_report, save_report
    from long_range_vision.video import VideoSettings, run_video
except ImportError:  # pragma: no cover - vendored root package absent in isolated test env
    LongRangePipeline = None  # type: ignore[assignment]
    draw_report = None  # type: ignore[assignment]
    save_report = None  # type: ignore[assignment]
    VideoSettings = None  # type: ignore[assignment]
    run_video = None  # type: ignore[assignment]

_ROOT_MODEL_META: dict[str, dict[str, Any]] = {
    "rfdetr": {
        "label": "RF-DETR Large",
        "adapter": "rfdetr",
        "role": "proposal",
        "variant": "large",
        "weight": 1.0,
    },
    "grounding_dino": {
        "label": "Grounding DINO Tiny",
        "adapter": "grounding_dino",
        "role": "proposal",
        "model_id": "IDEA-Research/grounding-dino-tiny",
        "text_threshold": 0.2,
        "weight": 0.9,
    },
    "sam3_verify": {
        "label": "SAM 3 verify (ROI crops)",
        "adapter": "sam3",
        "role": "segment_verify",
        "model_id": "facebook/sam3",
        "mask_threshold": 0.5,
    },
}


def _split_prompts(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", "\n").splitlines() if item.strip()]


class RootPipelineService:
    """Drives the vendored long_range_vision root pipeline, untouched, from the dashboard.

    Kept deliberately separate from the lab's own YOLO/RF-DETR/SAM adapters: this
    tab runs the root project's tile -> propose -> fuse -> ROI-verify -> temporal
    tracker architecture as-is, with every knob exposed.
    """

    def __init__(self, config: LabConfig) -> None:
        self.config = config
        self.ready = LongRangePipeline is not None

    def _job(self, kind: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.config.outputs_dir / "playground" / f"{stamp}_root_{kind}_{uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=False)
        return path

    def _build_config(
        self,
        *,
        prompts: list[str],
        models: list[str],
        threshold: float,
        tile_size: int,
        tile_overlap: float,
        nms_iou: float,
        ensemble_iou: float,
        roi_padding: float,
        device: str,
    ) -> dict[str, Any]:
        model_sections: dict[str, Any] = {}
        for key in models:
            meta = _ROOT_MODEL_META[key]
            section: dict[str, Any] = {
                "enabled": True,
                "adapter": meta["adapter"],
                "role": meta["role"],
                "threshold": threshold,
                "weight": meta.get("weight", 1.0),
            }
            section.update({k: v for k, v in meta.items() if k not in {"label", "adapter", "role", "weight"}})
            if key == "sam3_verify":
                section["roi_padding"] = roi_padding
            model_sections[key] = section
        return {
            "run": {
                "prompt": prompts,
                "threshold": threshold,
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
                "nms_iou": nms_iou,
                "ensemble_iou": ensemble_iou,
                "device": device,
            },
            "models": model_sections,
        }

    def _stage_gallery(self, image_path: Path, output: Path, report: dict[str, Any], models: list[str]) -> list[str]:
        raw = report.get("raw_model_detections", [])
        stage_paths: list[str] = []
        for key in models:
            stage = output / f"stage_{key}_proposals.jpg"
            draw_report(
                image_path,
                {"detections": [item for item in raw if item.get("model") == key]},
                stage,
            )
            stage_paths.append(str(stage))
        if any(_ROOT_MODEL_META[key]["role"] == "segment_verify" for key in models):
            verified = output / "stage_sam3_verified.jpg"
            draw_report(
                image_path,
                {"detections": [item for item in raw if item.get("model") == "sam3_verify"]},
                verified,
            )
            stage_paths.append(str(verified))
        fused = output / "stage_fused.jpg"
        draw_report(image_path, {"detections": report.get("detections", [])}, fused)
        stage_paths.append(str(fused))
        return stage_paths

    def run_image(
        self,
        image: str,
        prompts: str,
        models: list[str],
        threshold: float,
        tile_size: int,
        tile_overlap: float,
        nms_iou: float,
        ensemble_iou: float,
        roi_padding: float,
        device: str,
    ) -> tuple[str, list[str], str | None, dict]:
        if not self.ready:
            return (
                "The vendored root pipeline is not installed. Run `pip install -e '.[all,root]'`.",
                [],
                None,
                {"error": "root pipeline unavailable"},
            )
        from PIL import Image

        output = self._job("image")
        config = self._build_config(
            prompts=_split_prompts(prompts) or ["person"],
            models=models or ["rfdetr", "grounding_dino"],
            threshold=threshold,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            nms_iou=nms_iou,
            ensemble_iou=ensemble_iou,
            roi_padding=roi_padding,
            device=device,
        )
        pipeline = LongRangePipeline(config)
        report = pipeline.run_pil_image(Image.open(image).convert("RGB"), input_id=str(image))
        save_report(report, output / "root_image.json")
        annotated = output / "annotated.jpg"
        draw_report(image, report, annotated)
        gallery = [str(annotated), *self._stage_gallery(Path(image), output, report, list(config["models"]))]

        lines = [
            "### Root pipeline image run",
            "| Stage | Models / detections | Time (s) |",
            "|---|---:|---:|",
        ]
        for key in config["models"]:
            lines.append(
                "| {label} proposals | {count} | {elapsed:.2f} |".format(
                    label=_ROOT_MODEL_META[key]["label"],
                    count=sum(1 for item in report["raw_model_detections"] if item.get("model") == key),
                    elapsed=report["timing_seconds"].get(key, 0.0),
                )
            )
        lines.append(
            "| Final fused detections | {count} | {elapsed:.2f} |".format(
                count=len(report["detections"]),
                elapsed=sum(report["timing_seconds"].values()),
            )
        )
        summary = "\n".join(lines)
        gallery = [path for path in gallery if Path(path).is_file()]
        json_path = output / "root_image.json"
        return summary, gallery, str(json_path) if json_path.is_file() else None, report

    def run_video_job(
        self,
        video: str,
        prompts: str,
        models: list[str],
        threshold: float,
        tile_size: int,
        tile_overlap: float,
        nms_iou: float,
        ensemble_iou: float,
        roi_padding: float,
        device: str,
        detection_interval: int,
        min_hits: int,
        max_missed: int,
        association_iou: float,
        appearance_encoder: str,
        appearance_weight: float,
        appearance_momentum: float,
        appearance_batch_size: int,
        appearance_roi_padding: float,
        start_frame: int,
        max_frames: int,
    ) -> tuple[str, str | None, str | None, dict]:
        if not self.ready:
            return (
                "The vendored root pipeline is not installed. Run `pip install -e '.[all,root]'`.",
                None,
                None,
                {"error": "root pipeline unavailable"},
            )
        output = self._job("video")
        config = self._build_config(
            prompts=_split_prompts(prompts) or ["person"],
            models=models or ["rfdetr", "grounding_dino"],
            threshold=threshold,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            nms_iou=nms_iou,
            ensemble_iou=ensemble_iou,
            roi_padding=roi_padding,
            device=device,
        )
        pipeline = LongRangePipeline(config)
        report = run_video(
            pipeline,
            video,
            output,
            VideoSettings(
                detection_interval=detection_interval,
                min_hits=min_hits,
                max_missed_keyframes=max_missed,
                association_iou=association_iou,
                start_frame=start_frame,
                max_frames=max_frames or None,
                appearance_encoder=appearance_encoder,
                appearance_batch_size=appearance_batch_size,
                appearance_roi_padding=appearance_roi_padding,
                appearance_weight=appearance_weight,
                appearance_momentum=appearance_momentum,
            ),
        )
        annotated = output / "annotated.mp4"
        tracks = output / "tracks.json"
        summary = (
            f"### Root pipeline video run\n\n"
            f"- **{report['confirmed_track_count']}** confirmed tracks, "
            f"**{report['tentative_track_count']}** tentative\n"
            f"- Appearance encoder: **{appearance_encoder}**\n"
            f"- Detection keyframe interval: **{detection_interval}** frames\n"
            f"- Output: `{annotated}`"
        )
        return summary, str(annotated) if annotated.is_file() else None, str(tracks) if tracks.is_file() else None, report