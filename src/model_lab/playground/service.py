from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from ..adapters.meta_sam3 import MetaSam3Adapter
from ..adapters.sam3_cpp import Sam3CppAdapter, parse_boxes, parse_points
from ..compare import compare_image
from ..config import LabConfig
from ..downloader import download_models, model_status
from ..rendering import manifest_mask_files, render_sam_manifest, render_video_manifest
from .sessions import MetaVideoSessionController


def _path(value: str | Path | None) -> Path:
    if not value:
        raise ValueError("Upload a file first")
    return Path(value).expanduser().resolve()


def _records(value: str) -> list[str]:
    return [item.strip() for item in value.splitlines() if item.strip()]


class PlaygroundService:
    def __init__(self, config: LabConfig) -> None:
        self.config = config
        self.sessions = MetaVideoSessionController(config)

    def _job(self, kind: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.config.outputs_dir / "playground" / f"{stamp}_{kind}_{uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=False)
        return path

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
            )
        elif backend == "q8":
            if removals.strip() or direction != "forward":
                raise ValueError("Removal and backward/both propagation require the official backend")
            adapter = Sam3CppAdapter(self.config)
            common.update(score_threshold=threshold)
        else:
            raise ValueError(f"Unknown backend {backend!r}")
        manifest, payload = adapter.run_video(video_path, output, **common)
        annotated = render_video_manifest(video_path, manifest, output / "annotated.mp4")
        archive = shutil.make_archive(str(output) + "_results", "zip", root_dir=output)
        return str(annotated), str(manifest), archive, payload

    def compare(self, image: str, models: list[str], sam_text: str, sam_backend: str) -> tuple[list[str], str, dict]:
        image_path = _path(image)
        output = self._job("comparison")
        payload = compare_image(
            self.config,
            image_path,
            output,
            models=models,
            sam_text=sam_text,
            sam_backend=sam_backend,
        )
        images = [str(path) for path in sorted(output.glob("*_annotated.jpg"))]
        return images, str(output / "comparison.json"), payload

    def status(self) -> tuple[list[list[Any]], list[dict[str, object]]]:
        status = model_status(self.config)
        rows = [[item["component"], item["ready"], item["size_mb"], item["path"]] for item in status]
        return rows, status

    def download(self, model: str) -> tuple[list[list[Any]], list[dict[str, object]]]:
        download_models(self.config, model)
        return self.status()
