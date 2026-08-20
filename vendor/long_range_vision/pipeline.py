from __future__ import annotations

import json
import platform
import time
import tomllib
from pathlib import Path
from typing import Any

from .fusion import non_max_suppression, weighted_box_fusion
from .metrics import detections_from_items, match_detections
from .models import create_adapter
from .tiling import generate_tiles, map_detection_to_image, padded_crop_box
from .types import Detection


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


class LongRangePipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.run_config = config.get("run", {})
        self.model_configs = {
            name: values for name, values in config.get("models", {}).items() if values.get("enabled", False)
        }
        if not self.model_configs:
            raise ValueError("No models are enabled in the configuration")
        device = str(self.run_config.get("device", "auto"))
        self.adapters = {
            name: create_adapter(name, values, device=device) for name, values in self.model_configs.items()
        }

    def run_image(self, image_path: str | Path) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for image inference") from exc

        path = Path(image_path)
        image = Image.open(path).convert("RGB")
        return self.run_pil_image(image, input_id=str(path))

    def run_pil_image(self, image: Any, input_id: str = "<memory>") -> dict[str, Any]:
        """Run inference on an already decoded PIL image.

        Video processing uses this path so source frames do not need a lossy
        temporary JPEG round trip before detection.
        """
        image = image.convert("RGB")
        prompts = [str(value) for value in self.run_config.get("prompt", ["person"])]
        default_tile_size = int(self.run_config.get("tile_size", 1008))
        overlap = float(self.run_config.get("tile_overlap", 0.2))
        all_detections: list[Detection] = []
        timing: dict[str, float] = {}

        proposal_names = [
            name for name, values in self.model_configs.items() if values.get("role", "proposal") != "segment_verify"
        ]
        verifier_names = [
            name for name, values in self.model_configs.items() if values.get("role") == "segment_verify"
        ]
        # A verifier can still operate as a tiled detector if it is the only enabled model.
        if not proposal_names:
            proposal_names, verifier_names = verifier_names, []

        for model_name in proposal_names:
            adapter = self.adapters[model_name]
            model_config = self.model_configs[model_name]
            tile_size = int(model_config.get("tile_size", default_tile_size))
            tiles = generate_tiles(image.width, image.height, tile_size, overlap)
            started = time.perf_counter()
            model_detections: list[Detection] = []
            for tile in tiles:
                crop = image.crop((tile.x1, tile.y1, tile.x2, tile.y2))
                for detection in adapter.detect(crop, prompts):
                    model_detections.append(map_detection_to_image(detection, tile, image.width, image.height))
            timing[model_name] = time.perf_counter() - started
            all_detections.extend(
                non_max_suppression(
                    model_detections,
                    iou_threshold=float(self.run_config.get("nms_iou", 0.45)),
                )
            )

        weights = {name: float(values.get("weight", 1.0)) for name, values in self.model_configs.items()}
        proposal_fused = weighted_box_fusion(
            all_detections,
            model_weights=weights,
            iou_threshold=float(self.run_config.get("ensemble_iou", 0.5)),
        )

        # Segmentation foundation models work on candidate crops. This preserves
        # the full source pixels and avoids shrinking a 4K/8K frame to one tensor.
        for model_name in verifier_names:
            adapter = self.adapters[model_name]
            model_config = self.model_configs[model_name]
            padding = float(model_config.get("roi_padding", 2.0))
            started = time.perf_counter()
            verifier_detections: list[Detection] = []
            for roi_index, proposal in enumerate(proposal_fused):
                roi = padded_crop_box(proposal.box, image.width, image.height, padding=padding)
                crop = image.crop((round(roi.x1), round(roi.y1), round(roi.x2), round(roi.y2)))
                for detection in adapter.detect(crop, prompts):
                    metadata = dict(detection.metadata)
                    metadata.update(
                        verification_roi_index=roi_index,
                        verification_of_box=proposal.box.to_list(),
                        source_crop_size_px=list(crop.size),
                    )
                    verifier_detections.append(
                        Detection(
                            box=detection.box.translate(roi.x1, roi.y1).clip(image.width, image.height),
                            score=detection.score,
                            label=detection.label,
                            model=detection.model,
                            tile_id=f"roi_{roi_index:05d}",
                            metadata=metadata,
                        )
                    )
            timing[model_name] = time.perf_counter() - started
            all_detections.extend(non_max_suppression(verifier_detections, iou_threshold=0.45))

        fused = weighted_box_fusion(
            all_detections,
            model_weights=weights,
            iou_threshold=float(self.run_config.get("ensemble_iou", 0.5)),
        )
        return {
            "schema_version": "1.0",
            "input": input_id,
            "image_width": image.width,
            "image_height": image.height,
            "prompts": prompts,
            "models": list(self.adapters),
            "proposal_models": proposal_names,
            "verification_models": verifier_names,
            "timing_seconds": timing,
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "detections": [detection.to_dict() for detection in fused],
            "raw_model_detections": [detection.to_dict() for detection in all_detections],
        }


def save_report(report: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")


def draw_report(image_path: str | Path, report: dict[str, Any], output_path: str | Path) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to draw predictions") from exc

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for item in report.get("detections", []):
        x1, y1, x2, y2 = item["box_xyxy"]
        agreement = int(item.get("metadata", {}).get("agreement_count", 1))
        color = "#00ff6a" if agreement > 1 else "#ffcc00"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, image.width // 1200))
        text = f"{item['label']} {item['score']:.2f} {item['object_height_px']:.0f}px"
        draw.text((x1, max(0, y1 - 14)), text, fill=color, stroke_width=2, stroke_fill="black")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def stress_test(
    pipeline: LongRangePipeline,
    image_path: str | Path,
    output_dir: str | Path,
    scales: list[float],
) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for the stress test") from exc

    source = Image.open(image_path).convert("RGB")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    baseline_report = pipeline.run_image(image_path)
    for scale in scales:
        if not 0.0 < scale <= 1.0:
            raise ValueError("Stress-test scales must be in (0, 1]")
        reduced = source.resize(
            (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
            resample=Image.Resampling.LANCZOS,
        )
        simulated = reduced.resize(source.size, resample=Image.Resampling.BICUBIC)
        simulated_path = destination / f"scale_{scale:.3f}.png"
        simulated.save(simulated_path)
        # Avoid a duplicate model call at scale 1.0; the source itself is the
        # highest-fidelity baseline and has the same dimensions as `simulated`.
        report = baseline_report if abs(scale - 1.0) < 1e-9 else pipeline.run_image(simulated_path)
        detections = report["detections"]
        survival = match_detections(
            detections_from_items(detections),
            detections_from_items(baseline_report["detections"], model="baseline_prediction"),
            iou_threshold=0.5,
        )["overall"]
        results.append(
            {
                "linear_scale": scale,
                "effective_area_scale": scale * scale,
                "detection_count": len(detections),
                "maximum_score": max((item["score"] for item in detections), default=0.0),
                "mean_detected_height_px": (
                    sum(item["object_height_px"] for item in detections) / len(detections) if detections else 0.0
                ),
                "mean_effective_source_height_px": (
                    sum(item["object_height_px"] for item in detections) * scale / len(detections)
                    if detections
                    else 0.0
                ),
                "baseline_pseudo_recall_at_iou_0_5": survival["recall"],
                "baseline_pseudo_precision_at_iou_0_5": survival["precision"],
                "baseline_matched_detections": survival["true_positive"],
                "report_path": str(destination / f"scale_{scale:.3f}.json"),
            }
        )
        save_report(report, destination / f"scale_{scale:.3f}.json")
    summary = {
        "input": str(image_path),
        "method": "Lanczos downsample followed by bicubic resize to isolate information loss from tensor size",
        "warning": (
            "This tests detector robustness, not recovery of real identity or texture. "
            "Pseudo-recall measures survival relative to the model's own scale-1 predictions, not ground-truth accuracy."
        ),
        "results": results,
    }
    save_report(summary, destination / "stress_summary.json")
    return summary
