from __future__ import annotations

import gc
import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image

from .adapters import MetaSam3Adapter, RFDetrAdapter, Sam3CppAdapter, YoloAdapter
from .compare import filter_detector_result
from .config import LabConfig
from .contracts import Detection, ModelResult, resolve_mask_path
from .fusion import non_max_suppression, weighted_box_fusion
from .rendering import render_result
from .tiling import clip_box, generate_tiles, padded_crop_box, translate_box

_PROPOSAL_ADAPTERS = {"yolo": YoloAdapter, "rfdetr": RFDetrAdapter}


def _run_tiled_proposals(
    adapter: Any,
    model_name: str,
    image: Image.Image,
    scratch_dir: Path,
    tile_size: int,
    overlap: float,
    nms_iou: float,
) -> tuple[list[Detection], float]:
    """Run a proposal detector on overlapping source-resolution tiles.

    A single downscaled full-frame pass can erase a small/distant object
    before the model ever sees it. Tiling keeps every tile at full source
    resolution, mirroring long_range_vision.pipeline's image architecture.
    """
    tiles = generate_tiles(image.width, image.height, tile_size, overlap)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    detections: list[Detection] = []
    started = perf_counter()
    for tile in tiles:
        crop = image.crop((tile.x1, tile.y1, tile.x2, tile.y2))
        tile_path = scratch_dir / f"{tile.tile_id}.png"
        crop.save(tile_path)
        try:
            result = adapter.predict_image(tile_path, scratch_dir)
        finally:
            tile_path.unlink(missing_ok=True)
        for detection in result.detections:
            box = clip_box(translate_box(detection.box, tile.x1, tile.y1), image.width, image.height)
            metadata = dict(detection.metadata)
            metadata.update(source_model=model_name, tile_id=tile.tile_id)
            detections.append(
                Detection(
                    box=box,
                    score=detection.score,
                    label=detection.label,
                    class_id=detection.class_id,
                    metadata=metadata,
                )
            )
    elapsed = perf_counter() - started
    return non_max_suppression(detections, iou_threshold=nms_iou), elapsed


def _composite_mask(crop_mask_path: Path, output_path: Path, x1: int, y1: int, width: int, height: int) -> None:
    """Paste a crop-resolution mask into a full-image-sized canvas at its ROI offset.

    Keeps mask files compatible with the existing render_result/render_video_manifest
    overlay code, which assumes a mask is already sized to the source image.
    """
    crop_mask = Image.open(crop_mask_path).convert("L")
    canvas = Image.new("L", (width, height), 0)
    canvas.paste(crop_mask, (x1, y1))
    canvas.save(output_path)


def _run_roi_verification(
    adapter: Any,
    image: Image.Image,
    proposals: list[Detection],
    sam_text: str,
    output_dir: Path,
    padding: float,
) -> tuple[list[Detection], float]:
    """Verify each fused proposal with SAM on its own padded, full-resolution crop.

    This is the step that actually protects small objects: SAM sees a tight
    crop at source resolution instead of the whole frame resized to its fixed
    model input size.
    """
    scratch_dir = output_dir / "_scratch" / "sam3_roi"
    mask_dir = output_dir / "sam3_cascade_masks"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    detections: list[Detection] = []
    started = perf_counter()
    for roi_index, proposal in enumerate(proposals):
        x1, y1, x2, y2 = padded_crop_box(proposal.box, image.width, image.height, padding=padding)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image.crop((x1, y1, x2, y2))
        crop_path = scratch_dir / f"roi_{roi_index:05d}.png"
        crop.save(crop_path)
        roi_output = scratch_dir / f"roi_{roi_index:05d}_out"
        try:
            result = adapter.predict_text_image(crop_path, roi_output, sam_text)
        finally:
            crop_path.unlink(missing_ok=True)
        manifest_path = Path(str(result.metadata.get("manifest", roi_output / "manifest.json")))
        for detection_index, detection in enumerate(result.detections):
            crop_mask = resolve_mask_path(manifest_path, detection.mask_path)
            mask_name = None
            if crop_mask and crop_mask.exists():
                mask_name = f"roi_{roi_index:05d}_{detection_index:02d}.png"
                _composite_mask(crop_mask, mask_dir / mask_name, x1, y1, image.width, image.height)
            box = clip_box(translate_box(detection.box, x1, y1), image.width, image.height)
            metadata = dict(detection.metadata)
            metadata.update(
                verification_roi_index=roi_index,
                verification_of_box=list(proposal.box),
                proposal_metadata=dict(proposal.metadata),
                source_crop_size_px=list(crop.size),
            )
            detections.append(
                Detection(
                    box=box,
                    score=detection.score,
                    label=detection.label,
                    mask_path=f"sam3_cascade_masks/{mask_name}" if mask_name else None,
                    instance_id=detection.instance_id,
                    metadata=metadata,
                )
            )
        shutil.rmtree(roi_output, ignore_errors=True)
    elapsed = perf_counter() - started
    return detections, elapsed


def compare_image_cascade(
    config: LabConfig,
    image: Path,
    output_dir: Path,
    *,
    proposal_models: Iterable[str] = ("yolo", "rfdetr"),
    sam_text: str = "object",
    sam_backend: str | None = None,
    tile_size: int | None = None,
    tile_overlap: float | None = None,
    roi_padding: float | None = None,
    per_model_nms_iou: float | None = None,
    ensemble_iou: float | None = None,
    detector_target: str | None = None,
) -> dict:
    """Tile -> propose -> fuse -> padded-ROI-crop -> SAM verify, on one full-resolution image.

    Ports long_range_vision.pipeline's image architecture into model_comparison_lab.
    Unlike compare_image (each model run once on the whole frame, for apples-to-apples
    comparison), this path is meant to recover the detail that a single whole-image
    resize would otherwise erase for small/distant objects.
    """
    image_path = image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cascade_config = dict(config.raw.get("cascade", {}))
    tile_size = int(tile_size if tile_size is not None else cascade_config.get("tile_size", 1008))
    tile_overlap = float(tile_overlap if tile_overlap is not None else cascade_config.get("tile_overlap", 0.2))
    roi_padding = float(roi_padding if roi_padding is not None else cascade_config.get("roi_padding", 2.0))
    per_model_nms_iou = float(
        per_model_nms_iou if per_model_nms_iou is not None else cascade_config.get("per_model_nms_iou", 0.45)
    )
    ensemble_iou = float(ensemble_iou if ensemble_iou is not None else cascade_config.get("ensemble_iou", 0.5))
    configured_weights = cascade_config.get("weights", {})

    requested = list(dict.fromkeys(proposal_models))
    unknown = [name for name in requested if name not in _PROPOSAL_ADAPTERS]
    if unknown:
        raise ValueError(f"Unsupported cascade proposal model(s): {', '.join(unknown)}")

    pil_image = Image.open(image_path).convert("RGB")
    timing: dict[str, float] = {}
    proposal_results: dict[str, list[Detection]] = {}
    errors: dict[str, str] = {}

    for name in requested:
        try:
            adapter = _PROPOSAL_ADAPTERS[name](config)
            detections, elapsed = _run_tiled_proposals(
                adapter,
                name,
                pil_image,
                output_dir / "_scratch" / name,
                tile_size,
                tile_overlap,
                per_model_nms_iou,
            )
            if detector_target:
                wrapped = ModelResult(
                    model=name,
                    source=str(image_path),
                    width=pil_image.width,
                    height=pil_image.height,
                    elapsed_seconds=elapsed,
                    detections=detections,
                )
                detections = filter_detector_result(wrapped, detector_target).detections
            proposal_results[name] = detections
            timing[name] = elapsed
            del adapter
            gc.collect()
        except Exception as exc:  # noqa: BLE001 - preserve other model results in a benchmark run
            errors[name] = f"{type(exc).__name__}: {exc}"

    all_proposals = [detection for detections in proposal_results.values() for detection in detections]
    weights = {name: float(configured_weights.get(name, 1.0)) for name in requested}
    fused_proposals = weighted_box_fusion(all_proposals, model_weights=weights, iou_threshold=ensemble_iou)

    selected_backend = sam_backend or config.raw["sam3"].get("backend", "official")
    if selected_backend == "official":
        sam_adapter: Any = MetaSam3Adapter(config)
    elif selected_backend == "q8":
        sam_adapter = Sam3CppAdapter(config)
    else:
        raise ValueError(f"Unknown SAM backend: {selected_backend}")

    verified: list[Detection] = []
    try:
        verified, sam_elapsed = _run_roi_verification(
            sam_adapter, pil_image, fused_proposals, sam_text, output_dir, roi_padding
        )
        timing["sam3_roi_verify"] = sam_elapsed
    except Exception as exc:  # noqa: BLE001 - preserve proposal results even if SAM fails
        errors["sam3"] = f"{type(exc).__name__}: {exc}"
    finally:
        del sam_adapter
        gc.collect()

    cascade_result = ModelResult(
        model=f"cascade({'+'.join(requested)}->sam3-{selected_backend})",
        source=str(image_path),
        width=pil_image.width,
        height=pil_image.height,
        elapsed_seconds=sum(timing.values()),
        detections=verified,
        metadata={
            "proposal_models": requested,
            "sam_backend": selected_backend,
            "fused_proposal_count": len(fused_proposals),
        },
    )
    render_result(image_path, cascade_result, output_dir / "cascade_annotated.jpg")
    shutil.rmtree(output_dir / "_scratch", ignore_errors=True)

    payload = {
        "schema_version": 1,
        "source": str(image_path),
        "sam_text_prompt": sam_text,
        "detector_target_filter": detector_target,
        "sam_backend": selected_backend,
        "proposal_models": requested,
        "tile_size": tile_size,
        "tile_overlap": tile_overlap,
        "roi_padding": roi_padding,
        "warning": (
            "Cascade mode preserves source-resolution detail through tiled proposal detection and "
            "padded ROI-crop SAM verification, mirroring the long_range_vision root image pipeline. "
            "This is an inference report, not an accuracy ranking."
        ),
        "timing_seconds": timing,
        "proposal_counts": {name: len(detections) for name, detections in proposal_results.items()},
        "fused_proposal_count": len(fused_proposals),
        "result": cascade_result.to_dict(),
        "results": [cascade_result.to_dict()],
        "errors": errors,
    }
    (output_dir / "cascade.json").write_text(json.dumps(payload, indent=2))
    return payload
