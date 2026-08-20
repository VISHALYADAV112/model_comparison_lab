from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from model_lab import cascade
from model_lab.cascade import (
    _composite_mask,
    _run_roi_verification,
    _run_tiled_proposals,
    compare_image_cascade,
)
from model_lab.config import LabConfig
from model_lab.contracts import Detection, ModelResult


class FakeDetectorAdapter:
    """Stands in for YoloAdapter/RFDetrAdapter: one fixed box near a crop's corner."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[Path] = []

    def predict_image(self, image: Path, output_dir: Path) -> ModelResult:
        self.calls.append(image)
        width, height = Image.open(image).size
        box = (2.0, 2.0, float(min(8, width - 1)), float(min(8, height - 1)))
        return ModelResult(
            model=self.label,
            source=str(image),
            width=width,
            height=height,
            elapsed_seconds=0.01,
            detections=[Detection(box=box, score=0.9, label="person")],
        )


class FakeSamAdapter:
    """Stands in for MetaSam3Adapter/Sam3CppAdapter: masks the center of whatever crop it gets."""

    def predict_text_image(self, image: Path, output_dir: Path, text: str) -> ModelResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        width, height = Image.open(image).size
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 255
        mask_path = output_dir / "mask_000.png"
        Image.fromarray(mask).save(mask_path)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(json.dumps({"schema_version": 1}))
        return ModelResult(
            model="fake-sam",
            source=str(image),
            width=width,
            height=height,
            elapsed_seconds=0.01,
            detections=[
                Detection(
                    box=(width * 0.25, height * 0.25, width * 0.75, height * 0.75),
                    score=0.95,
                    label=text,
                    mask_path="mask_000.png",
                )
            ],
            metadata={"manifest": str(manifest_path)},
        )


def test_run_tiled_proposals_maps_tile_local_boxes_to_image_coordinates(tmp_path: Path) -> None:
    image = Image.new("RGB", (200, 100), "gray")
    adapter = FakeDetectorAdapter("fake")

    detections, elapsed = _run_tiled_proposals(
        adapter, "fake", image, tmp_path / "scratch", tile_size=100, overlap=0.2, nms_iou=0.45
    )

    assert elapsed >= 0
    assert len(adapter.calls) >= 2  # width 200 needs more than one 100px tile
    assert any(det.box[0] > 50 for det in detections)
    assert all(det.metadata["source_model"] == "fake" for det in detections)


def test_composite_mask_places_crop_at_the_correct_offset(tmp_path: Path) -> None:
    crop_mask = np.zeros((10, 10), dtype=np.uint8)
    crop_mask[2:8, 2:8] = 255
    crop_path = tmp_path / "crop_mask.png"
    Image.fromarray(crop_mask).save(crop_path)

    output_path = tmp_path / "full_mask.png"
    _composite_mask(crop_path, output_path, x1=50, y1=30, width=200, height=100)

    full = np.array(Image.open(output_path))
    assert full.shape == (100, 200)
    assert full[32, 52] == 255
    assert full[0, 0] == 0


def test_run_roi_verification_offsets_boxes_and_composites_mask(tmp_path: Path) -> None:
    image = Image.new("RGB", (200, 150), "gray")
    proposal = Detection(
        box=(90.0, 60.0, 110.0, 90.0), score=0.8, label="person", metadata={"source_model": "ensemble"}
    )

    detections, elapsed = _run_roi_verification(FakeSamAdapter(), image, [proposal], "person", tmp_path, padding=1.0)

    assert elapsed >= 0
    assert len(detections) == 1
    result = detections[0]
    assert result.metadata["verification_of_box"] == [90.0, 60.0, 110.0, 90.0]
    assert result.mask_path is not None
    mask_file = tmp_path / result.mask_path
    assert mask_file.exists()
    composited = Image.open(mask_file)
    assert composited.size == image.size  # composited back to full image size, not crop size
    assert np.array(composited).max() == 255


def test_compare_image_cascade_fuses_agreeing_models_and_verifies_with_sam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (200, 150), "gray").save(image_path)
    output_dir = tmp_path / "out"

    monkeypatch.setitem(cascade._PROPOSAL_ADAPTERS, "yolo", lambda config: FakeDetectorAdapter("yolo"))
    monkeypatch.setitem(cascade._PROPOSAL_ADAPTERS, "rfdetr", lambda config: FakeDetectorAdapter("rfdetr"))
    monkeypatch.setattr(cascade, "MetaSam3Adapter", lambda config: FakeSamAdapter())

    config = LabConfig(root=tmp_path, raw={"sam3": {"backend": "official"}})

    payload = compare_image_cascade(
        config,
        image_path,
        output_dir,
        proposal_models=["yolo", "rfdetr"],
        sam_text="person",
        tile_size=1008,
    )

    assert payload["proposal_counts"]["yolo"] == 1
    assert payload["proposal_counts"]["rfdetr"] == 1
    assert payload["fused_proposal_count"] == 1  # identical boxes from both models merge
    assert payload["errors"] == {}
    assert len(payload["result"]["detections"]) == 1
    assert (output_dir / "cascade.json").exists()
    assert (output_dir / "cascade_annotated.jpg").exists()
    assert not (output_dir / "_scratch").exists()


def test_compare_image_cascade_rejects_unknown_proposal_model(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (50, 50), "gray").save(image_path)
    config = LabConfig(root=tmp_path, raw={"sam3": {"backend": "official"}})

    with pytest.raises(ValueError):
        compare_image_cascade(config, image_path, tmp_path / "out", proposal_models=["not_a_model"])


def test_compare_image_cascade_can_tile_sam_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (200, 150), "gray").save(image_path)
    output_dir = tmp_path / "out"

    monkeypatch.setitem(cascade._PROPOSAL_ADAPTERS, "yolo", lambda config: FakeDetectorAdapter("yolo"))
    monkeypatch.setitem(cascade._PROPOSAL_ADAPTERS, "rfdetr", lambda config: FakeDetectorAdapter("rfdetr"))
    monkeypatch.setattr(cascade, "MetaSam3Adapter", lambda config: FakeSamAdapter())

    config = LabConfig(root=tmp_path, raw={"sam3": {"backend": "official"}})

    payload = compare_image_cascade(
        config,
        image_path,
        output_dir,
        proposal_models=["yolo", "rfdetr"],
        sam_text="person",
        tile_size=1008,
        fuse_models=False,
        tile_sam=True,
    )

    assert payload["tile_sam"] is True
    assert payload["errors"] == {}
    assert any(result["model"] == "cascade(sam3-official-tiled)" for result in payload["results"])
    tiled = next(result for result in payload["results"] if result["model"] == "cascade(sam3-official-tiled)")
    assert tiled["metadata"]["sam_mode"] == "tiled"
    assert tiled["metadata"]["tile_count"] == 1
    assert len(tiled["detections"]) == 1
    assert (output_dir / "sam3_cascade_annotated.jpg").exists()
    assert any((output_dir / "sam3_tiled_masks").glob("tile_000_*.png"))
    assert not (output_dir / "_scratch").exists()


def test_compare_image_cascade_without_fusion_reports_each_model_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (200, 150), "gray").save(image_path)
    output_dir = tmp_path / "out"

    monkeypatch.setitem(cascade._PROPOSAL_ADAPTERS, "yolo", lambda config: FakeDetectorAdapter("yolo"))
    monkeypatch.setitem(cascade._PROPOSAL_ADAPTERS, "rfdetr", lambda config: FakeDetectorAdapter("rfdetr"))
    monkeypatch.setattr(cascade, "MetaSam3Adapter", lambda config: FakeSamAdapter())

    config = LabConfig(root=tmp_path, raw={"sam3": {"backend": "official"}})

    payload = compare_image_cascade(
        config,
        image_path,
        output_dir,
        proposal_models=["yolo", "rfdetr"],
        sam_text="person",
        tile_size=1008,
        fuse_models=False,
    )

    assert payload["fuse_models"] is False
    assert payload["fused_proposal_count"] is None
    assert payload["errors"] == {}
    assert len(payload["results"]) == 2
    models = {result["model"] for result in payload["results"]}
    assert models == {"cascade(yolo->sam3-official)", "cascade(rfdetr->sam3-official)"}
    assert all(len(result["detections"]) == 1 for result in payload["results"])
    assert (output_dir / "yolo_cascade_annotated.jpg").exists()
    assert (output_dir / "rfdetr_cascade_annotated.jpg").exists()
    assert not (output_dir / "cascade_annotated.jpg").exists()
    assert (output_dir / "cascade.json").exists()
    assert not (output_dir / "_scratch").exists()
