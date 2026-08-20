from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from model_lab.config import LabConfig
from model_lab.playground import root_service


def _stub_root_modules(monkeypatch) -> dict:
    captured: dict = {}

    class FakePipeline:
        def __init__(self, config):
            captured["config"] = config

        def run_pil_image(self, image, input_id=None):
            return {
                "raw_model_detections": [
                    {"model": "rfdetr", "box_xyxy": [0, 0, 10, 20], "label": "person", "score": 0.9, "metadata": {}},
                    {"model": "grounding_dino", "box_xyxy": [0, 0, 10, 20], "label": "person", "score": 0.8, "metadata": {}},
                    {"model": "sam3_verify", "box_xyxy": [0, 0, 10, 20], "label": "person", "score": 0.95, "metadata": {}},
                ],
                "detections": [{"box_xyxy": [0, 0, 10, 20], "label": "person", "score": 0.95}],
                "timing_seconds": {"rfdetr": 0.5, "grounding_dino": 0.7, "sam3_verify": 0.3},
            }

    def fake_draw_report(image, report, output_path):
        Path(output_path).write_bytes(b"jpeg")

    def fake_save_report(report, path):
        Path(path).write_text(json.dumps(report), encoding="utf-8")

    def fake_run_video(pipeline, video, output_dir, settings):
        captured["settings"] = settings
        Path(output_dir, "annotated.mp4").write_bytes(b"mp4")
        Path(output_dir, "tracks.json").write_text(json.dumps({"tracks": []}), encoding="utf-8")
        return {
            "confirmed_track_count": 3,
            "tentative_track_count": 1,
            "outputs": {"annotated_video": str(Path(output_dir, "annotated.mp4"))},
        }

    class FakeVideoSettings:
        def __init__(self, **kwargs):
            captured["video_kwargs"] = kwargs

    monkeypatch.setattr(root_service, "LongRangePipeline", FakePipeline)
    monkeypatch.setattr(root_service, "draw_report", fake_draw_report)
    monkeypatch.setattr(root_service, "save_report", fake_save_report)
    monkeypatch.setattr(root_service, "run_video", fake_run_video)
    monkeypatch.setattr(root_service, "VideoSettings", FakeVideoSettings)
    return captured


def _config(tmp_path: Path) -> LabConfig:
    return LabConfig(root=tmp_path, raw={"paths": {"outputs_dir": str(tmp_path)}})


def _image(tmp_path: Path) -> str:
    path = tmp_path / "input.png"
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(path)
    return str(path)


def test_run_image_builds_root_config_and_stage_gallery(tmp_path, monkeypatch) -> None:
    captured = _stub_root_modules(monkeypatch)
    service = root_service.RootPipelineService(_config(tmp_path))
    summary, gallery, report_path, report = service.run_image(
        image=_image(tmp_path),
        prompts="person, vehicle",
        models=["rfdetr", "grounding_dino", "sam3_verify"],
        threshold=0.24,
        tile_size=1008,
        tile_overlap=0.2,
        nms_iou=0.45,
        ensemble_iou=0.5,
        roi_padding=2.0,
        device="auto",
    )
    config = captured["config"]
    assert config["run"]["prompt"] == ["person", "vehicle"]
    assert config["run"]["threshold"] == 0.24
    assert config["run"]["tile_size"] == 1008
    assert config["run"]["nms_iou"] == 0.45
    assert config["run"]["ensemble_iou"] == 0.5
    assert config["models"]["rfdetr"]["adapter"] == "rfdetr"
    assert config["models"]["rfdetr"]["role"] == "proposal"
    assert config["models"]["grounding_dino"]["text_threshold"] == 0.2
    assert config["models"]["sam3_verify"]["roi_padding"] == 2.0
    assert config["models"]["sam3_verify"]["role"] == "segment_verify"
    assert len(report["detections"]) == 1
    assert "RF-DETR Large" in summary
    assert "SAM 3 verify" in summary
    assert any(name.endswith("_proposals.jpg") for name in gallery)
    assert any(name.endswith("_fused.jpg") for name in gallery)
    assert Path(report_path).is_file()


def test_run_image_defaults_prompt_and_models(tmp_path, monkeypatch) -> None:
    captured = _stub_root_modules(monkeypatch)
    service = root_service.RootPipelineService(_config(tmp_path))
    service.run_image(
        image=_image(tmp_path),
        prompts="",
        models=[],
        threshold=0.24,
        tile_size=1008,
        tile_overlap=0.2,
        nms_iou=0.45,
        ensemble_iou=0.5,
        roi_padding=2.0,
        device="auto",
    )
    assert captured["config"]["run"]["prompt"] == ["person"]
    assert list(captured["config"]["models"]) == ["rfdetr", "grounding_dino"]


def test_run_video_maps_tracker_knobs(tmp_path, monkeypatch) -> None:
    captured = _stub_root_modules(monkeypatch)
    service = root_service.RootPipelineService(_config(tmp_path))
    video = tmp_path / "input.mp4"
    video.write_bytes(b"mp4")
    summary, annotated, tracks, report = service.run_video_job(
        video=str(video),
        prompts="person",
        models=["rfdetr"],
        threshold=0.24,
        tile_size=1008,
        tile_overlap=0.2,
        nms_iou=0.45,
        ensemble_iou=0.5,
        roi_padding=2.0,
        device="auto",
        detection_interval=5,
        min_hits=2,
        max_missed=2,
        association_iou=0.2,
        appearance_encoder="histogram",
        appearance_weight=0.35,
        appearance_momentum=0.85,
        appearance_batch_size=64,
        appearance_roi_padding=0.35,
        start_frame=0,
        max_frames=0,
    )
    kwargs = captured["video_kwargs"]
    assert kwargs["detection_interval"] == 5
    assert kwargs["min_hits"] == 2
    assert kwargs["max_missed_keyframes"] == 2
    assert kwargs["association_iou"] == 0.2
    assert kwargs["appearance_encoder"] == "histogram"
    assert kwargs["max_frames"] is None
    assert "3" in summary and "confirmed tracks" in summary
    assert Path(annotated).is_file()
    assert Path(tracks).is_file()
    assert report["confirmed_track_count"] == 3


def test_root_service_reports_missing_package_gracefully(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(root_service, "LongRangePipeline", None)
    service = root_service.RootPipelineService(_config(tmp_path))
    summary, gallery, report_path, payload = service.run_image(
        image=_image(tmp_path),
        prompts="person",
        models=["rfdetr"],
        threshold=0.24,
        tile_size=1008,
        tile_overlap=0.2,
        nms_iou=0.45,
        ensemble_iou=0.5,
        roi_padding=2.0,
        device="auto",
    )
    assert "not installed" in summary
    assert gallery == []
    assert report_path == ""
    assert payload["error"] == "root pipeline unavailable"
