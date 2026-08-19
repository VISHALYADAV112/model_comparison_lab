from model_lab.config import load_config


def test_default_config_paths_are_isolated() -> None:
    config = load_config()
    assert config.root.name == "model_comparison_lab"
    assert config.yolo_model.name == "yolo26l.pt"
    assert config.rfdetr_model.name == "rf-detr-large-2026.pth"
    assert config.sam3_q8_model.name == "sam3-q8_0.ggml"
    assert config.sam3_official_image_model.name == "sam3.pt"
    assert config.sam3_official_video_model.name == "sam3.1_multiplex.pt"
    assert config.models_dir.is_absolute()
    assert config.root in config.models_dir.parents


def test_sam_default_is_official() -> None:
    assert load_config().raw["sam3"]["backend"] == "official"


def test_bounded_video_defaults_are_conservative() -> None:
    settings = load_config().raw["bounded_video"]
    assert settings["chunk_frames"] == 60
    assert settings["overlap_frames"] == 8
    assert settings["grounding_batch_size"] == 1
    assert settings["max_active_objects"] == 16
    assert settings["rtsp_queue_capacity"] == 2
    assert settings["worker_timeout_seconds"] == 1800
    assert settings["state_history_frames"] == 32
    assert settings["frame_buffer_frames"] == 96
    assert settings["rtsp_frame_queue_capacity"] == 64
