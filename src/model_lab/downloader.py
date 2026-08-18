from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Iterator

from .config import LabConfig


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def download_yolo(config: LabConfig) -> Path:
    target = config.yolo_model
    if target.exists():
        return target
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the YOLO extra first: pip install -e '.[yolo]'") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config.root / ".cache" / "ultralytics"))
    with working_directory(target.parent):
        YOLO(target.name)
    if not target.exists():
        raise RuntimeError(f"Ultralytics completed but {target} was not created")
    return target


def download_rfdetr(config: LabConfig) -> Path:
    target = config.rfdetr_model
    if target.exists():
        return target
    os.environ["RF_HOME"] = str(target.parent)
    try:
        from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
    except ImportError as exc:
        raise RuntimeError("Install the RF-DETR extra first: pip install -e '.[rfdetr]'") from exc
    variants = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }
    variant = str(config.raw["rfdetr"]["variant"]).lower()
    variants[variant](pretrain_weights=str(target))
    if not target.exists():
        raise RuntimeError(f"RF-DETR completed but {target} was not created")
    return target


def _hf_download(config: LabConfig, *, repo_id: str, filename: str, target: Path, token: bool | None) -> Path:
    if target.exists():
        return target
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface-hub first: pip install -e .") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=target.parent,
        token=token,
    )
    path = Path(downloaded)
    if not target.exists() and path.exists():
        raise RuntimeError(f"Hugging Face placed the file at {path}, expected {target}")
    return target


def download_sam3_q8(config: LabConfig) -> Path:
    settings = config.raw["sam3"]
    return _hf_download(
        config,
        repo_id=settings["q8_repo_id"],
        filename=settings["q8_filename"],
        target=config.sam3_q8_model,
        token=False,
    )


def download_sam3_official(config: LabConfig) -> Path:
    settings = config.raw["sam3"]
    _hf_download(
        config,
        repo_id=settings["official_image_repo_id"],
        filename=settings["official_image_filename"],
        target=config.sam3_official_image_model,
        token=None,
    )
    return _hf_download(
        config,
        repo_id=settings["official_video_repo_id"],
        filename=settings["official_video_filename"],
        target=config.sam3_official_video_model,
        token=None,
    )


def download_models(config: LabConfig, model: str) -> dict[str, Path]:
    functions = {
        "yolo": download_yolo,
        "rfdetr": download_rfdetr,
        "sam3-official": download_sam3_official,
        "sam3-q8": download_sam3_q8,
    }
    if model == "sam3":
        results = {
            "sam3-official": download_sam3_official(config),
            "sam3-q8": download_sam3_q8(config),
        }
        return results
    names = list(functions) if model == "all" else [model]
    if any(name not in functions for name in names):
        raise ValueError(f"Unknown model {model!r}")
    return {name: functions[name](config) for name in names}


def model_status(config: LabConfig) -> list[dict[str, object]]:
    values = [
        ("YOLO", config.yolo_model),
        ("RF-DETR", config.rfdetr_model),
        ("SAM 3 official image", config.sam3_official_image_model),
        ("SAM 3.1 official video", config.sam3_official_video_model),
        ("SAM 3 public Q8_0", config.sam3_q8_model),
        ("SAM 3 Q8 bridge", config.sam3_bridge),
    ]
    return [
        {
            "component": name,
            "ready": path.exists(),
            "path": str(path),
            "size_mb": round(path.stat().st_size / 1_000_000, 1) if path.is_file() else None,
        }
        for name, path in values
    ]
