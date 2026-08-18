from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any


LAB_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = LAB_ROOT / "configs" / "models.toml"


@dataclass(frozen=True)
class LabConfig:
    root: Path
    raw: dict[str, Any]

    @property
    def models_dir(self) -> Path:
        return self.resolve(self.raw["paths"]["models_dir"])

    @property
    def outputs_dir(self) -> Path:
        return self.resolve(self.raw["paths"]["outputs_dir"])

    @property
    def sam3_bridge(self) -> Path:
        return self.resolve(self.raw["paths"]["sam3_bridge"])

    @property
    def sam3_model(self) -> Path:
        return self.sam3_q8_model

    @property
    def sam3_q8_model(self) -> Path:
        return self.models_dir / "sam3" / "q8" / self.raw["sam3"]["q8_filename"]

    @property
    def sam3_official_image_model(self) -> Path:
        return self.models_dir / "sam3" / "official" / self.raw["sam3"]["official_image_filename"]

    @property
    def sam3_official_video_model(self) -> Path:
        return self.models_dir / "sam3" / "official" / self.raw["sam3"]["official_video_filename"]

    @property
    def rfdetr_model(self) -> Path:
        return self.models_dir / "rfdetr" / self.raw["rfdetr"]["checkpoint"]

    @property
    def yolo_model(self) -> Path:
        return self.models_dir / "yolo" / self.raw["yolo"]["checkpoint"]

    def resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()


def load_config(path: str | Path | None = None) -> LabConfig:
    config_path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    required = {"paths", "yolo", "rfdetr", "sam3", "playground"}
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(sorted(missing))}")
    root = config_path.parent.parent.resolve()
    return LabConfig(root=root, raw=raw)
