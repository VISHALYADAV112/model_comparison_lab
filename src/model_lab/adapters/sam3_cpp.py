from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

from ..config import LabConfig
from ..contracts import Detection, ModelResult


def parse_points(value: str | None) -> list[tuple[float, float]]:
    if not value:
        return []
    points: list[tuple[float, float]] = []
    for item in filter(None, (part.strip() for part in value.split(";"))):
        fields = [float(field.strip()) for field in item.split(",")]
        if len(fields) != 2:
            raise ValueError(f"Point must be x,y; got {item!r}")
        points.append((fields[0], fields[1]))
    return points


def parse_boxes(value: str | None) -> list[tuple[float, float, float, float]]:
    if not value:
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for item in filter(None, (part.strip() for part in value.split(";"))):
        fields = [float(field.strip()) for field in item.split(",")]
        if len(fields) != 4:
            raise ValueError(f"Box must be x0,y0,x1,y1; got {item!r}")
        boxes.append(tuple(fields))
    return boxes


class Sam3CppAdapter:
    def __init__(self, config: LabConfig) -> None:
        self.config = config

    def _base_command(self, mode: str, output_dir: Path, use_gpu: bool | None) -> list[str]:
        if not self.config.sam3_bridge.exists():
            raise FileNotFoundError(
                f"Missing SAM 3 runtime {self.config.sam3_bridge}. Run: scripts/build_sam3_cpp.sh"
            )
        if not self.config.sam3_model.exists():
            raise FileNotFoundError(
                f"Missing SAM 3 Q8 model {self.config.sam3_model}. "
                "Run: model-lab models download --model sam3"
            )
        sam = self.config.raw["sam3"]
        command = [
            str(self.config.sam3_bridge),
            mode,
            "--model",
            str(self.config.sam3_model),
            "--output-dir",
            str(output_dir),
            "--threads",
            str(sam["threads"]),
        ]
        encode_size = int(sam.get("encode_image_size", 0))
        if encode_size:
            command.extend(["--encode-img-size", str(encode_size)])
        enabled = bool(sam.get("use_gpu", True)) if use_gpu is None else use_gpu
        if not enabled:
            command.append("--no-gpu")
        return command

    @staticmethod
    def _append_many(command: list[str], flag: str, values: Iterable[tuple[float, ...]]) -> None:
        for value in values:
            command.extend([flag, ",".join(f"{field:g}" for field in value)])

    def run_image(
        self,
        image: Path,
        output_dir: Path,
        *,
        mode: str = "text",
        text: str = "",
        positive_points: Iterable[tuple[float, float]] = (),
        negative_points: Iterable[tuple[float, float]] = (),
        box: tuple[float, float, float, float] | None = None,
        positive_exemplars: Iterable[tuple[float, float, float, float]] = (),
        negative_exemplars: Iterable[tuple[float, float, float, float]] = (),
        multimask: bool = False,
        score_threshold: float | None = None,
        nms_threshold: float | None = None,
        use_gpu: bool | None = None,
    ) -> tuple[Path, dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        command = self._base_command("image", output_dir, use_gpu)
        command.extend(["--image", str(image), "--prompt-mode", mode])
        if text:
            command.extend(["--text", text])
        self._append_many(command, "--positive", positive_points)
        self._append_many(command, "--negative", negative_points)
        self._append_many(command, "--pos-exemplar", positive_exemplars)
        self._append_many(command, "--neg-exemplar", negative_exemplars)
        if box:
            command.extend(["--box", ",".join(f"{value:g}" for value in box)])
        if multimask:
            command.append("--multimask")
        sam = self.config.raw["sam3"]
        command.extend(
            [
                "--score-threshold",
                str(score_threshold if score_threshold is not None else sam["score_threshold"]),
                "--nms-threshold",
                str(nms_threshold if nms_threshold is not None else sam["nms_threshold"]),
            ]
        )
        subprocess.run(command, check=True)
        manifest = output_dir / "manifest.json"
        return manifest, json.loads(manifest.read_text())

    def run_video(
        self,
        video: Path,
        output_dir: Path,
        *,
        mode: str,
        text: str = "",
        objects: Iterable[str] = (),
        refinements: Iterable[str] = (),
        start_frame: int = 0,
        max_frames: int = 0,
        score_threshold: float | None = None,
        nms_threshold: float | None = None,
        association_iou: float = 0.1,
        max_keep_alive: int = 30,
        recondition_every: int = 16,
        fill_hole_area: int = 16,
        use_gpu: bool | None = None,
    ) -> tuple[Path, dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        command = self._base_command("video", output_dir, use_gpu)
        command.extend(
            [
                "--video",
                str(video),
                "--prompt-mode",
                mode,
                "--start-frame",
                str(start_frame),
                "--max-frames",
                str(max_frames),
                "--assoc-iou",
                str(association_iou),
                "--max-keep-alive",
                str(max_keep_alive),
                "--recondition-every",
                str(recondition_every),
                "--fill-hole-area",
                str(fill_hole_area),
            ]
        )
        if text:
            command.extend(["--text", text])
        for item in objects:
            command.extend(["--object", item])
        for item in refinements:
            command.extend(["--refine", item])
        sam = self.config.raw["sam3"]
        command.extend(
            [
                "--score-threshold",
                str(score_threshold if score_threshold is not None else sam["score_threshold"]),
                "--nms-threshold",
                str(nms_threshold if nms_threshold is not None else sam["nms_threshold"]),
            ]
        )
        subprocess.run(command, check=True)
        manifest = output_dir / "manifest.json"
        return manifest, json.loads(manifest.read_text())

    def image_result(self, image: Path, manifest_path: Path, payload: dict, elapsed: float) -> ModelResult:
        frame = payload["frames"][0]
        detections = [
            Detection(
                box=tuple(item["box"]),
                score=float(item["score"]),
                label=str(payload.get("prompt") or "SAM 3 object"),
                mask_path=item.get("mask"),
                instance_id=item.get("instance_id"),
                metadata={"iou_score": item.get("iou_score")},
            )
            for item in frame["detections"]
        ]
        return ModelResult(
            model="SAM 3/sam3.cpp Q8_0",
            source=str(image),
            width=int(payload["width"]),
            height=int(payload["height"]),
            elapsed_seconds=elapsed,
            detections=detections,
            metadata={"task": "promptable segmentation", "manifest": str(manifest_path)},
        )

    def predict_text_image(self, image: Path, output_dir: Path, text: str) -> ModelResult:
        start = perf_counter()
        manifest, payload = self.run_image(image, output_dir, mode="text", text=text)
        return self.image_result(image, manifest, payload, perf_counter() - start)
