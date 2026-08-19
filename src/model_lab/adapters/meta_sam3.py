from __future__ import annotations

import gc
import importlib.util
import json
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from ..config import LabConfig
from ..contracts import Detection, ModelResult


def _box_from_mask(mask: np.ndarray) -> tuple[float, float, float, float]:
    rows, columns = np.where(mask)
    if not len(columns):
        return (0.0, 0.0, 0.0, 0.0)
    return (float(columns.min()), float(rows.min()), float(columns.max() + 1), float(rows.max() + 1))


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # NumPy has no bfloat16 dtype. SAM runs under BF16 autocast on CUDA,
        # so convert floating outputs to FP32 only after inference is finished.
        if value.is_floating_point():
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def _xyxy_to_normalized_cxcywh(box: tuple[float, float, float, float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = box
    return [
        ((x0 + x1) / 2) / width,
        ((y0 + y1) / 2) / height,
        (x1 - x0) / width,
        (y1 - y0) / height,
    ]


def parse_prompt_spec(value: str) -> dict[str, Any]:
    """Parse `id:2;frame:10;p:1,2;n:3,4;b:0,0,20,30` prompt syntax."""
    result: dict[str, Any] = {"positive": [], "negative": [], "box": None}
    for raw in filter(None, (item.strip() for item in value.split(";"))):
        if ":" not in raw:
            raise ValueError(f"Prompt item needs key:value syntax: {raw!r}")
        key, raw_value = raw.split(":", 1)
        key = key.strip().lower()
        if key in {"id", "frame"}:
            result[key] = int(raw_value)
        elif key in {"p", "n"}:
            point = tuple(float(item) for item in raw_value.split(","))
            if len(point) != 2:
                raise ValueError(f"Point must be x,y: {raw!r}")
            result["positive" if key == "p" else "negative"].append(point)
        elif key == "b":
            box = tuple(float(item) for item in raw_value.split(","))
            if len(box) != 4:
                raise ValueError(f"Box must be x0,y0,x1,y1: {raw!r}")
            result["box"] = box
        else:
            raise ValueError(f"Unknown prompt key {key!r}")
    return result


class MetaSam3Adapter:
    """Official Meta SAM 3 image and SAM 3.1 Object Multiplex video adapter."""

    def __init__(self, config: LabConfig) -> None:
        self.config = config

    @staticmethod
    def _require_cuda() -> None:
        if importlib.util.find_spec("pkg_resources") is None:
            raise RuntimeError(
                "Official SAM compatibility dependency is missing. Run: "
                ".venv/bin/python -m pip install 'setuptools<82'"
            )
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch/SAM 3 is not installed. Run scripts/install_meta_sam3.sh") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("The official SAM 3.1 backend requires an NVIDIA CUDA server")

    @staticmethod
    def _release_cuda(*objects: Any) -> None:
        del objects
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def _image_inference_context() -> Any:
        """Match Meta's required mixed-precision context for SAM 3 images."""
        import torch

        # Meta enables these settings in its official image example for Ampere
        # and newer GPUs. Scope autocast to this request so its precision state
        # cannot leak into YOLO, RF-DETR, or another Gradio worker.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    def _write_image_manifest(
        self,
        *,
        image: Path,
        output_dir: Path,
        prompt: str,
        prompt_mode: str,
        masks: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        elapsed: float,
        low_res_logits: np.ndarray | None = None,
    ) -> tuple[Path, dict]:
        width, height = Image.open(image).size
        records: list[dict[str, Any]] = []
        for index, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            mask = np.asarray(mask).squeeze().astype(bool)
            mask_name = f"mask_{index:03d}.png"
            _save_mask(mask, output_dir / mask_name)
            records.append(
                {
                    "box": [float(value) for value in box],
                    "score": float(score),
                    "iou_score": float(score) if prompt_mode == "visual" else None,
                    "instance_id": index,
                    "mask": mask_name,
                }
            )
        logits_name = None
        if low_res_logits is not None:
            logits_name = "low_res_logits.npy"
            logits = np.asarray(low_res_logits)
            best_index = int(np.argmax(scores)) if len(scores) else 0
            np.save(output_dir / logits_name, logits[best_index : best_index + 1])
            if len(logits) > 1:
                np.save(output_dir / "low_res_logits_all.npy", logits)
        payload = {
            "schema_version": 1,
            "runtime": "official-meta-sam3",
            "model": "SAM 3 image",
            "prompt_mode": prompt_mode,
            "prompt": prompt,
            "source": str(image),
            "width": width,
            "height": height,
            "elapsed_seconds": elapsed,
            "low_res_logits": logits_name,
            "frames": [{"frame_index": 0, "detections": records}],
        }
        manifest = output_dir / "manifest.json"
        manifest.write_text(json.dumps(payload, indent=2))
        return manifest, payload

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
        mask_input: Path | None = None,
        confidence_threshold: float | None = None,
    ) -> tuple[Path, dict]:
        self._require_cuda()
        if not self.config.sam3_official_image_model.exists():
            raise FileNotFoundError(
                f"Missing {self.config.sam3_official_image_model}. "
                "Run: model-lab models download --model sam3-official"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        pil_image = Image.open(image).convert("RGB")
        positive_exemplars = list(positive_exemplars)
        negative_exemplars = list(negative_exemplars)
        threshold = float(
            confidence_threshold
            if confidence_threshold is not None
            else self.config.raw["sam3"]["score_threshold"]
        )
        start = perf_counter()
        model = build_sam3_image_model(
            checkpoint_path=str(self.config.sam3_official_image_model),
            load_from_HF=False,
            device="cuda",
            enable_inst_interactivity=(mode == "visual"),
        )
        try:
            with self._image_inference_context():
                if mode == "text":
                    if not text and not positive_exemplars and not negative_exemplars:
                        raise ValueError("Text mode needs text and/or an exemplar box")
                    processor = Sam3Processor(model, device="cuda", confidence_threshold=threshold)
                    state = processor.set_image(pil_image)
                    if text:
                        state = processor.set_text_prompt(text, state)
                    for exemplar in positive_exemplars:
                        state = processor.add_geometric_prompt(
                            _xyxy_to_normalized_cxcywh(exemplar, pil_image.width, pil_image.height), True, state
                        )
                    for exemplar in negative_exemplars:
                        state = processor.add_geometric_prompt(
                            _xyxy_to_normalized_cxcywh(exemplar, pil_image.width, pil_image.height), False, state
                        )
                    masks = _numpy(state["masks"])
                    boxes = _numpy(state["boxes"])
                    scores = _numpy(state["scores"])
                    low_res = None
                elif mode == "visual":
                    predictor = model.inst_interactive_predictor
                    predictor.set_image(pil_image)
                    positives = list(positive_points)
                    negatives = list(negative_points)
                    all_points = positives + negatives
                    point_coords = np.asarray(all_points, dtype=np.float32) if all_points else None
                    point_labels = (
                        np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
                        if all_points
                        else None
                    )
                    prompt_box = np.asarray(box, dtype=np.float32) if box else None
                    previous_logits = np.load(mask_input) if mask_input else None
                    if point_coords is None and prompt_box is None and previous_logits is None:
                        raise ValueError("Visual mode needs a point, box, or previous low-resolution mask")
                    masks, scores, low_res = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=prompt_box,
                        mask_input=previous_logits,
                        multimask_output=multimask,
                    )
                    masks = _numpy(masks)
                    scores = _numpy(scores)
                    low_res = _numpy(low_res) if low_res is not None else None
                    boxes = np.asarray([_box_from_mask(np.asarray(mask).squeeze().astype(bool)) for mask in masks])
                else:
                    raise ValueError(f"Unknown SAM 3 image mode {mode!r}")
            return self._write_image_manifest(
                image=image,
                output_dir=output_dir,
                prompt=text,
                prompt_mode=mode,
                masks=masks,
                boxes=boxes,
                scores=scores,
                elapsed=perf_counter() - start,
                low_res_logits=low_res,
            )
        finally:
            del model
            self._release_cuda()

    @staticmethod
    def _video_info(video: Path) -> tuple[int, int, float]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Video support requires opencv-python-headless") from exc
        capture = cv2.VideoCapture(str(video))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open video {video}")
            return (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                float(capture.get(cv2.CAP_PROP_FPS) or 25.0),
            )
        finally:
            capture.release()

    @staticmethod
    def _prompt_request(session_id: str, spec: dict[str, Any], *, default_frame: int, obj_id: int) -> dict:
        positive, negative = spec["positive"], spec["negative"]
        points = positive + negative
        request: dict[str, Any] = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": int(spec.get("frame", default_frame)),
            "obj_id": int(spec.get("id", obj_id)),
            "rel_coordinates": False,
        }
        if points:
            request["points"] = points
            request["point_labels"] = [1] * len(positive) + [0] * len(negative)
        if spec.get("box"):
            x0, y0, x1, y1 = spec["box"]
            request["bounding_boxes"] = [[x0, y0, x1 - x0, y1 - y0]]
            request["bounding_box_labels"] = [1]
        return request

    def run_video(
        self,
        video: Path,
        output_dir: Path,
        *,
        mode: str = "text",
        text: str = "",
        objects: Iterable[str] = (),
        refinements: Iterable[str] = (),
        removals: Iterable[str] = (),
        start_frame: int = 0,
        max_frames: int = 0,
        propagation_direction: str = "forward",
        output_prob_threshold: float = 0.5,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
    ) -> tuple[Path, dict]:
        self._require_cuda()
        if not self.config.sam3_official_video_model.exists():
            raise FileNotFoundError(
                f"Missing {self.config.sam3_official_video_model}. "
                "Run: model-lab models download --model sam3-official"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        from sam3.model_builder import build_sam3_predictor

        width, height, fps = self._video_info(video)
        settings = self.config.raw["sam3"]
        start = perf_counter()
        predictor = build_sam3_predictor(
            checkpoint_path=str(self.config.sam3_official_video_model),
            version="sam3.1",
            compile=bool(settings.get("compile", False)),
            max_num_objects=int(settings.get("max_num_objects", 64)),
            multiplex_count=int(settings.get("multiplex_count", 16)),
            use_fa3=bool(settings.get("use_flash_attention_3", False)),
        )
        session_id: str | None = None
        frames: list[dict[str, Any]] = []
        try:
            response = predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(video),
                    "offload_video_to_cpu": offload_video_to_cpu,
                    "offload_state_to_cpu": offload_state_to_cpu,
                }
            )
            session_id = response["session_id"]
            if mode == "text":
                if not text:
                    raise ValueError("Text video mode requires --text")
                predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": start_frame,
                        "text": text,
                        "output_prob_thresh": output_prob_threshold,
                    }
                )
            elif mode == "visual":
                object_specs = [parse_prompt_spec(value) for value in objects]
                if not object_specs:
                    raise ValueError("Visual video mode needs at least one object prompt")
                for index, spec in enumerate(object_specs):
                    predictor.handle_request(self._prompt_request(session_id, spec, default_frame=start_frame, obj_id=index))
            else:
                raise ValueError(f"Unknown video mode {mode!r}")

            for value in refinements:
                spec = parse_prompt_spec(value)
                if "id" not in spec:
                    raise ValueError("A refinement must include id:<object id>")
                predictor.handle_request(
                    self._prompt_request(session_id, spec, default_frame=start_frame, obj_id=int(spec["id"]))
                )
            for value in removals:
                frame_value, object_value = (int(item.strip()) for item in value.split(":"))
                predictor.handle_request(
                    {
                        "type": "remove_object",
                        "session_id": session_id,
                        "frame_index": frame_value,
                        "obj_id": object_value,
                    }
                )

            stream_request = {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": propagation_direction,
                "start_frame_index": start_frame,
                "max_frame_num_to_track": max_frames or None,
                "output_prob_thresh": output_prob_threshold,
            }
            seen: set[int] = set()
            for response in predictor.handle_stream_request(stream_request):
                frame_index = int(response["frame_index"])
                if frame_index in seen:
                    continue
                seen.add(frame_index)
                output = response["outputs"]
                object_ids = _numpy(output["out_obj_ids"])
                scores = _numpy(output["out_probs"])
                boxes_xywh = _numpy(output["out_boxes_xywh"])
                masks = _numpy(output["out_binary_masks"])
                detections: list[dict[str, Any]] = []
                for index, (object_id, score, box_xywh, mask) in enumerate(
                    zip(object_ids, scores, boxes_xywh, masks)
                ):
                    x, y, box_width, box_height = [float(value) for value in box_xywh]
                    # Meta video outputs normalized xywh.
                    box = [x * width, y * height, (x + box_width) * width, (y + box_height) * height]
                    mask_name = f"frame_{frame_index:06d}_object_{int(object_id):04d}.png"
                    _save_mask(np.asarray(mask).squeeze().astype(bool), output_dir / mask_name)
                    detections.append(
                        {
                            "box": box,
                            "score": float(score),
                            "iou_score": None,
                            "instance_id": int(object_id),
                            "mask": mask_name,
                        }
                    )
                frames.append({"frame_index": frame_index, "detections": detections})
        finally:
            if session_id is not None:
                predictor.handle_request({"type": "close_session", "session_id": session_id})
            del predictor
            self._release_cuda()
        frames.sort(key=lambda item: item["frame_index"])
        payload = {
            "schema_version": 1,
            "runtime": "official-meta-sam3",
            "model": "SAM 3.1 Object Multiplex",
            "prompt_mode": mode,
            "prompt": text,
            "source": str(video),
            "width": width,
            "height": height,
            "fps": fps,
            "elapsed_seconds": perf_counter() - start,
            "frames": frames,
        }
        manifest = output_dir / "manifest.json"
        manifest.write_text(json.dumps(payload, indent=2))
        return manifest, payload

    def predict_text_image(self, image: Path, output_dir: Path, text: str) -> ModelResult:
        start = perf_counter()
        manifest, payload = self.run_image(image, output_dir, mode="text", text=text)
        frame = payload["frames"][0]
        return ModelResult(
            model="SAM 3 official",
            source=str(image),
            width=int(payload["width"]),
            height=int(payload["height"]),
            elapsed_seconds=perf_counter() - start,
            detections=[
                Detection(
                    box=tuple(item["box"]),
                    score=float(item["score"]),
                    label=text,
                    mask_path=item["mask"],
                    instance_id=item["instance_id"],
                )
                for item in frame["detections"]
            ],
            metadata={"task": "open-vocabulary segmentation", "manifest": str(manifest)},
        )
