from __future__ import annotations

import gc
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from ..adapters.meta_sam3 import (
    MetaSam3Adapter,
    _numpy,
    close_video_session,
    configure_video_predictor,
    start_video_session,
    stream_video_responses,
)
from ..adapters.sam3_cpp import parse_boxes, parse_points
from ..config import LabConfig
from ..rendering import render_video_manifest


def _summary(response: dict[str, Any]) -> dict[str, Any]:
    output = response.get("outputs", {})
    ids = _numpy(output.get("out_obj_ids", [])).tolist()
    probabilities = _numpy(output.get("out_probs", [])).tolist()
    boxes = _numpy(output.get("out_boxes_xywh", [])).tolist()
    masks = _numpy(output.get("out_binary_masks", []))
    return {
        "frame_index": response.get("frame_index"),
        "object_ids": ids,
        "probabilities": probabilities,
        "boxes_xywh_normalized": boxes,
        "mask_array_shape": list(masks.shape),
    }


class MetaVideoSessionController:
    """Expose Meta's complete stable handle_request session surface to Gradio."""

    def __init__(self, config: LabConfig) -> None:
        self.config = config
        self.predictor = None
        self.sessions: dict[str, dict[str, Any]] = {}

    def _ensure_predictor(self):
        MetaSam3Adapter._require_cuda()
        if not self.config.sam3_official_video_model.exists():
            raise FileNotFoundError(
                f"Missing {self.config.sam3_official_video_model}. Download sam3-official first."
            )
        if self.predictor is None:
            from sam3.model_builder import build_sam3_predictor

            settings = self.config.raw["sam3"]
            self.predictor = build_sam3_predictor(
                checkpoint_path=str(self.config.sam3_official_video_model),
                version="sam3.1",
                compile=bool(settings.get("compile", False)),
                max_num_objects=int(settings.get("max_num_objects", 64)),
                multiplex_count=int(settings.get("multiplex_count", 16)),
                use_fa3=bool(settings.get("use_flash_attention_3", False)),
            )
            configure_video_predictor(
                self.predictor, int(settings.get("grounding_batch_size", 4))
            )
        return self.predictor

    def start(self, video: str, offload_video: bool, offload_state: bool) -> tuple[str, dict]:
        video_path = Path(video).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        predictor = self._ensure_predictor()
        response = start_video_session(
            predictor,
            video_path,
            offload_video_to_cpu=offload_video,
            offload_state_to_cpu=offload_state,
        )
        session_id = response["session_id"]
        width, height, fps = MetaSam3Adapter._video_info(video_path)
        self.sessions[session_id] = {
            "video": video_path,
            "width": width,
            "height": height,
            "fps": fps,
        }
        return session_id, {"operation": "start_session", "session_id": session_id, "active_sessions": len(self.sessions)}

    def _session(self, session_id: str) -> tuple[Any, dict[str, Any]]:
        if not session_id or session_id not in self.sessions:
            raise ValueError("Start a session or enter an active session ID")
        return self._ensure_predictor(), self.sessions[session_id]

    def add_prompt(
        self,
        session_id: str,
        frame_index: int,
        object_id: int,
        text: str,
        positive: str,
        negative: str,
        box: str,
        clear_points: bool,
        clear_boxes: bool,
        threshold: float,
    ) -> dict:
        predictor, _ = self._session(session_id)
        positives, negatives = parse_points(positive), parse_points(negative)
        points = positives + negatives
        boxes = parse_boxes(box)
        request: dict[str, Any] = {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": int(frame_index),
            "obj_id": int(object_id),
            "clear_old_points": clear_points,
            "clear_old_boxes": clear_boxes,
            "output_prob_thresh": threshold,
            "rel_coordinates": False,
        }
        if (text or "").strip():
            request["text"] = text.strip()
        if points:
            request["points"] = points
            request["point_labels"] = [1] * len(positives) + [0] * len(negatives)
        if boxes:
            x0, y0, x1, y1 = boxes[0]
            request["bounding_boxes"] = [[x0, y0, x1 - x0, y1 - y0]]
            request["bounding_box_labels"] = [1]
        if not any(key in request for key in ("text", "points", "bounding_boxes")):
            raise ValueError("Enter text, point(s), or a box")
        response = predictor.handle_request(request)
        return {"operation": "add_prompt", "session_id": session_id, **_summary(response)}

    def remove(self, session_id: str, frame_index: int, object_id: int) -> dict:
        predictor, _ = self._session(session_id)
        response = predictor.handle_request(
            {
                "type": "remove_object",
                "session_id": session_id,
                "frame_index": int(frame_index),
                "obj_id": int(object_id),
            }
        )
        return {"operation": "remove_object", "session_id": session_id, **_summary(response)}

    def reset(self, session_id: str) -> dict:
        predictor, _ = self._session(session_id)
        response = predictor.handle_request({"type": "reset_session", "session_id": session_id})
        return {"operation": "reset_session", "session_id": session_id, **response}

    def cancel(self, session_id: str) -> dict:
        predictor, _ = self._session(session_id)
        response = predictor.handle_request({"type": "cancel_propagation", "session_id": session_id})
        return {"operation": "cancel_propagation", "session_id": session_id, **response}

    def close(self, session_id: str) -> tuple[str, dict]:
        predictor, _ = self._session(session_id)
        close_video_session(predictor, session_id)
        response = {"is_success": True}
        self.sessions.pop(session_id, None)
        if not self.sessions:
            self.predictor = None
            del predictor
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        return "", {"operation": "close_session", "session_id": session_id, **response}

    def propagate(
        self,
        session_id: str,
        direction: str,
        start_frame: int,
        max_frames: int,
        threshold: float,
    ) -> tuple[str, str, str, dict]:
        predictor, session = self._session(session_id)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = self.config.outputs_dir / "playground" / f"{stamp}_live_session_{uuid4().hex[:8]}"
        output_dir.mkdir(parents=True, exist_ok=False)
        frames: list[dict[str, Any]] = []
        seen: set[int] = set()
        request = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": direction,
            "start_frame_index": int(start_frame),
            "output_prob_thresh": threshold,
        }
        for response in stream_video_responses(
            predictor, request, max_frames=int(max_frames)
        ):
            frame_index = int(response["frame_index"])
            if frame_index in seen:
                continue
            seen.add(frame_index)
            outputs = response["outputs"]
            records: list[dict[str, Any]] = []
            for object_id, probability, box_xywh, mask in zip(
                _numpy(outputs["out_obj_ids"]),
                _numpy(outputs["out_probs"]),
                _numpy(outputs["out_boxes_xywh"]),
                _numpy(outputs["out_binary_masks"]),
            ):
                x, y, width, height = [float(value) for value in box_xywh]
                box = [
                    x * session["width"],
                    y * session["height"],
                    (x + width) * session["width"],
                    (y + height) * session["height"],
                ]
                mask_name = f"frame_{frame_index:06d}_object_{int(object_id):04d}.png"
                Image.fromarray((np.asarray(mask).squeeze().astype(np.uint8) * 255), mode="L").save(
                    output_dir / mask_name
                )
                records.append(
                    {
                        "box": box,
                        "score": float(probability),
                        "iou_score": None,
                        "instance_id": int(object_id),
                        "mask": mask_name,
                    }
                )
            frames.append({"frame_index": frame_index, "detections": records})
        frames.sort(key=lambda value: value["frame_index"])
        payload = {
            "schema_version": 1,
            "runtime": "official-meta-sam3",
            "model": "SAM 3.1 Object Multiplex live session",
            "session_id": session_id,
            "source": str(session["video"]),
            "width": session["width"],
            "height": session["height"],
            "fps": session["fps"],
            "frames": frames,
        }
        manifest = output_dir / "manifest.json"
        manifest.write_text(json.dumps(payload, indent=2))
        annotated = render_video_manifest(session["video"], manifest, output_dir / "annotated.mp4")
        archive = shutil.make_archive(str(output_dir) + "_results", "zip", root_dir=output_dir)
        return str(annotated), str(manifest), archive, payload
