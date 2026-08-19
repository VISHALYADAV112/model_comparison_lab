from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .contracts import ModelResult, resolve_mask_path

COLORS = [
    (255, 80, 80),
    (80, 220, 120),
    (70, 150, 255),
    (255, 190, 70),
    (190, 100, 255),
    (60, 220, 220),
]


def _overlay_mask(canvas: Image.Image, mask_path: Path, color: tuple[int, int, int], alpha: float = 0.42) -> None:
    mask = Image.open(mask_path).convert("L")
    if mask.size != canvas.size:
        mask = mask.resize(canvas.size, Image.Resampling.NEAREST)
    layer = Image.new("RGBA", canvas.size, color + (0,))
    layer.putalpha(mask.point(lambda value: int(alpha * value)))
    canvas.alpha_composite(layer)


def render_result(image_path: Path, result: ModelResult, output_path: Path) -> Path:
    canvas = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(canvas)
    manifest_path = Path(str(result.metadata.get("manifest", output_path.parent / "manifest.json")))
    for index, detection in enumerate(result.detections):
        color = COLORS[index % len(COLORS)]
        mask = resolve_mask_path(manifest_path, detection.mask_path)
        if mask and mask.exists():
            _overlay_mask(canvas, mask, color)
        draw.rectangle(detection.box, outline=color + (255,), width=3)
        text = f"{detection.label} {detection.score:.2f}"
        x0, y0, _, _ = detection.box
        draw.text((x0 + 3, max(0, y0 - 15)), text, fill=color + (255,), font=ImageFont.load_default())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output_path, quality=94)
    return output_path


def render_sam_manifest(image_path: Path, manifest_path: Path, output_path: Path) -> Path:
    payload = json.loads(manifest_path.read_text())
    frame = payload["frames"][0]
    result = ModelResult(
        model="SAM 3",
        source=str(image_path),
        width=payload["width"],
        height=payload["height"],
        elapsed_seconds=payload.get("elapsed_seconds", 0),
        detections=[],
        metadata={"manifest": str(manifest_path)},
    )
    from .contracts import Detection

    result.detections = [
        Detection(
            box=tuple(item["box"]),
            score=float(item["score"]),
            label=payload.get("prompt") or f"instance {item.get('instance_id', index)}",
            mask_path=item.get("mask"),
            instance_id=item.get("instance_id"),
        )
        for index, item in enumerate(frame["detections"])
    ]
    return render_result(image_path, result, output_path)


def manifest_mask_files(manifest_path: Path) -> list[Path]:
    payload = json.loads(manifest_path.read_text())
    files: list[Path] = []
    for frame in payload.get("frames", []):
        for detection in frame.get("detections", []):
            mask = resolve_mask_path(manifest_path, detection.get("mask"))
            if mask and mask.exists():
                files.append(mask)
    return files


def _encode_browser_video(raw_video: Path, source_video: Path, output_path: Path) -> None:
    """Convert OpenCV's intermediate into an H.264 MP4 Gradio can play."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "Cannot create a browser-compatible result because ffmpeg is not on PATH. "
            "Activate model-lab-bootstrap and restart the dashboard."
        )
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(raw_video),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"FFmpeg could not encode the dashboard video: {message}") from exc
    finally:
        raw_video.unlink(missing_ok=True)


def render_video_manifest(video_path: Path, manifest_path: Path, output_path: Path) -> Path:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video rendering requires: pip install -e '.[playground]'") from exc
    payload = json.loads(manifest_path.read_text())
    indexed = {int(frame["frame_index"]): frame for frame in payload.get("frames", [])}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or float(payload.get("fps", 25.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output_path.with_name(f"{output_path.stem}.opencv.mp4")
    writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not create the annotated video")
    frame_index = 0
    last_index = max(indexed) if indexed else -1
    try:
        while frame_index <= last_index:
            ok, frame = capture.read()
            if not ok:
                break
            record = indexed.get(frame_index)
            if record:
                for index, detection in enumerate(record.get("detections", [])):
                    color = COLORS[index % len(COLORS)]
                    color_bgr = (color[2], color[1], color[0])
                    mask_path = resolve_mask_path(manifest_path, detection.get("mask"))
                    if mask_path and mask_path.exists():
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            if mask.shape[:2] != frame.shape[:2]:
                                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                            selected = mask > 127
                            overlay = np.empty_like(frame)
                            overlay[:] = color_bgr
                            frame[selected] = (0.58 * frame[selected] + 0.42 * overlay[selected]).astype(np.uint8)
                    x0, y0, x1, y1 = (int(value) for value in detection["box"])
                    cv2.rectangle(frame, (x0, y0), (x1, y1), color_bgr, 2)
                    cv2.putText(
                        frame,
                        f"id={detection.get('instance_id', index)} {detection['score']:.2f}",
                        (x0, max(15, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color_bgr,
                        1,
                        cv2.LINE_AA,
                    )
            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
    _encode_browser_video(raw_output, video_path, output_path)
    return output_path
