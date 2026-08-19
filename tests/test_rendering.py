import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from model_lab.rendering import _encode_browser_video, render_sam_manifest


def test_manifest_renderer_overlays_relative_mask(tmp_path: Path) -> None:
    image = tmp_path / "source.png"
    mask = tmp_path / "mask.png"
    manifest = tmp_path / "manifest.json"
    Image.new("RGB", (20, 20), "black").save(image)
    mask_array = np.zeros((20, 20), dtype=np.uint8)
    mask_array[5:15, 5:15] = 255
    Image.fromarray(mask_array).save(mask)
    manifest.write_text(
        json.dumps(
            {
                "width": 20,
                "height": 20,
                "prompt": "target",
                "frames": [
                    {
                        "frame_index": 0,
                        "detections": [
                            {
                                "box": [5, 5, 15, 15],
                                "score": 0.9,
                                "instance_id": 1,
                                "mask": "mask.png",
                            }
                        ],
                    }
                ],
            }
        )
    )
    output = render_sam_manifest(image, manifest, tmp_path / "annotated.jpg")
    assert output.exists()
    assert Image.open(output).size == (20, 20)


@pytest.mark.parametrize("include_source_audio", [False, True])
def test_browser_encoder_supports_rtsp_video_only_and_file_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_source_audio: bool,
) -> None:
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"video")
    source = tmp_path / "source.mp4" if include_source_audio else None
    if source is not None:
        source.write_bytes(b"source")
    commands: list[list[str]] = []

    monkeypatch.setattr("model_lab.rendering.shutil.which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "model_lab.rendering.subprocess.run",
        lambda command, **_: commands.append(command),
    )

    _encode_browser_video(raw, source, tmp_path / "output.mp4")

    assert not raw.exists()
    assert len(commands) == 1
    assert commands[0].count("-i") == (2 if include_source_audio else 1)
    assert ("1:a:0?" in commands[0]) is include_source_audio
