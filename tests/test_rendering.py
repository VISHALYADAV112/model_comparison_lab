import json
from pathlib import Path

import numpy as np
from PIL import Image

from model_lab.rendering import render_sam_manifest


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

