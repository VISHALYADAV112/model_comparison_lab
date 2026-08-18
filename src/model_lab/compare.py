from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Iterable

from .adapters import MetaSam3Adapter, RFDetrAdapter, Sam3CppAdapter, YoloAdapter
from .config import LabConfig
from .contracts import ModelResult
from .rendering import render_result


def compare_image(
    config: LabConfig,
    image: Path,
    output_dir: Path,
    *,
    models: Iterable[str] = ("yolo", "rfdetr", "sam3"),
    sam_text: str = "object",
    sam_backend: str | None = None,
) -> dict:
    image = image.expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = list(dict.fromkeys(models))
    results: list[ModelResult] = []
    errors: dict[str, str] = {}
    for name in requested:
        try:
            if name == "yolo":
                adapter = YoloAdapter(config)
                result = adapter.predict_image(image, output_dir / name)
            elif name == "rfdetr":
                adapter = RFDetrAdapter(config)
                result = adapter.predict_image(image, output_dir / name)
            elif name == "sam3":
                selected_backend = sam_backend or config.raw["sam3"].get("backend", "official")
                if selected_backend == "official":
                    adapter = MetaSam3Adapter(config)
                elif selected_backend == "q8":
                    adapter = Sam3CppAdapter(config)
                else:
                    raise ValueError(f"Unknown SAM backend: {selected_backend}")
                result = adapter.predict_text_image(image, output_dir / name, sam_text)
            else:
                raise ValueError(f"Unknown model: {name}")
            render_result(image, result, output_dir / f"{name}_annotated.jpg")
            results.append(result)
            del adapter
            gc.collect()
        except Exception as exc:  # preserve other models' results in a benchmark run
            errors[name] = f"{type(exc).__name__}: {exc}"
    payload = {
        "schema_version": 1,
        "source": str(image),
        "sam_text_prompt": sam_text,
        "sam_backend": sam_backend or config.raw["sam3"].get("backend", "official"),
        "warning": (
            "This is an inference report, not an accuracy ranking. Add ground-truth annotations and run the "
            "evaluation command before drawing quality conclusions."
        ),
        "results": [result.to_dict() for result in results],
        "errors": errors,
    }
    (output_dir / "comparison.json").write_text(json.dumps(payload, indent=2))
    return payload
