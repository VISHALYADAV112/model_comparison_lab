from __future__ import annotations

import importlib
import platform
import shutil
import subprocess
import sys

from .config import LabConfig
from .downloader import model_status


def _package_report() -> tuple[dict[str, bool], dict[str, str]]:
    modules = {
        "ultralytics": "ultralytics",
        "rfdetr": "rfdetr",
        "huggingface_hub": "huggingface_hub",
        "gradio": "gradio",
        "opencv": "cv2",
        # Import the real image entry point. A shallow `import sam3` can leave
        # partially initialized modules behind after a missing dependency.
        "official_sam3": "sam3.model.sam3_image_processor",
        "official_sam3_pkg_resources": "pkg_resources",
    }
    ready: dict[str, bool] = {}
    errors: dict[str, str] = {}
    for name, module in modules.items():
        try:
            importlib.import_module(module)
            ready[name] = True
        except Exception as exc:  # noqa: BLE001 - diagnostics must preserve every import failure
            ready[name] = False
            errors[name] = f"{type(exc).__name__}: {exc}"
    return ready, errors


def doctor_report(config: LabConfig) -> dict:
    commands = {
        name: shutil.which(name)
        for name in ("git", "cmake", "ffmpeg", "ffprobe", "nvidia-smi", "nvcc")
    }
    cuda = None
    if commands["nvidia-smi"]:
        check = subprocess.run(
            [commands["nvidia-smi"], "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            check=False,
            text=True,
        )
        cuda = check.stdout.strip() if check.returncode == 0 else check.stderr.strip()
    packages, package_errors = _package_report()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "package_errors": package_errors,
        "commands": commands,
        "gpu": cuda,
        "models": model_status(config),
    }
