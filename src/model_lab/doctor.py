from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys

from .config import LabConfig
from .downloader import model_status


def doctor_report(config: LabConfig) -> dict:
    commands = {name: shutil.which(name) for name in ("git", "cmake", "ffmpeg", "nvidia-smi", "nvcc")}
    cuda = None
    if commands["nvidia-smi"]:
        check = subprocess.run(
            [commands["nvidia-smi"], "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        cuda = check.stdout.strip() if check.returncode == 0 else check.stderr.strip()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: importlib.util.find_spec(module) is not None
            for name, module in {
                "ultralytics": "ultralytics",
                "rfdetr": "rfdetr",
                "huggingface_hub": "huggingface_hub",
                "gradio": "gradio",
                "opencv": "cv2",
                "official_sam3": "sam3",
            }.items()
        },
        "commands": commands,
        "gpu": cuda,
        "models": model_status(config),
    }
