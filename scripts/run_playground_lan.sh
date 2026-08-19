#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${MODEL_LAB_HOST:-0.0.0.0}"
PORT="${MODEL_LAB_PORT:-7860}"
PYTHON_BIN="${LAB_ROOT}/.venv/bin/python"

# Reduce CUDA allocator fragmentation during SAM's changing video batch shapes.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# The Rocky Linux installation keeps ffmpeg/ffprobe in the bootstrap Conda
# environment. A fresh SSH login does not activate that environment, so make
# the dashboard independent of the current shell prompt when possible.
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  MEDIA_TOOL_DIRS=(
    "${CONDA_PREFIX:-}/bin"
    "${HOME}/miniforge3/envs/model-lab-bootstrap/bin"
    "${HOME}/miniforge3/bin"
    "${HOME}/miniconda3/envs/model-lab-bootstrap/bin"
    "${HOME}/miniconda3/bin"
  )
  for tool_dir in "${MEDIA_TOOL_DIRS[@]}"; do
    if [[ -x "${tool_dir}/ffmpeg" && -x "${tool_dir}/ffprobe" ]]; then
      export PATH="${tool_dir}:${PATH}"
      break
    fi
  done
fi

if [[ ! -x "${LAB_ROOT}/.venv/bin/model-lab" ]]; then
  echo "Run scripts/bootstrap_server.sh first." >&2
  exit 2
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "Video support needs both ffmpeg and ffprobe, but they are not on PATH." >&2
  echo "For this server, activate the existing environment and retry:" >&2
  echo "  source \"\$HOME/miniforge3/etc/profile.d/conda.sh\"" >&2
  echo "  conda activate model-lab-bootstrap" >&2
  exit 3
fi

if ! "${PYTHON_BIN}" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("pkg_resources") else 1)'; then
  echo "Official SAM needs the pkg_resources compatibility module." >&2
  echo "Repair it with: .venv/bin/python -m pip install 'setuptools<82'" >&2
  exit 3
fi

if ! "${PYTHON_BIN}" -c 'from sam3.model.sam3_image_processor import Sam3Processor'; then
  echo "The official SAM image runtime is incomplete." >&2
  echo "Repair it with: .venv/bin/python -m pip install -e \".[all]\"" >&2
  exit 3
fi

echo "Starting the vision dashboard on ${HOST}:${PORT}."
if [[ "${HOST}" == "127.0.0.1" || "${HOST}" == "localhost" ]]; then
  echo "Keep this terminal open. On the laptop, run in a second terminal:"
  echo "ssh -N -L ${PORT}:127.0.0.1:${PORT} USER@SERVER_LAN_IP"
  echo "Then open http://127.0.0.1:${PORT} in the laptop browser."
else
  echo "Direct LAN URL: http://SERVER_LAN_IP:${PORT}"
fi
exec "${LAB_ROOT}/.venv/bin/model-lab" playground --host "${HOST}" --port "${PORT}"
