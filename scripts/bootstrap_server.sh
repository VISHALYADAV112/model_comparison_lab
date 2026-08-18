#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3.12}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"

if ! command -v "${PYTHON_COMMAND}" >/dev/null; then
  echo "Python 3.12 is required for official SAM 3. Install it, then rerun." >&2
  exit 2
fi
for command in git cmake ffmpeg; do
  if ! command -v "${command}" >/dev/null; then
    echo "Missing ${command}. On Ubuntu: sudo apt install -y git cmake build-essential ffmpeg" >&2
    exit 2
  fi
done

"${PYTHON_COMMAND}" -m venv "${LAB_ROOT}/.venv"
PYTHON_BIN="${LAB_ROOT}/.venv/bin/python"
"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel

# Version follows Meta's current SAM 3.1 installation guide (CUDA 12.8 wheel).
"${PYTHON_BIN}" -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
"${PYTHON_BIN}" -m pip install -e "${LAB_ROOT}[all]"

MODEL_LAB_PYTHON="${PYTHON_BIN}" "${LAB_ROOT}/scripts/install_meta_sam3.sh"
"${LAB_ROOT}/scripts/build_sam3_cpp.sh"

if [[ "${DOWNLOAD_MODELS}" == "1" ]]; then
  if ! "${LAB_ROOT}/.venv/bin/hf" auth whoami >/dev/null 2>&1 && [[ -z "${HF_TOKEN:-}" ]]; then
    echo "Authenticate on this server first: ${LAB_ROOT}/.venv/bin/hf auth login" >&2
    echo "Then rerun with: ${LAB_ROOT}/.venv/bin/model-lab models download --model all" >&2
    exit 3
  fi
  "${LAB_ROOT}/.venv/bin/model-lab" models download --model all
fi

"${LAB_ROOT}/.venv/bin/model-lab" doctor
echo "Bootstrap complete. Start the LAN UI with scripts/run_playground_lan.sh"

