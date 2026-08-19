#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${MODEL_LAB_HOST:-0.0.0.0}"
PORT="${MODEL_LAB_PORT:-7860}"
PYTHON_BIN="${LAB_ROOT}/.venv/bin/python"

if [[ ! -x "${LAB_ROOT}/.venv/bin/model-lab" ]]; then
  echo "Run scripts/bootstrap_server.sh first." >&2
  exit 2
fi

if ! "${PYTHON_BIN}" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("pkg_resources") else 1)'; then
  echo "Official SAM needs the pkg_resources compatibility module." >&2
  echo "Repair it with: .venv/bin/python -m pip install 'setuptools<82'" >&2
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
