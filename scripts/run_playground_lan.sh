#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${MODEL_LAB_HOST:-0.0.0.0}"
PORT="${MODEL_LAB_PORT:-7860}"

if [[ ! -x "${LAB_ROOT}/.venv/bin/model-lab" ]]; then
  echo "Run scripts/bootstrap_server.sh first." >&2
  exit 2
fi

echo "Starting on ${HOST}:${PORT}. Direct LAN URL: http://SERVER_LAN_IP:${PORT}"
echo "Safer SSH tunnel: ssh -N -L ${PORT}:127.0.0.1:${PORT} USER@SERVER_LAN_IP"
exec "${LAB_ROOT}/.venv/bin/model-lab" playground --host "${HOST}" --port "${PORT}"

