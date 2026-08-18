#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${LAB_ROOT}/runtime/meta_sam3/source"
REPOSITORY="https://github.com/facebookresearch/sam3.git"
REVISION="8f0b7f4d4e7eda2ed606ebde6702c93359ad01da"
PYTHON_BIN="${MODEL_LAB_PYTHON:-${LAB_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found at ${PYTHON_BIN}. Run scripts/bootstrap_server.sh first." >&2
  exit 2
fi

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${SOURCE_DIR}")"
  git clone "${REPOSITORY}" "${SOURCE_DIR}"
  git -C "${SOURCE_DIR}" checkout "${REVISION}"
else
  ACTUAL_REVISION="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
  if [[ "${ACTUAL_REVISION}" != "${REVISION}" ]]; then
    echo "Existing Meta SAM 3 checkout is ${ACTUAL_REVISION}; expected ${REVISION}." >&2
    echo "Move runtime/meta_sam3/source aside and run this script again." >&2
    exit 2
  fi
fi

"${PYTHON_BIN}" -m pip install -e "${SOURCE_DIR}"
echo "Installed official Meta SAM 3 from pinned revision ${REVISION}"

