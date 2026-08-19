#!/usr/bin/env bash
set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${LAB_ROOT}/runtime/sam3_cpp/source"
BUILD_DIR="${LAB_ROOT}/runtime/sam3_cpp/build"
REPOSITORY="https://github.com/PABannier/sam3.cpp.git"
REVISION="01832ef85fcc8eb6488f1d01cd247f07e96ff5a9"
BACKEND="${SAM3_CPP_BACKEND:-auto}"

if ! command -v git >/dev/null || ! command -v cmake >/dev/null; then
  echo "git and cmake are required" >&2
  exit 2
fi

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${SOURCE_DIR}")"
  git clone --recursive "${REPOSITORY}" "${SOURCE_DIR}"
  git -C "${SOURCE_DIR}" checkout "${REVISION}"
  git -C "${SOURCE_DIR}" submodule update --init --recursive
else
  ACTUAL_REVISION="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
  if [[ "${ACTUAL_REVISION}" != "${REVISION}" ]]; then
    echo "Existing sam3.cpp checkout is ${ACTUAL_REVISION}; expected ${REVISION}." >&2
    echo "Move runtime/sam3_cpp/source aside and run this script again." >&2
    exit 2
  fi
fi

if [[ "${BACKEND}" == "auto" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    BACKEND="metal"
  else
    BACKEND="cpu"
  fi
fi

if [[ "${BACKEND}" == "cuda" ]]; then
  echo "The pinned sam3.cpp runtime initializes only CPU and Apple Metal backends." >&2
  echo "Use the official SAM 3.1 backend for NVIDIA CUDA video inference." >&2
  exit 2
fi

CMAKE_OPTIONS=(
  -S "${LAB_ROOT}/cpp"
  -B "${BUILD_DIR}"
  -DSAM3_CPP_ROOT="${SOURCE_DIR}"
  -DCMAKE_BUILD_TYPE=Release
)
if [[ "${BACKEND}" == "metal" ]]; then
  CMAKE_OPTIONS+=( -DGGML_CUDA=OFF -DSAM3_METAL=ON -DGGML_METAL=ON )
else
  CMAKE_OPTIONS+=( -DGGML_CUDA=OFF -DSAM3_METAL=OFF -DGGML_METAL=OFF )
fi

cmake "${CMAKE_OPTIONS[@]}"
cmake --build "${BUILD_DIR}" --config Release --parallel
echo "Built ${BUILD_DIR}/sam3_bridge (${BACKEND})"
