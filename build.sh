#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to project root directory (where the script is located)
cd "${SCRIPT_DIR}" || exit 1
echo "Working directory: $(pwd)"

BUILD_DIR=build

if [ ! -d "${BUILD_DIR}" ]; then
  mkdir "${BUILD_DIR}"
  echo "creating build dir: ${BUILD_DIR} ..."
else
  echo "build dir: ${BUILD_DIR} directory exist! ..."
fi

cd "${BUILD_DIR}" && pwd

# ── 环境配置 ────────────────────────────────────────────────────────────────
ENV_PREFIX="/eibot/environment/waybill_perception_cpp_env"
CUDA_DIR="/usr/local/cuda-12.9"
TENSORRT_DIR="${ENV_PREFIX}/TensorRT-10.13.3.9"
# 留空则自动下载到 third_party/，非空则使用已有安装（支持包风格和系统前缀风格）
ONNXRUNTIME_DIR="${ENV_PREFIX}"
# ────────────────────────────────────────────────────────────────────────────

if [ "$1" == "tensorrt" ]; then
  cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
           -DCMAKE_INSTALL_PREFIX="${ENV_PREFIX}" \
           -DCMAKE_PREFIX_PATH="${ENV_PREFIX}" \
           -DENABLE_TENSORRT=ON \
           -DCUDA_DIR="${CUDA_DIR}" \
           -DTensorRT_DIR="${TENSORRT_DIR}" \
           -DOnnxRuntime_DIR="${ONNXRUNTIME_DIR}" \
           -DENABLE_TEST=OFF

else
  cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
           -DCMAKE_INSTALL_PREFIX="${ENV_PREFIX}" \
           -DCMAKE_PREFIX_PATH="${ENV_PREFIX}" \
           -DOnnxRuntime_DIR="${ONNXRUNTIME_DIR}" \
           -DENABLE_TEST=ON
fi

make -j16 || { echo "Build failed, skipping install."; exit 1; }
sudo make install

# bash ./build.sh
# bash ./build.sh tensorrt
