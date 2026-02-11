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
if [ "$1" == "tensorrt" ]; then
  cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
           -DCMAKE_INSTALL_PREFIX=/eibot/environment/waybill_perception_cpp_env \
           -DENABLE_TENSORRT=ON \
           -DCUDA_DIR=/usr/local/cuda-12.9 \
           -DOpenCV_DIR=/eibot/environment/waybill_perception_cpp_env/lib/cmake/opencv4 \
           -DTensorRT_DIR=/usr \
           -DENABLE_TEST=ON

else
  cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
           -DCMAKE_INSTALL_PREFIX=/eibot/environment/waybill_perception_cpp_env \
           -DOpenCV_DIR=/eibot/environment/waybill_perception_cpp_env/lib/cmake/opencv4 \
           -DENABLE_TEST=ON
fi

make -j16
sudo make install

# bash ./build.sh
# bash ./build.sh tensorrt
