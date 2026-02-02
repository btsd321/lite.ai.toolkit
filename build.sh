#!/bin/bash

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
           -DTensorRT_DIR=/usr \
           -DENABLE_TEST=ON

else
  cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo \
           -DCMAKE_INSTALL_PREFIX=/eibot/environment/waybill_perception_cpp_env \
           -DENABLE_TEST=ON
fi

make -j16
sudo make install

# bash ./build.sh
# bash ./build.sh tensorrt
