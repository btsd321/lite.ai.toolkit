cmake_minimum_required(VERSION 3.14)
project(yolo26_obb_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ENV_PREFIX /eibot/environment/waybill_perception_cpp_env)

# ---- lite.ai.toolkit ----
set(lite.ai.toolkit_DIR ${ENV_PREFIX}/lib/cmake/lite.ai.toolkit)
find_package(lite.ai.toolkit REQUIRED)

# ---- OpenCV ----
set(OpenCV_DIR ${ENV_PREFIX}/lib/cmake/opencv4)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc highgui)

# ---- argparse (header-only) ----
include_directories(${ENV_PREFIX}/include)

# ---- CUDA stub (libcuda.so is only in stubs on non-driver machines) ----
link_directories(/usr/local/cuda-12.9/targets/x86_64-linux/lib/stubs)

# ---- compile definitions ----
# ENABLE_TENSORRT is required for lite/models.h to expose trt namespace
add_compile_definitions(ENABLE_TENSORRT)

# ---- executable ----
add_executable(yolo26_obb_demo src/main.cpp)

target_include_directories(yolo26_obb_demo PRIVATE
    ${lite.ai.toolkit_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(yolo26_obb_demo PRIVATE
    ${lite.ai.toolkit_LIBS_DIRS}
)

target_link_libraries(yolo26_obb_demo PRIVATE
    ${lite.ai.toolkit_LIBS}
    ${OpenCV_LIBS}
)
