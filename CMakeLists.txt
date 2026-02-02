cmake_minimum_required(VERSION 3.10)
project(lite.ai.toolkit)

set(CMAKE_CXX_STANDARD 17)
set(VERSION_STRING 0.3.1)
set(SOVERSION_STRING 0.3.1)

# Set default install prefix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/usr/local/lite.ai.toolkit" CACHE PATH "Default install prefix" FORCE)
endif()

include(cmake/utils.cmake)

if (NOT (UNIX AND NOT APPLE))
  message(FATAL_ERROR "lite.ai.toolkit>=0.2 not support for windows/mac now!")
endif()

set(THIRD_PARTY_PATH "${CMAKE_SOURCE_DIR}/third_party")
if(NOT EXISTS ${THIRD_PARTY_PATH})
    file(MAKE_DIRECTORY ${THIRD_PARTY_PATH})
endif()

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-std=c++17 -Wno-deprecated ${CMAKE_CXX_FLAGS} ")
    message(STATUS "[Linux GCC Compiler Options]+:-std=c++17 -Wno-deprecated")
endif ()
# 指定CUDA编译器
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 89) # For RTX 20xx series

# 启用 CUDA
enable_language(CUDA)
set(LITE_AI_ROOT_DIR ${CMAKE_SOURCE_DIR})
# 将 CMAKE_SOURCE_DIR 传递给代码
add_definitions(-DSOURCE_PATH="${CMAKE_SOURCE_DIR}")

option(ENABLE_TEST "build test examples." OFF)
option(ENABLE_DEBUG_STRING "enable DEBUG string or not" ON)
option(ENABLE_ONNXRUNTIME "enable ONNXRuntime engine" ON)
option(ENABLE_TENSORRT "enable TensorRT engine" ON)
option(ENABLE_MNN "enable MNN engine" OFF)
option(ENABLE_NCNN "enable NCNN engine" OFF) 
option(ENABLE_TNN "enable TNN engine" OFF)
option(ENABLE_ONNXRUNTIME_CUDA "enable ONNXRuntime engine with CUDA provider" OFF) # for future use
option(ENABLE_OPENCV_VIDEOIO "enable opencv videoio modules for detect_video apis" ON) # now, ON only

# Set TensorRT and CUDA paths
set(TensorRT_DIR "/usr" CACHE PATH "TensorRT installation directory")
set(CUDA_DIR "/usr/local/cuda" CACHE PATH "CUDA installation directory")

if ((NOT ENABLE_ONNXRUNTIME) AND (NOT ENABLE_MNN))
    message(FATAL_ERROR "One of ONNXRuntime/MNN Backend must be enable!")
endif()

if ((NOT ENABLE_ONNXRUNTIME) AND ENABLE_TEST)
    set(ENABLE_TEST OFF CACHE BOOL "ENABLE_TEST" FORCE)
    message(WARNING "Only support build tests while ONNXRuntime backend is enable, force ENABLE_TEST=OFF.")
endif()

include_directories(${LITE_AI_ROOT_DIR})
configure_file(cmake/lite.ai.toolkit.cmake.in ${CMAKE_BINARY_DIR}/lite.ai.toolkit.cmake @ONLY)

include(cmake/opencv.cmake)

add_lite_ai_toolkit_shared_library(${VERSION_STRING} ${SOVERSION_STRING})
install(TARGETS lite.ai.toolkit LIBRARY DESTINATION lib)

# Install all third-party libraries to lib directory
if(ENABLE_ONNXRUNTIME)
  file(GLOB ONNXRUNTIME_LIBS "${OnnxRuntime_DIR}/lib/*.so*")
  install(FILES ${ONNXRUNTIME_LIBS} DESTINATION lib)
  message(STATUS "Will install onnxruntime libraries to ${CMAKE_INSTALL_PREFIX}/lib")
endif()

if(EXISTS ${CMAKE_SOURCE_DIR}/lite/bin/libddim_scheduler_cpp.so)
  install(FILES ${CMAKE_SOURCE_DIR}/lite/bin/libddim_scheduler_cpp.so 
          DESTINATION lib)
  message(STATUS "Will install libddim_scheduler_cpp.so to ${CMAKE_INSTALL_PREFIX}/lib")
endif()

install(DIRECTORY ${CMAKE_SOURCE_DIR}/lite 
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include
        FILES_MATCHING
        PATTERN "*.h")
# install(DIRECTORY ${THIRD_PARTY_PATH} DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${CMAKE_BINARY_DIR}/lite.ai.toolkit.cmake 
        DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${CMAKE_SOURCE_DIR}/cmake/lite.ai.toolkit-config.cmake 
        DESTINATION ${CMAKE_INSTALL_PREFIX})

# Add uninstall target
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

if (ENABLE_TEST)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_INSTALL_PREFIX}/bin)
    add_subdirectory(examples/lite)
    # if ((UNIX AND NOT APPLE))
    #     file(GLOB_RECURSE ALL_THIRD_LIBS ${THIRD_PARTY_PATH} FOLLOW_SYMLINKS *.so*)
    #     install(FILES ${ALL_THIRD_LIBS} DESTINATION ${EXECUTABLE_OUTPUT_PATH})
    # endif()
endif ()

message(STATUS "-------------------------- lite.ai.toolkit Configuration Summary --------------------------")
message(STATUS "        Version: ${VERSION_STRING}")
message(STATUS "     SO Version: ${SOVERSION_STRING}")
message(STATUS "     Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "      Root Path: ${CMAKE_SOURCE_DIR}")
message(STATUS "         OpenCV: ON Version: ${OpenCV_Version}")
message(STATUS "    ONNXRUNTIME: ${ENABLE_ONNXRUNTIME} Version: ${OnnxRuntime_Version}")
message(STATUS "       TENSORRT: ${ENABLE_TENSORRT} Version: ${TensorRT_Version}")
message(STATUS "            MNN: ${ENABLE_MNN} Version: ${MNN_Version}")
message(STATUS "           NCNN: ${ENABLE_NCNN} Version: ${NCNN_Version}")
message(STATUS "            TNN: ${ENABLE_TNN} Version: ${TNN_Version}")
message(STATUS "        INSTALL:  ${CMAKE_INSTALL_PREFIX}")
message(STATUS "-------------------------- lite.ai.toolkit Configuration Summary --------------------------")
