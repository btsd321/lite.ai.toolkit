set(OpenCV_Version "4.9.0-ffmpeg4.2.2" CACHE STRING "OpenCV version" FORCE)

# 优先检查部署目录是否已有 OpenCV（直接安装在 lib/ 和 include/ 下）
if (EXISTS ${CMAKE_INSTALL_PREFIX}/lib/libopencv_core.so AND EXISTS ${CMAKE_INSTALL_PREFIX}/include/opencv4)
    set(OpenCV_DIR ${CMAKE_INSTALL_PREFIX})
    message("[Lite.AI.Toolkit][I] Found OpenCV in deployment directory: ${OpenCV_DIR}")
# 其次检查部署目录的 third_party 子目录
elseif (EXISTS ${CMAKE_INSTALL_PREFIX}/third_party/opencv/lib/libopencv_core.so)
    set(OpenCV_DIR ${CMAKE_INSTALL_PREFIX}/third_party/opencv)
    message("[Lite.AI.Toolkit][I] Found OpenCV in deployment directory: ${OpenCV_DIR}")
else()
    # 最后检查本地 third_party 目录
    set(OpenCV_DIR ${THIRD_PARTY_PATH}/opencv)
    # download from github if opencv library is not exists
    if (NOT EXISTS ${OpenCV_DIR})
        set(OpenCV_Filename "opencv.tar.gz")
        set(OpenCV_URL https://ghfast.top/https://github.com/xlite-dev/lite.ai.toolkit/releases/download/v0.0.1/opencv.tar.gz)
        message("[Lite.AI.Toolkit][I] Downloading library: ${OpenCV_URL}")
        download_and_decompress(${OpenCV_URL} ${OpenCV_Filename} ${OpenCV_DIR})
        create_ffmpeg_syslinks_if_not_found(${OpenCV_DIR}/lib)
    else()
        message("[Lite.AI.Toolkit][I] Found local OpenCV library: ${OpenCV_DIR}")
    endif()
    if(NOT EXISTS ${OpenCV_DIR})
        message(FATAL_ERROR "[Lite.AI.Toolkit][E] ${OpenCV_DIR} is not exists!")
    endif()
endif()

include_directories(${OpenCV_DIR}/include/opencv4)
link_directories(${OpenCV_DIR}/lib)

if(NOT WIN32)
    if(ENABLE_OPENCV_VIDEOIO OR ENABLE_TEST)
        set(OpenCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_video opencv_videoio opencv_calib3d)
    else()
        set(OpenCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d) # no videoio, video module
    endif()
else()
    set(OpenCV_LIBS opencv_world490)
endif()
