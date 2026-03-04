set(OnnxRuntime_Version "1.17.1" CACHE STRING "OnnxRuntime version" FORCE)
# Detect onnxruntime from user-provided OnnxRuntime_DIR.
# Supports two layout styles:
#   1. Package style  (downloaded tgz): <DIR>/include/onnxruntime_cxx_api.h  -> include_dir = <DIR>/include
#   2. System prefix style:             <DIR>/include/onnxruntime/onnxruntime_cxx_api.h -> include_dir = <DIR>/include/onnxruntime
set(_ORT_FOUND FALSE)
if (OnnxRuntime_DIR)
    if (EXISTS "${OnnxRuntime_DIR}/include/onnxruntime_cxx_api.h")
        # Package style
        set(_ORT_INCLUDE_DIR "${OnnxRuntime_DIR}/include")
        set(_ORT_LIB_DIR     "${OnnxRuntime_DIR}/lib")
        set(_ORT_FOUND TRUE)
        message("[Lite.AI.Toolkit][I] Using user-provided onnxruntime (package style): ${OnnxRuntime_DIR}")
    elseif (EXISTS "${OnnxRuntime_DIR}/include/onnxruntime/onnxruntime_cxx_api.h")
        # System prefix style
        set(_ORT_INCLUDE_DIR "${OnnxRuntime_DIR}/include/onnxruntime")
        set(_ORT_LIB_DIR     "${OnnxRuntime_DIR}/lib")
        set(_ORT_FOUND TRUE)
        message("[Lite.AI.Toolkit][I] Using user-provided onnxruntime (system prefix style): ${OnnxRuntime_DIR}")
    else()
        message("[Lite.AI.Toolkit][W] OnnxRuntime_DIR=${OnnxRuntime_DIR} does not contain a valid onnxruntime installation, falling back to third_party auto-download.")
    endif()
endif()

if (NOT _ORT_FOUND)
    # Fall back to third_party directory with auto-download
    set(OnnxRuntime_DIR ${THIRD_PARTY_PATH}/onnxruntime)
    if (NOT EXISTS ${OnnxRuntime_DIR}/include)
        if (EXISTS ${OnnxRuntime_DIR})
            message("[Lite.AI.Toolkit][W] Found incomplete onnxruntime dir (missing include/), removing and re-downloading ...")
            file(REMOVE_RECURSE ${OnnxRuntime_DIR})
        endif()
        set(OnnxRuntime_Filename "onnxruntime-linux-x64-${OnnxRuntime_Version}.tgz")
        set(OnnxRuntime_URL https://ghfast.top/https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/${OnnxRuntime_Filename})
        message("[Lite.AI.Toolkit][I] Downloading onnxruntime library: ${OnnxRuntime_URL}")
        download_and_decompress(${OnnxRuntime_URL} ${OnnxRuntime_Filename} ${OnnxRuntime_DIR})
    else()
        message("[Lite.AI.Toolkit][I] Found local onnxruntime library: ${OnnxRuntime_DIR}")
    endif()
    set(_ORT_INCLUDE_DIR "${OnnxRuntime_DIR}/include")
    set(_ORT_LIB_DIR     "${OnnxRuntime_DIR}/lib")
endif()

if(NOT EXISTS ${_ORT_INCLUDE_DIR})
    message(FATAL_ERROR "[Lite.AI.Toolkit][E] onnxruntime include dir not found: ${_ORT_INCLUDE_DIR}")
endif()
include_directories(${_ORT_INCLUDE_DIR})
link_directories(${_ORT_LIB_DIR})

# 1. glob sources files
file(GLOB ONNXRUNTIME_CORE_SRCS ${CMAKE_SOURCE_DIR}/lite/ort/core/*.cpp)
file(GLOB ONNXRUNTIME_CV_SRCS ${CMAKE_SOURCE_DIR}/lite/ort/cv/*.cpp)
file(GLOB ONNXRUNTIME_NLP_SRCS ${CMAKE_SOURCE_DIR}/lite/ort/nlp/*.cpp)
file(GLOB ONNXRUNTIME_ASR_SRCS ${CMAKE_SOURCE_DIR}/lite/ort/asr/*.cpp)
file(GLOB ONNXRUNTIME_SD_SRCS ${CMAKE_SOURCE_DIR}/lite/ort/sd/*.cpp)

# 2. glob headers files
file(GLOB ONNXRUNTIME_CORE_HEAD ${CMAKE_SOURCE_DIR}/lite/ort/core/*.h)
file(GLOB ONNXRUNTIME_CV_HEAD ${CMAKE_SOURCE_DIR}/lite/ort/cv/*.h)
file(GLOB ONNXRUNTIME_NLP_HEAD ${CMAKE_SOURCE_DIR}/lite/ort/nlp/*.h)
file(GLOB ONNXRUNTIME_ASR_HEAD ${CMAKE_SOURCE_DIR}/lite/ort/asr/*.h)
file(GLOB ONNXRUNTIME_SD_HEAD ${CMAKE_SOURCE_DIR}/lite/ort/sd/*.h)

set(ORT_SRCS ${ONNXRUNTIME_CV_SRCS} ${ONNXRUNTIME_NLP_SRCS} ${ONNXRUNTIME_ASR_SRCS} ${ONNXRUNTIME_CORE_SRCS} ${ONNXRUNTIME_SD_SRCS})
# 3. prepare install (headers will be installed at 'make install' stage, not configure stage)
message("[Lite.AI.Toolkit][I] Preparing Lite.AI.ToolKit Headers for ONNXRuntime Backend ...")
set(ORT_HEADERS ${ONNXRUNTIME_CORE_HEAD} ${ONNXRUNTIME_CV_HEAD} ${ONNXRUNTIME_ASR_HEAD} ${ONNXRUNTIME_NLP_HEAD} ${ONNXRUNTIME_SD_HEAD})

# Export variables for installation
set(ONNXRUNTIME_INSTALL_DIR ${OnnxRuntime_DIR} PARENT_SCOPE)
# If onnxruntime was found from a user-provided directory, skip re-installing
set(ONNXRUNTIME_USER_PROVIDED ${_ORT_FOUND} PARENT_SCOPE)
