# lite.ai.toolkit CMake Config
#
# 提供 IMPORTED 目标：
#   lite.ai.toolkit::lite.ai.toolkit
#
# 兼容旧式变量（已弃用）：
#   lite.ai.toolkit_INCLUDE_DIRS / lite.ai.toolkit_LIBS_DIRS / lite.ai.toolkit_LIBS

# 引入依赖查找和路径变量
include(${CMAKE_CURRENT_LIST_DIR}/lite.ai.toolkit.cmake)

# ===== IMPORTED target =====
if(NOT TARGET lite.ai.toolkit::lite.ai.toolkit)
  add_library(lite.ai.toolkit::lite.ai.toolkit SHARED IMPORTED)

  set_target_properties(lite.ai.toolkit::lite.ai.toolkit PROPERTIES
    IMPORTED_LOCATION "${LITE_AI_LIB_DIR}/liblite.ai.toolkit.so"
    IMPORTED_NO_SONAME TRUE
    INTERFACE_INCLUDE_DIRECTORIES "${Lite_AI_INCLUDE_DIRS}"
    INTERFACE_LINK_DIRECTORIES "${Lite_AI_LIBS_DIRS}"
  )

  # 传递依赖库（排除主库自身，避免循环引用）
  set(_lite_ai_deps ${Lite_AI_LIBS})
  list(REMOVE_ITEM _lite_ai_deps lite.ai.toolkit)
  if(_lite_ai_deps)
    set_target_properties(lite.ai.toolkit::lite.ai.toolkit PROPERTIES
      INTERFACE_LINK_LIBRARIES "${_lite_ai_deps}"
    )
  endif()
  unset(_lite_ai_deps)
endif()
