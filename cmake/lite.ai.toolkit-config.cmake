#    This file will define the following for find_package:
#
#    lite.ai.toolkit::lite.ai.toolkit     : IMPORTED SHARED target
#

include(${CMAKE_CURRENT_LIST_DIR}/lite.ai.toolkit.cmake)

# ===== IMPORTED target =====
if(NOT TARGET lite.ai.toolkit::lite.ai.toolkit)
  add_library(lite.ai.toolkit::lite.ai.toolkit SHARED IMPORTED)

  # 主库 .so 路径
  set_target_properties(lite.ai.toolkit::lite.ai.toolkit PROPERTIES
    IMPORTED_LOCATION "${LITE_AI_LIB_DIR}/liblite.ai.toolkit.so"
    IMPORTED_NO_SONAME TRUE
  )

  # 头文件搜索路径
  set_property(TARGET lite.ai.toolkit::lite.ai.toolkit PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${Lite_AI_INCLUDE_DIRS}"
  )

  # 库搜索路径（用于解析依赖库的短名称）
  set_property(TARGET lite.ai.toolkit::lite.ai.toolkit PROPERTY
    INTERFACE_LINK_DIRECTORIES "${Lite_AI_LIBS_DIRS}"
  )

  # 依赖库（排除主库自身，避免循环引用）
  set(_lite_ai_deps ${Lite_AI_LIBS})
  list(REMOVE_ITEM _lite_ai_deps lite.ai.toolkit)
  if(_lite_ai_deps)
    set_property(TARGET lite.ai.toolkit::lite.ai.toolkit PROPERTY
      INTERFACE_LINK_LIBRARIES "${_lite_ai_deps}"
    )
  endif()
  unset(_lite_ai_deps)
endif()
