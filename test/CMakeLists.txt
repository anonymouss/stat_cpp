# include_directories (include)
# aux_source_directory (source SRC_DIR)

file (GLOB MAIN_FILES ./*.cpp)
foreach (SRC_MAIN ${MAIN_FILES})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" TARGET ${SRC_MAIN})
    add_executable (${TARGET} ${SRC_MAIN})
    #target_link_libraries (${TARGET} stat)
endforeach ()