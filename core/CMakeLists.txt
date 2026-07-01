cmake_minimum_required(VERSION 3.6)

include(GNUInstallDirs)


add_custom_target(VersionCpp ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_SOURCE_DIR} -P ${CMAKE_CURRENT_LIST_DIR}/version.cmake)
set_source_files_properties(version.cpp PROPERTIES GENERATED 1)

add_library(libcamera_app libcamera_app.cpp post_processor.cpp post_processing_stage.cpp version.cpp)
add_dependencies(libcamera_app VersionCpp)

set_target_properties(libcamera_app PROPERTIES PREFIX "" IMPORT_PREFIX "")
target_link_libraries(libcamera_app pthread preview ${LIBCAMERA_LINK_LIBRARIES})

install(TARGETS libcamera_app LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
