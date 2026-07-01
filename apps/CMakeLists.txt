cmake_minimum_required(VERSION 3.6)


project(libcamera_CMS)
add_executable(libcamera_CMS libcamera_CMS.cpp)
target_link_libraries(libcamera_CMS libcamera_app)


set(EXECUTABLES libcamera_CMS)

set(EXECUTABLE_OUTPUT_PATH  ${CMAKE_BINARY_DIR})
install(TARGETS ${EXECUTABLES} RUNTIME DESTINATION bin)