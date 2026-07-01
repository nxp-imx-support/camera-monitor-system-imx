cmake_minimum_required(VERSION 3.6)

include(GNUInstallDirs)

pkg_check_modules(LIBDRM QUIET libdrm)
pkg_check_modules(X11 QUIET x11)
pkg_check_modules(EPOXY QUIET epoxy)
#find_library(LIBGBM NAMES libgbm HINTS /usr/lib/aarch64-linux-gnu/)

set(SRC "preview.cpp")
set(TARGET_LIBS "")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

IF (NOT DEFINED ENABLE_DRM)
    SET(ENABLE_DRM 1)
endif()
set(DRM_FOUND 0)
if (ENABLE_DRM AND LIBDRM_FOUND)
    message(STATUS "LIBDRM_LINK_LIBRARIES=${LIBDRM_LINK_LIBRARIES}")
    include_directories(${LIBDRM_INCLUDE_DIRS})
    #message(STATUS "LIBGBM_LINK_LIBRARIES=/usr/lib/aarch64-linux-gnu/libgbm.so")
    set(TARGET_LIBS ${TARGET_LIBS} ${LIBDRM_LIBRARIES} ${EPOXY_LIBRARIES} ${LIBGBM_LIBRARIES})
    set(SRC ${SRC} drm_preview.cpp 
                egl_preview.cpp 
                mesh.cpp 
                ssd_detector_cpu.cpp 
                ssd_detector_npu.cpp 
                gpu_scaler.cpp 
                g2d_scaler.cpp 
                vpu_encoder.cpp 
                dirty_detector.cpp)
    set(DRM_FOUND 1)
    message(STATUS "LIBDRM display mode enabled")
else()
    message(STATUS "LIBDRM display mode will be unavailable!")
endif()

pkg_check_modules(LIBAVCODEC QUIET libavcodec)
pkg_check_modules(LIBAVFORMAT QUIET libavformat)
pkg_check_modules(LIBAVUTIL QUIET libavutil)
pkg_check_modules(OPENCL QUIET OpenCL)

# Add OpenCL support
if(OPENCL_FOUND)
    message(STATUS "OpenCL found: ${OPENCL_VERSION}")
    include_directories(${OPENCL_INCLUDE_DIRS})
    set(TARGET_LIBS ${TARGET_LIBS} ${OPENCL_LIBRARIES})
else()
    # Try to find OpenCL manually
    find_library(OPENCL_LIB NAMES OpenCL)
    if(OPENCL_LIB)
        message(STATUS "OpenCL library found: ${OPENCL_LIB}")
        set(TARGET_LIBS ${TARGET_LIBS} ${OPENCL_LIB})
    else()
        message(WARNING "OpenCL not found, some features may not work")
    endif()
endif()

add_library(preview null_preview.cpp ${SRC})
target_link_libraries(preview
    ${TARGET_LIBS}
    ${OpenCV_LIBS}
    tensorflow-lite
    pthread
    g2d
    dl
    avcodec
    avformat
    avutil)

target_compile_definitions(preview PUBLIC LIBDRM_PRESENT=${DRM_FOUND} )
install(TARGETS preview LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
