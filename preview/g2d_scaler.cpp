/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025 NXP
 */

#include "g2d_scaler.hpp"
#include <g2d.h>
#include <g2dExt.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <linux/dma-heap.h> 

namespace imx95 {

G2DScaler::G2DScaler() 
    : g2d_handle_(nullptr), 
      src_width_(0), src_height_(0), src_stride_(0),
      dst_width_(0), dst_height_(0),
      initialized_(false),
      debug_dump_enabled_(false),
      dump_interval_(60),
      frame_count_(0),
      yuyv_buf_(nullptr)
{
}

G2DScaler::~G2DScaler()
{
    if (yuyv_buf_) {
        g2d_free(static_cast<struct g2d_buf*>(yuyv_buf_));
        yuyv_buf_ = nullptr;
    }

    if (g2d_handle_) {
        g2d_close(g2d_handle_);
        g2d_handle_ = nullptr;
    }
}

bool G2DScaler::setup(int src_width, int src_height, int src_stride,
                      int dst_width, int dst_height)
{
    if (g2d_open(&g2d_handle_) != 0) {
        std::cerr << "[G2DScaler] Failed to open G2D device" << std::endl;
        return false;
    }
    
    int scaling_supported = 0;
    g2d_query_feature(g2d_handle_, G2D_SCALING, &scaling_supported);
    
    if (!scaling_supported) {
        std::cerr << "[G2DScaler] ERROR: G2D_SCALING not supported!" << std::endl;
        g2d_close(g2d_handle_);
        return false;
    }
    
    src_width_ = src_width;
    src_height_ = src_height;
    src_stride_ = src_stride;
    dst_width_ = dst_width;
    dst_height_ = dst_height;
    initialized_ = true;
    
    std::cout << "[G2DScaler] Initialized successfully:" << std::endl;
    std::cout << "  Source: " << src_width << "x" << src_height 
              << " (stride=" << src_stride << ")" << std::endl;
    std::cout << "  Target: " << dst_width << "x" << dst_height << std::endl;
    
    return true;
}

void G2DScaler::dumpNV12(const void* y_data, const void* uv_data, 
                         int width, int height, int stride, 
                         const char* filename)
{
    std::string yuv_filename = std::string(filename) + ".yuv";
    std::ofstream yuv_file(yuv_filename, std::ios::binary);
    
    if (yuv_file.is_open()) {
        const uint8_t* y_ptr = static_cast<const uint8_t*>(y_data);
        for (int i = 0; i < height; i++) {
            yuv_file.write(reinterpret_cast<const char*>(y_ptr + i * stride), width);
        }
        
        const uint8_t* uv_ptr = static_cast<const uint8_t*>(uv_data);
        for (int i = 0; i < height / 2; i++) {
            yuv_file.write(reinterpret_cast<const char*>(uv_ptr + i * stride), width);
        }
        
        yuv_file.close();
        std::cout << "[G2DScaler] Saved NV12: " << yuv_filename << std::endl;
    }
}

void G2DScaler::dumpRGB888(const void* rgb_data, 
                           int width, int height, 
                           const char* filename)
{
    try {
        cv::Mat rgb_mat(height, width, CV_8UC3, const_cast<void*>(rgb_data));
        cv::Mat bgr_mat;
        cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);
        
        std::string png_filename = std::string(filename) + ".png";
        cv::imwrite(png_filename, bgr_mat);
        std::cout << "[G2DScaler] Saved RGB888: " << png_filename << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[G2DScaler] Failed to save RGB888: " << e.what() << std::endl;
    }
}

bool G2DScaler::scaleAndConvertNV12ToRGB_DMABuf(
    int y_dma_fd,
    int uv_dma_fd,
    size_t y_size,
    size_t uv_size,
    int dst_dma_fd,
    size_t dst_size,
    int camera_id)
{
    if (!initialized_) {
        std::cerr << "[G2DScaler] Not initialized" << std::endl;
        return false;
    }
    
    if (y_dma_fd < 0 || uv_dma_fd < 0 || dst_dma_fd < 0) {
        std::cerr << "[G2DScaler] Invalid DMA-BUF fds: y_fd=" << y_dma_fd 
                  << ", uv_fd=" << uv_dma_fd 
                  << ", dst_fd=" << dst_dma_fd
                  << " (camera " << camera_id << ")" << std::endl;
        return false;
    }
    
    frame_count_++;
    bool should_dump = debug_dump_enabled_ && (frame_count_ % dump_interval_ == 0);

    struct g2d_surface_dmabuf src_dma;
    memset(&src_dma, 0, sizeof(src_dma));
    
    src_dma.base.format = G2D_NV12;
    src_dma.base.left = 0;
    src_dma.base.top = 0;
    src_dma.base.right = src_width_;
    src_dma.base.bottom = src_height_;
    src_dma.base.stride = src_stride_;
    src_dma.base.width = src_width_;
    src_dma.base.height = src_height_;
    src_dma.base.rot = G2D_ROTATION_0;
    
    src_dma.plane_fd[0] = y_dma_fd;
    src_dma.plane_fd[1] = uv_dma_fd;
    src_dma.plane_offset[0] = 0;
    src_dma.plane_offset[1] = 0;
    
    struct g2d_surface_dmabuf dst_dma;
    memset(&dst_dma, 0, sizeof(dst_dma));
    
    dst_dma.base.format = G2D_RGB888;
    dst_dma.base.left = 0;
    dst_dma.base.top = 0;
    dst_dma.base.right = dst_width_;
    dst_dma.base.bottom = dst_height_;
    dst_dma.base.stride = dst_width_;
    dst_dma.base.width = dst_width_;
    dst_dma.base.height = dst_height_;
    dst_dma.base.rot = G2D_ROTATION_0;
    
    dst_dma.plane_fd[0] = dst_dma_fd;
    dst_dma.plane_offset[0] = 0;
    
    if (should_dump) {
        std::cout << "\n========== G2D DMA-BUF Blit (Camera " << camera_id << ") ==========" << std::endl;
        std::cout << "Source (NV12):" << std::endl;
        std::cout << "  Y  fd: " << src_dma.plane_fd[0] << std::endl;
        std::cout << "  UV fd: " << src_dma.plane_fd[1] << std::endl;
        std::cout << "  Size: " << src_width_ << "x" << src_height_ << std::endl;
        std::cout << "Destination (RGB888):" << std::endl;
        std::cout << "  fd: " << dst_dma.plane_fd[0] << std::endl;
        std::cout << "  Size: " << dst_width_ << "x" << dst_height_ << std::endl;
        std::cout << "==========================================================\n" << std::endl;
    }
    
    int ret = g2d_blit_dmabuf(g2d_handle_, &src_dma, &dst_dma);
    
    if (ret != 0) {
        std::cerr << "[G2DScaler] g2d_blit_dmabuf failed for camera " << camera_id 
                  << " with error: " << ret << std::endl;
        std::cerr << "  errno: " << errno << " (" << strerror(errno) << ")" << std::endl;
    } else {
        ret = g2d_finish(g2d_handle_);
        if (ret != 0) {
            std::cerr << "[G2DScaler] g2d_finish failed for camera " << camera_id 
                      << " with error: " << ret << std::endl;
        }
    }

    if (should_dump && ret == 0) {
        void* y_mapped = mmap(nullptr, y_size, PROT_READ, MAP_SHARED, y_dma_fd, 0);
        void* uv_mapped = mmap(nullptr, uv_size, PROT_READ, MAP_SHARED, uv_dma_fd, 0);
        void* dst_mapped = mmap(nullptr, dst_size, PROT_READ, MAP_SHARED, dst_dma_fd, 0);
        
        if (y_mapped != MAP_FAILED && uv_mapped != MAP_FAILED) {
            char input_filename[256];
            snprintf(input_filename, sizeof(input_filename), 
                     "%d_g2d_input_nv12_%04d", camera_id, frame_count_);
            dumpNV12(y_mapped, uv_mapped, src_width_, src_height_, 
                     src_stride_, input_filename);
        }
        
        if (dst_mapped != MAP_FAILED) {
            char output_filename[256];
            snprintf(output_filename, sizeof(output_filename), 
                     "%d_g2d_output_rgb888_%04d", camera_id, frame_count_);
            dumpRGB888(dst_mapped, dst_width_, dst_height_, output_filename);
        }
        
        if (y_mapped != MAP_FAILED) munmap(y_mapped, y_size);
        if (uv_mapped != MAP_FAILED) munmap(uv_mapped, uv_size);
        if (dst_mapped != MAP_FAILED) munmap(dst_mapped, dst_size);
    }
 
    return (ret == 0);
}

bool G2DScaler::convertXRGB8888ToYUYV_DMABuf(
    int xrgb_dma_fd,
    int yuyv_dma_fd,
    int width,
    int height)
{
    if (!g2d_handle_) {
        std::cerr << "[G2D] Not initialized" << std::endl;
        return false;
    }
    
    if (xrgb_dma_fd < 0 || yuyv_dma_fd < 0) {
        std::cerr << "[G2D] Invalid DMA-BUF fds: xrgb_fd=" << xrgb_dma_fd 
                  << ", yuyv_fd=" << yuyv_dma_fd << std::endl;
        return false;
    }
    
    frame_count_++;
    bool should_dump = debug_dump_enabled_ && (frame_count_ % dump_interval_ == 0);
    
    if (should_dump) {
        std::cout << "\n[G2D] ===============================================" << std::endl;
        std::cout << "[G2D] XRGB8888->YUYV DMA-BUF Conversion (Frame " << frame_count_ << ")" << std::endl;
        std::cout << "[G2D] Resolution: " << width << "x" << height << std::endl;
        std::cout << "[G2D] ===============================================" << std::endl;
    }
    
    struct g2d_surface_dmabuf src_dma;
    memset(&src_dma, 0, sizeof(src_dma));
    
    src_dma.base.format = G2D_BGRX8888;
    src_dma.base.left = 0;
    src_dma.base.top = 0;
    src_dma.base.right = width;
    src_dma.base.bottom = height;
    src_dma.base.stride = width;
    src_dma.base.width = width;
    src_dma.base.height = height;
    src_dma.base.rot = G2D_ROTATION_0;
    src_dma.plane_fd[0] = xrgb_dma_fd;
    src_dma.plane_offset[0] = 0;
    
    struct g2d_surface_dmabuf dst_dma;
    memset(&dst_dma, 0, sizeof(dst_dma));
    
    dst_dma.base.format = G2D_YUYV;
    dst_dma.base.left = 0;
    dst_dma.base.top = 0;
    dst_dma.base.right = width;
    dst_dma.base.bottom = height;
    dst_dma.base.stride = width;  // YUYV: 2 bytes per pixel
    dst_dma.base.width = width;
    dst_dma.base.height = height;
    dst_dma.base.rot = G2D_ROTATION_0;
    dst_dma.plane_fd[0] = yuyv_dma_fd;
    dst_dma.plane_offset[0] = 0;
    
    if (should_dump) {
        std::cout << "[G2D] Source XRGB8888:" << std::endl;
        std::cout << "  fd: " << xrgb_dma_fd << std::endl;
        std::cout << "  size: " << width << "x" << height << std::endl;
        std::cout << "  stride: " << (width * 4) << " bytes" << std::endl;
        std::cout << "[G2D] Destination YUYV:" << std::endl;
        std::cout << "  fd: " << yuyv_dma_fd << std::endl;
        std::cout << "  size: " << width << "x" << height << std::endl;
        std::cout << "  stride: " << (width * 2) << " bytes" << std::endl;
    }
    
    int ret = g2d_blit_dmabuf(g2d_handle_, &src_dma, &dst_dma);
    
    if (ret != 0) {
        std::cerr << "[G2D] g2d_blit_dmabuf (XRGB->YUYV) failed: " << ret 
                  << " (" << strerror(errno) << ")" << std::endl;
        return false;
    }
    
    ret = g2d_finish(g2d_handle_);
    if (ret != 0) {
        std::cerr << "[G2D] g2d_finish failed: " << ret << std::endl;
        return false;
    }
    
    if (should_dump) {
        std::cout << "[G2D] XRGB->YUYV DMA-BUF conversion successful (zero-copy)" << std::endl;
        
        size_t yuyv_size = width * height * 2;
        void* yuyv_mapped = mmap(nullptr, yuyv_size, PROT_READ, MAP_SHARED, yuyv_dma_fd, 0);
        
        if (yuyv_mapped != MAP_FAILED) {
            try {
                cv::Mat yuyv_mat(height, width, CV_8UC2, yuyv_mapped);
                cv::Mat bgr_mat;
                cv::cvtColor(yuyv_mat, bgr_mat, cv::COLOR_YUV2BGR_YUYV);
                
                char filename[256];
                snprintf(filename, sizeof(filename), 
                         "g2d_output_yuyv_%04d.png", frame_count_);
                cv::imwrite(filename, bgr_mat);
                std::cout << "[G2D] Saved YUYV: " << filename << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "[G2D] Failed to save YUYV: " << e.what() << std::endl;
            }
            munmap(yuyv_mapped, yuyv_size);
        }
        
        std::cout << "[G2D] ===============================================\n" << std::endl;
    }
    
    return true;
}


} // namespace imx95
