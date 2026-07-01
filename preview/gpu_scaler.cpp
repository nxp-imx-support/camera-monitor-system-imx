/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 *
 * gpu_scaler.cpp - GPU-accelerated NV12 image scaling using OpenCL.
 */
#include "gpu_scaler.hpp"
#include <iostream>
#include <cstring>
#include <sys/mman.h>

// OpenCL kernel: NV12 -> RGBA
static const char* nv12_scale_kernel_source = R"(
__kernel void nv12_to_rgba_scale(
    __global const uchar* y_plane,
    __global const uchar* uv_plane,
    __global uchar4* output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    int y_stride,
    int uv_stride)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= dst_width || y >= dst_height)
        return;
    
    float scale_x = (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;
    
    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;
    
    int x0 = max((int)floor(src_x), 0);
    int y0 = max((int)floor(src_y), 0);
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);
    
    float fx = src_x - x0;
    float fy = src_y - y0;
    
    float y00 = y_plane[y0 * y_stride + x0];
    float y10 = y_plane[y0 * y_stride + x1];
    float y01 = y_plane[y1 * y_stride + x0];
    float y11 = y_plane[y1 * y_stride + x1];
    
    float y_val = (1-fx)*(1-fy)*y00 + fx*(1-fy)*y10 + (1-fx)*fy*y01 + fx*fy*y11;
    
    int uv_x = ((int)src_x / 2) * 2;
    int uv_y = (int)src_y / 2;
    int uv_idx = uv_y * uv_stride + uv_x;
    
    float u_val = uv_plane[uv_idx] - 128.0f;
    float v_val = uv_plane[uv_idx + 1] - 128.0f;
    
    // YUV -> RGB (BT.601)
    y_val = y_val - 16.0f;
    float r = (149.0f * y_val + 204.0f * v_val) / 128.0f;
    float g = (149.0f * y_val - 50.0f * u_val - 104.0f * v_val) / 128.0f;
    float b = (149.0f * y_val + 258.0f * u_val) / 128.0f;
    
    uchar4 result;
    result.x = clamp((int)r, 0, 255);
    result.y = clamp((int)g, 0, 255);
    result.z = clamp((int)b, 0, 255);
    result.w = 255;
    
    output[y * dst_width + x] = result;
}
)";

namespace imx95 {

GPUScaler::GPUScaler()
    : initialized_(false)
    , context_(nullptr)
    , queue_(nullptr)
    , kernel_(nullptr)
    , program_(nullptr)
    , y_buffer_(nullptr)
    , uv_buffer_(nullptr)
    , dst_buffer_(nullptr)
    , src_width_(0)
    , src_height_(0)
    , src_stride_(0)
    , dst_width_(0)
    , dst_height_(0)
{
}

GPUScaler::~GPUScaler()
{
    cleanup();
}

bool GPUScaler::setup(int src_width, int src_height, int src_stride,
                      int dst_width, int dst_height)
{
    if (initialized_) {
        std::cerr << "GPUScaler already initialized" << std::endl;
        return false;
    }
    
    src_width_ = src_width;
    src_height_ = src_height;
    src_stride_ = src_stride;
    dst_width_ = dst_width;
    dst_height_ = dst_height;
    
    std::cout << "\n========================================================" << std::endl;
    std::cout << "           GPU Scaler Setup (NV12M -> RGBA)              " << std::endl;
    std::cout << "========================================================" << std::endl;
    printf("  Source:  %4dx%4d (stride: %4d)                    \n", 
           src_width_, src_height_, src_stride_);
    printf("  Dest:    %4dx%4d (RGBA)                               \n", 
           dst_width_, dst_height_);
    std::cout << "========================================================\n" << std::endl;
    
    // Get platform
    cl_platform_id platform;
    cl_int err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Scaler] Failed to get OpenCL platform: " << err << std::endl;
        return false;
    }
    
    // Get GPU device
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Scaler] Failed to get GPU device: " << err << std::endl;
        return false;
    }
    
    char device_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    std::cout << "[GPU Scaler] Using device: " << device_name << std::endl;
    
    // Create context
    context_ = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Scaler] Failed to create context: " << err << std::endl;
        return false;
    }
    
    // Create command queue
#ifdef CL_VERSION_2_0
    cl_queue_properties properties[] = {0};
    queue_ = clCreateCommandQueueWithProperties(context_, device, properties, &err);
    if (err != CL_SUCCESS) {
        #pragma GCC diagnostic push
        #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        queue_ = clCreateCommandQueue(context_, device, 0, &err);
        #pragma GCC diagnostic pop
    }
#else
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    queue_ = clCreateCommandQueue(context_, device, 0, &err);
    #pragma GCC diagnostic pop
#endif
    
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Scaler] Failed to create command queue: " << err << std::endl;
        return false;
    }
    
    if (!createKernel()) {
        return false;
    }
    
    if (!allocateBuffers()) {
        return false;
    }
    
    output_buffer_.resize(dst_width_ * dst_height_ * 4);
    
    initialized_ = true;
    std::cout << " GPU Scaler initialized successfully (NV12M->RGBA)\n" << std::endl;
    
    return true;
}

bool GPUScaler::createKernel()
{
    cl_int err;
    
    // Create program
    program_ = clCreateProgramWithSource(context_, 1, &nv12_scale_kernel_source, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Scaler] Failed to create program: " << err << std::endl;
        return false;
    }
    
    // Build program
    err = clBuildProgram(program_, 0, nullptr, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program_, 0, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program_, 0, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "[GPU Scaler] Build error: " << log.data() << std::endl;
        return false;
    }
    
    // Create kernel
    kernel_ = clCreateKernel(program_, "nv12_to_rgba_scale", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Scaler] Failed to create kernel: " << err << std::endl;
        return false;
    }
    
    std::cout << " Created NV12->RGBA kernel" << std::endl;
    
    return true;
}

bool GPUScaler::allocateBuffers()
{
    cl_int err;
    
    size_t y_size = src_stride_ * src_height_;
    y_buffer_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, y_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Scaler] Failed to create Y buffer: " << err << std::endl;
        return false;
    }
    
    size_t uv_size = src_stride_ * src_height_ / 2;
    uv_buffer_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, uv_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Scaler] Failed to create UV buffer: " << err << std::endl;
        return false;
    }
    
    size_t output_size = dst_width_ * dst_height_ * 4;
    dst_buffer_ = clCreateBuffer(context_, CL_MEM_WRITE_ONLY, output_size, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Scaler] Failed to create output buffer: " << err << std::endl;
        return false;
    }
    
    std::cout << " Allocated GPU buffers:" << std::endl;
    std::cout << "  Y:      " << y_size << " bytes" << std::endl;
    std::cout << "  UV:     " << uv_size << " bytes" << std::endl;
    std::cout << "  Output: " << output_size << " bytes (RGBA)" << std::endl;
    
    return true;
}

std::vector<uint8_t> GPUScaler::scaleNV12(int y_fd, int uv_fd, 
                                           size_t y_size, size_t uv_size)
{
    if (!initialized_) {
        std::cerr << "[GPU Scaler] Not initialized" << std::endl;
        return {};
    }
    
    cl_int err;
    
    // ===== Step 1: mmap DMA-BUF =====
    void* y_mapped = mmap(nullptr, y_size, PROT_READ, MAP_SHARED, y_fd, 0);
    void* uv_mapped = mmap(nullptr, uv_size, PROT_READ, MAP_SHARED, uv_fd, 0);
    
    if (y_mapped == MAP_FAILED || uv_mapped == MAP_FAILED) {
        std::cerr << "[GPU Scaler] Failed to mmap DMA-BUF" << std::endl;
        if (y_mapped != MAP_FAILED) munmap(y_mapped, y_size);
        if (uv_mapped != MAP_FAILED) munmap(uv_mapped, uv_size);
        return {};
    }
    
    // ===== Step 2:Upload Y and UV data to GPU=====
    err = clEnqueueWriteBuffer(queue_, y_buffer_, CL_FALSE, 0, 
                               y_size, y_mapped, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Scaler] Failed to write Y buffer: " << err << std::endl;
        munmap(y_mapped, y_size);
        munmap(uv_mapped, uv_size);
        return {};
    }
    
    err = clEnqueueWriteBuffer(queue_, uv_buffer_, CL_FALSE, 0, 
                               uv_size, uv_mapped, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Scaler] Failed to write UV buffer: " << err << std::endl;
        munmap(y_mapped, y_size);
        munmap(uv_mapped, uv_size);
        return {};
    }
    
    // ===== Step 3: Set kernel parameters =====
    err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &y_buffer_);
    err |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &uv_buffer_);
    err |= clSetKernelArg(kernel_, 2, sizeof(cl_mem), &dst_buffer_);
    err |= clSetKernelArg(kernel_, 3, sizeof(int), &src_width_);
    err |= clSetKernelArg(kernel_, 4, sizeof(int), &src_height_);
    err |= clSetKernelArg(kernel_, 5, sizeof(int), &dst_width_);
    err |= clSetKernelArg(kernel_, 6, sizeof(int), &dst_height_);
    err |= clSetKernelArg(kernel_, 7, sizeof(int), &src_stride_);
    err |= clSetKernelArg(kernel_, 8, sizeof(int), &src_stride_);
    
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Scaler] Failed to set kernel arguments: " << err << std::endl;
        munmap(y_mapped, y_size);
        munmap(uv_mapped, uv_size);
        return {};
    }
    
    // ===== Step 4: execute kernel =====
    size_t global_work_size[2] = { 
        (size_t)((dst_width_ + 15) & ~15),
        (size_t)((dst_height_ + 15) & ~15)
    };
    size_t local_work_size[2] = { 16, 16 };
    
    err = clEnqueueNDRangeKernel(queue_, kernel_, 2, nullptr,
                                global_work_size, local_work_size,
                                0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Scaler] Failed to enqueue kernel: " << err << std::endl;
        munmap(y_mapped, y_size);
        munmap(uv_mapped, uv_size);
        return {};
    }
    
    // ===== Step 5: result =====
    err = clEnqueueReadBuffer(queue_, dst_buffer_, CL_TRUE, 0,
                             output_buffer_.size(), output_buffer_.data(),
                             0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU Scaler] Failed to read output buffer: " << err << std::endl;
        munmap(y_mapped, y_size);
        munmap(uv_mapped, uv_size);
        return {};
    }
    
    // ===== Step 6: clear =====
    munmap(y_mapped, y_size);
    munmap(uv_mapped, uv_size);
    
    return output_buffer_;
}

void GPUScaler::cleanup()
{
    if (kernel_) {
        clReleaseKernel(kernel_);
        kernel_ = nullptr;
    }
    
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }
    
    if (y_buffer_) {
        clReleaseMemObject(y_buffer_);
        y_buffer_ = nullptr;
    }
    
    if (uv_buffer_) {
        clReleaseMemObject(uv_buffer_);
        uv_buffer_ = nullptr;
    }
    
    if (dst_buffer_) {
        clReleaseMemObject(dst_buffer_);
        dst_buffer_ = nullptr;
    }
    
    if (queue_) {
        clReleaseCommandQueue(queue_);
        queue_ = nullptr;
    }
    
    if (context_) {
        clReleaseContext(context_);
        context_ = nullptr;
    }
    
    output_buffer_.clear();
    initialized_ = false;
}

} // namespace imx95
