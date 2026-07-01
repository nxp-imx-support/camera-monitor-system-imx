/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 *
 * gpu_scaler.hpp - GPU-accelerated NV12 image scaling using OpenCL.
 */
#pragma once

#include <CL/cl.h>
#include <memory>
#include <vector>
#include <cstdint>

namespace imx95 {

class GPUScaler {
public:
    GPUScaler();
    ~GPUScaler();
    
    bool setup(int src_width, int src_height, int src_stride,
               int dst_width, int dst_height);
    
    std::vector<uint8_t> scaleNV12(int y_fd, int uv_fd, 
                                    size_t y_size, size_t uv_size);
    
    bool isInitialized() const { return initialized_; }
    
    void cleanup();

private:
    bool initialized_;
    
    cl_context context_;
    cl_command_queue queue_;
    cl_kernel kernel_;
    cl_program program_;
    
    cl_mem y_buffer_;
    cl_mem uv_buffer_;
    cl_mem dst_buffer_;
    
    int src_width_, src_height_, src_stride_;
    int dst_width_, dst_height_;
    
    std::vector<uint8_t> output_buffer_;
    
    bool createKernel();
    bool allocateBuffers();
};

} // namespace imx95
