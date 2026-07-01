/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 *
 * vpu_encoder.hpp - VPU encoder interface for video encoding
 */
#pragma once
#include <string>
#include <cstdint>
#include <memory>
#include <atomic>
#include <vector>
#include <linux/videodev2.h>

// Forward declarations for MP4 muxing
struct AVFormatContext;
struct AVStream;
struct AVPacket;

namespace imx95 {

enum class EncoderMode {
    DMABUF,
    MMAP
};

enum class InputFormat {
    NV12M,
    YUYV
};

class VPUEncoder {
public:
    VPUEncoder();
    ~VPUEncoder();

    // Setup encoder with mode and format selection
    bool setup(int width, int height, const std::string& codec, 
               int bitrate, const std::string& output_file,
               EncoderMode mode = EncoderMode::DMABUF,
               InputFormat input_format = InputFormat::NV12M);

    // Encode from DMA-BUF fds (NV12M dual-plane, zero-copy) - DMABUF mode only
    bool encodeFromDMABuf(int y_fd, int uv_fd, 
                          size_t y_size, size_t uv_size,
                          int stride, uint64_t timestamp);

    // Encode from DMA-BUF fd (YUYV single-plane, zero-copy) - DMABUF mode only
    bool encodeFromDMABuf_YUYV(int yuyv_fd, size_t yuyv_size,
                               int stride, uint64_t timestamp);

    void stop();
    
    bool isInitialized() const { return initialized_; }
    EncoderMode getMode() const { return mode_; }
    InputFormat getInputFormat() const { return input_format_; }

private:
    bool openV4L2Device();
    bool setupV4L2Format();
    bool allocateBuffers();
    bool startStreaming();
    void stopStreaming();
    
    // DMABUF mode encoding
    bool encodeFrameV4L2_DMABUF(int y_fd, int uv_fd);
    
    bool writeH264Output();
    
    // MP4 muxing functions
    bool setupMP4Muxer();
    bool writeMP4Packet(const uint8_t* data, size_t size, bool is_keyframe);
    void closeMP4Muxer();
    void neonMemcpy(void* dst, const void* src, size_t size);
    
    
    std::atomic<bool> initialized_;
    int width_;
    int height_;
    int fps_;
    int bitrate_;
    std::string codec_;
    std::string output_file_;
    
    EncoderMode mode_;
    InputFormat input_format_;
    
    // V4L2 resources
    int v4l2_fd_;
    
    struct V4L2Buffer {
        int index;
        void* start[2];   // Y and UV planes (MMAP mode)
        size_t length[2];
        int dmabuf_fd[2]; // DMA-BUF fds (DMABUF mode, unused)
        bool queued;
    };
    
    std::vector<V4L2Buffer> output_buffers_;
    std::vector<V4L2Buffer> capture_buffers_;
    
    uint32_t output_buffer_count_;
    uint32_t capture_buffer_count_;
    
    bool streaming_;
    uint64_t frame_count_;
    uint64_t encoded_bytes_;
    
    // MP4 muxer resources
    AVFormatContext* format_ctx_;
    AVStream* video_stream_;
    int64_t next_pts_;
};

} // namespace imx95
