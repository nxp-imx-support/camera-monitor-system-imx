/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 *
 * vpu_encoder.cpp - VPU encoder implementation using V4L2 M2M
 */
#include "vpu_encoder.hpp"
#include <iostream>
#include <cstring>
#include <chrono>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <cerrno>
#include <arm_neon.h> 

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
}

namespace imx95 {

VPUEncoder::VPUEncoder()
    : initialized_(false)
    , width_(0)
    , height_(0)
    , fps_(60)
    , bitrate_(4000000)
    , mode_(EncoderMode::DMABUF)
    , v4l2_fd_(-1)
    , output_buffer_count_(4)
    , capture_buffer_count_(4)
    , streaming_(false)
    , frame_count_(0)
    , encoded_bytes_(0)
    , format_ctx_(nullptr)
    , video_stream_(nullptr)
    , next_pts_(0)
{
}

VPUEncoder::~VPUEncoder()
{
    stop();
}


bool VPUEncoder::setup(int width, int height, const std::string& codec, 
                       int bitrate, const std::string& output_file,
                       EncoderMode mode, InputFormat input_format)
{
    if (initialized_) {
        std::cerr << "[VPUEncoder] Already initialized" << std::endl;
        return false;
    }
    
    width_ = width;
    height_ = height;
    codec_ = codec;
    bitrate_ = bitrate;
    output_file_ = output_file;
    mode_ = mode;
    input_format_ = input_format; 

    if (!openV4L2Device()) {
        std::cerr << "[VPUEncoder] Failed to open V4L2 device" << std::endl;
        return false;
    }
    
    if (!setupV4L2Format()) {
        std::cerr << "[VPUEncoder] Failed to setup V4L2 format" << std::endl;
        return false;
    }
    
    if (!allocateBuffers()) {
        std::cerr << "[VPUEncoder] Failed to allocate buffers" << std::endl;
        return false;
    }
    
    if (!setupMP4Muxer()) {
        std::cerr << "[VPUEncoder] Failed to setup MP4 muxer" << std::endl;
        return false;
    }
    
    if (!startStreaming()) {
        std::cerr << "[VPUEncoder] Failed to start streaming" << std::endl;
        return false;
    }
    
    initialized_ = true;
    
    std::cout << "\n========================================================" << std::endl;
    std::cout << "           VPU Encoder Initialized                      " << std::endl;
    std::cout << "========================================================" << std::endl;
    printf("  Mode:       %-40s \n", mode_ == EncoderMode::DMABUF ? "DMABUF (zero-copy)" : "MMAP (NEON-optimized copy)");
    printf("  Format:     %-40s \n", input_format_ == InputFormat::YUYV ? "YUYV (packed)" : "NV12M (dual-plane)");
    printf("  Resolution: %dx%d                                   \n", width_, height_);
    printf("  Codec:      %-40s \n", codec_.c_str());
    printf("  Bitrate:    %d bps                                  \n", bitrate_);
    printf("  Output:     %-40s \n", output_file_.c_str());
    std::cout << "========================================================\n" << std::endl;
    
    return true;
}


bool VPUEncoder::openV4L2Device()
{
    const char* device_paths[] = {
        "/dev/video0", "/dev/video1", "/dev/video2",
        "/dev/video3", "/dev/video4", "/dev/video5",
        "/dev/video6", "/dev/video7", "/dev/video8",
        "/dev/video9", "/dev/video10", "/dev/video11",
        "/dev/video12", "/dev/video13", "/dev/video14",
        "/dev/video15", "/dev/video16", "/dev/video17",
        "/dev/video18", "/dev/video19", "/dev/video20",
        "/dev/video21", "/dev/video22", "/dev/video23",
        nullptr
    };
    
    for (int i = 0; device_paths[i] != nullptr; i++) {
        v4l2_fd_ = open(device_paths[i], O_RDWR | O_NONBLOCK);
        if (v4l2_fd_ < 0) {
            continue;
        }
        
        struct v4l2_capability cap;
        if (ioctl(v4l2_fd_, VIDIOC_QUERYCAP, &cap) < 0) {
            close(v4l2_fd_);
            v4l2_fd_ = -1;
            continue;
        }
        
        std::string driver_name(reinterpret_cast<const char*>(cap.driver));
        std::string card_name(reinterpret_cast<const char*>(cap.card));
        
        if (driver_name.find("wave") == std::string::npos &&
            driver_name.find("vpu") == std::string::npos &&
            driver_name.find("hantro") == std::string::npos &&
            card_name.find("enc") == std::string::npos) {
            close(v4l2_fd_);
            v4l2_fd_ = -1;
            continue;
        }
        
        if (!(cap.capabilities & V4L2_CAP_VIDEO_M2M_MPLANE) ||
            !(cap.capabilities & V4L2_CAP_STREAMING)) {
            close(v4l2_fd_);
            v4l2_fd_ = -1;
            continue;
        }
        
        struct v4l2_fmtdesc fmtdesc;
        memset(&fmtdesc, 0, sizeof(fmtdesc));
        fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        
        bool supports_h264 = false;
        while (ioctl(v4l2_fd_, VIDIOC_ENUM_FMT, &fmtdesc) == 0) {
            if (fmtdesc.pixelformat == V4L2_PIX_FMT_H264 ||
                fmtdesc.pixelformat == V4L2_PIX_FMT_H264_NO_SC ||
                fmtdesc.pixelformat == V4L2_PIX_FMT_HEVC) {
                supports_h264 = true;
                break;
            }
            fmtdesc.index++;
        }
        
        if (!supports_h264) {
            close(v4l2_fd_);
            v4l2_fd_ = -1;
            continue;
        }
        
        memset(&fmtdesc, 0, sizeof(fmtdesc));
        fmtdesc.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        
        bool supports_nv12m = false;
        while (ioctl(v4l2_fd_, VIDIOC_ENUM_FMT, &fmtdesc) == 0) {
            if (fmtdesc.pixelformat == V4L2_PIX_FMT_NV12M ||
                fmtdesc.pixelformat == V4L2_PIX_FMT_NV12) {
                supports_nv12m = true;
            }
            fmtdesc.index++;
        }
        
        if (!supports_nv12m) {
            close(v4l2_fd_);
            v4l2_fd_ = -1;
            continue;
        }
        
        std::cout << "[VPUEncoder] Found encoder device: " << device_paths[i] << std::endl;
        std::cout << "  Driver: " << cap.driver << std::endl;
        std::cout << "  Card: " << cap.card << std::endl;
        
        return true;
    }
    
    std::cerr << "[VPUEncoder] No suitable encoder device found" << std::endl;
    return false;
}

void VPUEncoder::neonMemcpy(void* dst, const void* src, size_t size)
{
    uint8_t* d = static_cast<uint8_t*>(dst);
    const uint8_t* s = static_cast<const uint8_t*>(src);
    
    size_t neon_size = size & ~63;  // Align to 64 bytes
    
    // NEON copy: 64 bytes per iteration
    for (size_t i = 0; i < neon_size; i += 64) {
        uint8x16x4_t data = vld1q_u8_x4(s + i);
        vst1q_u8_x4(d + i, data);
    }
    
    // Copy remaining bytes
    for (size_t i = neon_size; i < size; i++) {
        d[i] = s[i];
    }
}

bool VPUEncoder::setupV4L2Format()
{
    // Set OUTPUT format (input to encoder)
    struct v4l2_format fmt_out;
    memset(&fmt_out, 0, sizeof(fmt_out));
    fmt_out.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    fmt_out.fmt.pix_mp.width = width_;
    fmt_out.fmt.pix_mp.height = height_;
    fmt_out.fmt.pix_mp.field = V4L2_FIELD_NONE;
    fmt_out.fmt.pix_mp.colorspace = V4L2_COLORSPACE_REC709;
    
    if (input_format_ == InputFormat::YUYV) {
        fmt_out.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt_out.fmt.pix_mp.num_planes = 1;
        fmt_out.fmt.pix_mp.plane_fmt[0].bytesperline = width_ * 2;  // 2 bytes/pixel
        fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage = width_ * height_ * 2;
        
        std::cout << "[VPUEncoder] Configuring YUYV format (single-plane)" << std::endl;
    } else {
        fmt_out.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12M;
        fmt_out.fmt.pix_mp.num_planes = 2;
        fmt_out.fmt.pix_mp.plane_fmt[0].bytesperline = width_;
        fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage = width_ * height_;
        fmt_out.fmt.pix_mp.plane_fmt[1].bytesperline = width_;
        fmt_out.fmt.pix_mp.plane_fmt[1].sizeimage = width_ * height_ / 2;
        
        std::cout << "[VPUEncoder] Configuring NV12M format (dual-plane)" << std::endl;
    }
    
    if (ioctl(v4l2_fd_, VIDIOC_S_FMT, &fmt_out) < 0) {
        std::cerr << "[VPUEncoder] Failed to set OUTPUT format: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    struct v4l2_format fmt_verify;
    memset(&fmt_verify, 0, sizeof(fmt_verify));
    fmt_verify.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    
    if (ioctl(v4l2_fd_, VIDIOC_G_FMT, &fmt_verify) == 0) {
        char fourcc[5] = {0};
        memcpy(fourcc, &fmt_verify.fmt.pix_mp.pixelformat, 4);
        std::cout << "[VPUEncoder] Actual OUTPUT format: " << fourcc 
                  << " (" << fmt_verify.fmt.pix_mp.num_planes << " planes)" << std::endl;
    }
    
    // Set CAPTURE format (encoded output) - H264/HEVC
    struct v4l2_format fmt_cap;
    memset(&fmt_cap, 0, sizeof(fmt_cap));
    fmt_cap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt_cap.fmt.pix_mp.width = width_;
    fmt_cap.fmt.pix_mp.height = height_;
    
    if (codec_ == "h265" || codec_ == "hevc") {
        fmt_cap.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_HEVC;
    } else {
        fmt_cap.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_H264;
    }
    
    fmt_cap.fmt.pix_mp.num_planes = 1;
    fmt_cap.fmt.pix_mp.plane_fmt[0].sizeimage = width_ * height_;
    fmt_cap.fmt.pix_mp.field = V4L2_FIELD_NONE;
    
    if (ioctl(v4l2_fd_, VIDIOC_S_FMT, &fmt_cap) < 0) {
        std::cerr << "[VPUEncoder] Failed to set CAPTURE format: " << strerror(errno) << std::endl;
        return false;
    }
    
    // Set bitrate
    struct v4l2_control ctrl;
    memset(&ctrl, 0, sizeof(ctrl));
    ctrl.id = V4L2_CID_MPEG_VIDEO_BITRATE;
    ctrl.value = bitrate_;
    
    if (ioctl(v4l2_fd_, VIDIOC_S_CTRL, &ctrl) < 0) {
        std::cerr << "[VPUEncoder] Warning: Failed to set bitrate" << std::endl;
    }
    
    // Set GOP size
    ctrl.id = V4L2_CID_MPEG_VIDEO_GOP_SIZE;
    ctrl.value = 240;
    
    if (ioctl(v4l2_fd_, VIDIOC_S_CTRL, &ctrl) < 0) {
        std::cerr << "[VPUEncoder] Warning: Failed to set GOP size" << std::endl;
    }
    
    // Set profile
    if (codec_ == "h265" || codec_ == "hevc") {
        ctrl.id = V4L2_CID_MPEG_VIDEO_HEVC_PROFILE;
        ctrl.value = V4L2_MPEG_VIDEO_HEVC_PROFILE_MAIN;
        ioctl(v4l2_fd_, VIDIOC_S_CTRL, &ctrl);
    } else {
        ctrl.id = V4L2_CID_MPEG_VIDEO_H264_PROFILE;
        ctrl.value = V4L2_MPEG_VIDEO_H264_PROFILE_BASELINE;
        ioctl(v4l2_fd_, VIDIOC_S_CTRL, &ctrl);
    }
    
    return true;
}


bool VPUEncoder::allocateBuffers()
{
    // ===== Request OUTPUT buffers =====
    struct v4l2_requestbuffers req_out;
    memset(&req_out, 0, sizeof(req_out));
    req_out.count = output_buffer_count_;
    req_out.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    
    if (mode_ == EncoderMode::DMABUF) {
        req_out.memory = V4L2_MEMORY_DMABUF;
    } else {
        req_out.memory = V4L2_MEMORY_MMAP;
    }
    
    if (ioctl(v4l2_fd_, VIDIOC_REQBUFS, &req_out) < 0) {
        std::cerr << "[VPUEncoder] Failed to request OUTPUT buffers: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    output_buffer_count_ = req_out.count;
    output_buffers_.resize(output_buffer_count_);
    
    int num_planes = (input_format_ == InputFormat::YUYV) ? 1 : 2;
    
    // ===== MMAP mode: map OUTPUT buffers =====
    if (mode_ == EncoderMode::MMAP) {
        for (uint32_t i = 0; i < output_buffer_count_; i++) {
            struct v4l2_buffer buf;
            struct v4l2_plane planes[2];
            
            memset(&buf, 0, sizeof(buf));
            memset(planes, 0, sizeof(planes));
            
            buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            buf.m.planes = planes;
            buf.length = num_planes;
            
            if (ioctl(v4l2_fd_, VIDIOC_QUERYBUF, &buf) < 0) {
                std::cerr << "[VPUEncoder] Failed to query OUTPUT buffer: " 
                          << strerror(errno) << std::endl;
                return false;
            }
            
            output_buffers_[i].index = i;
            
            // Map all planes
            for (int p = 0; p < num_planes; p++) {
                output_buffers_[i].length[p] = planes[p].length;
                output_buffers_[i].start[p] = mmap(NULL, planes[p].length,
                                                   PROT_READ | PROT_WRITE, MAP_SHARED,
                                                   v4l2_fd_, planes[p].m.mem_offset);
                
                if (output_buffers_[i].start[p] == MAP_FAILED) {
                    std::cerr << "[VPUEncoder] Failed to mmap OUTPUT plane " << p 
                              << ": " << strerror(errno) << std::endl;
                    
                    // Cleanup
                    for (int j = 0; j < p; j++) {
                        munmap(output_buffers_[i].start[j], output_buffers_[i].length[j]);
                    }
                    return false;
                }
            }
            
            output_buffers_[i].queued = false;
        }
        
        std::cout << "[VPUEncoder] Mapped " << output_buffer_count_ 
                  << " OUTPUT buffers (MMAP, " << num_planes << " planes)" << std::endl;
    } else {
        // DMABUF mode
        for (uint32_t i = 0; i < output_buffer_count_; i++) {
            output_buffers_[i].index = i;
            output_buffers_[i].queued = false;
            output_buffers_[i].start[0] = nullptr;
            output_buffers_[i].start[1] = nullptr;
        }
        
        std::cout << "[VPUEncoder] Allocated " << output_buffer_count_ 
                  << " OUTPUT buffers (DMABUF, " << num_planes << " planes)" << std::endl;
    }
    
    // ===== Request CAPTURE buffers (always MMAP) =====
    struct v4l2_requestbuffers req_cap;
    memset(&req_cap, 0, sizeof(req_cap));
    req_cap.count = capture_buffer_count_;
    req_cap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    req_cap.memory = V4L2_MEMORY_MMAP;
    
    if (ioctl(v4l2_fd_, VIDIOC_REQBUFS, &req_cap) < 0) {
        std::cerr << "[VPUEncoder] Failed to request CAPTURE buffers: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    capture_buffer_count_ = req_cap.count;
    capture_buffers_.resize(capture_buffer_count_);
    
    // Map CAPTURE buffers
    for (uint32_t i = 0; i < capture_buffer_count_; i++) {
        struct v4l2_buffer buf;
        struct v4l2_plane planes[1];
        
        memset(&buf, 0, sizeof(buf));
        memset(planes, 0, sizeof(planes));
        
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        buf.m.planes = planes;
        buf.length = 1;
        
        if (ioctl(v4l2_fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            std::cerr << "[VPUEncoder] Failed to query CAPTURE buffer: " 
                      << strerror(errno) << std::endl;
            return false;
        }
        
        capture_buffers_[i].index = i;
        capture_buffers_[i].length[0] = planes[0].length;
        capture_buffers_[i].start[0] = mmap(NULL, planes[0].length,
                                            PROT_READ | PROT_WRITE, MAP_SHARED,
                                            v4l2_fd_, planes[0].m.mem_offset);
        
        if (capture_buffers_[i].start[0] == MAP_FAILED) {
            std::cerr << "[VPUEncoder] Failed to mmap CAPTURE buffer: " 
                      << strerror(errno) << std::endl;
            return false;
        }
        
        capture_buffers_[i].queued = false;
    }
    
    std::cout << "[VPUEncoder] Allocated " << capture_buffer_count_ 
              << " CAPTURE buffers (MMAP)" << std::endl;
    
    return true;
}

bool VPUEncoder::setupMP4Muxer()
{
    int ret = avformat_alloc_output_context2(&format_ctx_, nullptr, "mp4", output_file_.c_str());
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        std::cerr << "[VPUEncoder] Failed to allocate output context: " << errbuf << std::endl;
        return false;
    }
    
    video_stream_ = avformat_new_stream(format_ctx_, nullptr);
    if (!video_stream_) {
        std::cerr << "[VPUEncoder] Failed to create video stream" << std::endl;
        return false;
    }
    
    video_stream_->id = 0;
    video_stream_->time_base = {1, fps_};
    video_stream_->avg_frame_rate = {fps_, 1};
    video_stream_->r_frame_rate = {fps_, 1};
    
    video_stream_->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    
    if (codec_ == "h265" || codec_ == "hevc") {
        video_stream_->codecpar->codec_id = AV_CODEC_ID_HEVC;
        video_stream_->codecpar->profile = AV_PROFILE_HEVC_MAIN;
        video_stream_->codecpar->level = 120;
    } else {
        video_stream_->codecpar->codec_id = AV_CODEC_ID_H264;
        video_stream_->codecpar->profile = AV_PROFILE_H264_BASELINE;
        video_stream_->codecpar->level = 40;
    }
    
    video_stream_->codecpar->width = width_;
    video_stream_->codecpar->height = height_;
    video_stream_->codecpar->format = AV_PIX_FMT_YUV420P;
    video_stream_->codecpar->bit_rate = bitrate_;

    if (!(format_ctx_->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&format_ctx_->pb, output_file_.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errbuf, sizeof(errbuf));
            std::cerr << "[VPUEncoder] Failed to open output file: " << errbuf << std::endl;
            return false;
        }
    }

    ret = avformat_write_header(format_ctx_, nullptr);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        std::cerr << "[VPUEncoder] Failed to write header: " << errbuf << std::endl;
        return false;
    }
    
    return true;
}

bool VPUEncoder::startStreaming()
{
    // Queue all CAPTURE buffers
    for (uint32_t i = 0; i < capture_buffer_count_; i++) {
        struct v4l2_buffer buf;
        struct v4l2_plane planes[1];
        
        memset(&buf, 0, sizeof(buf));
        memset(planes, 0, sizeof(planes));
        
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        buf.m.planes = planes;
        buf.length = 1;
        
        if (ioctl(v4l2_fd_, VIDIOC_QBUF, &buf) < 0) {
            std::cerr << "[VPUEncoder] Failed to queue CAPTURE buffer: " 
                      << strerror(errno) << std::endl;
            return false;
        }
        
        capture_buffers_[i].queued = true;
    }
    
    // Start streaming
    enum v4l2_buf_type type;
    
    type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    if (ioctl(v4l2_fd_, VIDIOC_STREAMON, &type) < 0) {
        std::cerr << "[VPUEncoder] Failed to start OUTPUT streaming: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    if (ioctl(v4l2_fd_, VIDIOC_STREAMON, &type) < 0) {
        std::cerr << "[VPUEncoder] Failed to start CAPTURE streaming: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    streaming_ = true;
    return true;
}

void VPUEncoder::stopStreaming()
{
    if (!streaming_) {
        return;
    }
    
    enum v4l2_buf_type type;
    
    
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    ioctl(v4l2_fd_, VIDIOC_STREAMOFF, &type);
    
    streaming_ = false;
}

bool VPUEncoder::encodeFromDMABuf_YUYV(int yuyv_fd, size_t yuyv_size,
                                       int stride, uint64_t timestamp)
{
    if (!initialized_ || mode_ != EncoderMode::DMABUF) {
        std::cerr << "[VPU] Not initialized or wrong mode" << std::endl;
        return false;
    }
    
    int free_index = -1;
    for (size_t i = 0; i < output_buffers_.size(); i++) {
        if (!output_buffers_[i].queued) {
            free_index = i;
            break;
        }
    }
    
    if (free_index < 0) {
        struct v4l2_buffer buf;
        struct v4l2_plane planes[1];
        memset(&buf, 0, sizeof(buf));
        memset(planes, 0, sizeof(planes));
        
        buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        buf.memory = V4L2_MEMORY_DMABUF;
        buf.m.planes = planes;
        buf.length = 1;
        
        if (ioctl(v4l2_fd_, VIDIOC_DQBUF, &buf) < 0) {
            if (errno != EAGAIN) {
                std::cerr << "[VPU] VIDIOC_DQBUF failed: " << strerror(errno) << std::endl;
            }
            return false;
        }
        
        free_index = buf.index;
        output_buffers_[free_index].queued = false;
    }
    
    struct v4l2_buffer buf;
    struct v4l2_plane planes[1];
    memset(&buf, 0, sizeof(buf));
    memset(planes, 0, sizeof(planes));
    
    buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    buf.memory = V4L2_MEMORY_DMABUF;
    buf.index = free_index;
    buf.m.planes = planes;
    buf.length = 1;
    buf.timestamp.tv_sec = timestamp / 1000000000ULL;
    buf.timestamp.tv_usec = (timestamp % 1000000000ULL) / 1000;
    
    planes[0].m.fd = yuyv_fd;
    planes[0].bytesused = yuyv_size;
    planes[0].length = yuyv_size;
    
    if (ioctl(v4l2_fd_, VIDIOC_QBUF, &buf) < 0) {
        std::cerr << "[VPU] VIDIOC_QBUF (YUYV DMABUF) failed: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    output_buffers_[free_index].queued = true;
    
    return writeH264Output();
}


// ===== DMABUF mode: encode from DMA-BUF fds =====
bool VPUEncoder::encodeFromDMABuf(int y_fd, int uv_fd, 
                                   size_t y_size, size_t uv_size,
                                   int stride, uint64_t timestamp)
{
    if (!initialized_) {
        std::cerr << "[VPUEncoder] Not initialized" << std::endl;
        return false;
    }
    
    if (mode_ != EncoderMode::DMABUF) {
        std::cerr << "[VPUEncoder] encodeFromDMABuf only works in DMABUF mode" << std::endl;
        return false;
    }
    
    //auto t0 = std::chrono::high_resolution_clock::now();
    
    bool success = encodeFrameV4L2_DMABUF(y_fd, uv_fd);
    
    //auto t1 = std::chrono::high_resolution_clock::now();
    
    if (success) {
        success = writeH264Output();
    }
    
    //auto t2 = std::chrono::high_resolution_clock::now();
    
    return success;
}

bool VPUEncoder::encodeFrameV4L2_DMABUF(int y_fd, int uv_fd)
{
    // Find a free OUTPUT buffer
    int buf_index = -1;
    for (uint32_t i = 0; i < output_buffer_count_; i++) {
        if (!output_buffers_[i].queued) {
            buf_index = i;
            break;
        }
    }
    
    if (buf_index < 0) {
        // Dequeue an OUTPUT buffer
        struct v4l2_buffer buf;
        struct v4l2_plane planes[2];
        
        memset(&buf, 0, sizeof(buf));
        memset(planes, 0, sizeof(planes));
        
        buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        buf.memory = V4L2_MEMORY_DMABUF;
        buf.m.planes = planes;
        buf.length = 2;
        
        if (ioctl(v4l2_fd_, VIDIOC_DQBUF, &buf) < 0) {
            if (errno != EAGAIN) {
                std::cerr << "[VPUEncoder] Failed to dequeue OUTPUT buffer: " 
                          << strerror(errno) << std::endl;
            }
            return false;
        }
        
        buf_index = buf.index;
        output_buffers_[buf_index].queued = false;
    }
    
    // Queue OUTPUT buffer with DMABUF fds
    struct v4l2_buffer buf;
    struct v4l2_plane planes[2];
    
    memset(&buf, 0, sizeof(buf));
    memset(planes, 0, sizeof(planes));
    
    buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    buf.memory = V4L2_MEMORY_DMABUF;
    buf.index = buf_index;
    buf.m.planes = planes;
    buf.length = 2;
    
    // Y plane
    planes[0].m.fd = y_fd;
    planes[0].bytesused = width_ * height_;
    planes[0].length = width_ * height_;
    
    // UV plane
    planes[1].m.fd = uv_fd;
    planes[1].bytesused = width_ * height_ / 2;
    planes[1].length = width_ * height_ / 2;
    
    buf.timestamp.tv_sec = frame_count_ / fps_;
    buf.timestamp.tv_usec = (frame_count_ % fps_) * (1000000 / fps_);
    
    if (ioctl(v4l2_fd_, VIDIOC_QBUF, &buf) < 0) {
        std::cerr << "[VPUEncoder] Failed to queue OUTPUT buffer: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    output_buffers_[buf_index].queued = true;
    frame_count_++;
    
    return true;
}

bool VPUEncoder::writeH264Output()
{
    // Dequeue CAPTURE buffer (encoded data)
    struct v4l2_buffer buf;
    struct v4l2_plane planes[1];
    
    memset(&buf, 0, sizeof(buf));
    memset(planes, 0, sizeof(planes));
    
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.m.planes = planes;
    buf.length = 1;
    
    if (ioctl(v4l2_fd_, VIDIOC_DQBUF, &buf) < 0) {
        if (errno != EAGAIN) {
            std::cerr << "[VPUEncoder] Failed to dequeue CAPTURE buffer: " 
                      << strerror(errno) << std::endl;
        }
        return false;
    }
    
    int buf_index = buf.index;
    size_t bytesused = planes[0].bytesused;
    
    if (bytesused > 0) {
        bool is_keyframe = (buf.flags & V4L2_BUF_FLAG_KEYFRAME) != 0;
        
        writeMP4Packet(static_cast<uint8_t*>(capture_buffers_[buf_index].start[0]), 
                       bytesused, is_keyframe);
        
        encoded_bytes_ += bytesused;
    }
    
    // Re-queue CAPTURE buffer
    memset(&buf, 0, sizeof(buf));
    memset(planes, 0, sizeof(planes));
    
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = buf_index;
    buf.m.planes = planes;
    buf.length = 1;
    
    if (ioctl(v4l2_fd_, VIDIOC_QBUF, &buf) < 0) {
        std::cerr << "[VPUEncoder] Failed to re-queue CAPTURE buffer: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    return true;
}

bool VPUEncoder::writeMP4Packet(const uint8_t* data, size_t size, bool is_keyframe)
{
    if (!format_ctx_ || !video_stream_) {
        return false;
    }
    
    AVPacket* pkt = av_packet_alloc();
    if (!pkt) {
        std::cerr << "[VPUEncoder] Failed to allocate packet" << std::endl;
        return false;
    }
    
    if (av_new_packet(pkt, size) < 0) {
        std::cerr << "[VPUEncoder] Failed to allocate packet data" << std::endl;
        av_packet_free(&pkt);
        return false;
    }
    
    // Copy H.264/HEVC data using NEON
    neonMemcpy(pkt->data, data, size);
    
    pkt->stream_index = video_stream_->index;
    pkt->pts = next_pts_;
    pkt->dts = next_pts_;
    pkt->duration = 1;
    
    if (is_keyframe) {
        pkt->flags |= AV_PKT_FLAG_KEY;
    }
    
    av_packet_rescale_ts(pkt, {1, fps_}, video_stream_->time_base);
    
    int ret = av_interleaved_write_frame(format_ctx_, pkt);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, errbuf, sizeof(errbuf));
        std::cerr << "[VPUEncoder] Failed to write packet: " << errbuf << std::endl;
        av_packet_free(&pkt);
        return false;
    }
    
    next_pts_++;
    av_packet_free(&pkt);
    
    return true;
}

void VPUEncoder::closeMP4Muxer()
{
    if (format_ctx_) {
        av_write_trailer(format_ctx_);
        
        if (format_ctx_->pb && !(format_ctx_->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&format_ctx_->pb);
        }
        
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
        video_stream_ = nullptr;
    }
}

void VPUEncoder::stop()
{
    if (!initialized_) {
        return;
    }
    
    std::cout << "[VPUEncoder] Stopping encoder..." << std::endl;
    
    stopStreaming();
    
    // Unmap OUTPUT buffers (MMAP mode only)
    if (mode_ == EncoderMode::MMAP) {
        for (auto& buf : output_buffers_) {
            if (buf.start[0] != nullptr && buf.start[0] != MAP_FAILED) {
                munmap(buf.start[0], buf.length[0]);
            }
            if (buf.start[1] != nullptr && buf.start[1] != MAP_FAILED) {
                munmap(buf.start[1], buf.length[1]);
            }
        }
    }
    
    // Unmap CAPTURE buffers
    for (auto& buf : capture_buffers_) {
        if (buf.start[0] != nullptr && buf.start[0] != MAP_FAILED) {
            munmap(buf.start[0], buf.length[0]);
        }
    }
    
    closeMP4Muxer();
    
    if (v4l2_fd_ >= 0) {
        close(v4l2_fd_);
        v4l2_fd_ = -1;
    }
    
    initialized_ = false;
    
    std::cout << "\n========================================================" << std::endl;
    std::cout << "           VPU Encoder Statistics                       " << std::endl;
    std::cout << "========================================================" << std::endl;
    printf("  Total frames:  %lu                                     \n", frame_count_);
    printf("  Total encoded: %.2f MB                                 \n", encoded_bytes_ / 1024.0 / 1024.0);
    printf("  Average size:  %.2f KB/frame                           \n", 
           frame_count_ > 0 ? (encoded_bytes_ / 1024.0 / frame_count_) : 0);
    std::cout << "========================================================\n" << std::endl;
}

} // namespace imx95

