/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 * Copyright (C) 2020, Raspberry Pi (Trading) Ltd.
 *
 * preview.hpp - preview window interface
 */

#pragma once

#include <functional>
#include <string>
#include <vector>

#include <libcamera/base/span.h>
#include <libcamera/framebuffer.h>

#include "core/stream_info.hpp"

class LibcameraApp;

// Add detector type enum
enum DetectorType {
	DETECTOR_NONE = 0,
	DETECTOR_SSD_CPU = 1,
	DETECTOR_SSD_NPU = 2,
	DETECTOR_DIRTY = 3,
};

class Preview
{
public:
	typedef std::function<void(int fd)> DoneCallback;
	typedef std::function<void(int fd, int fd2)> DoneCallbackDual;

	Preview()  {}
	virtual ~Preview() {}
	
	
	// This is where the application sets the callback it gets whenever the viewfinder
	// is no longer displaying the buffer and it can be safely recycled.
	void SetDoneCallback(DoneCallback callback) { done_callback_ = callback; }
	void SetDoneCallbackDual(DoneCallbackDual callback) { done_callback_dual_ = callback; }
	virtual void ShowBuffer(libcamera::FrameBuffer* buffer, 
	                        libcamera::Span<uint8_t> span, 
	                        StreamInfo const &info) 
	{
		int fd = buffer->planes()[0].fd.get();
		Show(fd, span, info);
	}
	
	virtual void ShowBufferDual(libcamera::FrameBuffer* buffer1,
	                            libcamera::Span<uint8_t> span1,
	                            StreamInfo const &info1,
	                            libcamera::FrameBuffer* buffer2,
	                            libcamera::Span<uint8_t> span2,
	                            StreamInfo const &info2)
	{
		int fd1 = buffer1->planes()[0].fd.get();
		int fd2 = buffer2->planes()[0].fd.get();
		ShowDual(fd1, span1, info1, fd2, span2, info2);
	}
	
	virtual void Show(int fd, libcamera::Span<uint8_t> span, StreamInfo const &info) = 0;
	virtual void ShowDual(int fd, libcamera::Span<uint8_t> span, StreamInfo const &info, 
	                      int fd2, libcamera::Span<uint8_t> span2, StreamInfo const &info2) = 0;
	
	// Reset the preview window, clearing the current buffers and being ready to
	// show new ones.
	virtual void Reset() = 0;
	// Check if preview window has been shut down.
	virtual bool Quit() { return false; }
	// Return the maximum image size allowed.
	virtual void MaxImageSize(unsigned int &w, unsigned int &h) const = 0;

	virtual void enableGPUScaler(int target_width, int target_height) {}
	virtual void enableVPUEncoder(const std::string& codec, int bitrate, const std::string& output_file) {}
	virtual void disableGPUScaler() {}
	virtual void disableVPUEncoder() {}

protected:
	DoneCallback done_callback_;
	DoneCallbackDual done_callback_dual_;
};

Preview *make_preview(int preview_type, int camera_num, int detector_type);
