/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 * Copyright (C) 2021, Raspberry Pi (Trading) Ltd.
 *
 * null_preview.cpp - dummy "show nothing" preview window.
 */

#include <iostream>

#include "ssd_detector.h"
#include "preview.hpp"

class NullPreview : public Preview
{
public:
	NullPreview(int camera_num, DetectorType detector_type = DETECTOR_NONE) : Preview()
	{

	}
	~NullPreview() {}
	// Display the buffer. You get given the fd back in the BufferDoneCallback
	// once its available for re-use.
	virtual void Show(int fd, libcamera::Span<uint8_t> span, StreamInfo const &info) override
	{
		done_callback_(fd);
	}
	virtual void ShowDual(int fd, libcamera::Span<uint8_t> span, StreamInfo const &info, int fd2, libcamera::Span<uint8_t> span2, StreamInfo const &info2) override
	{
		done_callback_dual_(fd, fd2);
	}
	// Reset the preview window, clearing the current buffers and being ready to
	// show new ones.
	void Reset() override {}
	// Return the maximum image size allowed. Zeroes mean "no limit".
	virtual void MaxImageSize(unsigned int &w, unsigned int &h) const override { w = h = 0; }

private:
};

Preview *make_null_preview(int camera_num, int detector_type)
{
	return new NullPreview(camera_num, static_cast<DetectorType>(detector_type));
}
