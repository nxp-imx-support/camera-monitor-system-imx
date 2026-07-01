/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 * Copyright (C) 2020-2021, Raspberry Pi (Trading) Ltd.
 *
 * libcamera_app.hpp - base class for libcamera apps.
 */

#pragma once

#include <sys/mman.h>

#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <variant>

#include <libcamera/base/span.h>
#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/control_ids.h>
#include <libcamera/controls.h>
#include <libcamera/formats.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/logging.h>
#include <libcamera/property_ids.h>

#include "core/completed_request.hpp"
#include "core/post_processor.hpp"
#include "core/stream_info.hpp"

class Preview;

namespace controls = libcamera::controls;
namespace properties = libcamera::properties;

struct Mode
{
	Mode() : Mode(0, 0, 0, false) {}
	Mode(unsigned int w, unsigned int h, unsigned int b, bool p) : width(w), height(h), bit_depth(b), packed(p) {}
	Mode(std::string const &mode_string);
	unsigned int width;
	unsigned int height;
	unsigned int bit_depth;
	bool packed;
	libcamera::Size Size() const { return libcamera::Size(width, height); }
	std::string ToString() const;
};

class LibcameraApp
{
public:
	using Stream = libcamera::Stream;
	using FrameBuffer = libcamera::FrameBuffer;
	using ControlList = libcamera::ControlList;
	using Request = libcamera::Request;
	using CameraManager = libcamera::CameraManager;
	using Camera = libcamera::Camera;
	using CameraConfiguration = libcamera::CameraConfiguration;
	using FrameBufferAllocator = libcamera::FrameBufferAllocator;
	using StreamRole = libcamera::StreamRole;
	using StreamRoles = std::vector<libcamera::StreamRole>;
	using PixelFormat = libcamera::PixelFormat;
	using StreamConfiguration = libcamera::StreamConfiguration;
	using BufferMap = Request::BufferMap;
	using Size = libcamera::Size;
	using Rectangle = libcamera::Rectangle;
	enum class MsgType
	{
		RequestComplete,
		RequestComplete1,
		RequestComplete2,
		Timeout,
		Quit
	};
	typedef std::variant<CompletedRequestPtr> MsgPayload;
	struct Msg
	{
		Msg(MsgType const &t) : type(t) {}
		template <typename T>
		Msg(MsgType const &t, T p) : type(t), payload(std::forward<T>(p))
		{
		}
		MsgType type;
		MsgPayload payload;
	};

	// Some flags that can be used to give hints to the camera configuration.
	static constexpr unsigned int FLAG_STILL_NONE = 0;
	static constexpr unsigned int FLAG_STILL_BGR = 1; // supply BGR images, not YUV
	static constexpr unsigned int FLAG_STILL_RGB = 2; // supply RGB images, not YUV
	static constexpr unsigned int FLAG_STILL_RAW = 4; // request raw image stream
	static constexpr unsigned int FLAG_STILL_DOUBLE_BUFFER = 8; // double-buffer stream
	static constexpr unsigned int FLAG_STILL_TRIPLE_BUFFER = 16; // triple-buffer stream
	static constexpr unsigned int FLAG_STILL_BUFFER_MASK = 24; // mask for buffer flags

	static constexpr unsigned int FLAG_VIDEO_NONE = 0;
	static constexpr unsigned int FLAG_VIDEO_RAW = 1; // request raw image stream
	static constexpr unsigned int FLAG_VIDEO_JPEG_COLOURSPACE = 2; // force JPEG colour space

	LibcameraApp();
	virtual ~LibcameraApp();

	std::string const &CameraId() const;
	std::string CameraModel() const;
	void OpenCamera(int camera_num, int preview_type, int detector_type);
	void CloseCamera();

	void ConfigureViewfinder();

	void Teardown();
	void StartCamera();
	void StopCamera();

	Msg Wait();
	Msg Wait_camera1();
	Msg Wait_camera2();
	void PostMessage(MsgType &t, MsgPayload &p);

	Stream *GetStream(std::string const &name, StreamInfo *info = nullptr) const;
	Stream *ViewfinderStream(StreamInfo *info = nullptr) const;
	Stream *Viewfinder2Stream(StreamInfo *info = nullptr) const;
	Stream *StillStream(StreamInfo *info = nullptr) const;
	Stream *RawStream(StreamInfo *info = nullptr) const;
	Stream *VideoStream(StreamInfo *info = nullptr) const;
	Stream *LoresStream(StreamInfo *info = nullptr) const;
	Stream *GetMainStream() const;

	std::vector<libcamera::Span<uint8_t>> Mmap(FrameBuffer *buffer) const;
	std::vector<libcamera::Span<uint8_t>> Mmap2(FrameBuffer *buffer) const;

	void ShowPreviewDual(CompletedRequestPtr &completed_request, CompletedRequestPtr &completed_request2, Stream *stream, Stream *stream2);
	void ShowPreview(CompletedRequestPtr &completed_request, Stream *stream);

	void SetControls(ControlList &controls);
	StreamInfo GetStreamInfo(Stream const *stream) const;

	// GPU Scaler and VPU Encoder control
	void enableGPUScaler(int target_width, int target_height);
	void enableVPUEncoder(const std::string& codec, int bitrate, const std::string& output_file);
	void disableGPUScaler();
	void disableVPUEncoder();

	static unsigned int verbosity;
	static unsigned int GetVerbosity() { return verbosity; }
	void setCameraNum(int num);

protected:

private:
	template <typename T>
	class MessageQueue
	{
	public:
		template <typename U>
		void Post(U &&msg)
		{
			std::unique_lock<std::mutex> lock(mutex_);
			queue_.push(std::forward<U>(msg));
			cond_.notify_one();
		}
		T Wait()
		{
			std::unique_lock<std::mutex> lock(mutex_);
			cond_.wait(lock, [this] { return !queue_.empty(); });
			T msg = std::move(queue_.front());
			queue_.pop();
			return msg;
		}
		void Clear()
		{
			std::unique_lock<std::mutex> lock(mutex_);
			queue_ = {};
		}

	private:
		std::queue<T> queue_;
		std::mutex mutex_;
		std::condition_variable cond_;
	};
	struct PreviewItem
	{
		PreviewItem() : stream(nullptr) {}
		PreviewItem(CompletedRequestPtr &b, Stream *s) : completed_request(b), stream(s) {}
		PreviewItem &operator=(PreviewItem &&other)
		{
			completed_request = std::move(other.completed_request);
			stream = other.stream;
			other.stream = nullptr;
			return *this;
		}
		CompletedRequestPtr completed_request;
		Stream *stream;
	};
	struct SensorMode
	{
		SensorMode()
			: size({}), format({}), fps(0)
		{
		}
		SensorMode(libcamera::Size _size, libcamera::PixelFormat _format, double _fps)
			: size(_size), format(_format), fps(_fps)
		{
		}
		unsigned int depth() const
		{
			// This is a really ugly way of getting the bit depth of the format.
			// But apart from duplicating the massive bayer format table, there is
			// no other way to determine this.
			std::string fmt = format.toString();
			unsigned int mode_depth = fmt.find("8") != std::string::npos ? 8 :
									  fmt.find("10") != std::string::npos ? 10 :
									  fmt.find("12") != std::string::npos ? 12 : 16;
			return mode_depth;
		}
		libcamera::Size size;
		libcamera::PixelFormat format;
		double fps;
	};

    int CameraNum = 1;
	void setupCapture();
	void makeRequests();
	void queueRequest(CompletedRequest *completed_request);
	void queueRequest2(CompletedRequest *completed_request);
	void requestComplete(Request *request);
	void requestComplete2(Request *request);
	void previewDoneCallback(int fd);
	void previewDoneCallbackDual(int fd, int fd2);
	void startPreview();
	void stopPreview();
	void previewThread();
	void configureDenoise(const std::string &denoise_mode);
	Mode selectModeForFramerate(const libcamera::Size &req, double fps);

	std::unique_ptr<CameraManager> camera_manager_;
	std::shared_ptr<Camera> camera_;
	std::shared_ptr<Camera> camera2_;
	bool camera_acquired_ = false;
	bool camera2_acquired_ = false;
	std::unique_ptr<CameraConfiguration> configuration_;
	std::unique_ptr<CameraConfiguration> configuration2_;
	std::map<FrameBuffer *, std::vector<libcamera::Span<uint8_t>>> mapped_buffers_;
	std::map<FrameBuffer *, std::vector<libcamera::Span<uint8_t>>> mapped_buffers2_;
	std::map<std::string, Stream *> streams_;
	FrameBufferAllocator *allocator_ = nullptr;
	FrameBufferAllocator *allocator2_ = nullptr;

	std::map<Stream *, std::queue<FrameBuffer *>> frame_buffers_;
	std::map<Stream *, std::queue<FrameBuffer *>> frame_buffers2_;
	std::vector<std::unique_ptr<Request>> requests_;
	std::vector<std::unique_ptr<Request>> requests2_;
	std::mutex completed_requests_mutex_;
	std::mutex completed_requests_mutex2_;
	std::set<CompletedRequest *> completed_requests_;
	std::set<CompletedRequest *> completed_requests2_;
	bool camera_started_ = false;
	bool camera2_started_ = false;
	std::mutex camera_stop_mutex_;
	std::mutex camera_stop_mutex2_;
	MessageQueue<Msg> msg_queue1_;
	MessageQueue<Msg> msg_queue2_;
	std::vector<SensorMode> sensor_modes_;
	// Related to the preview window.
	std::unique_ptr<Preview> preview_;
	std::map<int, CompletedRequestPtr> preview_completed_requests_;
	std::map<int, CompletedRequestPtr> preview_completed_requests2_;
	std::mutex preview_mutex_;
	std::mutex preview_item_mutex_;
	std::mutex preview_item_mutex2_;
	PreviewItem preview_item_;
	PreviewItem preview_item2_;
	std::condition_variable preview_cond_var1_;
	std::condition_variable preview_cond_var2_;
	bool preview_abort_ = false;
	uint32_t preview_frames_displayed_ = 0;
	uint32_t preview_frames_dropped_ = 0;
	std::thread preview_thread_;
	// For setting camera controls.
	std::mutex control_mutex_;
	ControlList controls_;
	// Other:
	uint64_t last_timestamp1_;
	uint64_t last_timestamp2_;
	uint64_t sequence_ = 0;
	uint64_t sequence2_ = 0;
	PostProcessor post_processor_;
	PostProcessor post_processor2_;

};
