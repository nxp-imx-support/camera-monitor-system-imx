/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 * Copyright (C) 2021, Raspberry Pi (Trading) Ltd.
 *
 * libcamera_app.cpp - base class for libcamera apps.
 */

#include "preview/preview.hpp"

#include "core/frame_info.hpp"
#include "core/libcamera_app.hpp"

#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/control_ids.h>
#include <libcamera/property_ids.h>
#include <libcamera/transform.h>
#include <unistd.h>


#include <cmath>
#include <fcntl.h>

#include <sys/ioctl.h>

#include <linux/videodev2.h>

unsigned int LibcameraApp::verbosity = 2;

LibcameraApp::LibcameraApp()
	:  controls_(controls::controls), post_processor_(this), post_processor2_(this)
{

}

LibcameraApp::~LibcameraApp()
{
	LOG(2, "Closing Libcamera application"
			   << "(frames displayed " << preview_frames_displayed_ << ", dropped " << preview_frames_dropped_
			   << ")");
	StopCamera();
	Teardown();
	CloseCamera();
}

std::string const &LibcameraApp::CameraId() const
{
	return camera_->id();
}

void LibcameraApp::setCameraNum(int num)
{
	CameraNum = num;
}

std::string LibcameraApp::CameraModel() const
{
	auto model = camera_->properties().get(properties::Model);
	return model ? std::string(*model) : camera_->id();
}

void LibcameraApp::OpenCamera(int camera_num, int preview_type, int detector_type)
{
	// Make a preview window.
	preview_ = std::unique_ptr<Preview>(make_preview(camera_num, preview_type, detector_type));

	if(CameraNum > 1)
		preview_->SetDoneCallbackDual(std::bind(&LibcameraApp::previewDoneCallbackDual, this, std::placeholders::_1, std::placeholders::_2));
	else 
		preview_->SetDoneCallback(std::bind(&LibcameraApp::previewDoneCallback, this, std::placeholders::_1));

	LOG(2, "Opening camera...");

	camera_manager_ = std::make_unique<CameraManager>();
	int ret = camera_manager_->start();
	if (ret)
		throw std::runtime_error("camera manager failed to start, code " + std::to_string(-ret));

	std::vector<std::shared_ptr<libcamera::Camera>> cameras = camera_manager_->cameras();
	// Do not show USB webcams as these are not supported in libcamera-apps!
	auto rem = std::remove_if(cameras.begin(), cameras.end(),
							  [](auto &cam) { return cam->id().find("/usb") != std::string::npos; });
	cameras.erase(rem, cameras.end());

	if (cameras.size() == 0)
		throw std::runtime_error("no cameras available");


	std::string const &cam_id = cameras[0]->id();
	camera_ = camera_manager_->get(cam_id);

	if (!camera_)
		throw std::runtime_error("failed to find camera " + cam_id);

	if (camera_->acquire())
		throw std::runtime_error("failed to acquire camera " + cam_id);

	camera_acquired_ = true;

	LOG(2, "Acquired camera " << cam_id );

	if(CameraNum > 1) {
		std::string const &cam_id2 = cameras[1]->id();
		camera2_ = camera_manager_->get(cam_id2);
		if (!camera2_)
			throw std::runtime_error("failed to find camera " + cam_id2);

		if (camera2_->acquire())
			throw std::runtime_error("failed to acquire camera " + cam_id2);

		camera2_acquired_ = true;

		LOG(2, "Acquired camera2 " << cam_id2);
	}

	// The queue takes over ownership from the post-processor.
	post_processor_.SetCallback(
		[this](CompletedRequestPtr &r) { this->msg_queue1_.Post(Msg(MsgType::RequestComplete1, std::move(r))); 
		});

	if(CameraNum > 1) {
		post_processor2_.SetCallback(
			[this](CompletedRequestPtr &r2) { this->msg_queue2_.Post(Msg(MsgType::RequestComplete2, std::move(r2))); });
	}
}

void LibcameraApp::CloseCamera()
{
	preview_.reset();

	if (camera_acquired_)
		camera_->release();

	camera_acquired_ = false;
	camera_.reset();

	if(CameraNum > 1) {
		if (camera2_acquired_)
			camera2_->release();

		camera2_acquired_ = false;
		camera2_.reset();
	}


	camera_manager_.reset();

	LOG(2, "Camera closed");
}

Mode LibcameraApp::selectModeForFramerate(const libcamera::Size &req, double fps)
{
	auto scoreFormat = [](double desired, double actual) -> double
	{
		double score = desired - actual;
		// Smaller desired dimensions are preferred.
		if (score < 0.0)
			score = (-score) / 8;
		// Penalise non-exact matches.
		if (actual != desired)
			score *= 2;

		return score;
	};

	constexpr float penalty_AR = 1500.0;
	constexpr float penalty_BD = 500.0;
	constexpr float penalty_FPS = 2000.0;

	double best_score = std::numeric_limits<double>::max(), score;
	SensorMode best_mode;

	LOG(1, "Mode selection:");
	for (const auto &mode : sensor_modes_)
	{
		double reqAr = static_cast<double>(req.width) / req.height;
		double fmtAr = static_cast<double>(mode.size.width) / mode.size.height;

		// Similar scoring mechanism that our pipeline handler does internally.
		score = scoreFormat(req.width, mode.size.width);
		score += scoreFormat(req.height, mode.size.height);
		score += penalty_AR * scoreFormat(reqAr, fmtAr);
		score += penalty_FPS * std::abs(fps - std::min(mode.fps, fps));
		score += penalty_BD * (16 - mode.depth());

		if (score <= best_score)
		{
			best_score = score;
			best_mode.size = mode.size;
			best_mode.format = mode.format;
		}

		LOG(1, "    " << mode.format.toString() << " " << mode.size.toString() << " - Score: " << score);
	}

	return { best_mode.size.width, best_mode.size.height, best_mode.depth(), true };
}

void LibcameraApp::ConfigureViewfinder()
{
    LOG(2, "Configuring viewfinder...");

    StreamRoles stream_roles = { StreamRole::Viewfinder };

    configuration_ = camera_->generateConfiguration(stream_roles);
    if (!configuration_)
        throw std::runtime_error("failed to generate viewfinder configuration for camera 1");

    if(CameraNum > 1) {
        configuration2_ = camera2_->generateConfiguration(stream_roles);
        if (!configuration2_)
            throw std::runtime_error("failed to generate viewfinder configuration for camera 2");
    }

    Size size(1920, 1080);

    // Finally trim the image size to the largest that the preview can handle.
    Size max_size;
    preview_->MaxImageSize(max_size.width, max_size.height);
    if (max_size.width && max_size.height)
    {
        size.boundTo(max_size.boundedToAspectRatio(size)).alignDownTo(2, 2);
        LOG(2, "Final viewfinder size is " << size.toString());
    }

    configuration_->at(0).pixelFormat = libcamera::formats::NV12;
    configuration_->at(0).size = size;
    
    StreamConfiguration &cfg = configuration_->at(0);
    
    size_t expected_size = cfg.stride * cfg.size.height * 3 / 2;
    
    std::cout << "=== Camera Configuration ===" << std::endl;
    std::cout << "  Size: " << cfg.size.width << "x" << cfg.size.height << std::endl;
    std::cout << "  Format: " << cfg.pixelFormat.toString() << std::endl;
    std::cout << "  Stride: " << cfg.stride << std::endl;
    std::cout << "  Frame Size: " << cfg.frameSize << std::endl;
    std::cout << "  Expected Size: " << expected_size << std::endl;
    std::cout << "=============================" << std::endl;

    if(CameraNum > 1) {
        configuration2_->at(0).pixelFormat = libcamera::formats::NV12;
        configuration2_->at(0).size = size;
    }

    post_processor_.AdjustConfig("viewfinder", &configuration_->at(0));

    if(CameraNum > 1)
        post_processor2_.AdjustConfig("viewfinder", &configuration2_->at(0));

    setupCapture();

    streams_["viewfinder"] = configuration_->at(0).stream();
    post_processor_.Configure();
    
    if(CameraNum > 1) {
        streams_["viewfinder2"] = configuration2_->at(0).stream();
        post_processor2_.Configure();
    }

}


void LibcameraApp::Teardown()
{
	stopPreview();

	post_processor_.Teardown();

	if(CameraNum > 1)
		post_processor2_.Teardown();

	for (auto &iter : mapped_buffers_)
	{
		// assert(iter.first->planes().size() == iter.second.size());
		// for (unsigned i = 0; i < iter.first->planes().size(); i++)
		for (auto &span : iter.second)
			munmap(span.data(), span.size());
	}
	mapped_buffers_.clear();
	delete allocator_;
	allocator_ = nullptr;
	configuration_.reset();

	if(CameraNum > 1) {
		mapped_buffers2_.clear();
		delete allocator2_;
		allocator2_ = nullptr;
		configuration2_.reset();
	}

	frame_buffers_.clear();

	streams_.clear();
}

void LibcameraApp::StartCamera()
{
	// This makes all the Request objects that we shall need.
	makeRequests();

	// Framerate is a bit weird. If it was set programmatically, we go with that, but
	// otherwise it applies only to preview/video modes. For stills capture we set it
	// as long as possible so that we get whatever the exposure profile wants.
	if (!controls_.get(controls::FrameDurationLimits)) {

			int64_t frame_time = 1000000 / 60; // in us
			controls_.set(controls::FrameDurationLimits,
						  libcamera::Span<const int64_t, 2>({ frame_time, frame_time }));

	}

	if (camera_->start(&controls_))
		throw std::runtime_error("failed to start camera");

	camera_started_ = true;

	if(CameraNum > 1) {
		if (camera2_->start(&controls_))
			throw std::runtime_error("failed to start camera 2");
		
		camera2_started_ = true;
	}

	controls_.clear();
	last_timestamp1_ = 0;
	last_timestamp2_ = 0;

	post_processor_.Start();
	
	if(CameraNum > 1)
		post_processor2_.Start();

	camera_->requestCompleted.connect(this, &LibcameraApp::requestComplete);

	if(CameraNum > 1) {
		camera2_->requestCompleted.connect(this, &LibcameraApp::requestComplete2);
	}

	for (std::unique_ptr<Request> &request : requests_)
	{
		if (camera_->queueRequest(request.get()) < 0)
			throw std::runtime_error("Failed to queue request");
	}

	if(CameraNum > 1) {
		for (std::unique_ptr<Request> &request : requests2_)
		{
			if (camera2_->queueRequest(request.get()) < 0)
				throw std::runtime_error("Failed to queue request");
		}
	}

	LOG(2, "Camera started!");
}

void LibcameraApp::StopCamera()
{
	{
		// We don't want QueueRequest to run asynchronously while we stop the camera.
		std::lock_guard<std::mutex> lock(camera_stop_mutex_);
		if (camera_started_) {
			if (camera_->stop())
				throw std::runtime_error("failed to stop camera");

			post_processor_.Stop();

			camera_started_ = false;
		}

	if (camera_) {
		camera_->requestCompleted.disconnect(this, &LibcameraApp::requestComplete);
	}

		if(CameraNum > 1) {
			if (camera2_started_) {
				if (camera2_->stop())
					throw std::runtime_error("failed to stop camera");

				post_processor2_.Stop();

				camera2_started_ = false;
			}
		}
	}

	if(CameraNum > 1) {
		if (camera2_) {
			camera2_->requestCompleted.disconnect(this, &LibcameraApp::requestComplete2);
		}
	}
	// An application might be holding a CompletedRequest, so queueRequest will get
	// called to delete it later, but we need to know not to try and re-queue it.
	completed_requests_.clear();
	msg_queue1_.Clear();
	requests_.clear();

	if(CameraNum > 1) {
		completed_requests2_.clear();
		msg_queue2_.Clear();
		requests2_.clear();
	}
	controls_.clear(); // no need for mutex here

	LOG(2, "Camera stopped!");
}

LibcameraApp::Msg LibcameraApp::Wait()
{
	return msg_queue1_.Wait();
}

LibcameraApp::Msg LibcameraApp::Wait_camera1()
{
	return msg_queue1_.Wait();
}

LibcameraApp::Msg LibcameraApp::Wait_camera2()
{
	return msg_queue2_.Wait();
}

void LibcameraApp::queueRequest(CompletedRequest *completed_request)
{
	BufferMap buffers(std::move(completed_request->buffers));

	// This function may run asynchronously so needs protection from the
	// camera stopping at the same time.
	std::lock_guard<std::mutex> stop_lock(camera_stop_mutex_);

	// An application could be holding a CompletedRequest while it stops and re-starts
	// the camera, after which we don't want to queue another request now.
	bool request_found;
	{
		std::lock_guard<std::mutex> lock(completed_requests_mutex_);
		auto it = completed_requests_.find(completed_request);
		if (it != completed_requests_.end())
		{
			request_found = true;
			completed_requests_.erase(it);
		}
		else
			request_found = false;
	}

	Request *request = completed_request->request;
	delete completed_request;
	assert(request);

	if (!camera_started_ || !request_found)
		return;

	for (auto const &p : buffers)
	{
		if (request->addBuffer(p.first, p.second) < 0)
			throw std::runtime_error("failed to add buffer to request in QueueRequest");
	}

	{
		std::lock_guard<std::mutex> lock(control_mutex_);
		request->controls() = std::move(controls_);
	}

	if (camera_->queueRequest(request) < 0)
		throw std::runtime_error("failed to queue request");
}

void LibcameraApp::queueRequest2(CompletedRequest *completed_request)
{
	BufferMap buffers(std::move(completed_request->buffers));

	// This function may run asynchronously so needs protection from the
	// camera stopping at the same time.
	std::lock_guard<std::mutex> stop_lock(camera_stop_mutex2_);

	// An application could be holding a CompletedRequest while it stops and re-starts
	// the camera, after which we don't want to queue another request now.
	bool request_found;
	{
		std::lock_guard<std::mutex> lock(completed_requests_mutex2_);
		auto it = completed_requests2_.find(completed_request);
		if (it != completed_requests2_.end())
		{
			request_found = true;
			completed_requests2_.erase(it);
		}
		else
			request_found = false;
	}

	Request *request = completed_request->request;
	delete completed_request;
	assert(request);

	if (!camera2_started_ || !request_found)
		return;

	for (auto const &p : buffers)
	{
		if (request->addBuffer(p.first, p.second) < 0)
			throw std::runtime_error("failed to add buffer to request in QueueRequest");
	}

	{
		std::lock_guard<std::mutex> lock(control_mutex_);
		request->controls() = std::move(controls_);
	}

	if (camera2_->queueRequest(request) < 0)
		throw std::runtime_error("failed to queue request");
}

void LibcameraApp::PostMessage(MsgType &t, MsgPayload &p)
{
	msg_queue1_.Post(Msg(t, std::move(p)));
}

libcamera::Stream *LibcameraApp::GetStream(std::string const &name, StreamInfo *info) const
{
	auto it = streams_.find(name);
	if (it == streams_.end())
		return nullptr;
	if (info)
		*info = GetStreamInfo(it->second);
	return it->second;
}

libcamera::Stream *LibcameraApp::ViewfinderStream(StreamInfo *info) const
{
	return GetStream("viewfinder", info);
}

libcamera::Stream *LibcameraApp::Viewfinder2Stream(StreamInfo *info) const
{
	return GetStream("viewfinder2", info);
}

libcamera::Stream *LibcameraApp::StillStream(StreamInfo *info) const
{
	return GetStream("still", info);
}

libcamera::Stream *LibcameraApp::RawStream(StreamInfo *info) const
{
	return GetStream("raw", info);
}

libcamera::Stream *LibcameraApp::VideoStream(StreamInfo *info) const
{
	return GetStream("video", info);
}

libcamera::Stream *LibcameraApp::LoresStream(StreamInfo *info) const
{
	return GetStream("lores", info);
}

libcamera::Stream *LibcameraApp::GetMainStream() const
{
	for (auto &p : streams_)
	{
		if (p.first == "viewfinder" || p.first == "still" || p.first == "video")
			return p.second;
	}

	return nullptr;
}

std::vector<libcamera::Span<uint8_t>> LibcameraApp::Mmap(FrameBuffer *buffer) const
{
	auto item = mapped_buffers_.find(buffer);
	if (item == mapped_buffers_.end())
		return {};
	return item->second;
}

std::vector<libcamera::Span<uint8_t>> LibcameraApp::Mmap2(FrameBuffer *buffer) const
{
	auto item = mapped_buffers2_.find(buffer);
	if (item == mapped_buffers2_.end())
		return {};
	return item->second;
}

static int fc = 0;
static struct timeval start_time1_;
static struct timeval last_time1_;
void LibcameraApp::ShowPreviewDual(CompletedRequestPtr &completed_request, CompletedRequestPtr &completed_request2, Stream *stream, Stream *stream2)
{
	std::lock_guard<std::mutex> lock(preview_item_mutex_);
	if (!preview_item_.stream)
		preview_item_ = PreviewItem(completed_request, stream); // copy the shared_ptr here

	else
		preview_frames_dropped_++;

	preview_cond_var1_.notify_one();

	std::lock_guard<std::mutex> lock2(preview_item_mutex2_);	
	if (!preview_item2_.stream)
		preview_item2_ = PreviewItem(completed_request2, stream2); // copy the shared_ptr here
	else
		preview_frames_dropped_++;

 	preview_cond_var2_.notify_one();
	
	gettimeofday(&start_time1_, NULL);
	if(fc % 300 == 0) {
		long diff= (start_time1_.tv_sec - last_time1_.tv_sec) *1000 + (start_time1_.tv_usec - last_time1_.tv_usec)/1000;
		printf("ShowPreviewDual frame_cnt=%d, fps =%2f\n",  fc, 300000.0/diff);
		last_time1_ = start_time1_;
	}
	fc++;
}

void LibcameraApp::ShowPreview(CompletedRequestPtr &completed_request, Stream *stream)
{
	std::lock_guard<std::mutex> lock(preview_item_mutex_);
	if (!preview_item_.stream)
		preview_item_ = PreviewItem(completed_request, stream); // copy the shared_ptr here

	else
		preview_frames_dropped_++;

	preview_cond_var1_.notify_one();

	gettimeofday(&start_time1_, NULL);
	if(fc % 300 == 0) {
		long diff= (start_time1_.tv_sec - last_time1_.tv_sec) *1000 + (start_time1_.tv_usec - last_time1_.tv_usec)/1000;
		printf("ShowPreview  frame_cnt=%d, fps =%2f, diff=%ld\n",  fc, 300000.0/diff, diff);
		last_time1_ = start_time1_;
	}
	fc++;
}

void LibcameraApp::SetControls(ControlList &controls)
{
	std::lock_guard<std::mutex> lock(control_mutex_);
	// Add new controls to the stored list. If a control is duplicated,
	// the value in the argument replaces the previously stored value.
	// These controls will be applied to the next StartCamera or request.
	for (const auto &c : controls)
		controls_.set(c.first, c.second);
}

StreamInfo LibcameraApp::GetStreamInfo(Stream const *stream) const
{
	StreamConfiguration const &cfg = stream->configuration();
	StreamInfo info;
	info.width = cfg.size.width;
	info.height = cfg.size.height;
	info.stride = cfg.stride;
	info.pixel_format = cfg.pixelFormat;
	info.colour_space = cfg.colorSpace;
	return info;
}

void LibcameraApp::setupCapture()
{
	// First finish setting up the configuration.
	CameraConfiguration::Status validation = configuration_->validate();

	if (validation == CameraConfiguration::Invalid)
		throw std::runtime_error("failed to valid stream configurations");
	else if (validation == CameraConfiguration::Adjusted)
		LOG(1, "Stream configuration adjusted");

	if (camera_->configure(configuration_.get()) < 0) 
		throw std::runtime_error("failed to configure streams");

	printf( "CameraNum CameraNum:%d\n", CameraNum);
	if(CameraNum > 1) {
		CameraConfiguration::Status validation2 = configuration2_->validate();

		if (validation2 == CameraConfiguration::Invalid)
			throw std::runtime_error("failed to valid stream configurations");
		else if (validation2 == CameraConfiguration::Adjusted)
			LOG(1, "Stream configuration adjusted");

		if (camera2_->configure(configuration2_.get()) < 0)
			throw std::runtime_error("failed to configure streams for camera 2");
	}

	LOG(2, "Camera streams configured");

	// Next allocate all the buffers we need, mmap them and store them on a free list.
    allocator_ = new FrameBufferAllocator(camera_);
    for (StreamConfiguration &config : *configuration_)
    {
        Stream *stream = config.stream();

        if (allocator_->allocate(stream) < 0)
            throw std::runtime_error("failed to allocate capture buffers");

        std::cout << "\n=== Stream Configuration ===" << std::endl;
        std::cout << "  Format: " << config.pixelFormat.toString() << std::endl;
        std::cout << "  Size: " << config.size.width << "x" << config.size.height << std::endl;
        std::cout << "  Stride: " << config.stride << std::endl;
        std::cout << "  Frame Size: " << config.frameSize << std::endl;
        std::cout << "============================\n" << std::endl;

        for (const std::unique_ptr<FrameBuffer> &buffer : allocator_->buffers(stream))
        {
            // "Single plane" buffers appear as multi-plane here, but we can spot them because then
            // planes all share the same fd. We accumulate them so as to mmap the buffer only once.
            size_t buffer_size = 0;
            for (unsigned i = 0; i < buffer->planes().size(); i++)
            {
                const FrameBuffer::Plane &plane = buffer->planes()[i];
                buffer_size += plane.length;
                if (i == buffer->planes().size() - 1 || plane.fd.get() != buffer->planes()[i + 1].fd.get())
                {
                    void *memory = mmap(NULL, buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
                    if (memory == MAP_FAILED) {
                        throw std::runtime_error("mmap failed: ");
                    }
                    
                    mapped_buffers_[buffer.get()].push_back(
                        libcamera::Span<uint8_t>(static_cast<uint8_t *>(memory), buffer_size));
                    
                    buffer_size = 0;
                }
            }
            frame_buffers_[stream].push(buffer.get());
        }
    }

	if(CameraNum > 1) {
		allocator2_ = new FrameBufferAllocator(camera2_);
		for (StreamConfiguration &config : *configuration2_)
		{
			Stream *stream = config.stream();
			if (allocator2_->allocate(stream) < 0)
				throw std::runtime_error("failed to allocate capture buffers");

			for (const std::unique_ptr<FrameBuffer> &buffer : allocator2_->buffers(stream))
			{
				// "Single plane" buffers appear as multi-plane here, but we can spot them because then
				// planes all share the same fd. We accumulate them so as to mmap the buffer only once.
				size_t buffer_size = 0;
				for (unsigned i = 0; i < buffer->planes().size(); i++)
				{
					const FrameBuffer::Plane &plane = buffer->planes()[i];
					buffer_size += plane.length;
					if (i == buffer->planes().size() - 1 || plane.fd.get() != buffer->planes()[i + 1].fd.get())
					{
						void *memory = mmap(NULL, buffer_size, PROT_READ | PROT_WRITE, MAP_SHARED, plane.fd.get(), 0);
						mapped_buffers2_[buffer.get()].push_back(
							libcamera::Span<uint8_t>(static_cast<uint8_t *>(memory), buffer_size));
						buffer_size = 0;
					}
				}
				frame_buffers2_[stream].push(buffer.get());
			}
		}
	}

	startPreview();

	// The requests will be made when StartCamera() is called.
}

void LibcameraApp::makeRequests()
{
	auto free_buffers(frame_buffers_);
	auto free_buffers2(frame_buffers2_);

	while (true)
	{
		for (StreamConfiguration &config : *configuration_)
		{
			Stream *stream = config.stream();
			if (stream == configuration_->at(0).stream())
			{
				if (free_buffers[stream].empty())
					return;

				std::unique_ptr<Request> request = camera_->createRequest();
				if (!request)
					throw std::runtime_error("failed to make request");
				requests_.push_back(std::move(request));
			}
			else if (free_buffers[stream].empty())
				throw std::runtime_error("concurrent streams need matching numbers of buffers");

			FrameBuffer *buffer = free_buffers[stream].front();
			free_buffers[stream].pop();

			if (requests_.back()->addBuffer(stream, buffer) < 0)
				throw std::runtime_error("failed to add buffer to request");
		}

	if(CameraNum > 1) {
		for (StreamConfiguration &config : *configuration2_)
		{
			Stream *stream = config.stream();
			if (stream == configuration2_->at(0).stream())
			{
				if (free_buffers2[stream].empty())
					return;

				std::unique_ptr<Request> request = camera2_->createRequest();
				if (!request)
					throw std::runtime_error("failed to make request");
				requests2_.push_back(std::move(request));
			}
			else if (free_buffers2[stream].empty())
				throw std::runtime_error("concurrent streams need matching numbers of buffers");

			FrameBuffer *buffer = free_buffers2[stream].front();
			free_buffers2[stream].pop();
			if (requests2_.back()->addBuffer(stream, buffer) < 0)
				throw std::runtime_error("failed to add buffer to request");
		}
	}
		//std::cout << "requests2_ " << requests2_.size() << std::endl;
	}
}
//static int frame_cnt1 = 0;
//static int frame_cnt2 = 0;
void LibcameraApp::requestComplete(Request *request)
{
	if (request->status() == Request::RequestCancelled)
	{
		// If the request is cancelled while the camera is still running, it indicates
		// a hardware timeout. Let the application handle this error.
		if (camera_started_)
			msg_queue1_.Post(Msg(MsgType::Timeout));

		return;
	}

	CompletedRequest *r = new CompletedRequest(sequence_++, request);
	CompletedRequestPtr payload(r, [this](CompletedRequest *cr) { this->queueRequest(cr); });
	{
		std::lock_guard<std::mutex> lock(completed_requests_mutex_);
		completed_requests_.insert(r);
		//std::cout << "completed_requests_ size " << completed_requests_.size() << std::endl;
	}

	// We calculate the instantaneous framerate in case anyone wants it.
	// Use the sensor timestamp if possible as it ought to be less glitchy than
	// the buffer timestamps.
	auto ts = payload->metadata.get(controls::SensorTimestamp);
	uint64_t timestamp = ts ? *ts : payload->buffers.begin()->second->metadata().timestamp;
	if (last_timestamp1_ == 0 || last_timestamp1_ == timestamp)
		payload->framerate = 0;
	else
		payload->framerate = 1e9 / (timestamp - last_timestamp1_);
/*
	if(frame_cnt1 % 300 == 0) {
		long diff = (timestamp - last_timestamp1_)/1000000;
		printf("----camera1: frame_cnt=%d %d,fps =%2f\n",  frame_cnt1, frame_cnt2, 300000.0/diff);
		last_timestamp1_ = timestamp;
	}
	printf("----camera1: frame_cnt1=%d frame_cnt2=%d\n",  frame_cnt1, frame_cnt2);
	frame_cnt1++;
*/
	post_processor_.Process(payload); // post-processor can re-use our shared_ptr
}

void LibcameraApp::requestComplete2(Request *request)
{
	if (request->status() == Request::RequestCancelled)
	{
		// If the request is cancelled while the camera is still running, it indicates
		// a hardware timeout. Let the application handle this error.
		if (camera_started_)
			msg_queue2_.Post(Msg(MsgType::Timeout));

		return;
	}

	CompletedRequest *r2 = new CompletedRequest(sequence2_++, request);
	CompletedRequestPtr payload2(r2, [this](CompletedRequest *cr2) { this->queueRequest2(cr2); });
	{
		std::lock_guard<std::mutex> lock(completed_requests_mutex2_);
		completed_requests2_.insert(r2);
		//std::cout << "completed_requests2_ size " << completed_requests2_.size() << std::endl;
	}

	// We calculate the instantaneous framerate in case anyone wants it.
	// Use the sensor timestamp if possible as it ought to be less glitchy than
	// the buffer timestamps.
	auto ts = payload2->metadata.get(controls::SensorTimestamp);
	uint64_t timestamp = ts ? *ts : payload2->buffers.begin()->second->metadata().timestamp;
	if (last_timestamp2_ == 0 || last_timestamp2_ == timestamp)
		payload2->framerate = 0;
	else
		payload2->framerate = 1e9 / (timestamp - last_timestamp2_);
/*
	if(frame_cnt2 % 300 == 0) {
		long diff = (timestamp - last_timestamp2_)/1000000;
		printf("***camera2 frame_cnt=%d %d, fps =%2f\n",  frame_cnt1, frame_cnt2, 300000.0/diff);
		last_timestamp2_ = timestamp;
	}
	printf("----camera2: frame_cnt1=%d frame_cnt2=%d\n",  frame_cnt1, frame_cnt2);
	frame_cnt2++;
*/
	post_processor2_.Process(payload2); // post-processor can re-use our shared_ptr
}

void LibcameraApp::previewDoneCallback(int fd)
{
	std::lock_guard<std::mutex> lock(preview_mutex_);
	auto it = preview_completed_requests_.find(fd);
	if (it == preview_completed_requests_.end())
		throw std::runtime_error("previewDoneCallback: missing fd " + std::to_string(fd));
	preview_completed_requests_.erase(it); // drop shared_ptr reference
}

void LibcameraApp::previewDoneCallbackDual(int fd, int fd2)
{
	std::lock_guard<std::mutex> lock(preview_mutex_);
	auto it = preview_completed_requests_.find(fd);
	if (it == preview_completed_requests_.end())
		throw std::runtime_error("previewDoneCallback: missing fd " + std::to_string(fd));
	preview_completed_requests_.erase(it); // drop shared_ptr reference

	auto it2 = preview_completed_requests2_.find(fd2);
	if (it2 == preview_completed_requests2_.end())
		throw std::runtime_error("previewDoneCallback: missing fd2 " + std::to_string(fd2));
	preview_completed_requests2_.erase(it2); // drop shared_ptr reference
}

void LibcameraApp::startPreview()
{
	preview_abort_ = false;
	preview_thread_ = std::thread(&LibcameraApp::previewThread, this);
}

void LibcameraApp::stopPreview()
{
	if (!preview_thread_.joinable()) // in case never started
		return;

	{
		std::lock_guard<std::mutex> lock(preview_item_mutex_);
		preview_abort_ = true;
		preview_cond_var1_.notify_one();

		if(CameraNum > 1)
			preview_cond_var2_.notify_one();

	}
	preview_thread_.join();
	preview_item_ = PreviewItem();

	if(CameraNum > 1)
		preview_item2_ = PreviewItem();

}
static struct timeval start_time_;
static struct timeval last_time_;
void LibcameraApp::previewThread()
{
    while (true)
    {
        PreviewItem item;
        while (!item.stream)
        {
            std::unique_lock<std::mutex> lock(preview_item_mutex_);
            if (preview_abort_) {
                preview_->Reset();
                return;
            }
            else if (preview_item_.stream) {
                item = std::move(preview_item_); // re-use existing shared_ptr reference
            }
            else {
                preview_cond_var1_.wait(lock);
            }
        }

        PreviewItem item2;
        if (CameraNum > 1) {
            while (!item2.stream)
            {
                std::unique_lock<std::mutex> lock(preview_item_mutex2_);
                if (preview_abort_) {
                    preview_->Reset();
                    return;
                }
                else if (preview_item2_.stream) {
                    item2 = std::move(preview_item2_); // re-use existing shared_ptr reference
                }
                else {
                    preview_cond_var2_.wait(lock);
                }
            }
        }

        gettimeofday(&start_time_, NULL);
        if (preview_frames_displayed_ % 300 == 0) {
            long diff = (start_time_.tv_sec - last_time_.tv_sec) * 1000 + 
                       (start_time_.tv_usec - last_time_.tv_usec) / 1000;
            printf("libcamera App previewThread frame_cnt=%d, fps=%2f\n", 
                   preview_frames_displayed_, 300000.0 / diff);
            last_time_ = start_time_;
        }

        StreamInfo info = GetStreamInfo(item.stream);
        FrameBuffer *buffer = item.completed_request->buffers[item.stream];
        libcamera::Span span = Mmap(buffer)[0];
        
        // Fill the frame info with the ControlList items and ancillary bits.
        FrameInfo frame_info(item.completed_request->metadata);
        frame_info.fps = item.completed_request->framerate;
        frame_info.sequence = item.completed_request->sequence;

        int fd = buffer->planes()[0].fd.get();
        {
            std::lock_guard<std::mutex> lock(preview_mutex_);
            // the reference to the shared_ptr moves to the map here
            preview_completed_requests_[fd] = std::move(item.completed_request);
        }

        if (CameraNum > 1) {
            StreamInfo info2 = GetStreamInfo(item2.stream);
            FrameBuffer *buffer2 = item2.completed_request->buffers[item2.stream];
            libcamera::Span<uint8_t> span2 = Mmap2(buffer2)[0];
            
            int fd2 = buffer2->planes()[0].fd.get();
            {
                std::lock_guard<std::mutex> lock(preview_mutex_);
                // the reference to the shared_ptr moves to the map here
                preview_completed_requests2_[fd2] = std::move(item2.completed_request);
            }

            if (preview_->Quit())
            {
                LOG(2, "Preview window has quit");
                msg_queue1_.Post(Msg(MsgType::Quit));
                msg_queue2_.Post(Msg(MsgType::Quit));
            }
            
            preview_frames_displayed_++;
            
              preview_->ShowBufferDual(buffer, span, info, buffer2, span2, info2);
        }
        else {
            if (preview_->Quit())
            {
                LOG(2, "Preview window has quit");
                msg_queue1_.Post(Msg(MsgType::Quit));
            }
            
            preview_frames_displayed_++;
            
            preview_->ShowBuffer(buffer, span, info);
        }
    }
}


void LibcameraApp::configureDenoise(const std::string &denoise_mode)
{
	using namespace libcamera::controls::draft;

	static const std::map<std::string, NoiseReductionModeEnum> denoise_table = {
		{ "off", NoiseReductionModeOff },
		{ "cdn_off", NoiseReductionModeMinimal },
		{ "cdn_fast", NoiseReductionModeFast },
		{ "cdn_hq", NoiseReductionModeHighQuality }
	};
	NoiseReductionModeEnum denoise;

	auto const mode = denoise_table.find(denoise_mode);
	if (mode == denoise_table.end())
		throw std::runtime_error("Invalid denoise mode " + denoise_mode);
	denoise = mode->second;

	controls_.set(NoiseReductionMode, denoise);
}

void LibcameraApp::enableGPUScaler(int target_width, int target_height)
{
	if (preview_)
		preview_->enableGPUScaler(target_width, target_height);
}

void LibcameraApp::enableVPUEncoder(const std::string& codec, int bitrate, const std::string& output_file)
{
	if (preview_)
		preview_->enableVPUEncoder(codec, bitrate, output_file);
}

void LibcameraApp::disableGPUScaler()
{
	if (preview_)
		preview_->disableGPUScaler();
}

void LibcameraApp::disableVPUEncoder()
{
	if (preview_)
		preview_->disableVPUEncoder();
}

