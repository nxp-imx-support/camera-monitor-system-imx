/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 * Copyright (C) 2020, Raspberry Pi (Trading) Ltd.
 *
 * drm_preview.cpp - DRM-based preview window with threaded detection.
 */

#include <drm.h>
#include <drm_fourcc.h>
#include <drm_mode.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <iomanip> 
#include <string.h>
#include <unistd.h>
#include <cmath> 
#include <sys/time.h>
#include <sys/ioctl.h>
#include <linux/dma-heap.h> 
#include <deque>
#include <array>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>

#include <g2d.h>

#include "core/logging.hpp"
#include "core/version.hpp"
#include "preview.hpp"

#include <opencv2/opencv.hpp>

#include "ssd_detector.h"
#include "gpu_scaler.hpp"
#include "vpu_encoder.hpp"
#include "dirty_detector.hpp"
#include "g2d_scaler.hpp"

#include <arm_neon.h>

using namespace imx95;

class DrmPreview : public Preview
{
public:
	DrmPreview(int camera_num, DetectorType detector_type = DETECTOR_NONE);
	~DrmPreview();
	virtual void Show(int fd, libcamera::Span<uint8_t> span, StreamInfo const &info) override;
	virtual void ShowDual(int fd, libcamera::Span<uint8_t> span, StreamInfo const &info, int fd2, libcamera::Span<uint8_t> span2, StreamInfo const &info2) override;
    virtual void ShowBuffer(libcamera::FrameBuffer* buffer, 
                           libcamera::Span<uint8_t> span, 
                           StreamInfo const &info) override;
    
    virtual void ShowBufferDual(libcamera::FrameBuffer* buffer1,
                               libcamera::Span<uint8_t> span1,
                               StreamInfo const &info1,
                               libcamera::FrameBuffer* buffer2,
                               libcamera::Span<uint8_t> span2,
                               StreamInfo const &info2) override;
	virtual void Reset() override;
	virtual void MaxImageSize(unsigned int &w, unsigned int &h) const override
	{
		w = max_image_width_;
		h = max_image_height_;
	}

	// GPU Scaler and VPU Encoder control
	void enableGPUScaler(int target_width, int target_height);
	void enableVPUEncoder(const std::string& codec, int bitrate, const std::string& output_file);
	void disableGPUScaler();
	void disableVPUEncoder();

private:
	int camera_num_;
	struct Buffer
	{
		 Buffer() : y_fd(-1), uv_fd(-1) {}
        int y_fd;
        int uv_fd;
		size_t size;
		StreamInfo info;
        uint32_t y_bo_handle;
        uint32_t uv_bo_handle;
		unsigned int fb_handle;
	};
	
	struct PlaneProperties {
		uint32_t crtc_id = 0;
		uint32_t fb_id = 0;
		uint32_t crtc_x = 0;
		uint32_t crtc_y = 0;
		uint32_t crtc_w = 0;
		uint32_t crtc_h = 0;
		uint32_t src_x = 0;
		uint32_t src_y = 0;
		uint32_t src_w = 0;
		uint32_t src_h = 0;
	};
	
    void makeBuffer(libcamera::FrameBuffer* framebuffer, 
                   size_t size, 
                   StreamInfo const &info, 
                   Buffer &buffer, 
                   uint32_t crtcId);
	std::map<libcamera::FrameBuffer*, Buffer> buffers_;
	libcamera::FrameBuffer* last_buffer_;
	 libcamera::FrameBuffer* last_buffer2_; 
	void addChangedProperties(drmModeAtomicReq *req, int plane_idx, uint32_t new_fb_id, 
									int x, int y, int w, int h, const StreamInfo &info);
	std::tuple<int, int, int, int> calculatePosition(const StreamInfo &info, int crtcIdx);
	void findCrtc();
	void findPlane();
	
	int drmfd_;
	int displayID = 0;
	int conId_[4];
	uint32_t crtcId_[4];
	int crtcIdx_1;
	int crtcIdx_2;
	uint32_t planeId_[4];
	unsigned int out_fourcc_;
	unsigned int x_;
	unsigned int y_;
	int crtc_x[2];
	int crtc_y[2];
	int crtc_w[2];
	int crtc_h[2];
	int src_x[2];
	int src_y[2];
	int src_w[2];
	int src_h[2];

	unsigned int width_;
	unsigned int height_;
	unsigned int screen_width_;
	unsigned int screen_height_;
	unsigned int max_image_width_;
	unsigned int max_image_height_;
	bool first_time_;

	PlaneProperties plane_props_[4];

	struct PlaneCache {
		uint32_t fb_id = 0;
		uint32_t crtc_id = 0;
		int crtc_x = -1, crtc_y = -1;
		int crtc_w = -1, crtc_h = -1;
		int src_x = 0, src_h = 0;
		int src_w = 0, src_y = 0;
	};
	PlaneCache plane_cache_[2];

	// Asynchronous detection
	std::thread detection_thread_;
	std::thread copy_thread_;
	std::mutex job_mutex_;
	std::mutex result_mutex_;
	std::mutex copy_mutex_;
	std::condition_variable job_cv_;
	std::condition_variable copy_cv_;
	std::atomic<bool> detection_running_{false};
	
	std::map<int, DetectionResult> latest_results_;
	
	// Performance statistics
	std::atomic<int> frames_processed_{0};
	std::atomic<int> detections_completed_{0};
	std::atomic<int> detections_skipped_{0};
	std::atomic<int> queue_full_count_{0};

	DetectorType detector_type_;
	std::unique_ptr<imx95::SSDDetectorCPU> ssd_detector_cpu_;
	std::unique_ptr<imx95::SSDDetectorNPU> ssd_detector_npu_;

	// GPU Scaler and VPU Encoder
	std::unique_ptr<GPUScaler> gpu_scaler_[2];  // Support dual camera
	std::unique_ptr<VPUEncoder> vpu_encoder_[2];
	bool gpu_scaler_enabled_ = false;
	bool vpu_encoder_enabled_ = false;
	int scaler_target_width_ = 0;
	int scaler_target_height_ = 0;
	std::string encoder_codec_;
	int encoder_bitrate_ = 0;
	std::string encoder_output_file_[2];

	// Encoding thread
	// Encoding job structure
	struct EncodingJob {
        int y_fd = -1;           // Y plane DMA-BUF fd
        int uv_fd = -1;          // UV plane DMA-BUF fd
        size_t y_size = 0;
        size_t uv_size = 0;
		int stride;
		int width;
		int height;
		uint64_t timestamp;
	};
	
	// Encoding thread
	std::thread encoding_thread_[2];
	std::mutex encoding_mutex_[2];
	std::condition_variable encoding_cv_[2];
	std::deque<EncodingJob> encoding_queue_[2];
	std::atomic<bool> encoding_running_{false};
	static constexpr int MAX_ENCODING_QUEUE_SIZE = 32;

	void encodingWorker(int camera_id);
    void submitEncodingJob(int y_fd, int uv_fd, size_t y_size, size_t uv_size,
                                    int stride, int width, int height, 
                                    int camera_id, uint64_t timestamp);
	// Ring buffer configuration
	static constexpr int RING_BUFFER_SIZE = 4;
	
    struct RingBuffer {
        const uint8_t* rgb_data_ptr = nullptr; 
        StreamInfo info;
        StreamInfo original_info;
        std::atomic<bool> in_use{false};
        int camera_id = -1;
        uint64_t timestamp = 0;
        int resized_width = 0;
        int resized_height = 0;
		bool resize_completed = false;
         int g2d_buffer_id = -1; 
    };
	
	struct DetectionJob {
		int buffer_index;
		int camera_id;
		uint64_t timestamp;
		bool valid = false;
	};

	struct CopyJob {
    	int y_fd = -1;           // Y plane DMA-BUF fd
    	int uv_fd = -1;          // UV plane DMA-BUF fd
        size_t y_size = 0;       // Y plane size in bytes
        size_t uv_size = 0;      // UV plane size in bytes
		StreamInfo info;
		int camera_id;
		int buffer_index;
		bool valid = false;
	};

	std::array<RingBuffer, RING_BUFFER_SIZE> ring_buffers_;
	std::atomic<int> write_index_{0};

	std::deque<DetectionJob> job_queue_;
	std::deque<CopyJob> copy_queue_;
	static constexpr int MAX_QUEUE_SIZE = 2;

	struct DetectionJobDual {
		int buffer_index1;
		int buffer_index2;
		int camera_id1;
		int camera_id2;
		uint64_t timestamp;
		bool valid = false;
	};

	std::deque<DetectionJobDual> job_queue_dual_;
	
	void submitDetectionJobDual(int fd1, libcamera::Span<uint8_t> span1, const StreamInfo &info1,
								int fd2, libcamera::Span<uint8_t> span2, const StreamInfo &info2);
	void detectionWorkerDual();
	
	std::thread detection_thread_dual_;
	
	int acquireBufferIndex();
	void releaseBuffer(int index);
	
	void detectionWorker();
	void copyWorker();
	void submitDetectionJob(int fd, libcamera::Span<uint8_t> span, 
						   const StreamInfo &info, int camera_id);
	void drawDetections(libcamera::Span<uint8_t> span, const StreamInfo &info, int camera_id);
	
	DetectionResult processImage(const uint8_t* image_data, int width, int height, int camera_id);

    // Dirty detector
    std::unique_ptr<imx95::DirtyDetector> dirty_detector_;
    std::thread dirty_detection_thread_;
    std::mutex dirty_job_mutex_;
    std::mutex dirty_result_mutex_;
    std::mutex dirty_detection_mutex_;  
    std::condition_variable dirty_job_cv_;
    std::atomic<int> dirty_frame_counter_{0};
    static constexpr int DIRTY_DETECTION_INTERVAL = 60;
    bool draw_dirty_contours_ = true;
    struct DirtyCopyBuffer {
        std::vector<uint8_t> y_data;
        int width = 0;
        int height = 0;
        int stride = 0;
        int camera_id = 0;
        uint64_t timestamp = 0;
        std::atomic<bool> in_use{false};
        bool copy_completed = false;
    };
    
    static constexpr int DIRTY_COPY_BUFFER_SIZE = 2;
    std::array<DirtyCopyBuffer, DIRTY_COPY_BUFFER_SIZE> dirty_copy_buffers_;
    std::atomic<int> dirty_write_index_{0};
    
    std::thread dirty_copy_thread_;
    std::mutex dirty_copy_mutex_;
    std::condition_variable dirty_copy_cv_;
    
    struct DirtyCopyJob {
        int y_fd = -1;
        size_t y_size = 0;
        int width = 0;
        int height = 0;
        int stride = 0;
        int camera_id = 0;
        uint64_t timestamp = 0;
        int buffer_index = -1;
        bool valid = false;
    };
    
    std::deque<DirtyCopyJob> dirty_copy_queue_;
    
    void dirtyCopyWorker();
    int acquireDirtyCopyBuffer();
    void releaseDirtyCopyBuffer(int index);

    struct DirtyDetectionJob {
        int y_fd = -1;
        size_t y_size = 0;
        int width = 0;
        int height = 0;
        int stride = 0;
        int camera_id = 0;
        uint64_t timestamp = 0;
        int buffer_index = -1;
        bool valid = false;
    };
    
    std::deque<DirtyDetectionJob> dirty_job_queue_;
    std::map<int, imx95::DirtyDetectionResult> latest_dirty_results_;
    imx95::DirtyDetectionResult dirty_detection_result_;

    void dirtyDetectionWorker();
    void submitDirtyDetectionJob(int y_fd, size_t y_size, 
                                 const StreamInfo &info, int camera_id);
    void drawDirtyDetections(libcamera::Span<uint8_t> span, 
                            const StreamInfo &info, int camera_id);

    static constexpr int SRC_WIDTH = 1920;
    static constexpr int SRC_HEIGHT = 1080;
    static constexpr int TARGET_WIDTH = 300;
    static constexpr int TARGET_HEIGHT = 300;

    std::unique_ptr<imx95::G2DScaler> g2d_scaler_;
    struct DRMBuffer {
        int dma_fd = -1;           // DMA-BUF fd
        void* virt_addr = nullptr;
        size_t size = 0;
        uint32_t bo_handle = 0;    // DRM BO handle
    };
    
    std::array<DRMBuffer, RING_BUFFER_SIZE> drm_buffers_;
    
    bool allocateDRMBuffer(int index, size_t size);
    void freeDRMBuffer(int index);
};

#define ERRSTR strerror(errno)

static int frame_cnt = 0;

void DrmPreview::findCrtc()
{
	int i;
	drmModeRes *res = drmModeGetResources(drmfd_);
	if (!res)
		throw std::runtime_error("drmModeGetResources failed: " + std::string(ERRSTR));

	if (res->count_crtcs <= 0)
	{
		drmModeFreeResources(res);
		throw std::runtime_error("drm: no crts");
	}

	max_image_width_ = res->max_width;
	max_image_height_ = res->max_height;

	if (!conId_[0])
	{
		LOG(2, "No connector ID specified.  Choosing default from list:");

		for (i = 0; i < res->count_connectors; i++)
		{
			drmModeConnector *con = drmModeGetConnector(drmfd_, res->connectors[i]);
			drmModeEncoder *enc = NULL;
			drmModeCrtc *crtc = NULL;

			if (con->encoder_id)
			{
				enc = drmModeGetEncoder(drmfd_, con->encoder_id);
				if (enc->crtc_id)
				{
					crtc = drmModeGetCrtc(drmfd_, enc->crtc_id);
				}
			}
			if (!conId_[displayID] && crtc)
			{
				conId_[displayID] = con->connector_id;
				crtcId_[displayID] = crtc->crtc_id;
				displayID++;
			}

			if (crtc)
			{
				screen_width_ = crtc->width;
				screen_height_ = crtc->height;
			}

			LOG(2, "Connector " << con->connector_id << " (crtc " << (crtc ? crtc->crtc_id : 0) << "): type "
								<< con->connector_type << ", " << (crtc ? crtc->width : 0) << "x"
								<< (crtc ? crtc->height : 0) << (conId_[0] == (int)con->connector_id ? " (chosen)" : ""));

			if (con->encoder_id)
			{
				drmModeFreeEncoder(enc);
				if (enc->crtc_id)
				{
					drmModeFreeCrtc(crtc);
				}
			}
			drmModeFreeConnector(con);
		}

		if (!conId_[0])
		{
			drmModeFreeResources(res);
			throw std::runtime_error("No suitable enabled connector found");
		}
	}

	crtcIdx_1 = -1;

	for (i = 0; i < res->count_crtcs; ++i)
	{
		if (crtcId_[0] == res->crtcs[i])
		{
			crtcIdx_1 = i;
			crtcId_[crtcIdx_1] = crtcId_[0];
			break;
		}
	}
	if(crtcId_[1])
	{
		for (i = 0; i < res->count_crtcs; ++i)
		{
			if (crtcId_[1] == res->crtcs[i])
			{
				crtcIdx_2 = i;
                crtcId_[crtcIdx_2] = crtcId_[1];
				break;
			}
		}
	}

	if (crtcIdx_1 == -1)
	{
		drmModeFreeResources(res);
		throw std::runtime_error("drm: CRTC " + std::to_string(crtcId_[0]) + " not found");
	}

	if (res->count_connectors <= 0)
	{
		drmModeFreeResources(res);
		throw std::runtime_error("drm: no connectors");
	}

	drmModeConnector *c;
	c = drmModeGetConnector(drmfd_, conId_[0]);
	if (!c)
	{
		drmModeFreeResources(res);
		throw std::runtime_error("drmModeGetConnector failed: " + std::string(ERRSTR));
	}

	if (!c->count_modes)
	{
		drmModeFreeConnector(c);
		drmModeFreeResources(res);
		throw std::runtime_error("connector supports no mode");
	}

	drmModeCrtc *crtc = drmModeGetCrtc(drmfd_, crtcId_[0]);
	x_ = crtc->x;
	y_ = crtc->y;
	width_ = crtc->width;
	height_ = crtc->height;
	drmModeFreeCrtc(crtc);

	drmModeFreeConnector(c);
	drmModeFreeResources(res);
}

int enable_atomic(int fd) {
	struct drm_set_client_cap cap;
	cap.capability = DRM_CLIENT_CAP_ATOMIC;
	cap.value = 1;
	return ioctl(fd, DRM_IOCTL_SET_CLIENT_CAP, &cap);
}

void DrmPreview::findPlane()
{
	drmModePlaneResPtr planes;
	drmModePlanePtr plane;
	unsigned int i;
	unsigned int j;

	enable_atomic(drmfd_);

	planes = drmModeGetPlaneResources(drmfd_);
	if (!planes)
		throw std::runtime_error("drmModeGetPlaneResources failed: " + std::string(ERRSTR));

	try
	{
		for (i = 0; i < planes->count_planes; ++i)
		{
			plane = drmModeGetPlane(drmfd_, planes->planes[i]);
			if (!plane)
				throw std::runtime_error("drmModeGetPlane failed: " + std::string(ERRSTR));

			if (plane->crtc_id != crtcId_[crtcIdx_1])
			{
				drmModeFreePlane(plane);
				continue;
			}

			for (j = 0; j < plane->count_formats; ++j)
			{
				if (plane->formats[j] == out_fourcc_)
				{
					break;
				}
			}

			if (j == plane->count_formats)
			{
				drmModeFreePlane(plane);
				continue;
			}

			planeId_[0] = plane->plane_id;
			drmModeObjectProperties *props = drmModeObjectGetProperties(
				drmfd_, plane->plane_id, DRM_MODE_OBJECT_PLANE);
			if (!props)
				throw std::runtime_error("drmModeObjectGetProperties failed");

			for (unsigned int k = 0; k < props->count_props; ++k)
			{
				drmModePropertyRes *prop = drmModeGetProperty(drmfd_, props->props[k]);
				if (!prop) continue;

				if (strcmp(prop->name, "CRTC_ID") == 0)
					plane_props_[0].crtc_id = prop->prop_id;
				else if (strcmp(prop->name, "FB_ID") == 0)
					plane_props_[0].fb_id = prop->prop_id;
				else if (strcmp(prop->name, "CRTC_X") == 0)
					plane_props_[0].crtc_x = prop->prop_id;
				else if (strcmp(prop->name, "CRTC_Y") == 0)
					plane_props_[0].crtc_y = prop->prop_id;
				else if (strcmp(prop->name, "CRTC_W") == 0)
					plane_props_[0].crtc_w = prop->prop_id;
				else if (strcmp(prop->name, "CRTC_H") == 0)
					plane_props_[0].crtc_h = prop->prop_id;
				else if (strcmp(prop->name, "SRC_X") == 0)
					plane_props_[0].src_x = prop->prop_id;
				else if (strcmp(prop->name, "SRC_Y") == 0)
					plane_props_[0].src_y = prop->prop_id;
				else if (strcmp(prop->name, "SRC_W") == 0)
					plane_props_[0].src_w = prop->prop_id;
				else if (strcmp(prop->name, "SRC_H") == 0)
					plane_props_[0].src_h = prop->prop_id;

				drmModeFreeProperty(prop);
			}
			drmModeFreeObjectProperties(props);
			drmModeFreePlane(plane);
			break;
		}
		if(displayID > 1)
		{
			for (i = 0; i < planes->count_planes; ++i)
			{
				plane = drmModeGetPlane(drmfd_, planes->planes[i]);
				if (!plane)
					throw std::runtime_error("drmModeGetPlane failed: " + std::string(ERRSTR));

				if (plane->crtc_id != crtcId_[crtcIdx_2])
				{
					drmModeFreePlane(plane);
					continue;
				}

				for (j = 0; j < plane->count_formats; ++j)
				{
					if (plane->formats[j] == out_fourcc_)
					{
						break;
					}
				}

				if (j == plane->count_formats)
				{
					drmModeFreePlane(plane);
					continue;
				}

				planeId_[1] = plane->plane_id;
			   drmModeObjectProperties *props = drmModeObjectGetProperties(
					drmfd_, plane->plane_id, DRM_MODE_OBJECT_PLANE);
				if (!props)
					throw std::runtime_error("drmModeObjectGetProperties failed");

				for (unsigned int k = 0; k < props->count_props; ++k)
				{
					drmModePropertyRes *prop = drmModeGetProperty(drmfd_, props->props[k]);
					if (!prop) continue;

					if (strcmp(prop->name, "CRTC_ID") == 0)
						plane_props_[1].crtc_id = prop->prop_id;
					else if (strcmp(prop->name, "FB_ID") == 0)
						plane_props_[1].fb_id = prop->prop_id;
					else if (strcmp(prop->name, "CRTC_X") == 0)
						plane_props_[1].crtc_x = prop->prop_id;
					else if (strcmp(prop->name, "CRTC_Y") == 0)
						plane_props_[1].crtc_y = prop->prop_id;
					else if (strcmp(prop->name, "CRTC_W") == 0)
						plane_props_[1].crtc_w = prop->prop_id;
					else if (strcmp(prop->name, "CRTC_H") == 0)
						plane_props_[1].crtc_h = prop->prop_id;
					else if (strcmp(prop->name, "SRC_X") == 0)
						plane_props_[1].src_x = prop->prop_id;
					else if (strcmp(prop->name, "SRC_Y") == 0)
						plane_props_[1].src_y = prop->prop_id;
					else if (strcmp(prop->name, "SRC_W") == 0)
						plane_props_[1].src_w = prop->prop_id;
					else if (strcmp(prop->name, "SRC_H") == 0)
						plane_props_[1].src_h = prop->prop_id;

					drmModeFreeProperty(prop);
				}
				drmModeFreeObjectProperties(props);
				drmModeFreePlane(plane);
				break;
			}
		}

	}
	catch (std::exception const &e)
	{
		drmModeFreePlaneResources(planes);
		throw;
	}

	drmModeFreePlaneResources(planes);
}

DrmPreview::DrmPreview(int camera_num, DetectorType detector_type) 
    : Preview(), last_buffer_(nullptr), last_buffer2_(nullptr),first_time_(true), 
    detector_type_(detector_type)
{
	drmfd_ = drmOpen("imx95-dpu", NULL);
	if (drmfd_ < 0)
		throw std::runtime_error("drmOpen failed: " + std::string(ERRSTR));

    // Add format capability check
    uint64_t has_dumb;
    if (drmGetCap(drmfd_, DRM_CAP_DUMB_BUFFER, &has_dumb) < 0 || !has_dumb) {
        close(drmfd_);
        throw std::runtime_error("DRM device doesn't support dumb buffers");
    }
    
    // Check if NV12 is supported
    drmModePlaneRes *plane_res = drmModeGetPlaneResources(drmfd_);
    if (plane_res) {
        bool nv12_supported = false;
        for (uint32_t i = 0; i < plane_res->count_planes; i++) {
            drmModePlane *plane = drmModeGetPlane(drmfd_, plane_res->planes[i]);
            if (plane) {
                for (uint32_t j = 0; j < plane->count_formats; j++) {
                    if (plane->formats[j] == DRM_FORMAT_NV12) {
                        nv12_supported = true;
                        break;
                    }
                }
                drmModeFreePlane(plane);
            }
            if (nv12_supported) break;
        }
        drmModeFreePlaneResources(plane_res);
        
        if (!nv12_supported) {
            std::cerr << "Warning: NV12 format may not be supported by DRM plane" << std::endl;
        }
    }		

	camera_num_ = camera_num;
	screen_width_ = 0;
	screen_height_ = 0;
	try
	{
		if (!drmIsMaster(drmfd_))
			throw std::runtime_error("DRM preview unavailable - not master");

		displayID = 0;
		for(int i = 0; i < 4; i++)
			conId_[i] = 0;

		findCrtc();
		out_fourcc_ = DRM_FORMAT_NV12;
		findPlane();
	}
	catch (std::exception const &e)
	{
		close(drmfd_);
		throw;
	}
	
	width_ = screen_width_;
	height_ = screen_height_;

	// Initialize detector based on type
	if (detector_type_ == DETECTOR_SSD_CPU) {
		printf("Initializing CPU Detector...\n");
		ssd_detector_cpu_ = std::make_unique<SSDDetectorCPU>();
		ssd_detector_cpu_->setNumThreads(6);
		if (!ssd_detector_cpu_->initialize(CPU_MODEL,  CPU_LABEL, 
									0.5, 0.5)) {
			std::cerr << "Failed to initialize detector" << std::endl;
			throw std::runtime_error("Detector initialization failed");
		}
	} else if (detector_type_ == DETECTOR_SSD_NPU) {
		printf("Initializing NPU Detector...\n");
		ssd_detector_npu_ = std::make_unique<SSDDetectorNPU>();
		ssd_detector_npu_->setNumThreads(1);
		if (!ssd_detector_npu_->initialize(NPU_MODEL, NPU_LABEL, 
								0.01, 0.8)) {
			std::cerr << "Failed to initialize detector" << std::endl;
			throw std::runtime_error("Detector initialization failed");
		}
	}

    if (detector_type_ == DETECTOR_SSD_CPU || detector_type_ == DETECTOR_SSD_NPU) {
        
        g2d_scaler_ = std::make_unique<imx95::G2DScaler>();
        
        // Assume source is 1920x1080 (will be updated on first frame if different)
        if (!g2d_scaler_->setup(SRC_WIDTH, SRC_HEIGHT, SRC_WIDTH, TARGET_WIDTH, TARGET_HEIGHT)) {
            std::cerr << "Failed to initialize G2D scaler" << std::endl;
            throw std::runtime_error("G2D scaler initialization failed");
        }
        
        // ===== Allocate G2D output buffers =====
        size_t output_size = g2d_scaler_->getOutputSize();  // TARGET_WIDTH * TARGET_HEIGHT * 3
        for (int i = 0; i < RING_BUFFER_SIZE; i++) {
            if (!allocateDRMBuffer(i, output_size)) {
                std::cerr << "Failed to allocate DRM buffer " << i << std::endl;
                
                // Cleanup
                for (int j = 0; j < i; j++) {
                    freeDRMBuffer(j);
                }
                throw std::runtime_error("DRM buffer allocation failed");
            }
        }
    }
	
	// Pre-allocated circular buffer
	for (auto& buffer : ring_buffers_) {
		buffer.in_use = false;
        buffer.g2d_buffer_id = -1;
	}

	if (detector_type_ == DETECTOR_SSD_CPU || detector_type_ == DETECTOR_SSD_NPU) {
		detection_running_ = true;
        if (camera_num_ == 1) {
		    detection_thread_ = std::thread(&DrmPreview::detectionWorker, this);
        } else if (camera_num_ == 2) {
            detection_thread_dual_ = std::thread(&DrmPreview::detectionWorkerDual, this);
        }
		copy_thread_ = std::thread(&DrmPreview::copyWorker, this);
	}

    if (detector_type_ == DETECTOR_DIRTY) {
        printf("Initializing Dirty Detector (Multi-Filter Debug Mode)...\n");
        dirty_detector_ = std::make_unique<imx95::DirtyDetector>();
        
        // Basic parameters
        const int GRID_SIZE = 64;
        const float CONTRAST_THRESHOLD = 0.18f;
        
        if (!dirty_detector_->initialize(GRID_SIZE, CONTRAST_THRESHOLD)) {
            std::cerr << "Failed to initialize dirty detector" << std::endl;
            throw std::runtime_error("Dirty detector initialization failed");
        }

        // Size filters
        const int MIN_GRID_COUNT = 20;      // Minimum 3 grids
        const int MAX_GRID_COUNT = 300;     // Maximum 30 grids (filter large desktop areas)
        dirty_detector_->setMinGridCount(MIN_GRID_COUNT);
        dirty_detector_->setMaxGridCount(MAX_GRID_COUNT);
        
        // Brightness filter (based on your data: dirt ~50-90, desktop ~110-140)
        const float MIN_BRIGHTNESS = 60.0f;
        const float MAX_BRIGHTNESS = 120.0f;
        dirty_detector_->setBrightnessRange(MIN_BRIGHTNESS, MAX_BRIGHTNESS);
        
        // Gradient filter
        const float GRADIENT_THRESHOLD = 8.0f;  // Adjust based on debug output
        dirty_detector_->setGradientThreshold(GRADIENT_THRESHOLD);
        
        // Position filter (exclude edges)
        const float EDGE_MARGIN_RATIO = 0.00f;  // Exclude 15% margin from edges
        dirty_detector_->setEdgeMarginRatio(EDGE_MARGIN_RATIO);
        
        // Shape filter (exclude long strips)
        const float MAX_ASPECT_RATIO = 2.5f;  // Max width:height ratio
        dirty_detector_->setMaxAspectRatio(MAX_ASPECT_RATIO);
        
        // Confidence filter
        const float CONFIDENCE_THRESHOLD_H = 0.9f;
        const float CONFIDENCE_THRESHOLD_L = 0.42f;
        dirty_detector_->setConfidenceThreshold(CONFIDENCE_THRESHOLD_H, CONFIDENCE_THRESHOLD_L);
        
        // Temporal stability (require N consecutive frames)
        const int TEMPORAL_STABILITY = 3;
        dirty_detector_->setTemporalStability(TEMPORAL_STABILITY);

        // Pre-allocate dirty copy buffers (1920x1080 Y plane)
        for (auto& buffer : dirty_copy_buffers_) {
            buffer.y_data.resize(SRC_WIDTH * SRC_HEIGHT);  // Max size
            buffer.in_use = false;
            buffer.copy_completed = false;
        }

        detection_running_ = true;
        dirty_detection_thread_ = std::thread(&DrmPreview::dirtyDetectionWorker, this);
        dirty_copy_thread_ = std::thread(&DrmPreview::dirtyCopyWorker, this);

        std::cout << " Dirty detector initialized with multi-filter strategy:" << std::endl;
        std::cout << "  ================================================================== " << std::endl;
        std::cout << "   Filter 1: Size          " << MIN_GRID_COUNT << " - " << MAX_GRID_COUNT << " grids              " << std::endl;
        std::cout << "   Filter 2: Brightness    " << MIN_BRIGHTNESS << " - " << MAX_BRIGHTNESS << " (Y value)         " << std::endl;
        std::cout << "   Filter 3: Gradient      < " << GRADIENT_THRESHOLD << "                      " << std::endl;
        std::cout << "   Filter 4: Edge margin   " << (EDGE_MARGIN_RATIO * 100) << "% exclusion           " << std::endl;
        std::cout << "   Filter 5: Aspect ratio  < " << MAX_ASPECT_RATIO << ":1                    " << std::endl;
        std::cout << "   Filter 6: Confidence_h    > " << CONFIDENCE_THRESHOLD_H << "                      " << std::endl;
        std::cout << "   Filter 7: Confidence_l    > " << CONFIDENCE_THRESHOLD_L << "                      " << std::endl;
        std::cout << "   Filter 8: Stability     " << TEMPORAL_STABILITY << " frames               " << std::endl;
        std::cout << "  ================================================================== \n" << std::endl;
    }

	//enableVPUEncoder("h264", 8000000, "output.mp4");
    //enableGPUScaler(100, 100);
}

DrmPreview::~DrmPreview()
{
	std::cout << "Stopping threads..." << std::endl;

	// Stop encoding threads
	encoding_running_ = false;
	for (int i = 0; i < 2; i++) {
		encoding_cv_[i].notify_all();
		if (encoding_thread_[i].joinable()) {
			encoding_thread_[i].join();
		}
	}

	detection_running_ = false;
	job_cv_.notify_all();
	copy_cv_.notify_all();
    dirty_job_cv_.notify_all();

	if(camera_num_ == 1){
		if (detection_thread_.joinable()) {
			detection_thread_.join();
		}
	}

	if(camera_num_ == 2){
		if (detection_thread_dual_.joinable()) {
			detection_thread_dual_.join();
		}
	}

	if (copy_thread_.joinable()) {
		copy_thread_.join();
	}

    if (dirty_detection_thread_.joinable()) {
        dirty_detection_thread_.join();
    }
    
    if (dirty_detector_) {
        dirty_detector_.reset();
    }

    // Stop dirty copy thread
    if (dirty_copy_thread_.joinable()) {
        dirty_copy_cv_.notify_all();
        dirty_copy_thread_.join();
    }

    std::cout << " Threads stopped" << std::endl;

	// Cleanup GPU scaler and VPU encoder
	for (int i = 0; i < 2; i++) {
		if (vpu_encoder_[i]) {
			vpu_encoder_[i]->stop();
		}
		gpu_scaler_[i].reset();
		vpu_encoder_[i].reset();
	}

    // ===== Cleanup G2D buffers =====
    if (g2d_scaler_) {
        std::cout << "Cleaning up DRM buffers..." << std::endl;
        
        for (int i = 0; i < RING_BUFFER_SIZE; i++) {
            freeDRMBuffer(i);
        }
        
        std::cout << " DRM buffers cleaned up" << std::endl;
        g2d_scaler_.reset();
    }

	close(drmfd_);
}

void DrmPreview::enableGPUScaler(int target_width, int target_height)
{
	scaler_target_width_ = target_width;
	scaler_target_height_ = target_height;
	gpu_scaler_enabled_ = true;
	
	std::cout << "GPU Scaler enabled: " << target_width << "x" << target_height << std::endl;
}

void DrmPreview::enableVPUEncoder(const std::string& codec, int bitrate, const std::string& output_file)
{
	encoder_codec_ = codec;
	encoder_bitrate_ = bitrate;
	
	if (camera_num_ == 1) {
		encoder_output_file_[0] = output_file;
	} else if (camera_num_ == 2) {
		// For dual camera, create separate output files
		size_t dot_pos = output_file.find_last_of('.');
		std::string base = output_file.substr(0, dot_pos);
		std::string ext = output_file.substr(dot_pos);
		encoder_output_file_[0] = base + "_cam1" + ext;
		encoder_output_file_[1] = base + "_cam2" + ext;
	}
	
	vpu_encoder_enabled_ = true;
	encoding_running_ = true;
	
	// Start encoding threads
	for (int i = 0; i < camera_num_; i++) {
		encoding_thread_[i] = std::thread(&DrmPreview::encodingWorker, this, i + 1);
	}
	
	std::cout << "VPU Encoder enabled: " << codec << " @ " << bitrate << " bps" << std::endl;
	std::cout << "Output file(s): " << encoder_output_file_[0];
	if (camera_num_ == 2) {
		std::cout << ", " << encoder_output_file_[1];
	}
	std::cout << std::endl;
}

void DrmPreview::encodingWorker(int camera_id)
{
    int idx = camera_id - 1;
    
    if (!vpu_encoder_[idx]) {
        vpu_encoder_[idx] = std::make_unique<imx95::VPUEncoder>();
        
        std::cout << "[EncodingWorker " << camera_id << "] Waiting for first frame..." << std::endl;
        
        std::unique_lock<std::mutex> lock(encoding_mutex_[idx]);
        encoding_cv_[idx].wait(lock, [this, idx] {
            return !encoding_queue_[idx].empty() || !encoding_running_;
        });
        
        if (!encoding_running_) {
            std::cout << "[EncodingWorker " << camera_id << "] Thread stopped (no init)" << std::endl;
            return;
        }
        
        auto& first_job = encoding_queue_[idx].front();
        lock.unlock();
        
        std::cout << "[EncodingWorker " << camera_id << "] Setting up encoder " 
                  << first_job.width << "x" << first_job.height << "..." << std::endl;
        
        // Use regular setup method (works with both mmap and direct encoding)
        if (!vpu_encoder_[idx]->setup(first_job.width, first_job.height, 
                                       encoder_codec_, encoder_bitrate_, 
                                       encoder_output_file_[idx])) {
            std::cerr << "Failed to setup VPU encoder for camera " << camera_id << std::endl;
            vpu_encoder_[idx].reset();
            encoding_running_ = false;
            return;
        }
        
        std::cout << "[EncodingWorker " << camera_id << "] Encoder initialized successfully" << std::endl;
    }
    
    int frame_processed = 0;
    
    while (encoding_running_) {
        EncodingJob job;
        bool has_job = false;
        
        {
            std::unique_lock<std::mutex> lock(encoding_mutex_[idx]);
            encoding_cv_[idx].wait(lock, [this, idx] {
                return !encoding_queue_[idx].empty() || !encoding_running_;
            });
            
            if (!encoding_running_) break;
            
            if (!encoding_queue_[idx].empty()) {
                job = encoding_queue_[idx].front();
                encoding_queue_[idx].pop_front();
                has_job = true;
            }
        }
        
        if (has_job && vpu_encoder_[idx] && vpu_encoder_[idx]->isInitialized()) {
            
            // Use DMA-BUF fds directly (mmap internally)
            vpu_encoder_[idx]->encodeFromDMABuf(
                job.y_fd, job.uv_fd,
                job.y_size, job.uv_size,
                job.stride,
                job.timestamp
            );

            frame_processed++;
        }
    }
    
    std::cout << "[EncodingWorker " << camera_id << "] Thread stopped, processed " 
              << frame_processed << " frames" << std::endl;
}


void DrmPreview::submitEncodingJob(int y_fd, int uv_fd, size_t y_size, size_t uv_size,
                                    int stride, int width, int height, 
                                    int camera_id, uint64_t timestamp)
{
    if (!vpu_encoder_enabled_ || !encoding_running_) {
        return;
    }
    
    int idx = camera_id - 1;
    
    bool dropped = false;
    {
        std::lock_guard<std::mutex> lock(encoding_mutex_[idx]);
        
        if (encoding_queue_[idx].size() >= MAX_ENCODING_QUEUE_SIZE) {
            static int drop_count[2] = {0, 0};
            if (++drop_count[idx] % 30 == 0) {
                std::cout << " Camera " << camera_id << " encoding too slow, dropped " 
                          << drop_count[idx] << " frames (queue full)" << std::endl;
            }
            dropped = true;
        } else {
            EncodingJob job;
            job.y_fd = y_fd;
            job.uv_fd = uv_fd;
            job.y_size = y_size;
            job.uv_size = uv_size;
            job.stride = stride;
            job.width = width;
            job.height = height;
            job.timestamp = timestamp;
            
            encoding_queue_[idx].push_back(job);
        }
    }
    
    if (!dropped) {
        encoding_cv_[idx].notify_one();
    }
}

void DrmPreview::disableGPUScaler()
{
	gpu_scaler_enabled_ = false;
	for (int i = 0; i < 2; i++) {
		gpu_scaler_[i].reset();
	}
	std::cout << "GPU Scaler disabled" << std::endl;
}

void DrmPreview::disableVPUEncoder()
{
	encoding_running_ = false;
	
	for (int i = 0; i < 2; i++) {
		encoding_cv_[i].notify_all();
		if (encoding_thread_[i].joinable()) {
			encoding_thread_[i].join();
		}
		if (vpu_encoder_[i]) {
			vpu_encoder_[i]->stop();
		}
		vpu_encoder_[i].reset();
	}
	
	vpu_encoder_enabled_ = false;
	std::cout << "VPU Encoder disabled" << std::endl;
}

/*allocateDRMBuffer used for 300*300 RGB image which is used for g2d output */
bool DrmPreview::allocateDRMBuffer(int index, size_t size)
{
    auto& buf = drm_buffers_[index];
    
    int heap_fd = open("/dev/dma_heap/linux,cma", O_RDWR);
    if (heap_fd < 0) {
        std::cerr << "Failed to open /dev/dma_heap/linux,cma: " << strerror(errno) << std::endl;
        return false;
    }
    
    struct dma_heap_allocation_data alloc_data = {};
    alloc_data.len = size;
    alloc_data.fd = 0;
    alloc_data.fd_flags = O_RDWR | O_CLOEXEC;
    alloc_data.heap_flags = 0;  // 0 = cached
    
    int ret = ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data);
    close(heap_fd);
    
    if (ret < 0) {
        std::cerr << "DMA_HEAP_IOCTL_ALLOC failed: " << strerror(errno) << std::endl;
        return false;
    }
    
    buf.dma_fd = alloc_data.fd;
    buf.size = size;
    
    // ===== cached mapping =====
    buf.virt_addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, 
                         buf.dma_fd, 0);
    
    if (buf.virt_addr == MAP_FAILED) {
        std::cerr << "Failed to mmap DMA-Heap buffer: " << strerror(errno) << std::endl;
        close(buf.dma_fd);
        buf.dma_fd = -1;
        return false;
    }
    
    std::cout << "[DrmPreview] Allocated DMA-Heap CMA buffer " << index << ":" << std::endl;
    std::cout << "  Size: " << buf.size << " bytes" << std::endl;
    std::cout << "  DMA-BUF fd: " << buf.dma_fd << std::endl;
    std::cout << "  Virt addr: " << buf.virt_addr << std::endl;
    std::cout << "  Type: CACHED (linux,cma)" << std::endl;
    
    return true;
}

void DrmPreview::freeDRMBuffer(int index)
{
    auto& buf = drm_buffers_[index];
    
    if (buf.virt_addr && buf.virt_addr != MAP_FAILED) {
        munmap(buf.virt_addr, buf.size);
        buf.virt_addr = nullptr;
    }
    
    if (buf.dma_fd >= 0) {
        close(buf.dma_fd);
        buf.dma_fd = -1;
    }
    
    buf.size = 0;
}

void DrmPreview::makeBuffer(libcamera::FrameBuffer* framebuffer,
                           size_t size, 
                           StreamInfo const &info, 
                           Buffer &buffer, 
                           uint32_t crtcId)
{
    if (first_time_)
    {
        first_time_ = false;
    }

    if (framebuffer->planes().size() < 2) {
        throw std::runtime_error("NV12 requires at least 2 planes, got " + 
                                std::to_string(framebuffer->planes().size()));
    }
    
    buffer.y_fd = framebuffer->planes()[0].fd.get();
    buffer.uv_fd = framebuffer->planes()[1].fd.get();
    buffer.size = size;
    buffer.info = info;

    if (drmPrimeFDToHandle(drmfd_, buffer.y_fd, &buffer.y_bo_handle))
        throw std::runtime_error("drmPrimeFDToHandle failed for Y plane fd " + 
                                std::to_string(buffer.y_fd));

    if (drmPrimeFDToHandle(drmfd_, buffer.uv_fd, &buffer.uv_bo_handle))
        throw std::runtime_error("drmPrimeFDToHandle failed for UV plane fd " + 
                                std::to_string(buffer.uv_fd));

    uint32_t offsets[4] = { 
        framebuffer->planes()[0].offset,
        framebuffer->planes()[1].offset,
        0, 
        0 
    };
    
    uint32_t pitches[4] = { 
        info.stride,     // Y plane pitch
        info.stride,     // UV plane pitch
        0, 
        0 
    };
    
    uint32_t bo_handles[4] = { 
        buffer.y_bo_handle,
        buffer.uv_bo_handle,
        0, 
        0 
    };

    uint64_t modifiers[4] = { 0, 0, 0, 0 };  // DRM_FORMAT_MOD_LINEAR
    
    int ret = drmModeAddFB2WithModifiers(drmfd_, info.width, info.height, out_fourcc_,
                                         bo_handles, pitches, offsets, modifiers, 
                                         &buffer.fb_handle, DRM_MODE_FB_MODIFIERS);
    
    if (ret) {
        std::cerr << "drmModeAddFB2WithModifiers failed, trying without modifiers..." << std::endl;
        
        ret = drmModeAddFB2(drmfd_, info.width, info.height, out_fourcc_,
                           bo_handles, pitches, offsets, &buffer.fb_handle, 0);
    }
    
    if (ret) {
        std::cerr << "drmModeAddFB2 failed with error: " << strerror(errno) << std::endl;
        std::cerr << "  Width: " << info.width << ", Height: " << info.height << std::endl;
        std::cerr << "  Format: NV12 (0x" << std::hex << out_fourcc_ << std::dec << ")" << std::endl;
        std::cerr << "  Y  BO Handle: " << buffer.y_bo_handle << " (fd=" << buffer.y_fd << ")" << std::endl;
        std::cerr << "  UV BO Handle: " << buffer.uv_bo_handle << " (fd=" << buffer.uv_fd << ")" << std::endl;
        
        throw std::runtime_error("drmModeAddFB2 failed: " + std::string(ERRSTR));
    }
    
}

void DrmPreview::ShowBuffer(libcamera::FrameBuffer* buffer, 
                           libcamera::Span<uint8_t> span, 
                           StreamInfo const &info)
{
    if (frame_cnt == 0) {
        std::cout << "\n=== First Frame Info (ShowBuffer) ===" << std::endl;
        std::cout << "  FrameBuffer planes: " << buffer->planes().size() << std::endl;
        for (size_t i = 0; i < buffer->planes().size(); i++) {
            std::cout << "    Plane " << i << ": fd=" << buffer->planes()[i].fd.get()
                      << ", offset=" << buffer->planes()[i].offset
                      << ", length=" << buffer->planes()[i].length << std::endl;
        }
        std::cout << "  Width: " << info.width << std::endl;
        std::cout << "  Height: " << info.height << std::endl;
        std::cout << "  Stride: " << info.stride << std::endl;
        std::cout << "  Pixel Format: " << info.pixel_format.toString() << std::endl;
        std::cout << "  Buffer Size: " << span.size() << std::endl;
        std::cout << "======================================\n" << std::endl;
    }

    Buffer &buf = buffers_[buffer];
    if (buf.y_fd == -1) {
        makeBuffer(buffer, span.size(), info, buf, crtcIdx_1);
    }
    last_buffer_ = buffer;
    

    // ===== GPU Scaler =====
    if (gpu_scaler_enabled_) {
        if (!gpu_scaler_[0]) {
            gpu_scaler_[0] = std::make_unique<imx95::GPUScaler>();
            gpu_scaler_[0]->setup(info.width, info.height, info.stride,
                                  scaler_target_width_, scaler_target_height_);
        }
        
        if (gpu_scaler_[0] && gpu_scaler_[0]->isInitialized()) {
            
            size_t y_size = info.stride * info.height;
            size_t uv_size = info.stride * info.height / 2;
            
            std::vector<uint8_t> scaled_data = gpu_scaler_[0]->scaleNV12(
                buf.y_fd, 
                buf.uv_fd,
                y_size, 
                uv_size
            );

            static int frame_counter = 0;
            static int saved_image_count = 0;
            
            frame_counter++;

            if (frame_counter % 60 == 0 && !scaled_data.empty()) {
                try {
                    int scaled_width = scaler_target_width_;
                    int scaled_height = scaler_target_height_;
                    
                    size_t expected_rgba = static_cast<size_t>(scaled_width * scaled_height * 4);
                    
                    if (scaled_data.size() == expected_rgba) {
                        cv::Mat rgba_mat(scaled_height, scaled_width, CV_8UC4, scaled_data.data());
                        cv::Mat bgr_mat;
                        cv::cvtColor(rgba_mat, bgr_mat, cv::COLOR_RGBA2BGR);

                        char filename[256];
                        snprintf(filename, sizeof(filename), "scaled_cam1_%04d.png", saved_image_count);

                        if (cv::imwrite(filename, bgr_mat)) {
                            std::cout << "[Cam1] Saved: " << filename << std::endl;
                            saved_image_count++;
                        }
                    } else {
                        std::cerr << "[Cam1] Unexpected size: " << scaled_data.size() 
                                  << " (expected " << expected_rgba << ")" << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error saving Camera 1 scaled image: " << e.what() << std::endl;
                }
            }
        }
    }

    // ===== VPU Encoder =====
    if (vpu_encoder_enabled_) {
        uint64_t timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        size_t y_size = info.stride * info.height;
        size_t uv_size = info.stride * info.height / 2;

        submitEncodingJob(buf.y_fd, buf.uv_fd, y_size, uv_size,
                          info.stride, info.width, info.height, 
                          1, timestamp);
    }

    // ===== Object Detection =====
    if (detector_type_ == DETECTOR_SSD_CPU || detector_type_ == DETECTOR_SSD_NPU) {
         const int DETECTION_INTERVAL = 1;
        if (frame_cnt % DETECTION_INTERVAL == 0) {
            submitDetectionJob(buf.y_fd, span, info, 1);
        }
        // ===== Draw Detections =====
        drawDetections(span, info, 1);
    }

    // ===== Dirty Detection (every 30 frames) =====
    if (detector_type_ == DETECTOR_DIRTY) {
        dirty_frame_counter_++;
        
        if (dirty_frame_counter_ % DIRTY_DETECTION_INTERVAL == 0) {
            size_t y_size = info.stride * info.height;
            submitDirtyDetectionJob(buf.y_fd, y_size, info, 1);
        }
        
        // Draw dirty detection results
        drawDirtyDetections(span, info, 1);
    }

    frames_processed_++;
    
    // ===== DRM Display =====
    auto [x, y, w, h] = calculatePosition(info, crtcIdx_1);

    drmModeAtomicReq *req = drmModeAtomicAlloc();

    try {
        addChangedProperties(req, 0, buf.fb_handle, x, y, w, h, info);

        const int max_retry = 5;
        int retry = 0;
        int commit_result = -1;
        
        while (retry < max_retry) {
            commit_result = drmModeAtomicCommit(drmfd_, req, DRM_MODE_ATOMIC_NONBLOCK, nullptr);
            if (commit_result == 0) {
                break;
            } else if (errno == EBUSY) {
                usleep(100 * (1 << retry));
                retry++;
            } else {
                printf("drmModeAtomicCommit failed: %s\n", std::string(ERRSTR).c_str());
                break;
            }
        }

        if (retry < max_retry) {
            frame_cnt++;
        } else {
            std::cout << " DRM commit failed after " << retry << " retries" << std::endl;
        }
    } catch (...) {
        drmModeAtomicFree(req);
        throw;
    }
    drmModeAtomicFree(req);
  
    done_callback_(buf.y_fd);
}


void DrmPreview::ShowBufferDual(libcamera::FrameBuffer* buffer1,
                               libcamera::Span<uint8_t> span1,
                               StreamInfo const &info1,
                               libcamera::FrameBuffer* buffer2,
                               libcamera::Span<uint8_t> span2,
                               StreamInfo const &info2)
{
    Buffer &buf1 = buffers_[buffer1];
    if (buf1.y_fd == -1) {
        makeBuffer(buffer1, span1.size(), info1, buf1, crtcIdx_1);
    }
    
    Buffer &buf2 = buffers_[buffer2];
    if (buf2.y_fd == -1) {
        makeBuffer(buffer2, span2.size(), info2, buf2, crtcIdx_2);
    }

    last_buffer_ = buffer1;
    last_buffer2_ = buffer2;

    // ===== Camera 1: GPU Scaler =====
    if (gpu_scaler_enabled_) {
        if (!gpu_scaler_[0]) {
            gpu_scaler_[0] = std::make_unique<imx95::GPUScaler>();
            gpu_scaler_[0]->setup(info1.width, info1.height, info1.stride,
                                  scaler_target_width_, scaler_target_height_);
        }
        
        if (gpu_scaler_[0] && gpu_scaler_[0]->isInitialized()) {
            size_t y_size = info1.stride * info1.height;
            size_t uv_size = info1.stride * info1.height / 2;
            
            std::vector<uint8_t> scaled_data1 = gpu_scaler_[0]->scaleNV12(
                buf1.y_fd, buf1.uv_fd, y_size, uv_size
            );
        }
    }

    // ===== Camera 2: GPU Scaler =====
    if (gpu_scaler_enabled_) {
        if (!gpu_scaler_[1]) {
            gpu_scaler_[1] = std::make_unique<imx95::GPUScaler>();
            gpu_scaler_[1]->setup(info2.width, info2.height, info2.stride,
                                  scaler_target_width_, scaler_target_height_);
        }
        
        if (gpu_scaler_[1] && gpu_scaler_[1]->isInitialized()) {
            size_t y_size = info2.stride * info2.height;
            size_t uv_size = info2.stride * info2.height / 2;
            std::vector<uint8_t> scaled_data2 = gpu_scaler_[1]->scaleNV12(
                buf2.y_fd, buf2.uv_fd, y_size, uv_size
            );
        }
    }

    // ===== VPU Encoder =====
    if (vpu_encoder_enabled_) {
        uint64_t timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        
        size_t y_size1 = info1.stride * info1.height;
        size_t uv_size1 = info1.stride * info1.height / 2;
        submitEncodingJob(buf1.y_fd, buf1.uv_fd, y_size1, uv_size1,
                          info1.stride, info1.width, info1.height, 
                          1, timestamp);
        
        size_t y_size2 = info2.stride * info2.height;
        size_t uv_size2 = info2.stride * info2.height / 2;
        submitEncodingJob(buf2.y_fd, buf2.uv_fd, y_size2, uv_size2,
                          info2.stride, info2.width, info2.height, 
                          2, timestamp);
    }

    // ===== Object Detection =====
    if (detector_type_ == DETECTOR_SSD_CPU || detector_type_ == DETECTOR_SSD_NPU) {
        submitDetectionJobDual(buf1.y_fd, span1, info1, buf2.y_fd, span2, info2);

        // ===== Draw Detections =====
        drawDetections(span1, info1, 1);
        drawDetections(span2, info2, 2);
    }

    // ===== Dirty Detection (every 30 frames) =====
    if (detector_type_ == DETECTOR_DIRTY) {
        dirty_frame_counter_++;
        
        if (dirty_frame_counter_ % DIRTY_DETECTION_INTERVAL == 0) {
            size_t y_size1 = info1.stride * info1.height;
            
            submitDirtyDetectionJob(buf1.y_fd, y_size1, info1, 1);
        }
 
    if (dirty_frame_counter_ % DIRTY_DETECTION_INTERVAL == (DIRTY_DETECTION_INTERVAL / 2)) {
            size_t y_size2 = info2.stride * info2.height;
            
            submitDirtyDetectionJob(buf2.y_fd, y_size2, info2, 2);
        }
        // Draw dirty detection results
        drawDirtyDetections(span1, info1, 1);
        drawDirtyDetections(span2, info2, 2);
    }

    frames_processed_++;

    // ===== DRM Display =====
    auto [x1, y1, w1, h1] = calculatePosition(info1, crtcIdx_1);
    auto [x2, y2, w2, h2] = calculatePosition(info2, crtcIdx_2);
    
    drmModeAtomicReq *req = drmModeAtomicAlloc();
    try {
        addChangedProperties(req, 0, buf1.fb_handle, x1, y1, w1, h1, info1);
        addChangedProperties(req, 1, buf2.fb_handle, x2, y2, w2, h2, info2);

        const int max_retry = 5;
        int retry = 0;
        while (retry < max_retry) {
            int ret = drmModeAtomicCommit(drmfd_, req, DRM_MODE_ATOMIC_NONBLOCK, nullptr);
            if (ret == 0) {
                break;
            } else if (errno == EBUSY) {
                usleep(100 * (1 << retry));
                retry++;
            } else {
                printf("drmModeAtomicCommit err: %s\n", ERRSTR);
                break;
            }
        }
        if (retry < max_retry) {
            frame_cnt++;
        }
    } catch (...) {
        drmModeAtomicFree(req);
        throw;
    }
    
    drmModeAtomicFree(req);

    done_callback_dual_(buf1.y_fd, buf2.y_fd);
}


//Deprecated
void DrmPreview::Show(int fd, libcamera::Span<uint8_t> span, StreamInfo const &info)
{
    std::cerr << "Warning: Show(int fd) is deprecated, use ShowBuffer(FrameBuffer*) instead" << std::endl;
    throw std::runtime_error("Show(int fd) cannot handle NV12 with separate planes");
}

void DrmPreview::ShowDual(int fd1, libcamera::Span<uint8_t> span, StreamInfo const &info, 
                          int fd2, libcamera::Span<uint8_t> span2, StreamInfo const &info2)
{
    std::cerr << "Warning: ShowDual(int fd) is deprecated, use ShowBufferDual(FrameBuffer*) instead" << std::endl;
    throw std::runtime_error("ShowDual(int fd) cannot handle NV12 with separate planes");
}



void DrmPreview::addChangedProperties(drmModeAtomicReq *req, int index, 
									 uint32_t new_fb_id, int x, int y, int w, int h, const StreamInfo &info) {
	auto &cache = plane_cache_[index];
	auto &props = plane_props_[index];
	uint32_t plane_id = planeId_[index];
	uint32_t crtc_id_ = crtcId_[index];

	crtc_x[index] = x;
	crtc_y[index] = y;
	crtc_w[index] = w;
	crtc_h[index] = h;

	if(frame_cnt < 2){
		src_w[index] = info.width;
		src_h[index] = info.height;
	}
	
	if (new_fb_id != cache.fb_id) {
		drmModeAtomicAddProperty(req, plane_id, props.fb_id, new_fb_id);
		cache.fb_id = new_fb_id;
	}

	if (cache.crtc_id != crtc_id_) {
		drmModeAtomicAddProperty(req, plane_id, props.crtc_id, crtc_id_);
		cache.crtc_id = crtc_id_;
	}

	if (cache.crtc_x != crtc_x[index]) {
		drmModeAtomicAddProperty(req, plane_id, props.crtc_x, crtc_x[index]);
		cache.crtc_x = crtc_x[index];
	}
	if (cache.crtc_y != crtc_y[index]) {
		drmModeAtomicAddProperty(req, plane_id, props.crtc_y, crtc_y[index]);
		cache.crtc_y = crtc_y[index];
	}
	if (cache.crtc_w != crtc_w[index]) {
		drmModeAtomicAddProperty(req, plane_id, props.crtc_w, crtc_w[index]);
		cache.crtc_w = crtc_w[index];
	}
	if (cache.crtc_h != crtc_h[index]) {
		drmModeAtomicAddProperty(req, plane_id, props.crtc_h, crtc_h[index]);
		cache.crtc_h = crtc_h[index];
	}

	if (cache.src_x != src_x[index]) {
		drmModeAtomicAddProperty(req, plane_id, props.src_x, src_x[index]);
		cache.src_x = src_x[index];
	}
	if (cache.src_y != src_y[index]) {
		drmModeAtomicAddProperty(req, plane_id, props.src_y, src_y[index]);
		cache.src_y = src_y[index];
	}
	if (cache.src_w != src_w[index]) {
		drmModeAtomicAddProperty(req, plane_id, props.src_w, src_w[index] << 16);
		cache.src_w = src_w[index];
	}
	if (cache.src_h != src_h[index]) {
		drmModeAtomicAddProperty(req, plane_id, props.src_h, src_h[index] << 16);
		cache.src_h = src_h[index];
	}
}

std::tuple<int, int, int, int> DrmPreview::calculatePosition(const StreamInfo &info, int crtcIdx) {
	unsigned int x_off = 0, y_off = 0;
	unsigned int w = width_, h = height_;

	if (info.width * h > w * info.height) {
		h = w * info.height / info.width;
		y_off = (height_ - h) / 2;
	} else {
		w = h * info.width / info.height;
		x_off = (width_ - w) / 2;
	}

	return {x_off + x_, y_off + y_, w, h};
}

int DrmPreview::acquireBufferIndex()
{
	for (int i = 0; i < RING_BUFFER_SIZE; i++) {
		int index = (write_index_ + i) % RING_BUFFER_SIZE;
		bool expected = false;
		
		if (ring_buffers_[index].in_use.compare_exchange_strong(expected, true)) {
			write_index_ = (index + 1) % RING_BUFFER_SIZE;
			return index;
		}
	}
	
	detections_skipped_++;
	return -1;
}

void DrmPreview::releaseBuffer(int index)
{
	if (index >= 0 && index < RING_BUFFER_SIZE) {
        ring_buffers_[index].in_use = false;
	}
}

void DrmPreview::copyWorker() 
{
    std::cout << "[CopyWorker] Thread started (G2D ZERO-COPY mode with g2d_buf_from_fd)" << std::endl;

    while (detection_running_) {
        CopyJob job;
        job.valid = false;
        
        {
            std::unique_lock<std::mutex> lock(copy_mutex_);
            copy_cv_.wait(lock, [this] { 
                return !copy_queue_.empty() || !detection_running_; 
            });
            
            if (!detection_running_) break;
            
            if (!copy_queue_.empty()) {
                job = copy_queue_.front();
                copy_queue_.pop_front();
                job.valid = true;
            }
        }
        
        if (job.valid && job.buffer_index >= 0) {
            auto start = std::chrono::high_resolution_clock::now();
            
            auto& buffer = ring_buffers_[job.buffer_index];
            auto& drm_buf = drm_buffers_[job.buffer_index]; 
            buffer.original_info = job.info;
            buffer.camera_id = job.camera_id;
            buffer.g2d_buffer_id = job.buffer_index;

            bool success = g2d_scaler_->scaleAndConvertNV12ToRGB_DMABuf(
                job.y_fd,
                job.uv_fd,
                job.y_size,
                job.uv_size,
                drm_buf.dma_fd,      // DRM DMA-BUF fd
                drm_buf.size,
                job.camera_id
            );
            
            if (!success) {
                std::cerr << "[CopyWorker] G2D conversion failed for buffer " 
                          << job.buffer_index << ", camera " << job.camera_id << std::endl;
                releaseBuffer(job.buffer_index);
                continue;
            }

            // ===== Store pointer to G2D output =====
            buffer.rgb_data_ptr = static_cast<const uint8_t*>(drm_buf.virt_addr);
            buffer.resized_width = TARGET_WIDTH;
            buffer.resized_height = TARGET_HEIGHT;
            buffer.info = job.info;
            buffer.resize_completed = true;

            auto end = std::chrono::high_resolution_clock::now();
            auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            // ===== Performance statistics =====
            static int copy_count = 0;
            static uint64_t total_g2d_us = 0;
            total_g2d_us += total_time.count();
            
            if (++copy_count % 300 == 0) {
                std::cout << "========================================================" << std::endl;
                std::cout << "  CopyWorker Performance (G2D ZERO-COPY, avg 300 frames) " << std::endl;
                std::cout << "========================================================" << std::endl;
                printf("  Total time:      %6.2f ms                            \n", 
                       total_g2d_us / 300000.0);
                printf("  Resolution:      %dx%d -> %dx%d              \n",
                       job.info.width, job.info.height, TARGET_WIDTH, TARGET_HEIGHT);
                printf("  Format:          NV12 -> RGB888                       \n");
                printf("  Method:          ZERO-COPY (g2d_buf_from_fd)          \n");
                std::cout << "========================================================" << std::endl;
                total_g2d_us = 0;
            }

            // ===== Submit to detection queue =====
            if (camera_num_ == 1) {
                std::lock_guard<std::mutex> lock(job_mutex_);
                DetectionJob det_job;
                det_job.buffer_index = job.buffer_index;
                det_job.camera_id = job.camera_id;
                det_job.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
                det_job.valid = true;
                job_queue_.push_back(det_job);

                job_cv_.notify_one();
            }
            else if (camera_num_ == 2) {
                std::lock_guard<std::mutex> lock(job_mutex_);
                bool should_notify = false;
                
                for (auto& dual_job : job_queue_dual_) {
                    if (dual_job.buffer_index1 == job.buffer_index) {
                        auto& buffer2 = ring_buffers_[dual_job.buffer_index2];

                        if (buffer2.resize_completed) {
                            should_notify = true;
                        }
                        break;
                    } 
                    else if (dual_job.buffer_index2 == job.buffer_index) {
                        auto& buffer1 = ring_buffers_[dual_job.buffer_index1];

                        if (buffer1.resize_completed) {
                            should_notify = true;
                        }
                        break;
                    }
                }

                if (should_notify) {
                    job_cv_.notify_one();
                }
            }
        }
    }

    std::cout << "[CopyWorker] Thread stopped" << std::endl;
}


void DrmPreview::submitDetectionJob(int fd, libcamera::Span<uint8_t> span, 
									const StreamInfo &info, int camera_id)
{
	// First check if the queue is full.
	{
		std::lock_guard<std::mutex> lock(job_mutex_);
		if (job_queue_.size() >= MAX_QUEUE_SIZE) {
			queue_full_count_++;
			return;  // If the queue is full, skip directly.
		}
	}
	
	int buffer_index = acquireBufferIndex();
	if (buffer_index < 0) {
		return;  // No buffer available
	}

    libcamera::FrameBuffer* current_fb = (camera_id == 1) ? last_buffer_ : last_buffer2_;
    
    if (!current_fb) {
        std::cerr << "[submitDetectionJob] No current FrameBuffer" << std::endl;
        releaseBuffer(buffer_index);
        return;
    }

    auto it = buffers_.find(current_fb);
    if (it == buffers_.end()) {
        std::cerr << "[submitDetectionJob] FrameBuffer not found in buffers_ map" << std::endl;
        releaseBuffer(buffer_index);
        return;
    }

    int y_fd = it->second.y_fd;
    int uv_fd = it->second.uv_fd;
    if (y_fd < 0 || uv_fd < 0) {
        std::cerr << "[submitDetectionJob] Invalid DMA-BUF fds: y_fd=" << y_fd 
                  << ", uv_fd=" << uv_fd << std::endl;
        releaseBuffer(buffer_index);
        return;
    }

    CopyJob job;
    job.y_fd = y_fd;
    job.uv_fd = uv_fd;
    job.y_size = info.stride * info.height;
    job.uv_size = info.stride * info.height / 2;
    job.info = info;
    job.camera_id = camera_id;
    job.buffer_index = buffer_index;
    job.valid = true;
	{
		std::lock_guard<std::mutex> lock(copy_mutex_);
		copy_queue_.push_back(job);
	}

	copy_cv_.notify_one();
}

// Perform the test and return the results.
DetectionResult DrmPreview::processImage(const uint8_t* image_data, int width, int height, int camera_id)
{
	DetectionResult result;
	result.camera_id = camera_id;
	result.valid = false;
	
	try {
		if (detector_type_ == DETECTOR_SSD_CPU) {
			result.detections = ssd_detector_cpu_->detect(image_data, width, height);
		} else {
			result.detections = ssd_detector_npu_->detect(image_data, width, height);
		}

		result.valid = true;
		
	} catch (const std::exception& e) {
		std::cerr << "Detection error for camera " << camera_id << ": " << e.what() << std::endl;
	}
	
	return result;
}

void DrmPreview::detectionWorker()
{
    std::cout << "Detection worker thread started" << std::endl;
    
    while (detection_running_) {
        DetectionJob job;
        bool has_job = false;
        
        {
            std::unique_lock<std::mutex> lock(job_mutex_);

            job_cv_.wait(lock, [this] { 
                return !job_queue_.empty() || !detection_running_; 
            });

            if (!detection_running_) break;
            
            if (!job_queue_.empty()) {
                job = job_queue_.front();
                job_queue_.pop_front();
                has_job = true;
            }
        }
        
        if (has_job && job.valid && job.buffer_index >= 0) {
            try {
                auto start = std::chrono::high_resolution_clock::now();
                
                auto& buffer = ring_buffers_[job.buffer_index];

                if (!buffer.rgb_data_ptr) {
                    std::cerr << "[DetectionWorker] ERROR: Invalid rgb_data_ptr for buffer " 
                              << job.buffer_index << std::endl;
                    releaseBuffer(job.buffer_index);
                    continue;
                }
                
                if (!buffer.resize_completed) {
                    std::cerr << "[DetectionWorker] ERROR: Resize not completed for buffer " 
                              << job.buffer_index << std::endl;
                    releaseBuffer(job.buffer_index);
                    continue;
                }
                
                DetectionResult result = processImage(buffer.rgb_data_ptr,
                                                      buffer.resized_width, 
                                                      buffer.resized_height, 
                                                      job.camera_id);
                
                result.timestamp = job.timestamp;
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                {
                    std::lock_guard<std::mutex> lock(result_mutex_);
                    latest_results_[job.camera_id] = result;
                }
                
                releaseBuffer(job.buffer_index);
                
                detections_completed_++;
                
                // Print every 300 times
                if (detections_completed_ % 300 == 0) {
                    std::cout << "========================================================" << std::endl;
                    std::cout << "  Detection Results (Camera " << job.camera_id << ")        " << std::endl;
                    std::cout << "========================================================" << std::endl;
                    std::cout << "  Detection time: " << duration.count() << " ms" << std::endl;
                    std::cout << "  Objects found:  " << result.detections.size() << std::endl;
                    std::cout << "  Queue size:     " << job_queue_.size() << std::endl;
                    
                    if (!result.detections.empty()) {
                        std::cout << "  Detected objects:" << std::endl;
                        for (size_t i = 0; i < std::min(result.detections.size(), size_t(3)); i++) {
                            const auto& det = result.detections[i];
                            std::cout << "    " << i+1 << ". " << det.class_name 
                                      << " (" << (det.score * 100) << "%)" << std::endl;
                        }
                    }
                    std::cout << "========================================================" << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Detection error: " << e.what() << std::endl;
                releaseBuffer(job.buffer_index);
            }
        }
    }
    
    std::cout << "Detection worker thread stopped" << std::endl;
}



void DrmPreview::drawDetections(libcamera::Span<uint8_t> span, const StreamInfo &info, int camera_id)
{
    std::lock_guard<std::mutex> lock(result_mutex_);
    
    auto it = latest_results_.find(camera_id);

    if (it == latest_results_.end() || !it->second.valid) {
        return;
    }
    
    const auto& result = it->second;

    imx95::DrawConfig config;
    
    if (camera_id == 1) {
        config.box_color = 0xFF00FF00;
        config.text_color = 0xFFFFFFFF;
        config.bg_color = 0xC0006600;
    } else {
        config.box_color = 0xFFFF0000;
        config.text_color = 0xFFFFFFFF;
        config.bg_color = 0xC0660000;
    }
    
    config.box_thickness = 10;
    config.draw_label = true;
    config.draw_confidence = true;
    
    imx95::ImageInfo image_info;
    image_info.data = span.data();
    image_info.width = info.width;
    image_info.height = info.height;
    image_info.stride = info.stride;
    image_info.pixel_format = 1;  // NV12
    
    if (detector_type_ == DETECTOR_SSD_CPU) {
        ssd_detector_cpu_->drawDetections(result.detections, image_info, config);
    } else {
        ssd_detector_npu_->drawDetections(result.detections, image_info, config);
    }
}


void DrmPreview::submitDetectionJobDual(int fd1, libcamera::Span<uint8_t> span1, const StreamInfo &info1,
                                       int fd2, libcamera::Span<uint8_t> span2, const StreamInfo &info2)
{
    {
        std::lock_guard<std::mutex> lock(job_mutex_);
        if (job_queue_dual_.size() >= MAX_QUEUE_SIZE) {
            queue_full_count_++;
            return;
        }
    }
    
    int buffer_index1 = acquireBufferIndex();
    if (buffer_index1 < 0) {
        return;
    }
    
    int buffer_index2 = acquireBufferIndex();
    if (buffer_index2 < 0) {
        releaseBuffer(buffer_index1);
        return;
    }
    
    auto it1 = buffers_.find(last_buffer_);
    auto it2 = buffers_.find(last_buffer2_);
    
    if (it1 == buffers_.end() || it2 == buffers_.end()) {
        std::cerr << " FrameBuffer not found in buffers_ map" << std::endl;
        releaseBuffer(buffer_index1);
        releaseBuffer(buffer_index2);
        return;
    }
    
    CopyJob job1;
    job1.y_fd = it1->second.y_fd;
    job1.uv_fd = it1->second.uv_fd;
    job1.y_size = info1.stride * info1.height;
    job1.uv_size = info1.stride * info1.height / 2;
    job1.info = info1;
    job1.camera_id = 1;
    job1.buffer_index = buffer_index1;
    job1.valid = true;
    
    CopyJob job2;
    job2.y_fd = it2->second.y_fd;
    job2.uv_fd = it2->second.uv_fd;
    job2.y_size = info2.stride * info2.height;
    job2.uv_size = info2.stride * info2.height / 2;
    job2.info = info2;
    job2.camera_id = 2;
    job2.buffer_index = buffer_index2;
    job2.valid = true;

    {
        std::lock_guard<std::mutex> lock(copy_mutex_);
        copy_queue_.push_back(job1);
        copy_queue_.push_back(job2);
    }
    copy_cv_.notify_all();
    
    DetectionJobDual dual_job;
    dual_job.buffer_index1 = buffer_index1;
    dual_job.buffer_index2 = buffer_index2;
    dual_job.camera_id1 = 1;
    dual_job.camera_id2 = 2;
    dual_job.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    dual_job.valid = true;
    
    {
        std::lock_guard<std::mutex> lock(job_mutex_);
        job_queue_dual_.push_back(dual_job);
    }
}

void DrmPreview::detectionWorkerDual()
{
    std::cout << "Detection worker thread (dual) started" << std::endl;
    while (detection_running_) {
        DetectionJobDual job;
        bool has_job = false;
        
        {
            std::unique_lock<std::mutex> lock(job_mutex_);
            job_cv_.wait(lock, [this] { 
                return !job_queue_dual_.empty() || !detection_running_; 
            });
            
            if (!detection_running_) break;
            
            if (!job_queue_dual_.empty()) {
                job = job_queue_dual_.front();
                job_queue_dual_.pop_front();
                has_job = true;
            }
        }
        
        if (has_job && job.valid && job.buffer_index1 >= 0 && job.buffer_index2 >= 0) {
            try {
                auto total_start = std::chrono::high_resolution_clock::now();
                
                auto& buffer1 = ring_buffers_[job.buffer_index1];
                auto& buffer2 = ring_buffers_[job.buffer_index2];

                while (!buffer1.resize_completed || !buffer2.resize_completed) {
                    if (!detection_running_) break;
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }

                if (!detection_running_) {
                    std::cout << "[DetectionWorkerDual] Detection stopped, releasing buffers" << std::endl;
                    releaseBuffer(job.buffer_index1);
                    releaseBuffer(job.buffer_index2);
                    return;
                }

                // Validate G2D output pointers
                if (!buffer1.rgb_data_ptr || buffer1.resized_width <= 0 || buffer1.resized_height <= 0) {
                    std::cerr << "[DetectionWorkerDual] Invalid resized data for camera 1:" << std::endl;
                    releaseBuffer(job.buffer_index1);
                    releaseBuffer(job.buffer_index2);
                    return;
                }
                
                if (!buffer2.rgb_data_ptr || buffer2.resized_width <= 0 || buffer2.resized_height <= 0) {
                    std::cerr << "[DetectionWorkerDual] Invalid resized data for camera 2:" << std::endl;
                    releaseBuffer(job.buffer_index1);
                    releaseBuffer(job.buffer_index2);
                    return;
                }

                // ===== Extract data before releasing buffers =====
                const uint8_t* data1_ptr = buffer1.rgb_data_ptr;
                const uint8_t* data2_ptr = buffer2.rgb_data_ptr;
                int width1 = buffer1.resized_width;
                int height1 = buffer1.resized_height;
                int width2 = buffer2.resized_width;
                int height2 = buffer2.resized_height;

                // Release buffers immediately after copying pointers
                releaseBuffer(job.buffer_index1);
                releaseBuffer(job.buffer_index2);
                
                DetectionResult result1, result2;

                if (detector_type_ == DETECTOR_SSD_NPU) {
                    // ===== NPU parallel detection (runs in separate thread) =====
                    std::thread npu_thread([this, job, data1_ptr, data2_ptr, width1, height1, width2, height2]() {
                        try {
                            auto detect1_start = std::chrono::high_resolution_clock::now();
                            DetectionResult result1 = processImage(
                                data1_ptr,
                                width1,
                                height1,
                                job.camera_id1
                            );
                            result1.timestamp = job.timestamp;

                            auto detect1_end = std::chrono::high_resolution_clock::now();
                            auto detect1_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                detect1_end - detect1_start);

                            auto detect2_start = std::chrono::high_resolution_clock::now();
                            DetectionResult result2 = processImage(
                                data2_ptr,
                                width2,
                                height2,
                                job.camera_id2
                            );
                            result2.timestamp = job.timestamp;

                            auto detect2_end = std::chrono::high_resolution_clock::now();
                            auto detect2_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                detect2_end - detect2_start);
                            
                            auto total_end = std::chrono::high_resolution_clock::now();
                            auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                                total_end - detect1_start);
                            {
                                std::lock_guard<std::mutex> lock(result_mutex_);
                                latest_results_[job.camera_id1] = result1;
                                latest_results_[job.camera_id2] = result2;
                            }
                            
                            detections_completed_ += 2;
                            
                            static int npu_count = 0;
                            static double sum_cam1 = 0, sum_cam2 = 0, sum_total = 0;
                            static std::mutex stats_mutex;
                            
                            {
                                std::lock_guard<std::mutex> lock(stats_mutex);
                                sum_cam1 += detect1_duration.count();
                                sum_cam2 += detect2_duration.count();
                                sum_total += total_duration.count();
                                
                                if (++npu_count % 300 == 0) {
                                    std::cout << "========================================================" << std::endl;
                                    std::cout << "  NPU Detection Performance (avg 300 frames, parallel)  " << std::endl;
                                    std::cout << "========================================================" << std::endl;
                                    printf("  Camera 1 NPU:  %6.2f ms  (%2zu objects)               \n", 
                                           sum_cam1 / 300.0, result1.detections.size());
                                    printf("  Camera 2 NPU:  %6.2f ms  (%2zu objects)               \n", 
                                           sum_cam2 / 300.0, result2.detections.size());
                                    printf("  Total (serial):%6.2f ms                              \n", 
                                           sum_total / 300.0);
                                    std::cout << "========================================================" << std::endl;
                                    printf("      Note: NPU runs in separate thread (parallel)       \n");
                                    std::cout << "========================================================" << std::endl;
                                    
                                    sum_cam1 = sum_cam2 = sum_total = 0;
                                }
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "[NPU Thread] Exception: " << e.what() << std::endl;
                        }
                    });
                    
                    npu_thread.detach();
                } else {
                    // ===== CPU serial detection =====
                    
                    auto detect1_start = std::chrono::high_resolution_clock::now();
                    result1 = processImage(data1_ptr, width1, height1, job.camera_id1);
                    result1.timestamp = job.timestamp;
                    auto detect1_end = std::chrono::high_resolution_clock::now();
                    auto detect1_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        detect1_end - detect1_start);
  
                    auto detect2_start = std::chrono::high_resolution_clock::now();
                    result2 = processImage(data2_ptr, width2, height2, job.camera_id2);
                    result2.timestamp = job.timestamp;
                    auto detect2_end = std::chrono::high_resolution_clock::now();
                    auto detect2_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        detect2_end - detect2_start);

                    auto total_end = std::chrono::high_resolution_clock::now();
                    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        total_end - total_start);
                    
                    {
                        std::lock_guard<std::mutex> lock(result_mutex_);
                        latest_results_[job.camera_id1] = result1;
                        latest_results_[job.camera_id2] = result2;
                    }
                    
                    detections_completed_ += 2;
                    
                    static int cpu_count = 0;
                    static double sum_cam1 = 0, sum_cam2 = 0, sum_total = 0;
                    static std::mutex stats_mutex;
                    
                    {
                        std::lock_guard<std::mutex> lock(stats_mutex);
                        sum_cam1 += detect1_duration.count();
                        sum_cam2 += detect2_duration.count();
                        sum_total += total_duration.count();
                        
                        if (++cpu_count % 300 == 0) {
                            std::cout << "========================================================" << std::endl;
                            std::cout << "  CPU Detection Performance (avg 300 frames, serial)    " << std::endl;
                            std::cout << "========================================================" << std::endl;
                            printf("  Camera 1 CPU:  %6.2f ms  (%2zu objects)               \n", 
                                   sum_cam1 / 300.0, result1.detections.size());
                            printf("  Camera 2 CPU:  %6.2f ms  (%2zu objects)               \n", 
                                   sum_cam2 / 300.0, result2.detections.size());
                            printf("  Total (serial):%6.2f ms                              \n", 
                                   sum_total / 300.0);
                            std::cout << "========================================================" << std::endl;
                            printf("      Note: CPU runs serially (avoid TFLite conflict)    \n");
                            std::cout << "========================================================" << std::endl;
                            
                            sum_cam1 = sum_cam2 = sum_total = 0;
                        }
                    }
                }
                
            } catch (const std::exception& e) {
                std::cerr << "[DetectionWorkerDual] Exception: " << e.what() << std::endl;
                releaseBuffer(job.buffer_index1);
                releaseBuffer(job.buffer_index2);
            }
        }
    }
    
    std::cout << "[DetectionWorkerDual] Thread stopped" << std::endl;
}


int DrmPreview::acquireDirtyCopyBuffer()
{
    for (int i = 0; i < DIRTY_COPY_BUFFER_SIZE; i++) {
        int index = (dirty_write_index_ + i) % DIRTY_COPY_BUFFER_SIZE;
        bool expected = false;
        
        if (dirty_copy_buffers_[index].in_use.compare_exchange_strong(expected, true)) {
            dirty_write_index_ = (index + 1) % DIRTY_COPY_BUFFER_SIZE;
            return index;
        }
    }
    return -1;  // All buffers busy
}

void DrmPreview::releaseDirtyCopyBuffer(int index)
{
    if (index >= 0 && index < DIRTY_COPY_BUFFER_SIZE) {
        dirty_copy_buffers_[index].in_use = false;
        dirty_copy_buffers_[index].copy_completed = false;
    }
}

void DrmPreview::dirtyCopyWorker()
{
    std::cout << "[DirtyCopyWorker] Thread started (NEON-optimized memcpy)" << std::endl;
    
    while (detection_running_) {
        DirtyCopyJob job;
        bool has_job = false;
        
        {
            std::unique_lock<std::mutex> lock(dirty_copy_mutex_);
            dirty_copy_cv_.wait(lock, [this] { 
                return !dirty_copy_queue_.empty() || !detection_running_; 
            });
            
            if (!detection_running_) {
                std::cout << "[DirtyCopyWorker] Received stop signal" << std::endl;
                break;
            }
            
            if (!dirty_copy_queue_.empty()) {
                job = dirty_copy_queue_.front();
                dirty_copy_queue_.pop_front();
                has_job = true;
            }
        }
        
        if (!has_job) {
            std::cout << " [DirtyCopyWorker] Woke up but no job available" << std::endl;
            continue;
        }
        
        static int copy_count = 0;
        if (++copy_count % 1 == 0) {
            std::cout << "[DirtyCopyWorker] Processing job " << copy_count 
                      << ", buffer_index=" << job.buffer_index << std::endl;
        }
        
        if (job.valid && job.buffer_index >= 0) {
            auto start = std::chrono::high_resolution_clock::now();
            
            auto& buffer = dirty_copy_buffers_[job.buffer_index];
            
            // ===== Step 1: mmap DMA-BUF =====
            void* y_mapped = mmap(nullptr, job.y_size, PROT_READ, MAP_SHARED, job.y_fd, 0);
            
            if (y_mapped == MAP_FAILED) {
                std::cerr << " [DirtyCopy] mmap failed: " << strerror(errno) << std::endl;
                releaseDirtyCopyBuffer(job.buffer_index);
                continue;
            }
            
            // ===== Step 2: Fast memcpy (NEON-optimized) =====
            const uint8_t* src = static_cast<const uint8_t*>(y_mapped);
            uint8_t* dst = buffer.y_data.data();
            
            // Use NEON for fast copy (64 bytes per iteration)
            size_t copy_size = job.y_size;
            size_t neon_size = copy_size & ~63;  // Align to 64 bytes
            
            for (size_t i = 0; i < neon_size; i += 64) {
                uint8x16x4_t data = vld1q_u8_x4(src + i);
                vst1q_u8_x4(dst + i, data);
            }
            
            // Copy remaining bytes
            if (neon_size < copy_size) {
                std::memcpy(dst + neon_size, src + neon_size, copy_size - neon_size);
            }
            
            // ===== Step 3: Unmap immediately =====
            munmap(y_mapped, job.y_size);
            
            // ===== Step 4: Update buffer metadata =====
            buffer.width = job.width;
            buffer.height = job.height;
            buffer.stride = job.stride;
            buffer.camera_id = job.camera_id;
            buffer.timestamp = job.timestamp;
            buffer.copy_completed = true;
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            if (copy_count % 60 == 0) {
                std::cout << "[DirtyCopyWorker] Copy completed in " 
                          << duration.count() / 1000.0 << " ms" << std::endl;
            }

            // ===== Submit to detection queue =====
            {
                std::lock_guard<std::mutex> lock(dirty_job_mutex_);

                DirtyDetectionJob det_job;
                det_job.y_fd = -1;  // Use copied data, not DMA-BUF
                det_job.y_size = job.y_size;
                det_job.width = job.width;
                det_job.height = job.height;
                det_job.stride = job.stride;
                det_job.camera_id = job.camera_id;
                det_job.timestamp = job.timestamp;
                det_job.valid = true;
                det_job.buffer_index = job.buffer_index;
                
                dirty_job_queue_.push_back(det_job);

                if (copy_count % 60 == 0) {
                    std::cout << "[DirtyCopyWorker] Submitted detection job, queue size=" 
                              << dirty_job_queue_.size() << std::endl;
                }
            }
            dirty_job_cv_.notify_one();
        }
    }
    
    std::cout << "[DirtyCopyWorker] Thread stopped" << std::endl;
}


void DrmPreview::dirtyDetectionWorker()
{
    std::cout << "[DirtyDetectionWorker] Thread started (Using copied data)" << std::endl;
    
    while (detection_running_) {
        DirtyDetectionJob job;
        bool has_job = false;
        
        {
            std::unique_lock<std::mutex> lock(dirty_job_mutex_);
            
            dirty_job_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] { 
                return !dirty_job_queue_.empty() || !detection_running_; 
            });
            
            if (!detection_running_) {
                std::cout << "[DirtyDetectionWorker] Received stop signal" << std::endl;
                break;
            }
            
            if (!dirty_job_queue_.empty()) {
                job = dirty_job_queue_.front();
                dirty_job_queue_.pop_front();
                has_job = true;
            }
        }

        static int detect_count = 0;
        if (has_job) {
            if (++detect_count % 60 == 0) {
                std::cout << "[DirtyDetectionWorker] Processing job " << detect_count 
                          << ", buffer_index=" << job.buffer_index << std::endl;
            }
        }

        if (has_job && job.valid && job.buffer_index >= 0) {
            try {
                auto& buffer = dirty_copy_buffers_[job.buffer_index];
                
                if (!buffer.copy_completed) {
                    std::cerr << " [DirtyDetectionWorker] Copy not completed (should not happen!)" << std::endl;
                    releaseDirtyCopyBuffer(job.buffer_index);
                    continue;
                }
                
                // ===== Detect from copied memory (no DMA-BUF access) =====
                auto result = dirty_detector_->detectFromNV12(
                    buffer.y_data.data(),
                    buffer.width,
                    buffer.height,
                    buffer.stride
                );
                
                result.camera_id = buffer.camera_id;
                
                // Release buffer immediately after detection
                releaseDirtyCopyBuffer(job.buffer_index);
                
                // Update result
                {
                    std::lock_guard<std::mutex> lock(dirty_result_mutex_);
                    latest_dirty_results_[buffer.camera_id] = result;
                }

                if (detect_count % 60 == 0) {
                    std::cout << "[DirtyDetectionWorker] Detection completed, defects=" 
                              << result.defects.size() << ", valid=" << result.valid << std::endl;
                }
                
            } catch (const std::exception& e) {
                std::cerr << "[DirtyDetectionWorker] Error: " << e.what() << std::endl;
                releaseDirtyCopyBuffer(job.buffer_index);
            }
        }
    }
    
    std::cout << "[DirtyDetectionWorker] Thread stopped" << std::endl;
}



void DrmPreview::submitDirtyDetectionJob(int y_fd, size_t y_size,
                                         const StreamInfo &info, int camera_id)
{
    // Check if copy queue is full
    {
        std::lock_guard<std::mutex> lock(dirty_copy_mutex_);
        if (dirty_copy_queue_.size() >= 2) {
            static int skip_count = 0;
            if (++skip_count % 30 == 0) {
                std::cout << " [submitDirtyDetectionJob] Copy queue full, skipped " 
                          << skip_count << " frames" << std::endl;
            }
            return;
        }
    }
    
    // Acquire copy buffer
    int buffer_index = acquireDirtyCopyBuffer();
    if (buffer_index < 0) {
        static int skip_count = 0;
        if (++skip_count % 30 == 0) {
            std::cout << " [submitDirtyDetectionJob] Dirty copy buffers full, skipped " 
                      << skip_count << " frames" << std::endl;
        }
        return;
    }

    static int submit_count = 0;
    if (++submit_count % 30 == 0) {
        std::cout << "[submitDirtyDetectionJob] Submitted " << submit_count 
                  << " jobs, buffer_index=" << buffer_index 
                  << ", y_fd=" << y_fd << std::endl;
    }

    // Submit copy job
    DirtyCopyJob job;
    job.y_fd = y_fd;
    job.y_size = y_size;
    job.width = info.width;
    job.height = info.height;
    job.stride = info.stride;
    job.camera_id = camera_id;
    job.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    job.buffer_index = buffer_index;
    job.valid = true;
    
    {
        std::lock_guard<std::mutex> lock(dirty_copy_mutex_);
        dirty_copy_queue_.push_back(job);
    }

    dirty_copy_cv_.notify_one();
}



void DrmPreview::drawDirtyDetections(libcamera::Span<uint8_t> span, 
                                     StreamInfo const &info, 
                                     int camera_id)
{
    if (!dirty_detector_) return;
    
    imx95::DirtyDetectionResult result;
    bool has_result = false;
    
    // Get detection result
    {
        std::lock_guard<std::mutex> lock(dirty_result_mutex_);
        auto it = latest_dirty_results_.find(camera_id);
        if (it != latest_dirty_results_.end() && it->second.valid) {
            result = it->second;
            has_result = true;
        }
    }
    
    static int draw_count = 0;
    if (++draw_count % 30 == 0) {
        std::cout << "[drawDirtyDetections] Camera " << camera_id 
                  << ", has_result=" << has_result;
        if (has_result) {
            std::cout << ", defects=" << result.defects.size();
        }
    }
    
    if (!has_result) return;
    
    uint8_t* y_plane = span.data();
    
    // Draw detections with current visualization mode
    dirty_detector_->drawDetections(result.defects, y_plane, 
                                    info.width, info.height, info.stride,
                                    VisualizationMode::DEFECT_ONLY); //DEBUG_GRID,DEFECT_ONLY
    
    // Print detection statistics
    if (!result.defects.empty()) {
        static int print_counter = 0;
        if (++print_counter % 30 == 0) {  // Print every 30 frames
            std::cout << "========================================================" << std::endl;
            std::cout << "         Dirty Detection Results (Camera " << camera_id << ")          " << std::endl;
            std::cout << "========================================================" << std::endl;
            printf("  Total defects:     %3zu                                 \n", 
                   result.defects.size());
            printf("  Grid size:         %3dx%3d                             \n", 
                   result.grid_width, result.grid_height);
            printf("  Processing time:   %6.2f ms                          \n", 
                   result.processing_time_ms);
            std::cout << "========================================================" << std::endl;
            
            for (size_t i = 0; i < std::min(result.defects.size(), size_t(5)); i++) {
                const auto& defect = result.defects[i];
                printf("  Defect %zu:                                            \n", i+1);
                printf("    Position:      (%4d, %4d)                         \n", 
                       defect.x, defect.y);
                printf("    Size:          %4dx%4d                            \n", 
                       defect.width, defect.height);
                printf("    Grid count:    %3d                                 \n", 
                       defect.grid_count);
                printf("    Grid cells:    %3zu                                 \n", 
                       defect.cells.size());
                printf("    Confidence:    %5.1f%%                             \n", 
                       defect.confidence * 100);
                printf("    Avg contrast:  %5.3f                              \n", 
                       defect.avg_contrast);
            }
            
            if (result.defects.size() > 5) {
                printf("  ... and %zu more defects                              \n", 
                       result.defects.size() - 5);
            }
            std::cout << "========================================================" << std::endl;
        }
    }
}

void DrmPreview::Reset()
{
    for (auto &it : buffers_)
    {
        drmModeRmFB(drmfd_, it.second.fb_handle);
        
        drm_gem_close gem_close = {};
        gem_close.handle = it.second.y_bo_handle;
        if (drmIoctl(drmfd_, DRM_IOCTL_GEM_CLOSE, &gem_close) < 0)
            LOG(1, "DRM_IOCTL_GEM_CLOSE failed for Y plane");
        
        gem_close.handle = it.second.uv_bo_handle;
        if (drmIoctl(drmfd_, DRM_IOCTL_GEM_CLOSE, &gem_close) < 0)
            LOG(1, "DRM_IOCTL_GEM_CLOSE failed for UV plane");
    }
    buffers_.clear();
    first_time_ = true;
}

Preview *make_drm_preview(int camera_num, int detector_type)
{
	return new DrmPreview(camera_num, static_cast<DetectorType>(detector_type));
}
