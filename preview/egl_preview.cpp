/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 * Copyright (C) 2020, Raspberry Pi (Trading) Ltd.
 *
 * egl_preview.cpp - DRM-based-egl preview window.
 */

#include <drm.h>
#include <drm_fourcc.h>
#include <drm_mode.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include <fstream>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <cmath> 
#include <sys/time.h>
#include <sys/ioctl.h>
#include <linux/dma-heap.h> 

#include "core/logging.hpp"
#include "core/version.hpp"
#include "preview.hpp"

#include <epoxy/egl.h>
#include <epoxy/gl.h>
#include <gbm.h>
#include "mesh.hpp"
#include <stdlib.h>
#include <fcntl.h>

#include <opencv2/opencv.hpp>

#include <g2d.h>

#include "ssd_detector.h"
#include "vpu_encoder.hpp"
#include "g2d_scaler.hpp"

using namespace imx95;

Mesh *SSQuad1;

class EglPreview : public Preview
{
public:
	EglPreview(int camera_num, DetectorType detector_type = DETECTOR_NONE);
	~EglPreview();
	// Display the buffer. You get given the fd back in the BufferDoneCallback
	// once its available for re-use.
    virtual void ShowBuffer(libcamera::FrameBuffer* buffer, 
                           libcamera::Span<uint8_t> span, 
                           StreamInfo const &info) override;
    
    virtual void ShowBufferDual(libcamera::FrameBuffer* buffer1,
                               libcamera::Span<uint8_t> span1,
                               StreamInfo const &info1,
                               libcamera::FrameBuffer* buffer2,
                               libcamera::Span<uint8_t> span2,
                               StreamInfo const &info2) override;
	virtual void Show(int fd, libcamera::Span<uint8_t> span, StreamInfo const &info) override;
	virtual void ShowDual(int fd1, libcamera::Span<uint8_t> span1, StreamInfo const &info1, int fd2, libcamera::Span<uint8_t> span2, StreamInfo const &info2) override;
	// Reset the preview window, clearing the current buffers and being ready to
	// show new ones.
	virtual void Reset() override;
	// Return the maximum image size allowed.
	virtual void MaxImageSize(unsigned int &w, unsigned int &h) const override
	{
		w = max_image_width_;
		h = max_image_height_;
	}
    void enableVPUEncoder(const std::string& codec, int bitrate, const std::string& output_file);
    void disableVPUEncoder();

private:
	struct Buffer
	{
        Buffer() : y_fd(-1), uv_fd(-1), size(0), y_bo_handle(0), uv_bo_handle(0), fb_handle(0), texture(0) {}
        int y_fd;
        int uv_fd;
		size_t size;
		StreamInfo info;
        uint32_t y_bo_handle;
        uint32_t uv_bo_handle;
		unsigned int fb_handle;
		GLuint texture;
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
    PlaneProperties plane_props_;

    struct PlaneCache {
        uint32_t fb_id = 0;
        uint32_t crtc_id = 0;
        int crtc_x = -1, crtc_y = -1;
        int crtc_w = -1, crtc_h = -1;
        int src_x = 0, src_y = 0;
        int src_w = 0, src_h = 0;
    };
    PlaneCache plane_cache_;

    void addChangedProperties(drmModeAtomicReq *req, uint32_t new_fb_id);

	void makeBuffer(libcamera::FrameBuffer* framebuffer, 
                   size_t size, 
                   StreamInfo const &info, 
                   Buffer &buffer);

    std::unique_ptr<imx95::G2DScaler> g2d_scaler_;

    static constexpr int RING_BUFFER_SIZE = 4;
    struct DRMBuffer {
        int dma_fd = -1;
        void* virt_addr = nullptr;
        size_t size = 0;
        uint32_t bo_handle = 0;
    };
    std::array<DRMBuffer, RING_BUFFER_SIZE> drm_buffers_;

    static constexpr int SRC_WIDTH = 1920;
    static constexpr int SRC_HEIGHT = 1080;
    static constexpr int TARGET_WIDTH = 300;
    static constexpr int TARGET_HEIGHT = 300;

    bool allocateDRMBuffer(int index, size_t size);
    void freeDRMBuffer(int index);

    EGLDisplay egl_display_;
	EGLContext egl_context_;
	EGLSurface egl_surface_;
	void findCrtc();
	void findPlane();
	int drmfd_;
	int conId_;
	uint32_t crtcId_;
	int crtcIdx_;
	uint32_t planeId_;
	unsigned int out_fourcc_;
	unsigned int x_;
	unsigned int y_;
	unsigned int width_;
	unsigned int height_;
	unsigned int screen_width_;
	unsigned int screen_height_;
    std::map<libcamera::FrameBuffer*, Buffer> buffers_;
    libcamera::FrameBuffer* last_buffer_;
	libcamera::FrameBuffer* last_buffer2_;

	unsigned int max_image_width_;
	unsigned int max_image_height_;
	bool first_time_;

	// Detector members
	std::thread detection_thread_;
	std::thread copy_thread_;
	std::mutex job_mutex_;
	std::mutex result_mutex_;
	std::mutex copy_mutex_;
	std::condition_variable job_cv_;
	std::condition_variable copy_cv_;
	std::atomic<bool> detection_running_{false};
	
	std::map<int, DetectionResult> latest_results_;
	
	std::atomic<int> frames_processed_{0};
	std::atomic<int> detections_completed_{0};
	std::atomic<int> detections_skipped_{0};
	std::atomic<int> queue_full_count_{0};

	DetectorType detector_type_;
	std::unique_ptr<SSDDetectorCPU> ssd_detector_cpu_;
	std::unique_ptr<SSDDetectorNPU> ssd_detector_npu_;
	
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
	
	int acquireBufferIndex();
	void releaseBuffer(int index);
	
	void detectionWorker();
	void copyWorker();
	void submitDetectionJob(int fd, libcamera::Span<uint8_t> span, 
						   const StreamInfo &info, int camera_id);
	void drawDetections(libcamera::Span<uint8_t> span, const StreamInfo &info, int camera_id);
	
	DetectionResult processImage(const uint8_t* image_data, int width, int height, int camera_id);

    // VPU Encoder
    std::unique_ptr<imx95::VPUEncoder> vpu_encoder_;
    bool vpu_encoder_enabled_ = false;
    std::string encoder_codec_;
    int encoder_bitrate_ = 0;
    std::string encoder_output_file_;
    
    // Encoding thread
    struct EncodingJob {
        int y_fd = -1;           // Y plane DMA-BUF fd
        int uv_fd = -1;          // UV plane DMA-BUF fd
        size_t y_size = 0;       // Y plane size in bytes
        size_t uv_size = 0;      // UV plane size in bytes
        int width;
        int height;
        int stride;
        uint64_t timestamp;
    };
    
    std::thread encoding_thread_;
    std::mutex encoding_mutex_;
    std::condition_variable encoding_cv_;
    std::deque<EncodingJob> encoding_queue_;
    std::atomic<bool> encoding_running_{false};
    static constexpr int MAX_ENCODING_QUEUE_SIZE = 32;
    
    void encodingWorker();
    void submitEncodingJobDMABuf(int yuyv_dma_fd, size_t yuyv_size,
                             int width, int height, uint64_t timestamp);

};

#define ERRSTR strerror(errno)
struct gbm_surface *gbm_surface;
struct gbm_device *gbm_device;
struct gbm_bo *bo;
uint32_t handle;
uint32_t pitch;
uint32_t fb;
uint64_t modifier;

static GLint compile_shader(GLenum target, const char *source)
{
	GLuint s = glCreateShader(target);
	glShaderSource(s, 1, (const GLchar **)&source, NULL);
	glCompileShader(s);

	GLint ok;
	glGetShaderiv(s, GL_COMPILE_STATUS, &ok);

	if (!ok)
	{
		GLchar *info;
		GLint size;

		glGetShaderiv(s, GL_INFO_LOG_LENGTH, &size);
		info = (GLchar *)malloc(size);

		glGetShaderInfoLog(s, size, NULL, info);
		throw std::runtime_error("failed to compile shader: " + std::string(info) + "\nsource:\n" +
								 std::string(source));
	}

	return s;
}

static GLint link_program(GLint vs, GLint fs)
{
	GLint prog = glCreateProgram();
	glAttachShader(prog, vs);
	glAttachShader(prog, fs);
	
	glBindAttribLocation(prog, 0, "pos");
	glBindAttribLocation(prog, 2, "tex");
		
	glLinkProgram(prog);

	GLint ok;
	glGetProgramiv(prog, GL_LINK_STATUS, &ok);
	if (!ok)
	{
		 //Some drivers return a size of 1 for an empty log.  This is the size
		  //of a log that contains only a terminating NUL character.
		 
		GLint size;
		GLchar *info = NULL;
		glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &size);
		if (size > 1)
		{
			info = (GLchar *)malloc(size);
			glGetProgramInfoLog(prog, size, NULL, info);
		}

		throw std::runtime_error("failed to link: " + std::string(info ? info : "<empty log>"));
	}

	return prog;
}

static int match_config_to_visual(EGLDisplay egl_display, EGLint visual_id, EGLConfig *configs, int count) {
	int j;
	for (j = 0; j < count; ++j) {
		EGLint id;
		EGLint blue_size, red_size, green_size, alpha_size;
		std::cout << j << "\n";
		if (!eglGetConfigAttrib(egl_display, configs[j], EGL_NATIVE_VISUAL_ID,&id)) 
			continue;
			
		eglGetConfigAttrib(egl_display, configs[j], EGL_RED_SIZE, &red_size);
        eglGetConfigAttrib(egl_display, configs[j], EGL_GREEN_SIZE, &green_size);
        eglGetConfigAttrib(egl_display, configs[j], EGL_BLUE_SIZE, &blue_size);
        eglGetConfigAttrib(egl_display, configs[j], EGL_ALPHA_SIZE, &alpha_size);	
        
        char gbm_format_str[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        memcpy(gbm_format_str, &id, sizeof(EGLint));
        printf("  %d-th GBM format: %s;  sizes(RGBA) = %d,%d,%d,%d,\n",
               j, gbm_format_str, red_size, green_size, blue_size, alpha_size);        
        
		if (id == visual_id) 
			return j;
	}
	return -1;
}

void EglPreview::findCrtc()
{
	int i;
	drmModeRes *res = drmModeGetResources(drmfd_);
	if (!res)
		throw std::runtime_error("drmModeGetResources failed: " + std::string(ERRSTR));

	if (res->count_crtcs <= 0)
		throw std::runtime_error("drm: no crts");

	max_image_width_ = res->max_width;
	max_image_height_ = res->max_height;

	if (!conId_)
	{

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

			if (!conId_ && crtc)
			{
				conId_ = con->connector_id;
				crtcId_ = crtc->crtc_id;
			}

			if (crtc)
			{
				screen_width_ = crtc->width;
				screen_height_ = crtc->height;
			}

			std::cerr << "Connector " << con->connector_id << " (crtc " << (crtc ? crtc->crtc_id : 0) << "): type "
					  << con->connector_type << ", " << (crtc ? crtc->width : 0) << "x" << (crtc ? crtc->height : 0)
					  << (conId_ == (int)con->connector_id ? " (chosen)" : "") << std::endl;
		}

		if (!conId_)
			throw std::runtime_error("No suitable enabled connector found");
	}

	crtcIdx_ = -1;

	for (i = 0; i < res->count_crtcs; ++i)
	{
		if (crtcId_ == res->crtcs[i])
		{
			crtcIdx_ = i;
			break;
		}
	}

	if (crtcIdx_ == -1)
	{
		drmModeFreeResources(res);
		throw std::runtime_error("drm: CRTC " + std::to_string(crtcId_) + " not found");
	}

	if (res->count_connectors <= 0)
	{
		drmModeFreeResources(res);
		throw std::runtime_error("drm: no connectors");
	}

	drmModeConnector *c;
	c = drmModeGetConnector(drmfd_, conId_);
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

	if (width_ == 0 || height_ == 0)
	{
		drmModeCrtc *crtc = drmModeGetCrtc(drmfd_, crtcId_);
		x_ = crtc->x;
		y_ = crtc->y;
		width_ = crtc->width;
		height_ = crtc->height;
		drmModeFreeCrtc(crtc);
	}
	gbm_device = gbm_create_device(drmfd_);
    assert(gbm != NULL);
	gbm_surface = gbm_surface_create(gbm_device, width_, height_, GBM_FORMAT_XRGB8888, GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
	if (!gbm_surface){
			throw std::runtime_error("Failed to create GBM surface\n");
	}

	PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT = 
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (!eglGetPlatformDisplayEXT) {
        gbm_device_destroy(gbm_device);
        close(drmfd_);
        throw std::runtime_error("EGL_EXT_platform_base not support\n");
    }

	egl_display_ = eglGetPlatformDisplayEXT(EGL_PLATFORM_GBM_KHR,
				gbm_device, NULL);
    if (egl_display_ == EGL_NO_DISPLAY){
		printf("Failed to get EGL Display 0x%x\n", eglGetError());
	}
	EGLint major, minor;
	if(!eglInitialize(egl_display_, &major, &minor))
		throw std::runtime_error("Failed to get EGL Display\n");
	printf("EGL Version \"%s\"\n", eglQueryString(egl_display_, EGL_VERSION));
	
	eglBindAPI(EGL_OPENGL_API);
	
	static const EGLint attribs[] =
		{
			//EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
			//EGL_BUFFER_SIZE, 32,
			//EGL_DEPTH_SIZE, EGL_DONT_CARE,
			//EGL_STENCIL_SIZE, EGL_DONT_CARE,
			EGL_RED_SIZE, 1,
			EGL_GREEN_SIZE, 1,
			EGL_BLUE_SIZE, 1,
			EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
			EGL_NONE
		};
		
	static const EGLint ctx_attribs[] = {
		EGL_CONTEXT_CLIENT_VERSION, 2,
		EGL_NONE
	};
		
	EGLint num_configs;
	EGLConfig *configs;
	int config_index;
	
	if(!eglGetConfigs(egl_display_, NULL, 0, &num_configs) || num_configs < 1){
		printf("cannot get any configs, error: 0x%x\n", eglGetError());
	}
	configs = (EGLConfig*)malloc(num_configs * sizeof(EGLConfig));
	
	if(!configs)
		throw std::runtime_error("no configs");
	
	if (!eglChooseConfig(egl_display_, attribs, configs, num_configs, &num_configs))
		throw std::runtime_error("couldn't get an EGL visual config");
	
	config_index = match_config_to_visual(egl_display_, GBM_FORMAT_XRGB8888, configs, num_configs);
	
	std::cout << config_index << "\n";
	
	//freezes here
		
	egl_context_ = eglCreateContext(egl_display_, configs[config_index], EGL_NO_CONTEXT, ctx_attribs);
	if (!egl_context_){
		printf("eglCreateContext failed 0x%x\n", eglGetError());
		throw std::runtime_error("context failed bro");
	}
		
	printf("im here\n");
		
	egl_surface_ = eglCreateWindowSurface(egl_display_, configs[config_index], (EGLNativeWindowType)gbm_surface, NULL);
	if(egl_surface_ == EGL_NO_SURFACE) {
		printf("failed to create EGL window surface, error: 0x%x\n", eglGetError());
	}
	
	free(configs);
	
	printf("setup egl i think\n");
}

void EglPreview::findPlane()
{
	drmModePlaneResPtr planes;
	drmModePlanePtr plane;
	unsigned int i;
	unsigned int j;

	planes = drmModeGetPlaneResources(drmfd_);
	if (!planes)
		throw std::runtime_error("drmModeGetPlaneResources failed: " + std::string(ERRSTR));

	try
	{
		for (i = 0; i < planes->count_planes; ++i)
		{
			plane = drmModeGetPlane(drmfd_, planes->planes[i]);
			if (!planes)
				throw std::runtime_error("drmModeGetPlane failed: " + std::string(ERRSTR));

			if (!(plane->possible_crtcs & (1 << crtcIdx_)))
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

			planeId_ = plane->plane_id;

			drmModeFreePlane(plane);
			break;
		}
	}
	catch (std::exception const &e)
	{
		drmModeFreePlaneResources(planes);
		throw;
	}

	drmModeFreePlaneResources(planes);

    struct drm_set_client_cap cap;
    cap.capability = DRM_CLIENT_CAP_ATOMIC;
    cap.value = 1;
    if (ioctl(drmfd_, DRM_IOCTL_SET_CLIENT_CAP, &cap) < 0) {
        std::cerr << "Failed to enable atomic mode: " << strerror(errno) << std::endl;
    } else {
        std::cout << "EGL: Atomic mode enabled" << std::endl;
    }

    drmModeObjectProperties *props = drmModeObjectGetProperties(
        drmfd_, planeId_, DRM_MODE_OBJECT_PLANE);

    if (!props) {
        std::cerr << "Failed to get plane properties" << std::endl;
        return;
    }

    for (unsigned int k = 0; k < props->count_props; ++k) {
        drmModePropertyRes *prop = drmModeGetProperty(drmfd_, props->props[k]);
        if (!prop) continue;

        if (strcmp(prop->name, "CRTC_ID") == 0)
            plane_props_.crtc_id = prop->prop_id;
        else if (strcmp(prop->name, "FB_ID") == 0)
            plane_props_.fb_id = prop->prop_id;
        else if (strcmp(prop->name, "CRTC_X") == 0)
            plane_props_.crtc_x = prop->prop_id;
        else if (strcmp(prop->name, "CRTC_Y") == 0)
            plane_props_.crtc_y = prop->prop_id;
        else if (strcmp(prop->name, "CRTC_W") == 0)
            plane_props_.crtc_w = prop->prop_id;
        else if (strcmp(prop->name, "CRTC_H") == 0)
            plane_props_.crtc_h = prop->prop_id;
        else if (strcmp(prop->name, "SRC_X") == 0)
            plane_props_.src_x = prop->prop_id;
        else if (strcmp(prop->name, "SRC_Y") == 0)
            plane_props_.src_y = prop->prop_id;
        else if (strcmp(prop->name, "SRC_W") == 0)
            plane_props_.src_w = prop->prop_id;
        else if (strcmp(prop->name, "SRC_H") == 0)
            plane_props_.src_h = prop->prop_id;

        drmModeFreeProperty(prop);
    }

    drmModeFreeObjectProperties(props);

}

EglPreview::EglPreview(int camera_num, DetectorType detector_type) : Preview(), last_buffer_(nullptr), last_buffer2_(nullptr), first_time_(true), detector_type_(detector_type)
{
	drmfd_ = drmOpen("imx95-dpu", NULL);
	if (drmfd_ < 0){
		throw std::runtime_error("drmOpen failed: " + std::string(ERRSTR));
	}

	try
	{
		if (!drmIsMaster(drmfd_))
			throw std::runtime_error("DRM preview unavailable - not master");

		conId_ = 0;
		findCrtc();
		out_fourcc_ = DRM_FORMAT_NV12;
		findPlane();

		// Initialize detector based on type
		if (detector_type_ == DETECTOR_SSD_CPU) {
			printf("Initializing SSD Detector...\n");
			ssd_detector_cpu_ = std::make_unique<SSDDetectorCPU>();
			ssd_detector_cpu_->setNumThreads(6);
			if (!ssd_detector_cpu_->initialize(CPU_MODEL,  CPU_LABEL, 
										0.5, 0.5)) {
				std::cerr << "Failed to initialize detector" << std::endl;
				throw std::runtime_error("Detector initialization failed");
			}
		} else if (detector_type_ == DETECTOR_SSD_NPU) {
			printf("Initializing SSD Anchor Detector...\n");
			ssd_detector_npu_ = std::make_unique<SSDDetectorNPU>();
			ssd_detector_npu_->setNumThreads(1);
			if (!ssd_detector_npu_->initialize(NPU_MODEL, NPU_LABEL, 
									0.5, 0.5)) {
				std::cerr << "Failed to initialize detector" << std::endl;
				throw std::runtime_error("Detector initialization failed");
			}
		}

	for (auto& buffer : ring_buffers_) {
		buffer.in_use = false;
        buffer.g2d_buffer_id = -1;
	}
	
	std::cout << " Ring buffers initialized (" << RING_BUFFER_SIZE << " buffers)" << std::endl;
	
	if (detector_type_ != DETECTOR_NONE ) {
		detection_running_ = true;
		detection_thread_ = std::thread(&EglPreview::detectionWorker, this);
		copy_thread_ = std::thread(&EglPreview::copyWorker, this);
	}

	}
	catch (std::exception const &e)
	{
		close(drmfd_);
		throw;
	}

    if (detector_type_ == DETECTOR_SSD_CPU || detector_type_ == DETECTOR_SSD_NPU) {

        g2d_scaler_ = std::make_unique<imx95::G2DScaler>();
        
        if (!g2d_scaler_->setup(SRC_WIDTH, SRC_HEIGHT, SRC_WIDTH, TARGET_WIDTH, TARGET_HEIGHT)) {
            std::cerr << "Failed to initialize G2D scaler" << std::endl;
            throw std::runtime_error("G2D scaler initialization failed");
        }
        
        size_t output_size = TARGET_WIDTH * TARGET_HEIGHT * 3;  // RGB888
        for (int i = 0; i < RING_BUFFER_SIZE; i++) {
            if (!allocateDRMBuffer(i, output_size)) {
                std::cerr << "Failed to allocate DRM buffer " << i << std::endl;
                for (int j = 0; j < i; j++) {
                    freeDRMBuffer(j);
                }
                throw std::runtime_error("DRM buffer allocation failed");
            }
        }
        
        std::cout << "[EGL] G2D scaler initialized with " << RING_BUFFER_SIZE 
                  << " DMA-BUF output buffers" << std::endl;
    }

}

EglPreview::~EglPreview()
{
    std::cout << "[EGL] Stopping threads..." << std::endl;

    // Stop encoding thread
    if (vpu_encoder_enabled_) {
        encoding_running_ = false;
        encoding_cv_.notify_all();
        if (encoding_thread_.joinable()) {
            encoding_thread_.join();
        }
        vpu_encoder_.reset();
    }

	detection_running_ = false;
    
    job_cv_.notify_all();
    copy_cv_.notify_all();

    if (detection_thread_.joinable()) {
        std::cout << "[EGL] Waiting for detection_thread_..." << std::endl;
        detection_thread_.join();
        std::cout << "[EGL]  detection_thread_ stopped" << std::endl;
    }

    if (copy_thread_.joinable()) {
        std::cout << "[EGL] Waiting for copy_thread_..." << std::endl;
        copy_thread_.join();
        std::cout << "[EGL]  copy_thread_ stopped" << std::endl;
    }

    std::cout << "[EGL]  All threads stopped" << std::endl;

    if (ssd_detector_cpu_) {
        std::cout << "[EGL] Cleaning up CPU detector..." << std::endl;
        ssd_detector_cpu_.reset();
    }
    
    if (ssd_detector_npu_) {
        std::cout << "[EGL] Cleaning up NPU detector..." << std::endl;
        ssd_detector_npu_.reset();
    }

    std::cout << "[EGL] Cleaning up EGL resources..." << std::endl;
    
    if (egl_display_ != EGL_NO_DISPLAY) {
        eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        
        if (egl_context_ != EGL_NO_CONTEXT) {
            eglDestroyContext(egl_display_, egl_context_);
            egl_context_ = EGL_NO_CONTEXT;
        }
        
        if (egl_surface_ != EGL_NO_SURFACE) {
            eglDestroySurface(egl_display_, egl_surface_);
            egl_surface_ = EGL_NO_SURFACE;
        }
        
        eglTerminate(egl_display_);
        egl_display_ = EGL_NO_DISPLAY;
    }

    if (gbm_surface) {
        std::cout << "[EGL] Destroying GBM surface..." << std::endl;
        gbm_surface_destroy(gbm_surface);
        gbm_surface = nullptr;
    }
    
    if (gbm_device) {
        std::cout << "[EGL] Destroying GBM device..." << std::endl;
        gbm_device_destroy(gbm_device);
        gbm_device = nullptr;
    }

    if (drmfd_ >= 0) {
        std::cout << "[EGL] Closing DRM fd..." << std::endl;
        close(drmfd_);
        drmfd_ = -1;
    }

    if (g2d_scaler_) {
        std::cout << "[EGL] Cleaning up DRM buffers..." << std::endl;

        for (int i = 0; i < RING_BUFFER_SIZE; i++) {
            freeDRMBuffer(i);
        }

        std::cout << " DRM buffers cleaned up" << std::endl;
        g2d_scaler_.reset();
    }

    std::cout << "[EGL]  Cleanup complete" << std::endl;
}

// DRM doesn't seem to have userspace definitions of its enums, but the properties
// contain enum-name-to-value tables. So the code below ends up using strings and
// searching for name matches. I suppose it works...
static void get_colour_space_info(std::optional<libcamera::ColorSpace> const &cs, char const *&encoding,
								  char const *&range)
{
	static char const encoding_601[] = "601", encoding_709[] = "709";
	static char const range_limited[] = "limited", range_full[] = "full";
	encoding = encoding_601;
	range = range_limited;

	if (cs == libcamera::ColorSpace::Sycc)
		range = range_full;
	else if (cs == libcamera::ColorSpace::Smpte170m)
		/* all good */;
	else if (cs == libcamera::ColorSpace::Rec709)
		encoding = encoding_709;
	else
		std::cerr << "EglPreview: unexpected colour space " << libcamera::ColorSpace::toString(cs) << std::endl;
}

void EglPreview::addChangedProperties(drmModeAtomicReq *req, uint32_t new_fb_id)
{
    auto &cache = plane_cache_;
    auto &props = plane_props_;

    if (new_fb_id != cache.fb_id) {
        drmModeAtomicAddProperty(req, planeId_, props.fb_id, new_fb_id);
        cache.fb_id = new_fb_id;
    }

    if (cache.crtc_id != crtcId_) {
        drmModeAtomicAddProperty(req, planeId_, props.crtc_id, crtcId_);
        cache.crtc_id = crtcId_;
    }

    if (cache.crtc_x != (int)x_) {
        drmModeAtomicAddProperty(req, planeId_, props.crtc_x, x_);
        cache.crtc_x = x_;
    }
    if (cache.crtc_y != (int)y_) {
        drmModeAtomicAddProperty(req, planeId_, props.crtc_y, y_);
        cache.crtc_y = y_;
    }
    if (cache.crtc_w != (int)width_) {
        drmModeAtomicAddProperty(req, planeId_, props.crtc_w, width_);
        cache.crtc_w = width_;
    }
    if (cache.crtc_h != (int)height_) {
        drmModeAtomicAddProperty(req, planeId_, props.crtc_h, height_);
        cache.crtc_h = height_;
    }

    if (cache.src_x != 0) {
        drmModeAtomicAddProperty(req, planeId_, props.src_x, 0);
        cache.src_x = 0;
    }
    if (cache.src_y != 0) {
        drmModeAtomicAddProperty(req, planeId_, props.src_y, 0);
        cache.src_y = 0;
    }
    if (cache.src_w != (int)width_) {
        drmModeAtomicAddProperty(req, planeId_, props.src_w, width_ << 16);
        cache.src_w = width_;
    }
    if (cache.src_h != (int)height_) {
        drmModeAtomicAddProperty(req, planeId_, props.src_h, height_ << 16);
        cache.src_h = height_;
    }
}


static int drm_set_property(int fd, int plane_id, char const *name, char const *val)
{
	drmModeObjectPropertiesPtr properties = nullptr;
	drmModePropertyPtr prop = nullptr;
	int ret = -1;
	properties = drmModeObjectGetProperties(fd, plane_id, DRM_MODE_OBJECT_PLANE);

	for (unsigned int i = 0; i < properties->count_props; i++)
	{
		int prop_id = properties->props[i];
		prop = drmModeGetProperty(fd, prop_id);
		if (!prop)
			continue;

		if (!drm_property_type_is(prop, DRM_MODE_PROP_ENUM) || !strstr(prop->name, name))
		{
			drmModeFreeProperty(prop);
			prop = nullptr;
			continue;
		}

		// We have found the right property from its name, now search the enum table
		// for the numerical value that corresponds to the value name that we have.
		for (int j = 0; j < prop->count_enums; j++)
		{
			if (!strstr(prop->enums[j].name, val))
				continue;

			ret = drmModeObjectSetProperty(fd, plane_id, DRM_MODE_OBJECT_PLANE, prop_id, prop->enums[j].value);
			if (ret < 0)
				std::cerr << "EglPreview: failed to set value " << val << " for property " << name << std::endl;
			goto done;
		}

		std::cerr << "EglPreview: failed to find value " << val << " for property " << name << std::endl;
		goto done;
	}

	std::cerr << "EglPreview: failed to find property " << name << std::endl;
done:
	if (prop)
		drmModeFreeProperty(prop);
	if (properties)
		drmModeFreeObjectProperties(properties);
	return ret;
}

static void setup_colour_space(int fd, int plane_id, std::optional<libcamera::ColorSpace> const &cs)
{
	char const *encoding, *range;
	get_colour_space_info(cs, encoding, range);

	drm_set_property(fd, plane_id, "COLOR_ENCODING", encoding);
	drm_set_property(fd, plane_id, "COLOR_RANGE", range);
	const char *vs = "#version 300 es\n"
					 "in vec3 pos;\n"
			         "in vec2 tex;\n"
			         "out vec2 texcoord;\n"
			         "\n"
			         "void main() {\n"
			         "  gl_Position = vec4(pos, 1.0);\n"
			         "  texcoord = tex;\n"
			         "}\n";
	GLint vs_s = compile_shader(GL_VERTEX_SHADER, vs);
	const char *fs = "#version 300 es\n"
					 "#extension GL_OES_EGL_image_external_essl3  : enable\n"
					 "precision mediump float;\n"
					 "uniform samplerExternalOES s;\n"
					 "in vec2 texcoord;\n"
					 "out vec4 out_color;\n"
					 "void main() {\n"
					 "  out_color = texture(s, texcoord);\n"
					 "}\n";
	GLint fs_s = compile_shader(GL_FRAGMENT_SHADER, fs);
	GLint prog = link_program(vs_s, fs_s);

	glUseProgram(prog);
	
    std::vector<float> vertices; 
    std::vector<unsigned short> indices;
        
	float N = 100;    //create an NxN grid of triangles (NxNx2 Triangles produced)
    float z = 0;      //empty z component for the POS vector
    
    for (float x = -1, a = 0; x <= 1 && a <= 1; x+= 2/N, a += 1/N)
    {
		for (float y = -1, b = 1; y <= 1 && b >= 0; y+= 2/N, b-= 1/N)
        {
			float theta = atan2(y, x);
			float r = sqrt(x*x + y*y);
			r = r -0.15*pow(r, 3.0) + 0.01*pow(r, 5.0);
			vertices.push_back(r*cos(theta));
			vertices.push_back(r*sin(theta));
			//vertices.push_back(x);
			//vertices.push_back(y);
			vertices.push_back(z);
			vertices.push_back(a);
			vertices.push_back(b);
        }
	}
	 
	for (int x = 0; x < N; x++)
	{
		for (int z = 0; z < N; z++)
		{
			int offset = x * (N+1) + z;
            indices.push_back((short)(offset+0));
            indices.push_back((short)(offset+1));
			indices.push_back((short)(offset+ (N+1) + 1));
            indices.push_back((short)(offset+0));
            indices.push_back((short)(offset+ (N+1)));
            indices.push_back((short)(offset+ (N+1) + 1));
        }
    }
	SSQuad1 = new Mesh({ POS, TEX }, vertices, indices);
}

void EglPreview::makeBuffer(libcamera::FrameBuffer* framebuffer, 
                           size_t size, 
                           StreamInfo const &info, 
                           Buffer &buffer)
{
    if (first_time_)
    {
        first_time_ = false;
        if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_))
            throw std::runtime_error("eglMakeCurrent failed");
        setup_colour_space(drmfd_, planeId_, info.colour_space);
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

    eglSwapBuffers(egl_display_, egl_surface_);
    bo = gbm_surface_lock_front_buffer(gbm_surface);
    
    uint32_t width = info.width;
    uint32_t height = info.height;
    uint32_t format = DRM_FORMAT_NV12;

    uint32_t offsets[4] = { 
        framebuffer->planes()[0].offset,
        framebuffer->planes()[1].offset,
        0, 
        0 
    };
    
    uint32_t handles[4] = { 
        buffer.y_bo_handle,   // Y plane BO
        buffer.uv_bo_handle,  // UV plane BO
        0, 
        0 
    };
    
    uint32_t strides[4] = { 
        info.stride,     // Y plane stride
        info.stride,     // UV plane stride
        0, 
        0 
    };
/*
    std::cout << "  Creating framebuffer with:" << std::endl;
    std::cout << "    Width: " << width << ", Height: " << height << std::endl;
    std::cout << "    Format: NV12 (0x" << std::hex << format << std::dec << ")" << std::endl;
    std::cout << "    Y  BO: " << buffer.y_bo_handle << ", stride: " << strides[0] << ", offset: " << offsets[0] << std::endl;
    std::cout << "    UV BO: " << buffer.uv_bo_handle << ", stride: " << strides[1] << ", offset: " << offsets[1] << std::endl;
*/
    int ret = drmModeAddFB2(drmfd_, width, height, format,
                           handles, strides, offsets, &buffer.fb_handle, 0);
    if (ret) {
        std::cerr << "drmModeAddFB2 failed with error: " << strerror(errno) << std::endl;
        std::cerr << "  Width: " << width << ", Height: " << height << std::endl;
        std::cerr << "  Format: " << format << " (NV12)" << std::endl;
        std::cerr << "  Y  BO Handle: " << buffer.y_bo_handle << " (fd=" << buffer.y_fd << ")" << std::endl;
        std::cerr << "  UV BO Handle: " << buffer.uv_bo_handle << " (fd=" << buffer.uv_fd << ")" << std::endl;
        std::cerr << "  Y  stride: " << strides[0] << ", offset: " << offsets[0] << std::endl;
        std::cerr << "  UV stride: " << strides[1] << ", offset: " << offsets[1] << std::endl;
        throw std::runtime_error("drmModeAddFB2 failed: " + std::string(ERRSTR));
    }

    //std::cout << "  Framebuffer created: " << buffer.fb_handle << std::endl;

    // ===== EGL Image create(NV12)=====
    EGLint attribs[] = {
        EGL_WIDTH, static_cast<EGLint>(info.width),
        EGL_HEIGHT, static_cast<EGLint>(info.height),
        EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_NV12,
        
        // Y plane (plane 0)
        EGL_DMA_BUF_PLANE0_FD_EXT, buffer.y_fd,
        EGL_DMA_BUF_PLANE0_OFFSET_EXT, static_cast<EGLint>(framebuffer->planes()[0].offset),
        EGL_DMA_BUF_PLANE0_PITCH_EXT, static_cast<EGLint>(info.stride),
        
        // UV plane (plane 1)
        EGL_DMA_BUF_PLANE1_FD_EXT, buffer.uv_fd,
        EGL_DMA_BUF_PLANE1_OFFSET_EXT, static_cast<EGLint>(framebuffer->planes()[1].offset),
        EGL_DMA_BUF_PLANE1_PITCH_EXT, static_cast<EGLint>(info.stride),
        
        EGL_YUV_COLOR_SPACE_HINT_EXT, EGL_ITU_REC601_EXT,
        EGL_SAMPLE_RANGE_HINT_EXT, EGL_YUV_NARROW_RANGE_EXT,
        EGL_NONE
    };
   
    EGLImage image = eglCreateImageKHR(egl_display_, EGL_NO_CONTEXT, 
                                       EGL_LINUX_DMA_BUF_EXT, NULL, attribs);
    if (!image) {
        printf("failed to create EGL image, error: 0x%x\n", eglGetError());
        throw std::runtime_error("no image made");
    }
    
    //std::cout << "  EGL Image created" << std::endl;
    
    glGenTextures(1, &buffer.texture);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, buffer.texture);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glEGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES, image);

    eglDestroyImageKHR(egl_display_, image);
    gbm_surface_release_buffer(gbm_surface, bo);
    
    /*LOG(2, " Created NV12 EGL texture: " << buffer.texture 
           << " (" << info.width << "x" << info.height 
           << ", Y_fd=" << buffer.y_fd << ", UV_fd=" << buffer.uv_fd << ")");*/
}

void EglPreview::enableVPUEncoder(const std::string& codec, int bitrate, const std::string& output_file)
{
    encoder_codec_ = codec;
    encoder_bitrate_ = bitrate;
    encoder_output_file_ = output_file;
    
    vpu_encoder_enabled_ = true;
    encoding_running_ = true;
    
    encoding_thread_ = std::thread(&EglPreview::encodingWorker, this);
    
    std::cout << "[EGL] VPU Encoder enabled: " << codec << " @ " << bitrate << " bps" << std::endl;
    std::cout << "[EGL] Output file: " << output_file << std::endl;
}

void EglPreview::disableVPUEncoder()
{
    encoding_running_ = false;
    encoding_cv_.notify_all();
    
    if (encoding_thread_.joinable()) {
        encoding_thread_.join();
    }
    
    if (vpu_encoder_) {
        vpu_encoder_->stop();
    }
    vpu_encoder_.reset();
    
    vpu_encoder_enabled_ = false;
    std::cout << "[EGL] VPU Encoder disabled" << std::endl;
}

// ===== Encoding worker thread =====
void EglPreview::encodingWorker()
{
    std::cout << "[EGL EncodingWorker] Thread started (DMA-BUF mode)" << std::endl;
    
    while (encoding_running_) {
        EncodingJob job;
        bool has_job = false;

        {
            std::unique_lock<std::mutex> lock(encoding_mutex_);
            encoding_cv_.wait(lock, [this] {
                return !encoding_queue_.empty() || !encoding_running_;
            });
            
            if (!encoding_running_) break;
            
            if (!encoding_queue_.empty()) {
                job = encoding_queue_.front();
                encoding_queue_.pop_front();
                has_job = true;
            }
        }

        if (has_job) {
            auto t0 = std::chrono::high_resolution_clock::now();

            if (!vpu_encoder_) {
                vpu_encoder_ = std::make_unique<imx95::VPUEncoder>();
                
                std::cout << "[EGL EncodingWorker] Setting up encoder " 
                          << job.width << "x" << job.height << "..." << std::endl;
                
               imx95::InputFormat input_fmt = (job.uv_fd < 0) 
                    ? imx95::InputFormat::YUYV 
                    : imx95::InputFormat::NV12M;
                
                if (!vpu_encoder_->setup(job.width, job.height, 
                                         encoder_codec_, encoder_bitrate_, 
                                         encoder_output_file_,
                                         imx95::EncoderMode::DMABUF,
                                         input_fmt)) { 
                    std::cerr << "[EGL] Failed to setup VPU encoder" << std::endl;
                    vpu_encoder_.reset();
                    encoding_running_ = false;
                    return;
                }
                
                std::cout << "[EGL] VPU encoder initialized (DMABUF mode): " 
                          << job.width << "x" << job.height << std::endl;
            }
            
            if (vpu_encoder_ && vpu_encoder_->isInitialized()) {
                auto t1 = std::chrono::high_resolution_clock::now();
                // YUYV formt
                vpu_encoder_->encodeFromDMABuf_YUYV(
                    job.y_fd,
                    job.y_size,
                    job.stride,
                    job.timestamp
                );

                auto t2 = std::chrono::high_resolution_clock::now();
                
                static int encode_count = 0;
                static uint64_t total_encode_us = 0;
                static uint64_t total_total_us = 0;
                
                auto encode_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
                auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0);
                
                total_encode_us += encode_time.count();
                total_total_us += total_time.count();
                
                if (++encode_count % 300 == 0) {
                    std::cout << "========================================================" << std::endl;
                    std::cout << "  EGL Encoding Worker Performance (avg 300 frames)       " << std::endl;
                    std::cout << "========================================================" << std::endl;
                    printf("  VPU encode:    %6.2f ms  (DMA-BUF zero-copy)        \n", 
                           total_encode_us / 300000.0);
                    printf("  Total:         %6.2f ms                              \n", 
                           total_total_us / 300000.0);
                    printf("  Max FPS:       %6.2f                                  \n", 
                           300000000.0 / total_total_us);
                    printf("  Queue size:    %zu                                    \n", 
                           encoding_queue_.size());
                    std::cout << "========================================================" << std::endl;
                    total_encode_us = 0;
                    total_total_us = 0;
                }
            }
        }
    }
    
    std::cout << "[EGL EncodingWorker] Thread stopped" << std::endl;
}

// ===== Submit encoding job =====
void EglPreview::submitEncodingJobDMABuf(int yuyv_dma_fd, size_t yuyv_size,
                                         int width, int height, uint64_t timestamp)
{
    if (!vpu_encoder_enabled_ || !encoding_running_) {
        static int skip_count = 0;
        if (++skip_count % 60 == 0) {
            std::cout << " [EGL submitEncodingJob] Skipped " << skip_count 
                      << " frames (enabled=" << vpu_encoder_enabled_ 
                      << ", running=" << encoding_running_ << ")" << std::endl;
        }
        return;
    }
    
    bool dropped = false;
    {
        std::lock_guard<std::mutex> lock(encoding_mutex_);
        
        if (encoding_queue_.size() >= MAX_ENCODING_QUEUE_SIZE) {
            static int drop_count = 0;
            if (++drop_count % 30 == 0) {
                std::cout << "[EGL] Encoding too slow, dropped " 
                          << drop_count << " frames (queue full)" << std::endl;
            }
            dropped = true;
        } else {
            EncodingJob job;
            job.y_fd = yuyv_dma_fd;
            job.uv_fd = -1;
            job.y_size = yuyv_size;
            job.uv_size = 0;
            job.width = width;
            job.height = height;
            job.stride = width * 2;      // YUYV stride
            job.timestamp = timestamp;
            
            encoding_queue_.push_back(job);
        }
    }
    
    if (!dropped) {
        encoding_cv_.notify_one();
    }
}

bool EglPreview::allocateDRMBuffer(int index, size_t size)
{
    auto& buf = drm_buffers_[index];
    
    int heap_fd = open("/dev/dma_heap/linux,cma", O_RDWR);
    if (heap_fd < 0) {
        std::cerr << "Failed to open /dev/dma_heap/linux,cma: " << strerror(errno) << std::endl;
        return false;
    }
    
    struct dma_heap_allocation_data {
        uint64_t len;
        uint32_t fd;
        uint32_t fd_flags;
        uint64_t heap_flags;
    };
    
    struct dma_heap_allocation_data alloc_data = {};
    alloc_data.len = size;
    alloc_data.fd = 0;
    alloc_data.fd_flags = O_RDWR | O_CLOEXEC;
    alloc_data.heap_flags = 0;  // cached
    
    int ret = ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data);
    close(heap_fd);
    
    if (ret < 0) {
        std::cerr << "DMA_HEAP_IOCTL_ALLOC failed: " << strerror(errno) << std::endl;
        return false;
    }
    
    buf.dma_fd = alloc_data.fd;
    buf.size = size;
    
    // mmap for CPU access
    buf.virt_addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, 
                         buf.dma_fd, 0);
    
    if (buf.virt_addr == MAP_FAILED) {
        std::cerr << "Failed to mmap DMA-Heap buffer: " << strerror(errno) << std::endl;
        close(buf.dma_fd);
        buf.dma_fd = -1;
        return false;
    }
    
    std::cout << "[EGL] Allocated DMA-BUF " << index << ": fd=" << buf.dma_fd 
              << ", size=" << size << " bytes" << std::endl;
    
    return true;
}

void EglPreview::freeDRMBuffer(int index)
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

static int frame_cnt = 0;
static struct timeval start_time_;
static struct timeval last_time_;

void EglPreview::ShowBuffer(libcamera::FrameBuffer* buffer, 
                           libcamera::Span<uint8_t> span, 
                           StreamInfo const &info)
{
    if (frame_cnt == 0) {
        std::cout << "\n=== First Frame Info (EGL ShowBuffer) ===" << std::endl;
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
    if (buf.y_fd == -1)
        makeBuffer(buffer, span.size(), info, buf);

    last_buffer_ = buffer;

	if (detector_type_ != DETECTOR_NONE) {
		const int DETECTION_INTERVAL = 1;
		
		if (frame_cnt % DETECTION_INTERVAL == 0) {
			submitDetectionJob(buf.y_fd, span, info, 1);
		}

		drawDetections(span, info, 1);
	}

	if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_))
        throw std::runtime_error("eglMakeCurrent failed");

    glClearColor(0, 0, 0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_EXTERNAL_OES, buf.texture);
    glViewport(0, 0, width_ / 2, height_);
    SSQuad1->draw();

    glFlush();

    EGLSyncKHR sync = eglCreateSyncKHR(egl_display_, EGL_SYNC_FENCE_KHR, NULL);
    if (sync != EGL_NO_SYNC_KHR) {
        EGLint result = eglClientWaitSyncKHR(egl_display_, sync, 
                                             EGL_SYNC_FLUSH_COMMANDS_BIT_KHR, 
                                             100000000);
        if (result == EGL_TIMEOUT_EXPIRED_KHR) {
            std::cerr << "Warning: EGL sync timeout" << std::endl;
        }
        eglDestroySyncKHR(egl_display_, sync);
    }

    eglSwapBuffers(egl_display_, egl_surface_);
    
    struct gbm_bo *bo = gbm_surface_lock_front_buffer(gbm_surface);
    if (!bo) {
        throw std::runtime_error("Failed to lock front buffer");
    }

    // ===== VPU Encoding: XRGB -> YUYV  =====
    if (vpu_encoder_enabled_) {
        int gbm_fd = gbm_bo_get_fd(bo);
        if (gbm_fd < 0) {
            std::cerr << "[EGL] Failed to get GBM BO fd" << std::endl;
        } else {
            static int yuyv_dma_fd = -1;
            static bool yuyv_allocated = false;
            
            if (!yuyv_allocated) {
                int heap_fd = open("/dev/dma_heap/linux,cma", O_RDWR);
                if (heap_fd < 0) {
                    std::cerr << "[EGL] Failed to open DMA-Heap" << std::endl;
                } else {
                    struct dma_heap_allocation_data {
                        uint64_t len;
                        uint32_t fd;
                        uint32_t fd_flags;
                        uint64_t heap_flags;
                    };
                    
                    struct dma_heap_allocation_data alloc_data = {};
                    alloc_data.len = width_ * height_ * 2;  // YUYV: 2 bytes/pixel
                    alloc_data.fd = 0;
                    alloc_data.fd_flags = O_RDWR | O_CLOEXEC;
                    alloc_data.heap_flags = 0;

                    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data) == 0) {
                        yuyv_dma_fd = alloc_data.fd;
                        yuyv_allocated = true;
                        std::cout << "[EGL ShowBuffer] Allocated YUYV DMA-BUF: fd=" << yuyv_dma_fd 
                                  << ", size=" << (width_ * height_ * 2) << " bytes" << std::endl;
                    }
                    close(heap_fd);
                }
            }
            
            auto t1 = std::chrono::high_resolution_clock::now();
            
            bool g2d_success = false;
            if (yuyv_dma_fd >= 0 && g2d_scaler_) {
                g2d_success = g2d_scaler_->convertXRGB8888ToYUYV_DMABuf(
                    gbm_fd,
                    yuyv_dma_fd,
                    width_,
                    height_
                );
            }
            
            auto t2 = std::chrono::high_resolution_clock::now();
            
            if (g2d_success) {
                uint64_t timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
                
                bool dropped = false;
                {
                    std::lock_guard<std::mutex> lock(encoding_mutex_);
                    
                    if (encoding_queue_.size() >= MAX_ENCODING_QUEUE_SIZE) {
                        static int drop_count = 0;
                        if (++drop_count % 30 == 0) {
                            std::cout << "[EGL ShowBuffer] Encoding too slow, dropped " 
                                      << drop_count << " frames (queue full)" << std::endl;
                        }
                        dropped = true;
                    } else {
                        EncodingJob job;
                        job.y_fd = yuyv_dma_fd;
                        job.uv_fd = -1;
                        job.y_size = width_ * height_ * 2;
                        job.uv_size = 0;
                        job.width = width_;
                        job.height = height_;
                        job.stride = width_ * 2;     // YUYV stride
                        job.timestamp = timestamp;
                        
                        encoding_queue_.push_back(job);
                        
                        static int queue_count = 0;
                        if (++queue_count % 60 == 0) {
                            std::cout << " [EGL ShowBuffer] Queued encoding job " << queue_count 
                                      << " (queue size=" << encoding_queue_.size() 
                                      << ", yuyv_fd=" << yuyv_dma_fd << ")" << std::endl;
                        }
                    }
                }
                
                if (!dropped) {
                    encoding_cv_.notify_one();
                }
            }
            
            auto t3 = std::chrono::high_resolution_clock::now();
            
            static int perf_count = 0;
            static uint64_t total_g2d_us = 0;
            static uint64_t total_submit_us = 0;
            
            auto g2d_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
            auto submit_time = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);
            
            total_g2d_us += g2d_time.count();
            total_submit_us += submit_time.count();
            
            if (++perf_count % 300 == 0) {
                std::cout << "========================================================" << std::endl;
                std::cout << "  G2D XRGB->YUYV Zero-Copy Performance (avg 300 frames)  " << std::endl;
                std::cout << "  (ShowBuffer - Single Camera)                          " << std::endl;
                std::cout << "========================================================" << std::endl;
                printf("  G2D convert:   %6.2f ms  (DMA-BUF zero-copy)         \n", 
                       total_g2d_us / 300000.0);
                printf("  Submit job:    %6.2f ms                              \n", 
                       total_submit_us / 300000.0);
                printf("  Total:         %6.2f ms                              \n", 
                       (total_g2d_us + total_submit_us) / 300000.0);
                printf("  Max FPS:       %6.2f                                  \n", 
                       300000000.0 / (total_g2d_us + total_submit_us));
                std::cout << "========================================================" << std::endl;
                total_g2d_us = 0;
                total_submit_us = 0;
            }
            
            close(gbm_fd);
        }
    }

    uint32_t fb_id = 0;
    uint32_t *fb_ptr = (uint32_t *)gbm_bo_get_user_data(bo);
    
    if (fb_ptr) {
        fb_id = *fb_ptr;
    } else {
        uint32_t handle = gbm_bo_get_handle(bo).u32;
        uint32_t stride = gbm_bo_get_stride(bo);
        uint32_t format = gbm_bo_get_format(bo);
        
        uint32_t handles[4] = {handle, 0, 0, 0};
        uint32_t strides[4] = {stride, 0, 0, 0};
        uint32_t offsets[4] = {0, 0, 0, 0};
        
        if (drmModeAddFB2(drmfd_, width_, height_, format,
                         handles, strides, offsets, &fb_id, 0)) {
            throw std::runtime_error("drmModeAddFB2 failed");
        }
        
        fb_ptr = new uint32_t(fb_id);
        gbm_bo_set_user_data(bo, fb_ptr, 
            [](struct gbm_bo *bo, void *data) {
                uint32_t *fb = (uint32_t *)data;
                delete fb;
            });
        
        std::cout << " Created single camera framebuffer: " << fb_id << std::endl;
    }

    drmModeAtomicReq *req = drmModeAtomicAlloc();
    if (!req) {
        std::cerr << "drmModeAtomicAlloc failed" << std::endl;
        gbm_surface_release_buffer(gbm_surface, bo);
        done_callback_(buf.y_fd);
        return;
    }

    try {
        addChangedProperties(req, fb_id);

        const int max_retry = 5;
        int retry = 0;
        int commit_result = -1;

        while (retry < max_retry) {
            commit_result = drmModeAtomicCommit(drmfd_, req,
                                                DRM_MODE_ATOMIC_NONBLOCK,
                                                nullptr);
            if (commit_result == 0) {
                break;
            } else if (errno == EBUSY) {
                usleep(100 * (1 << retry));
                retry++;
            } else {
                std::cerr << "drmModeAtomicCommit failed: " << strerror(errno) << std::endl;
                break;
            }
        }

        if (retry > 0 && retry < max_retry) {
            static int retry_count = 0;
            if (++retry_count % 100 == 0) {
                std::cout << "EGL: DRM commit retried " << retry << " times (total: " 
                          << retry_count << ")" << std::endl;
            }
        }
    } catch (...) {
        drmModeAtomicFree(req);
        throw;
    }

    drmModeAtomicFree(req);

    gbm_surface_release_buffer(gbm_surface, bo);

    if (frame_cnt % 300 == 0) {
        gettimeofday(&start_time_, NULL);
        long diff = (start_time_.tv_sec - last_time_.tv_sec) * 1000 + 
                   (start_time_.tv_usec - last_time_.tv_usec) / 1000;
        printf("EGL_preview ShowBuffer (GPU Composite) frame_cnt=%d, fps=%2f\n", 
               frame_cnt, 300000.0 / diff);
        last_time_ = start_time_;
    }

    frame_cnt++;
    done_callback_(buf.y_fd);
}


void EglPreview::ShowBufferDual(libcamera::FrameBuffer* buffer1,
                               libcamera::Span<uint8_t> span1,
                               StreamInfo const &info1,
                               libcamera::FrameBuffer* buffer2,
                               libcamera::Span<uint8_t> span2,
                               StreamInfo const &info2)
{
    gettimeofday(&start_time_, NULL);
    
    Buffer &buf1 = buffers_[buffer1];
    if (buf1.y_fd == -1)
        makeBuffer(buffer1, span1.size(), info1, buf1);

    Buffer &buf2 = buffers_[buffer2];
    if (buf2.y_fd == -1)
        makeBuffer(buffer2, span2.size(), info2, buf2);

    last_buffer_ = buffer1;
    last_buffer2_ = buffer2;

	if (detector_type_ != DETECTOR_NONE) {
		const int DETECTION_INTERVAL = 2;
		if (frame_cnt % DETECTION_INTERVAL == 0) {
			submitDetectionJob(buf1.y_fd, span1, info1, 1);
		} else if (frame_cnt % DETECTION_INTERVAL == 1) {
			submitDetectionJob(buf2.y_fd, span2, info2, 2);
		}

		drawDetections(span1, info1, 1);
		drawDetections(span2, info2, 2);
	}

	if (!eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_))
        throw std::runtime_error("eglMakeCurrent failed");

    glClearColor(0, 0, 0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindTexture(GL_TEXTURE_EXTERNAL_OES, buf1.texture);
    glViewport(0, 0, width_ / 2, height_);
    SSQuad1->draw();

    glBindTexture(GL_TEXTURE_EXTERNAL_OES, buf2.texture);
    glViewport(width_ / 2, 0, width_ / 2, height_);
    SSQuad1->draw();

    glFlush();

    EGLSyncKHR sync = eglCreateSyncKHR(egl_display_, EGL_SYNC_FENCE_KHR, NULL);
    if (sync != EGL_NO_SYNC_KHR) {
        EGLint result = eglClientWaitSyncKHR(egl_display_, sync, 
                                             EGL_SYNC_FLUSH_COMMANDS_BIT_KHR, 
                                             100000000);
        if (result == EGL_TIMEOUT_EXPIRED_KHR) {
            std::cerr << "Warning: EGL sync timeout" << std::endl;
        }
        eglDestroySyncKHR(egl_display_, sync);
    }

    eglSwapBuffers(egl_display_, egl_surface_);
    
    struct gbm_bo *bo = gbm_surface_lock_front_buffer(gbm_surface);
    if (!bo) {
        throw std::runtime_error("Failed to lock front buffer");
    }

    // ===== VPU Encoding: XRGB -> YUYV =====
    if (vpu_encoder_enabled_) {
        //auto t0 = std::chrono::high_resolution_clock::now();
        
        int gbm_fd = gbm_bo_get_fd(bo);
        if (gbm_fd < 0) {
            std::cerr << "[EGL] Failed to get GBM BO fd" << std::endl;
        } else {
            static int yuyv_dma_fd = -1;
            static bool yuyv_allocated = false;
            
            if (!yuyv_allocated) {
                int heap_fd = open("/dev/dma_heap/linux,cma", O_RDWR);
                if (heap_fd < 0) {
                    std::cerr << "[EGL] Failed to open DMA-Heap" << std::endl;
                } else {
                    struct dma_heap_allocation_data {
                        uint64_t len;
                        uint32_t fd;
                        uint32_t fd_flags;
                        uint64_t heap_flags;
                    };
                    
                    struct dma_heap_allocation_data alloc_data = {};
                    alloc_data.len = width_ * height_ * 2;  // YUYV: 2 bytes/pixel
                    alloc_data.fd = 0;
                    alloc_data.fd_flags = O_RDWR | O_CLOEXEC;
                    alloc_data.heap_flags = 0;

                    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data) == 0) {
                        yuyv_dma_fd = alloc_data.fd;
                        yuyv_allocated = true;
                        std::cout << "[EGL] Allocated YUYV DMA-BUF: fd=" << yuyv_dma_fd 
                                  << ", size=" << (width_ * height_ * 2) << " bytes" << std::endl;
                    }
                    close(heap_fd);
                }
            }
            
            auto t1 = std::chrono::high_resolution_clock::now();
            
            bool g2d_success = false;
            if (yuyv_dma_fd >= 0 && g2d_scaler_) {
                g2d_success = g2d_scaler_->convertXRGB8888ToYUYV_DMABuf(
                    gbm_fd,
                    yuyv_dma_fd,
                    width_,
                    height_
                );
            }
            
            auto t2 = std::chrono::high_resolution_clock::now();
            
            if (g2d_success) {
                uint64_t timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
                
                submitEncodingJobDMABuf(
                    yuyv_dma_fd,    // YUYV DMA-BUF fd
                    width_ * height_ * 2,  // YUYV size
                    width_,
                    height_,
                    timestamp
                );
            }
            
            auto t3 = std::chrono::high_resolution_clock::now();
            
            static int perf_count = 0;
            static uint64_t total_g2d_us = 0;
            static uint64_t total_submit_us = 0;
            
            auto g2d_time = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
            auto submit_time = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);
            
            total_g2d_us += g2d_time.count();
            total_submit_us += submit_time.count();
            
            if (++perf_count % 300 == 0) {
                std::cout << "========================================================" << std::endl;
                std::cout << "  G2D XRGB->YUYV Zero-Copy Performance (avg 300 frames)  " << std::endl;
                std::cout << "========================================================" << std::endl;
                printf("  G2D convert:   %6.2f ms  (DMA-BUF zero-copy)         \n", 
                       total_g2d_us / 300000.0);
                printf("  Submit job:    %6.2f ms                              \n", 
                       total_submit_us / 300000.0);
                printf("  Total:         %6.2f ms                              \n", 
                       (total_g2d_us + total_submit_us) / 300000.0);
                printf("  Max FPS:       %6.2f                                  \n", 
                       300000000.0 / (total_g2d_us + total_submit_us));
                std::cout << "========================================================" << std::endl;
                total_g2d_us = 0;
                total_submit_us = 0;
            }
            
            close(gbm_fd);
        }
    }

    static uint32_t composite_fb = 0;
    if (composite_fb == 0) {
        uint32_t handle = gbm_bo_get_handle(bo).u32;
        uint32_t stride = gbm_bo_get_stride(bo);
        uint32_t format = gbm_bo_get_format(bo);
        
        uint32_t handles[4] = {handle, 0, 0, 0};
        uint32_t strides[4] = {stride, 0, 0, 0};
        uint32_t offsets[4] = {0, 0, 0, 0};
        
        if (drmModeAddFB2(drmfd_, width_, height_, format,
                         handles, strides, offsets, &composite_fb, 0)) {
            throw std::runtime_error("drmModeAddFB2 failed for composite");
        }
        
        std::cout << " Created composite framebuffer: " << composite_fb 
                  << " (" << width_ << "x" << height_ << ")" << std::endl;
    }

    drmModeAtomicReq *req = drmModeAtomicAlloc();
    if (!req) {
        std::cerr << "drmModeAtomicAlloc failed" << std::endl;
        gbm_surface_release_buffer(gbm_surface, bo);
        done_callback_dual_(buf1.y_fd, buf2.y_fd);
        return;
    }

    try {
        addChangedProperties(req, composite_fb);

        const int max_retry = 5;
        int retry = 0;

        while (retry < max_retry) {
            int ret = drmModeAtomicCommit(drmfd_, req,
                                          DRM_MODE_ATOMIC_NONBLOCK,
                                          nullptr);
            if (ret == 0) {
                break;
            } else if (errno == EBUSY) {
                usleep(100 * (1 << retry));
                retry++;
            } else {
                std::cerr << "drmModeAtomicCommit failed: " << strerror(errno) << std::endl;
                break;
            }
        }
    } catch (...) {
        drmModeAtomicFree(req);
        throw;
    }

    drmModeAtomicFree(req);

    gbm_surface_release_buffer(gbm_surface, bo);

    if (frame_cnt % 300 == 0) {
        long diff = (start_time_.tv_sec - last_time_.tv_sec) * 1000 + 
                   (start_time_.tv_usec - last_time_.tv_usec) / 1000;
        printf("EGL_preview ShowBufferDual (GPU Composite) frame_cnt=%d, fps=%2f\n", 
               frame_cnt, 300000.0 / diff);
        last_time_ = start_time_;
    }
    frame_cnt++;

    done_callback_dual_(buf1.y_fd, buf2.y_fd);
}


int EglPreview::acquireBufferIndex()
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

void EglPreview::releaseBuffer(int index)
{
	if (index >= 0 && index < RING_BUFFER_SIZE) {
		ring_buffers_[index].in_use = false;
	}
}

void EglPreview::copyWorker() 
{
    std::cout << "[EGL CopyWorker] Thread started (Ultra-optimized NV12->RGB)" << std::endl;
    
    std::vector<uint8_t> rgb_full_size;
    std::vector<uint8_t> rgb_resized(TARGET_WIDTH * TARGET_HEIGHT * 3);

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
                drm_buf.dma_fd,
                drm_buf.size,
                job.camera_id
            );
            
            if (!success) {
                std::cerr << "[EGL CopyWorker] G2D conversion failed for buffer " 
                          << job.buffer_index << std::endl;
                releaseBuffer(job.buffer_index);
                continue;
            }

            buffer.rgb_data_ptr = static_cast<const uint8_t*>(drm_buf.virt_addr);
            buffer.resized_width = TARGET_WIDTH;
            buffer.resized_height = TARGET_HEIGHT;
            buffer.info = job.info;
            buffer.resize_completed = true;

            auto end = std::chrono::high_resolution_clock::now();
            auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            static int copy_count = 0;
            static uint64_t total_g2d_us = 0;
            total_g2d_us += total_time.count();
            
            if (++copy_count % 300 == 0) {
                std::cout << "========================================================" << std::endl;
                std::cout << "  EGL CopyWorker Performance (G2D ZERO-COPY, avg 300)   " << std::endl;
                std::cout << "========================================================" << std::endl;
                printf("  Total time:      %6.2f ms                            \n", 
                       total_g2d_us / 300000.0);
                printf("  Resolution:      %dx%d -> %dx%d              \n",
                       job.info.width, job.info.height, TARGET_WIDTH, TARGET_HEIGHT);
                printf("  Format:          NV12 -> RGB888                       \n");
                printf("  Method:          G2D ZERO-COPY (hardware accelerated) \n");
                std::cout << "========================================================" << std::endl;
                total_g2d_us = 0;
            }

            // ===== Submit to the detection queue =====
            {
                std::lock_guard<std::mutex> lock(job_mutex_);
                DetectionJob det_job;
                det_job.buffer_index = job.buffer_index;
                det_job.camera_id = job.camera_id;
                det_job.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
                det_job.valid = true;
                job_queue_.push_back(det_job);
            }
            job_cv_.notify_one();
        }
    }
    
    std::cout << "[EGL CopyWorker] Thread stopped" << std::endl;
}

void EglPreview::submitDetectionJob(int fd, libcamera::Span<uint8_t> span, 
                                   const StreamInfo &info, int camera_id)
{
    {
        std::lock_guard<std::mutex> lock(job_mutex_);
        if (job_queue_.size() >= MAX_QUEUE_SIZE) {
            queue_full_count_++;
            return;
        }
    }
    
    int buffer_index = acquireBufferIndex();
    if (buffer_index < 0) {
        return;
    }

    auto it = buffers_.find(last_buffer_);
    if (it == buffers_.end()) {
        std::cerr << "[EGL] FrameBuffer not found in buffers_ map" << std::endl;
        releaseBuffer(buffer_index);
        return;
    }

    CopyJob job;
    job.y_fd = it->second.y_fd;
    job.uv_fd = it->second.uv_fd;
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

DetectionResult EglPreview::processImage(const uint8_t* image_data, int width, int height, int camera_id)
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

void EglPreview::detectionWorker()
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
                    std::cerr << "[EGL DetectionWorker] ERROR: Invalid rgb_data_ptr for buffer " 
                              << job.buffer_index << std::endl;
                    releaseBuffer(job.buffer_index);
                    continue;
                }

                if (!buffer.resize_completed) {
                    std::cerr << "[EGL DetectionWorker] ERROR: G2D conversion not completed for buffer " 
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
				
				// Print once every 30 times
				/*if (detections_completed_ % 30 == 0) {
					std::cout << "Camera " << job.camera_id 
							  << " detection: " << duration.count() << "ms, "
							  << "objects: " << result.detections.size()
							  << ", queue: " << job_queue_.size()
							  << std::endl;
				}*/
				
			} catch (const std::exception& e) {
				std::cerr << "Detection error: " << e.what() << std::endl;
				releaseBuffer(job.buffer_index);
			}
		}
	}
	
	std::cout << "Detection worker thread stopped" << std::endl;
}

void EglPreview::drawDetections(libcamera::Span<uint8_t> span, const StreamInfo &info, int camera_id)
{
    std::lock_guard<std::mutex> lock(result_mutex_);
    
    auto it = latest_results_.find(camera_id);
    if (it == latest_results_.end() || !it->second.valid) {
        return;
    }
    
    const auto& result = it->second;
    
    imx95::DrawConfig config;
    
    if (camera_id == 1) {
        // Camera 1:  (RGB: 0, 255, 0)
        config.box_color = 0xFF00FF00;      // ARGB: Alpha=255, R=0, G=255, B=0
        config.text_color = 0xFFFFFFFF;
        config.bg_color = 0xC0006600;
    } else {
        // Camera 2:  (RGB: 255, 0, 0)
        config.box_color = 0xFFFF0000;      // ARGB: Alpha=255, R=255, G=0, B=0
        config.text_color = 0xFFFFFFFF;
        config.bg_color = 0xC0660000;
    }
    
    config.box_thickness = 10;
    config.draw_label = false;
    config.draw_confidence = false;
    
    imx95::ImageInfo image_info;
    image_info.data = span.data();
    image_info.width = info.width;
    image_info.height = info.height;
    image_info.stride = info.stride;
    image_info.pixel_format = 1;

    if (detector_type_ == DETECTOR_SSD_CPU) {
        ssd_detector_cpu_->drawDetections(result.detections, image_info, config);
    } else {
        ssd_detector_npu_->drawDetections(result.detections, image_info, config);
    }
}


void EglPreview::Show(int fd, libcamera::Span<uint8_t> span, StreamInfo const &info)
{
    std::cerr << "Warning: Show(int fd) is deprecated, use ShowBuffer(FrameBuffer*) instead" << std::endl;
    throw std::runtime_error("Show(int fd) cannot handle NV12 with separate planes");
}

void EglPreview::ShowDual(int fd1, libcamera::Span<uint8_t> span1, StreamInfo const &info1, 
                          int fd2, libcamera::Span<uint8_t> span2, StreamInfo const &info2)
{
    std::cerr << "Warning: ShowDual(int fd) is deprecated, use ShowBufferDual(FrameBuffer*) instead" << std::endl;
    throw std::runtime_error("ShowDual(int fd) cannot handle NV12 with separate planes");
}

void EglPreview::Reset()
{
    for (auto &it : buffers_) {
        drmModeRmFB(drmfd_, it.second.fb_handle);
        glDeleteTextures(1, &it.second.texture);
        
        drm_gem_close gem_close = {};
        gem_close.handle = it.second.y_bo_handle;
        if (drmIoctl(drmfd_, DRM_IOCTL_GEM_CLOSE, &gem_close) < 0)
            LOG(1, "DRM_IOCTL_GEM_CLOSE failed for Y plane");
        
        gem_close.handle = it.second.uv_bo_handle;
        if (drmIoctl(drmfd_, DRM_IOCTL_GEM_CLOSE, &gem_close) < 0)
            LOG(1, "DRM_IOCTL_GEM_CLOSE failed for UV plane");
    }
    buffers_.clear();
    last_buffer_ = nullptr;
    eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    first_time_ = true;
}

Preview *make_egl_preview(int camera_num, int detector_type)
{
	return new EglPreview(camera_num, static_cast<DetectorType>(detector_type));
}
