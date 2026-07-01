/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 *
 * ssd_detector.h - SSD object detection interface
 */
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

#define NPU_MODEL "ssdlite_mobilenet_v2_coco_quant_uint8_float32_no_postprocess_neutron.tflite"
#define NPU_LABEL "coco_labels_list_npu.txt"

#define CPU_MODEL "ssd_mobilenet_v1_quant.tflite"
#define CPU_LABEL "coco_labels_list_cpu.txt"

namespace imx95 {

// bounding box structure
struct BBox {
    float x1, y1, x2, y2;
    
    BBox() : x1(0), y1(0), x2(0), y2(0) {}
    BBox(float _x1, float _y1, float _x2, float _y2)
        : x1(_x1), y1(_y1), x2(_x2), y2(_y2) {}
};

struct Detection {
    BBox bbox;
    int class_id;
    float score;
    std::string class_name;
    
    Detection() : class_id(0), score(0.0f) {}
    Detection(const BBox& box, int id, float s, const std::string& name)
        : bbox(box), class_id(id), score(s), class_name(name) {}
};

// Detection result structure
struct DetectionResult {
	std::vector<imx95::Detection> detections;
	int camera_id;
	uint64_t timestamp;
	bool valid;
	
	DetectionResult() : camera_id(0), timestamp(0), valid(false) {}
};

struct PerformanceStats {
    double preprocess_time_ms;
    double inference_time_ms;
    double postprocess_time_ms;
    double total_time_ms;
    
    void print() const;
};

struct DrawConfig {
    uint32_t box_color;      // border color
    uint32_t text_color;     // text color
    uint32_t bg_color;       // background color
    int box_thickness;       // Border thickness
    bool draw_label;         // Whether to draw labels
    bool draw_confidence;    // Whether to display confidence level
    
    DrawConfig()
        : box_color(0xFF00FF00)
        , text_color(0xFFFFFFFF)
        , bg_color(0xC0006600)
        , box_thickness(3)
        , draw_label(true)
        , draw_confidence(true)
    {}
};

struct ImageInfo {
    uint8_t* data;
    int width;
    int height;
    int stride;
    int pixel_format;  // 1=YUV420
    
    ImageInfo()
        : data(nullptr), width(0), height(0), stride(0), pixel_format(0)
    {}
};

class SSDDetectorCPU {
public:
    SSDDetectorCPU();
    ~SSDDetectorCPU();

    bool initialize(const std::string& model_path,
                   const std::string& label_path,
                   float score_threshold = 0.5f,
                   float iou_threshold = 0.5f);
    
    void setNumThreads(int num_threads);
    
    std::vector<Detection> detect(const uint8_t* image_data, int width, int height);
    
    const PerformanceStats& getPerformanceStats() const { return perf_stats_; }
    
    // Plotting the detection results to the image buffer
    void drawDetections(const std::vector<Detection>& detections,
                       ImageInfo& image,
                       const DrawConfig& config = DrawConfig());
    
    static void drawRectangle(ImageInfo& image,
                             int x, int y, int width, int height,
                             uint32_t color, int thickness = 3);
    
    static void drawText(ImageInfo& image,
                        const std::string& text,
                        int x, int y,
                        uint32_t text_color,
                        uint32_t bg_color);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    PerformanceStats perf_stats_;
    
    static int getCharIndex(char c);
    static void drawCharacter(ImageInfo& image, char c, int x, int y, uint32_t color);
};

class SSDAnchors {
public:
    struct AnchorSpec {
        int feature_map_size;
        float scale;
        std::vector<float> aspect_ratios;
    };
    
    SSDAnchors();
    
    const std::vector<std::array<float, 4>>& getAnchors() const {
        return anchors_;
    }
    
    std::vector<std::array<float, 4>> decodeBoxes(
        const float* box_encodings,
        int num_boxes) const;

private:
    std::vector<AnchorSpec> specs_;
    std::vector<std::array<float, 4>> anchors_;  // [cy, cx, h, w]
    
    void generateAnchors();
};

class SSDDetectorNPU {
public:
    SSDDetectorNPU();
    ~SSDDetectorNPU();
    
    bool initialize(const std::string& model_path,
                   const std::string& label_path,
                   float score_threshold = 0.3f,
                   float iou_threshold = 0.5f);
    
    void setNumThreads(int num_threads);
    
    std::vector<Detection> detect(const uint8_t* image_data, int width, int height);
    
    const PerformanceStats& getPerformanceStats() const { return perf_stats_; }
    
    void drawDetections(const std::vector<Detection>& detections,
                       ImageInfo& image,
                       const DrawConfig& config = DrawConfig());
    
    static void drawRectangle(ImageInfo& image,
                             int x, int y, int width, int height,
                             uint32_t color, int thickness = 3);
    
    static void drawText(ImageInfo& image,
                        const std::string& text,
                        int x, int y,
                        uint32_t text_color,
                        uint32_t bg_color);

    struct AnchorSpec {
        int feature_map_size;
        float scale;
        std::vector<float> aspect_ratios;
    };

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    PerformanceStats perf_stats_;
    
    static int getCharIndex(char c);
    static void drawCharacter(ImageInfo& image, char c, int x, int y, uint32_t color);
    void preprocessOptimized(const uint8_t* image_data, int width, int height);
    std::vector<Detection> postprocessOptimized();
};

} // namespace imx95
