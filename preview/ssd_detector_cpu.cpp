/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 *
 * ssd_detector_cpu.cpp - CPU-based SSD object detection implementation.
 */
#include "ssd_detector.h"
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <map>

namespace imx95 {

// ============================================================================
// RGB to Y value lookup table (pre-calculated)
// ============================================================================
static const std::map<uint32_t, uint8_t> RGB_TO_Y_LUT = {
    {0xFFFFFFFF, 255},  // white
    {0xFF000000, 0},    // black
    {0xFFFF0000, 76},   // red (R=255, G=0, B=0)
    {0xFF00FF00, 149},  // green (R=0, G=255, B=0)
    {0xFF0000FF, 29},   // blue (R=0, G=0, B=255)
    {0xFFFFFF00, 225},  // yellow (R=255, G=255, B=0)
    {0xFFFF00FF, 105},  // magenta (R=255, G=0, B=255)
    {0xFF00FFFF, 178},  // cyan (R=0, G=255, B=255)
};

static uint8_t rgb_to_y_fast(uint32_t rgb_color) {
    auto it = RGB_TO_Y_LUT.find(rgb_color);
    if (it != RGB_TO_Y_LUT.end()) {
        return it->second;
    }
    
    uint8_t r = (rgb_color >> 16) & 0xFF;
    uint8_t g = (rgb_color >> 8) & 0xFF;
    uint8_t b = rgb_color & 0xFF;
    
    return static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);
}

// ============================================================================
// Font data (8x12 bitmap for larger text)
// ============================================================================
static const uint8_t font_8x12[][12] = {
    // A (0)
    {0x00, 0x00, 0x18, 0x24, 0x24, 0x42, 0x42, 0x7E, 0x42, 0x42, 0x42, 0x00},
    // B (1)
    {0x00, 0x00, 0x7C, 0x42, 0x42, 0x7C, 0x42, 0x42, 0x42, 0x42, 0x7C, 0x00},
    // C (2)
    {0x00, 0x00, 0x3C, 0x42, 0x42, 0x40, 0x40, 0x40, 0x42, 0x42, 0x3C, 0x00},
    // D (3)
    {0x00, 0x00, 0x78, 0x44, 0x42, 0x42, 0x42, 0x42, 0x42, 0x44, 0x78, 0x00},
    // E (4)
    {0x00, 0x00, 0x7E, 0x40, 0x40, 0x40, 0x7C, 0x40, 0x40, 0x40, 0x7E, 0x00},
    // F (5)
    {0x00, 0x00, 0x7E, 0x40, 0x40, 0x40, 0x7C, 0x40, 0x40, 0x40, 0x40, 0x00},
    // G (6)
    {0x00, 0x00, 0x3C, 0x42, 0x40, 0x40, 0x4E, 0x42, 0x42, 0x46, 0x3A, 0x00},
    // H (7)
    {0x00, 0x00, 0x42, 0x42, 0x42, 0x42, 0x7E, 0x42, 0x42, 0x42, 0x42, 0x00},
    // I (8)
    {0x00, 0x00, 0x3E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x3E, 0x00},
    // J (9)
    {0x00, 0x00, 0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x44, 0x44, 0x38, 0x00},
    // K (10)
    {0x00, 0x00, 0x42, 0x44, 0x48, 0x50, 0x60, 0x50, 0x48, 0x44, 0x42, 0x00},
    // L (11)
    {0x00, 0x00, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x7E, 0x00},
    // M (12)
    {0x00, 0x00, 0x42, 0x66, 0x5A, 0x5A, 0x42, 0x42, 0x42, 0x42, 0x42, 0x00},
    // N (13)
    {0x00, 0x00, 0x42, 0x62, 0x52, 0x52, 0x4A, 0x4A, 0x46, 0x46, 0x42, 0x00},
    // O (14)
    {0x00, 0x00, 0x3C, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00},
    // P (15)
    {0x00, 0x00, 0x7C, 0x42, 0x42, 0x42, 0x7C, 0x40, 0x40, 0x40, 0x40, 0x00},
    // Q (16)
    {0x00, 0x00, 0x3C, 0x42, 0x42, 0x42, 0x42, 0x42, 0x5A, 0x66, 0x3C, 0x03},
    // R (17)
    {0x00, 0x00, 0x7C, 0x42, 0x42, 0x42, 0x7C, 0x48, 0x44, 0x44, 0x42, 0x00},
    // S (18)
    {0x00, 0x00, 0x3C, 0x42, 0x40, 0x40, 0x3C, 0x02, 0x02, 0x42, 0x3C, 0x00},
    // T (19)
    {0x00, 0x00, 0x7F, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x00},
    // U (20)
    {0x00, 0x00, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00},
    // V (21)
    {0x00, 0x00, 0x41, 0x41, 0x41, 0x22, 0x22, 0x22, 0x14, 0x14, 0x08, 0x00},
    // W (22)
    {0x00, 0x00, 0x42, 0x42, 0x42, 0x42, 0x5A, 0x5A, 0x66, 0x66, 0x42, 0x00},
    // X (23)
    {0x00, 0x00, 0x42, 0x42, 0x24, 0x24, 0x18, 0x24, 0x24, 0x42, 0x42, 0x00},
    // Y (24)
    {0x00, 0x00, 0x41, 0x41, 0x22, 0x22, 0x14, 0x08, 0x08, 0x08, 0x08, 0x00},
    // Z (25)
    {0x00, 0x00, 0x7E, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x40, 0x7E, 0x00},
    // 0 (26)
    {0x00, 0x00, 0x3C, 0x42, 0x46, 0x4A, 0x52, 0x62, 0x42, 0x42, 0x3C, 0x00},
    // 1 (27)
    {0x00, 0x00, 0x08, 0x18, 0x28, 0x08, 0x08, 0x08, 0x08, 0x08, 0x3E, 0x00},
    // 2 (28)
    {0x00, 0x00, 0x3C, 0x42, 0x02, 0x02, 0x0C, 0x30, 0x40, 0x40, 0x7E, 0x00},
    // 3 (29)
    {0x00, 0x00, 0x3C, 0x42, 0x02, 0x02, 0x1C, 0x02, 0x02, 0x42, 0x3C, 0x00},
    // 4 (30)
    {0x00, 0x00, 0x04, 0x0C, 0x14, 0x24, 0x44, 0x7E, 0x04, 0x04, 0x04, 0x00},
    // 5 (31)
    {0x00, 0x00, 0x7E, 0x40, 0x40, 0x7C, 0x02, 0x02, 0x02, 0x42, 0x3C, 0x00},
    // 6 (32)
    {0x00, 0x00, 0x1C, 0x20, 0x40, 0x40, 0x7C, 0x42, 0x42, 0x42, 0x3C, 0x00},
    // 7 (33)
    {0x00, 0x00, 0x7E, 0x02, 0x04, 0x08, 0x10, 0x10, 0x10, 0x10, 0x10, 0x00},
    // 8 (34)
    {0x00, 0x00, 0x3C, 0x42, 0x42, 0x42, 0x3C, 0x42, 0x42, 0x42, 0x3C, 0x00},
    // 9 (35)
    {0x00, 0x00, 0x3C, 0x42, 0x42, 0x42, 0x3E, 0x02, 0x02, 0x04, 0x38, 0x00},
    // space (36)
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
    // - (37)
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00, 0x00},
    // . (38)
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00},
    // : (39)
    {0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x00, 0x00},
    // % (40)
    {0x00, 0x00, 0x62, 0x64, 0x08, 0x10, 0x10, 0x20, 0x26, 0x46, 0x00, 0x00},
};

int SSDDetectorCPU::getCharIndex(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c >= '0' && c <= '9') return c - '0' + 26;
    
    switch (c) {
        case ' ': return 36;
        case '-': return 37;
        case '.': return 38;
        case ':': return 39;
        case '%': return 40;
        default: return 36;
    }
}

void SSDDetectorCPU::drawRectangle(ImageInfo& image,
                                   int x, int y, int width, int height,
                                   uint32_t color, int thickness) {
    if (image.pixel_format != 1 || image.data == nullptr) return;
    
    if (x < 0 || y < 0 || width <= 0 || height <= 0 ||
        x + width > image.width || y + height > image.height) {
        return;
    }
    
    uint8_t* y_plane = image.data;
    int stride = image.stride;
    
    uint8_t y_val = rgb_to_y_fast(color);
    
    // Only draw Y plane (grayscale)
    for (int i = 0; i < thickness; i++) {
        // Top and bottom
        for (int px = x; px < x + width && px < image.width; px++) {
            int py = y + i;
            if (px >= 0 && py >= 0 && py < image.height) {
                y_plane[py * stride + px] = y_val;
            }
            
            py = y + height - 1 - i;
            if (px >= 0 && py >= 0 && py < image.height) {
                y_plane[py * stride + px] = y_val;
            }
        }
        
        // Left and right
        for (int py = y; py < y + height && py < image.height; py++) {
            int px = x + i;
            if (px >= 0 && py >= 0 && px < image.width) {
                y_plane[py * stride + px] = y_val;
            }
            
            px = x + width - 1 - i;
            if (px >= 0 && py >= 0 && px < image.width) {
                y_plane[py * stride + px] = y_val;
            }
        }
    }
}

void SSDDetectorCPU::drawCharacter(ImageInfo& image, char c, int x, int y, uint32_t color) {
    if (image.data == nullptr || image.pixel_format != 1) return;
    
    int char_idx = getCharIndex(c);
    if (char_idx < 0 || char_idx >= 41) return;
    
    const int char_width = 8;
    const int char_height = 12;
    
    uint8_t y_val = rgb_to_y_fast(color);
    
    uint8_t* y_plane = image.data;
    int stride = image.stride;
    
    for (int row = 0; row < char_height; row++) {
        uint8_t bitmap = font_8x12[char_idx][row];
        for (int col = 0; col < char_width; col++) {
            if (bitmap & (0x80 >> col)) {
                int px = x + col;
                int py = y + row;
                
                if (px >= 0 && px < image.width && py >= 0 && py < image.height) {
                    y_plane[py * stride + px] = y_val;
                }
            }
        }
    }
}

void SSDDetectorCPU::drawText(ImageInfo& image,
                              const std::string& text,
                              int x, int y,
                              uint32_t text_color,
                              uint32_t bg_color) {
    if (image.data == nullptr || image.pixel_format != 1) return;
    
    const int char_width = 8;
    const int char_height = 12;
    const int padding = 2;
    
    int text_width = text.length() * char_width + padding * 2;
    int text_height = char_height + padding * 2;
    
    // Clamp to image bounds
    if (y < 0) y = 0;
    if (x < 0) x = 0;
    if (y + text_height > image.height) y = image.height - text_height;
    if (x + text_width > image.width) x = image.width - text_width;
    
    uint8_t bg_y = rgb_to_y_fast(bg_color);
    
    uint8_t* y_plane = image.data;
    int stride = image.stride;
    
    // Draw background
    for (int py = y; py < y + text_height && py < image.height; py++) {
        for (int px = x; px < x + text_width && px < image.width; px++) {
            if (px >= 0 && py >= 0) {
                y_plane[py * stride + px] = bg_y;
            }
        }
    }
    
    // Draw text
    for (size_t i = 0; i < text.length(); i++) {
        drawCharacter(image, text[i], x + padding + i * char_width, y + padding, text_color);
    }
}

void SSDDetectorCPU::drawDetections(const std::vector<Detection>& detections,
                                    ImageInfo& image,
                                    const DrawConfig& config) {
    if (image.data == nullptr || image.pixel_format != 1) {
        return;
    }

    for (size_t idx = 0; idx < detections.size(); idx++) {
        const auto& det = detections[idx];
        
        int x = static_cast<int>(det.bbox.x1 * image.width);
        int y = static_cast<int>(det.bbox.y1 * image.height);
        int w = static_cast<int>((det.bbox.x2 - det.bbox.x1) * image.width);
        int h = static_cast<int>((det.bbox.y2 - det.bbox.y1) * image.height);
        
        if (x < 0 || y < 0 || w <= 0 || h <= 0 || 
            x + w > image.width || y + h > image.height) {
            continue;
        }
        
        // Draw bounding box
        drawRectangle(image, x, y, w, h, config.box_color, config.box_thickness);
        
        // Draw label and confidence
        if (config.draw_label || config.draw_confidence) {
            std::string label_text;
            
            if (config.draw_label) {
                label_text = det.class_name;
            }
            
            if (config.draw_confidence) {
                char conf_str[16];
                snprintf(conf_str, sizeof(conf_str), " %.0f%%", det.score * 100);
                label_text += conf_str;
            }
            
            // Draw text above the box
            int text_y = y - 18;
            if (text_y < 0) text_y = y + h + 2;
            
            drawText(image, label_text, x, text_y, config.text_color, config.bg_color);
        }
    }
}

// ============================================================================
// SSDDetectorCPU::Impl - TensorFlow Lite
// ============================================================================
class SSDDetectorCPU::Impl {
public:
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    std::vector<std::string> labels_;
    float score_threshold_;
    float iou_threshold_;
    int num_threads_;
    
    Impl() : score_threshold_(0.5f), iou_threshold_(0.5f), num_threads_(4) {}
    
    bool loadLabels(const std::string& label_path) {
        std::ifstream file(label_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open label file: " << label_path << std::endl;
            return false;
        }
        
        labels_.clear();
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            labels_.push_back(line);
        }
        
        std::cout << "Loaded " << labels_.size() << " labels" << std::endl;
        return !labels_.empty();
    }
    
    float calculateIoU(const BBox& box1, const BBox& box2) {
        float x1 = std::max(box1.x1, box2.x1);
        float y1 = std::max(box1.y1, box2.y1);
        float x2 = std::min(box1.x2, box2.x2);
        float y2 = std::min(box1.y2, box2.y2);
        
        float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
        float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
        float union_area = area1 + area2 - intersection;
        
        return union_area > 0 ? intersection / union_area : 0;
    }
    
    std::vector<Detection> nms(std::vector<Detection>& detections) {
        std::sort(detections.begin(), detections.end(),
                 [](const Detection& a, const Detection& b) {
                     return a.score > b.score;
                 });
        
        std::vector<Detection> result;
        std::vector<bool> suppressed(detections.size(), false);
        
        for (size_t i = 0; i < detections.size(); i++) {
            if (suppressed[i]) continue;
            
            result.push_back(detections[i]);
            
            for (size_t j = i + 1; j < detections.size(); j++) {
                if (suppressed[j]) continue;
                
                if (detections[i].class_id == detections[j].class_id) {
                    float iou = calculateIoU(detections[i].bbox, detections[j].bbox);
                    if (iou > iou_threshold_) {
                        suppressed[j] = true;
                    }
                }
            }
        }
        
        return result;
    }
};

SSDDetectorCPU::SSDDetectorCPU() : impl_(std::make_unique<Impl>()) {
    perf_stats_ = {0, 0, 0, 0};
}

SSDDetectorCPU::~SSDDetectorCPU() = default;

bool SSDDetectorCPU::initialize(const std::string& model_path,
                             const std::string& label_path,
                             float score_threshold,
                             float iou_threshold) {
    impl_->score_threshold_ = score_threshold;
    impl_->iou_threshold_ = iou_threshold;
    
    if (!impl_->loadLabels(label_path)) {
        return false;
    }
    
    impl_->model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!impl_->model_) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return false;
    }
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*impl_->model_, resolver);
    builder(&impl_->interpreter_);
    
    if (!impl_->interpreter_) {
        std::cerr << "Failed to create interpreter" << std::endl;
        return false;
    }
    
    impl_->interpreter_->SetNumThreads(impl_->num_threads_);
    
    if (impl_->interpreter_->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        return false;
    }
    
    std::cout << "Model initialized successfully" << std::endl;
    return true;
}

void SSDDetectorCPU::setNumThreads(int num_threads) {
    impl_->num_threads_ = num_threads;
    if (impl_->interpreter_) {
        impl_->interpreter_->SetNumThreads(num_threads);
    }
}

std::vector<Detection> SSDDetectorCPU::detect(const uint8_t* image_data, int width, int height) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    if (!image_data) {
        return {};
    }
    
    if (!impl_->interpreter_) {
        return {};
    }
    
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    
    cv::Mat image(height, width, CV_8UC3, const_cast<uint8_t*>(image_data));
    
    if (image.empty()) {
        return {};
    }
    
    TfLiteTensor* input_tensor = impl_->interpreter_->input_tensor(0);
    int input_height = input_tensor->dims->data[1];
    int input_width = input_tensor->dims->data[2];
    
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_width, input_height));
    
    uint8_t* input_data = impl_->interpreter_->typed_input_tensor<uint8_t>(0);
    std::memcpy(input_data, resized.data, resized.total() * resized.elemSize());
    
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    perf_stats_.preprocess_time_ms = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
    
    auto inference_start = std::chrono::high_resolution_clock::now();
    
    if (impl_->interpreter_->Invoke() != kTfLiteOk) {
        return {};
    }
    
    auto inference_end = std::chrono::high_resolution_clock::now();
    perf_stats_.inference_time_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
    
    auto postprocess_start = std::chrono::high_resolution_clock::now();
    
    const float* detection_boxes = impl_->interpreter_->typed_output_tensor<float>(0);
    const float* detection_classes = impl_->interpreter_->typed_output_tensor<float>(1);
    const float* detection_scores = impl_->interpreter_->typed_output_tensor<float>(2);
    const float* num_detections = impl_->interpreter_->typed_output_tensor<float>(3);
    
    int num_det = static_cast<int>(num_detections[0]);
    
    std::vector<Detection> detections;
    
    for (int i = 0; i < num_det; i++) {
        float score = detection_scores[i];
        if (score < impl_->score_threshold_) continue;
        
        int class_id = static_cast<int>(detection_classes[i]);
        
        float ymin = detection_boxes[i * 4 + 0];
        float xmin = detection_boxes[i * 4 + 1];
        float ymax = detection_boxes[i * 4 + 2];
        float xmax = detection_boxes[i * 4 + 3];
        
        BBox bbox(xmin, ymin, xmax, ymax);
        
        std::string class_name = (class_id >= 0 && class_id < static_cast<int>(impl_->labels_.size())) 
                                ? impl_->labels_[class_id] 
                                : "unknown";
        
        detections.emplace_back(bbox, class_id, score, class_name);
    }
    
    detections = impl_->nms(detections);
    
    auto postprocess_end = std::chrono::high_resolution_clock::now();
    perf_stats_.postprocess_time_ms = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
    
    auto total_end = std::chrono::high_resolution_clock::now();
    perf_stats_.total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    
    return detections;
}

} // namespace imx95
