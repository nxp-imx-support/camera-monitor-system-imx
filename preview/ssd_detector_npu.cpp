/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 *
 * ssd_detector_npu.cpp - NPU-based SSD object detection implementation.
 */
#include "ssd_detector.h"
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/delegates/external/external_delegate.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <map>

namespace imx95 {

// ============================================================================
// RGB to Y value lookup table (pre-calculated)
// ============================================================================
static const std::map<uint32_t, uint8_t> RGB_TO_Y_LUT_NPU = {
    {0xFFFFFFFF, 255},  // white
    {0xFF000000, 0},    // black
    {0xFFFF0000, 76},   // red (R=255, G=0, B=0)
    {0xFF00FF00, 149},  // green (R=0, G=255, B=0)
    {0xFF0000FF, 29},   // blue (R=0, G=0, B=255)
    {0xFFFFFF00, 225},  // yellow (R=255, G=255, B=0)
    {0xFFFF00FF, 105},  // magenta (R=255, G=0, B=255)
    {0xFF00FFFF, 178},  // cyan (R=0, G=255, B=255)
};

static uint8_t rgb_to_y_fast_npu(uint32_t rgb_color) {
    auto it = RGB_TO_Y_LUT_NPU.find(rgb_color);
    if (it != RGB_TO_Y_LUT_NPU.end()) {
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
static const uint8_t font_8x12_npu[][12] = {
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

// ============================================================================
// SSD Anchor Generator
// ============================================================================
SSDAnchors::SSDAnchors() {
    // SSDLite MobileNet V2 COCO configuration
    //  1917 anchors
    // 1917 = 19*19*3 + 10*10*6 + 5*5*6 + 3*3*6 + 2*2*6 + 1*1*6
    specs_ = {
        // Layer 1: 19x19, 3 boxes per location
        {19, 0.2f, {1.0f, 2.0f, 0.5f}},
        
        // Layer 2-6: 6 boxes per location
        {10, 0.35f, {1.0f, 2.0f, 0.5f, 3.0f, 0.3333f, 1.0f}},
        {5, 0.5f, {1.0f, 2.0f, 0.5f, 3.0f, 0.3333f, 1.0f}},
        {3, 0.65f, {1.0f, 2.0f, 0.5f, 3.0f, 0.3333f, 1.0f}},
        {2, 0.8f, {1.0f, 2.0f, 0.5f, 3.0f, 0.3333f, 1.0f}},
        {1, 0.95f, {1.0f, 2.0f, 0.5f, 3.0f, 0.3333f, 1.0f}},
    };
    
    generateAnchors();
    
    int expected = 19*19*3 + 10*10*6 + 5*5*6 + 3*3*6 + 2*2*6 + 1*1*6;
    std::cout << " Generated " << anchors_.size() << " anchor boxes (expected " << expected << ")" << std::endl;
    
    if (anchors_.size() != 1917) {
        std::cerr << " ERROR: Expected 1917 anchors, got " << anchors_.size() << std::endl;
        std::cerr << "   Breakdown:" << std::endl;
        for (size_t i = 0; i < specs_.size(); i++) {
            int count = specs_[i].feature_map_size * specs_[i].feature_map_size * specs_[i].aspect_ratios.size();
            std::cerr << "   Layer " << (i+1) << ": " << specs_[i].feature_map_size << "x" 
                     << specs_[i].feature_map_size << " x " << specs_[i].aspect_ratios.size() 
                     << " = " << count << std::endl;
        }
    }
}

void SSDAnchors::generateAnchors() {
    anchors_.clear();
    
    for (const auto& spec : specs_) {
        int feature_map_size = spec.feature_map_size;
        float scale = spec.scale;
        
        for (int y = 0; y < feature_map_size; y++) {
            for (int x = 0; x < feature_map_size; x++) {
                // Center point coordinates (normalized to [0, 1])
                float cx = (x + 0.5f) / feature_map_size;
                float cy = (y + 0.5f) / feature_map_size;
                
                // Generate anchors for each aspect ratio
                for (float aspect_ratio : spec.aspect_ratios) {
                    float w = scale * std::sqrt(aspect_ratio);
                    float h = scale / std::sqrt(aspect_ratio);
                    
                    // format: [cy, cx, h, w]
                    anchors_.push_back({cy, cx, h, w});
                }
            }
        }
    }
}

std::vector<std::array<float, 4>> SSDAnchors::decodeBoxes(
    const float* box_encodings,
    int num_boxes) const {
    
    if (num_boxes != static_cast<int>(anchors_.size())) {
        std::cerr << " Anchor mismatch: " << anchors_.size() 
                 << " anchors vs " << num_boxes << " boxes" << std::endl;
        return {};
    }
    
    // SSD standard decoding parameters
    const float y_scale = 10.0f;
    const float x_scale = 10.0f;
    const float h_scale = 5.0f;
    const float w_scale = 5.0f;
    
    std::vector<std::array<float, 4>> decoded_boxes;
    decoded_boxes.reserve(num_boxes);
    
    for (int i = 0; i < num_boxes; i++) {
        // Read the encoded offset [dy, dx, dh, dw]
        float dy = box_encodings[i * 4 + 0] / y_scale;
        float dx = box_encodings[i * 4 + 1] / x_scale;
        float dh = box_encodings[i * 4 + 2] / h_scale;
        float dw = box_encodings[i * 4 + 3] / w_scale;
        
        // The center point and dimensions of the anchor [cy, cx, h, w]
        float anchor_cy = anchors_[i][0];
        float anchor_cx = anchors_[i][1];
        float anchor_h = anchors_[i][2];
        float anchor_w = anchors_[i][3];
        
        // Decoding center point and size
        float cy = dy * anchor_h + anchor_cy;
        float cx = dx * anchor_w + anchor_cx;
        float h = std::exp(std::clamp(dh, -10.0f, 10.0f)) * anchor_h;
        float w = std::exp(std::clamp(dw, -10.0f, 10.0f)) * anchor_w;
        
        // Convert to corner point format [y_min, x_min, y_max, x_max]
        float y_min = std::clamp(cy - h / 2.0f, 0.0f, 1.0f);
        float x_min = std::clamp(cx - w / 2.0f, 0.0f, 1.0f);
        float y_max = std::clamp(cy + h / 2.0f, 0.0f, 1.0f);
        float x_max = std::clamp(cx + w / 2.0f, 0.0f, 1.0f);
        
        decoded_boxes.push_back({y_min, x_min, y_max, x_max});
    }
    
    return decoded_boxes;
}

// ============================================================================
// Performance statistics
// ============================================================================
void PerformanceStats::print() const {
    std::cout << "Performance Stats:" << std::endl;
    std::cout << "  Preprocess:  " << std::fixed << std::setprecision(2) 
              << preprocess_time_ms << " ms" << std::endl;
    std::cout << "  Inference:   " << inference_time_ms << " ms" << std::endl;
    std::cout << "  Postprocess: " << postprocess_time_ms << " ms" << std::endl;
    std::cout << "  Total:       " << total_time_ms << " ms" << std::endl;
}

// ============================================================================
// character map
// ============================================================================
int SSDDetectorNPU::getCharIndex(char c) {
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

// ============================================================================
// Draw a single character
// ============================================================================
void SSDDetectorNPU::drawCharacter(ImageInfo& image, char c, int x, int y, uint32_t color) {
    int char_idx = getCharIndex(c);
    if (char_idx < 0 || char_idx >= 41) return;
    
    const int char_width = 8;
    const int char_height = 12;
    
    if (image.pixel_format == 1) {
        if (image.data == nullptr) return;
        
        uint8_t y_val = rgb_to_y_fast_npu(color);
        uint8_t* y_plane = image.data;
        int stride = image.stride;
        
        for (int row = 0; row < char_height; row++) {
            uint8_t bitmap = font_8x12_npu[char_idx][row];
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
        return;
    }
    
    // XRGB8888 format
    if (image.pixel_format != 0) return;
    
    uint32_t* pixels = reinterpret_cast<uint32_t*>(image.data);
    int stride_pixels = image.stride / 4;
    
    const uint8_t* glyph = font_8x12_npu[char_idx];
    
    for (int col = 0; col < char_width; col++) {
        uint8_t column_data = glyph[col];
        
        for (int row = 0; row < char_height; row++) {
            if (column_data & (1 << row)) {
                int px = x + col;
                int py = y + row;
                
                if (px >= 0 && px < image.width && py >= 0 && py < image.height) {
                    pixels[py * stride_pixels + px] = color;
                }
            }
        }
    }
}

// ============================================================================
// draw text
// ============================================================================
void SSDDetectorNPU::drawText(ImageInfo& image,
                          const std::string& text,
                          int x, int y,
                          uint32_t text_color,
                          uint32_t bg_color) {
    if (image.data == nullptr) return;
    
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
    
    if (image.pixel_format == 1) {
        uint8_t bg_y = rgb_to_y_fast_npu(bg_color);
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
        return;
    }
    
    // XRGB8888 format
    if (image.pixel_format != 0) return;
    
    uint32_t* pixels = reinterpret_cast<uint32_t*>(image.data);
    int stride_pixels = image.stride / 4;
    
    uint8_t bg_r = (bg_color >> 16) & 0xFF;
    uint8_t bg_g = (bg_color >> 8) & 0xFF;
    uint8_t bg_b = bg_color & 0xFF;
    
    for (int py = y; py < y + text_height && py < image.height; py++) {
        for (int px = x; px < x + text_width && px < image.width; px++) {
            if (px >= 0 && py >= 0) {
                uint32_t old_color = pixels[py * stride_pixels + px];
                uint8_t old_r = (old_color >> 16) & 0xFF;
                uint8_t old_g = (old_color >> 8) & 0xFF;
                uint8_t old_b = old_color & 0xFF;
                
                uint8_t new_r = (bg_r * 3 + old_r * 1) / 4;
                uint8_t new_g = (bg_g * 3 + old_g * 1) / 4;
                uint8_t new_b = (bg_b * 3 + old_b * 1) / 4;
                
                pixels[py * stride_pixels + px] = 0xFF000000 | (new_r << 16) | (new_g << 8) | new_b;
            }
        }
    }
    
    int cursor_x = x + padding;
    int cursor_y = y + padding;
    
    for (size_t i = 0; i < text.length(); i++) {
        drawCharacter(image, text[i], cursor_x, cursor_y, text_color);
        cursor_x += char_width;
        
        if (cursor_x >= image.width - char_width) {
            break;
        }
    }
}

// ============================================================================
// draw rectangle
// ============================================================================
void SSDDetectorNPU::drawRectangle(ImageInfo& image,
                               int x, int y, int width, int height,
                               uint32_t color, int thickness) {
    if (image.data == nullptr) return;
    
    if (image.pixel_format == 1) {
        if (x < 0 || y < 0 || width <= 0 || height <= 0 ||
            x + width > image.width || y + height > image.height) {
            return;
        }
        
        uint8_t* y_plane = image.data;
        int stride = image.stride;
        
        uint8_t y_val = rgb_to_y_fast_npu(color);
        
        for (int i = 0; i < thickness; i++) {
            for (int px = x; px < x + width && px < image.width; px++) {
                int py = y + i;
                if (px >= 0 && py >= 0 && py < image.height) {
                    y_plane[py * stride + px] = y_val;
                }
            }
            
            for (int px = x; px < x + width && px < image.width; px++) {
                int py = y + height - 1 - i;
                if (px >= 0 && py >= 0 && py < image.height) {
                    y_plane[py * stride + px] = y_val;
                }
            }
            
            for (int py = y; py < y + height && py < image.height; py++) {
                int px = x + i;
                if (px >= 0 && py >= 0 && px < image.width) {
                    y_plane[py * stride + px] = y_val;
                }
            }
            
            for (int py = y; py < y + height && py < image.height; py++) {
                int px = x + width - 1 - i;
                if (px >= 0 && py >= 0 && px < image.width) {
                    y_plane[py * stride + px] = y_val;
                }
            }
        }
        return;
    }
    
    // ============================================================================
    // XRGB8888
    // ============================================================================
    if (image.pixel_format != 0) return;
    
    uint32_t* pixels = reinterpret_cast<uint32_t*>(image.data);
    int stride_pixels = image.stride / 4;
    
    if (x < 0 || y < 0 || x + width > image.width || y + height > image.height)
        return;
    
    for (int i = 0; i < thickness; i++) {
        for (int px = x; px < x + width && px < image.width; px++) {
            if (px >= 0 && y + i >= 0 && y + i < image.height)
                pixels[(y + i) * stride_pixels + px] = color;
        }
        
        for (int px = x; px < x + width && px < image.width; px++) {
            if (px >= 0 && y + height - 1 - i >= 0 && y + height - 1 - i < image.height)
                pixels[(y + height - 1 - i) * stride_pixels + px] = color;
        }
        
        for (int py = y; py < y + height && py < image.height; py++) {
            if (x + i >= 0 && py >= 0 && x + i < image.width)
                pixels[py * stride_pixels + x + i] = color;
        }
        
        for (int py = y; py < y + height && py < image.height; py++) {
            if (x + width - 1 - i >= 0 && py >= 0 && x + width - 1 - i < image.width)
                pixels[py * stride_pixels + x + width - 1 - i] = color;
        }
    }
}

// ============================================================================
// draw all detection results
// ============================================================================
void SSDDetectorNPU::drawDetections(const std::vector<Detection>& detections,
                                ImageInfo& image,
                                const DrawConfig& config) {
    if (image.data == nullptr) return;
    
    for (const auto& det : detections) {
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
// SSDDetectorNPU::Impl - TensorFlow Lite
// ============================================================================
class SSDDetectorNPU::Impl {
public:
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    TfLiteDelegate* neutron_delegate_ = nullptr;
    std::vector<std::string> labels_;
    float score_threshold_;
    float iou_threshold_;
    int num_threads_;

    std::unique_ptr<SSDAnchors> anchor_generator_;

    Impl() : score_threshold_(0.3f), iou_threshold_(0.5f), num_threads_(4) {
        anchor_generator_ = std::make_unique<SSDAnchors>();
    }

    ~Impl() {
        if (neutron_delegate_) {
            TfLiteExternalDelegateDelete(neutron_delegate_);
            neutron_delegate_ = nullptr;
        }
    }

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
    
    float calculateIoU(const std::array<float, 4>& box1, const std::array<float, 4>& box2) {
        float y1 = std::max(box1[0], box2[0]);
        float x1 = std::max(box1[1], box2[1]);
        float y2 = std::min(box1[2], box2[2]);
        float x2 = std::min(box1[3], box2[3]);
        
        float intersection = std::max(0.0f, y2 - y1) * std::max(0.0f, x2 - x1);
        float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
        float union_area = area1 + area2 - intersection;
        
        return union_area > 0 ? intersection / union_area : 0;
    }
    
    std::vector<int> nms(const std::vector<std::array<float, 4>>& boxes,
                         const std::vector<float>& scores) {
        std::vector<int> indices(scores.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        std::sort(indices.begin(), indices.end(),
                 [&scores](int a, int b) { return scores[a] > scores[b]; });
        
        std::vector<int> keep;
        std::vector<bool> suppressed(scores.size(), false);
        
        for (size_t i = 0; i < indices.size(); i++) {
            int idx = indices[i];
            if (suppressed[idx]) continue;
            
            keep.push_back(idx);
            
            for (size_t j = i + 1; j < indices.size(); j++) {
                int idx2 = indices[j];
                if (suppressed[idx2]) continue;
                
                float iou = calculateIoU(boxes[idx], boxes[idx2]);
                if (iou > iou_threshold_) {
                    suppressed[idx2] = true;
                }
            }
        }
        
        return keep;
    }
};

// ============================================================================
// SSDDetectorNPU Public Interface Implementation
// ============================================================================
SSDDetectorNPU::SSDDetectorNPU() : impl_(std::make_unique<Impl>()) {
    perf_stats_ = {0, 0, 0, 0};
}

SSDDetectorNPU::~SSDDetectorNPU() = default;

bool SSDDetectorNPU::initialize(const std::string& model_path,
                             const std::string& label_path,
                             float score_threshold,
                             float iou_threshold) {
    impl_->score_threshold_ = score_threshold;
    impl_->iou_threshold_ = iou_threshold;

    cv::setNumThreads(1);

    if (!impl_->loadLabels(label_path)) {
        return false;
    }
    
    impl_->model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!impl_->model_) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return false;
    }
    
    std::cout << " Model loaded: " << model_path << std::endl;
    
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*impl_->model_, resolver);
    builder(&impl_->interpreter_);
    
    if (!impl_->interpreter_) {
        std::cerr << "Failed to create interpreter" << std::endl;
        return false;
    }
    
    std::cout << " Interpreter created" << std::endl;
    
    impl_->num_threads_ = 2;
    impl_->interpreter_->SetNumThreads(impl_->num_threads_);
    std::cout << " Using " << impl_->num_threads_ << " CPU threads" << std::endl;

    bool use_neutron = false;
    bool force_neutron = (model_path.find("neutron") != std::string::npos);
    
    std::cout << "=== Neutron Delegate Detection ===" << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Force Neutron (from filename): " << (force_neutron ? "YES" : "NO") << std::endl;
    
    const char* neutron_env = std::getenv("USE_NEUTRON");
    std::cout << "USE_NEUTRON env: " << (neutron_env ? neutron_env : "not set") << std::endl;
    
    if ((neutron_env && std::string(neutron_env) == "1") || force_neutron) {
        const char* neutron_lib_path = "/usr/lib/libneutron_delegate.so";
        
        std::ifstream lib_check(neutron_lib_path);
        if (!lib_check.good()) {
            std::cerr << " Neutron library not found at: " << neutron_lib_path << std::endl;
            if (force_neutron) return false;
        } else {
            std::cout << " Neutron library found at: " << neutron_lib_path << std::endl;
        }
        
        std::cout << "Creating Neutron delegate..." << std::endl;
        TfLiteExternalDelegateOptions options = TfLiteExternalDelegateOptionsDefault(neutron_lib_path);
        impl_->neutron_delegate_ = TfLiteExternalDelegateCreate(&options);
        
        if (impl_->neutron_delegate_) {
            std::cout << " Neutron delegate created" << std::endl;
            std::cout << "Applying delegate to interpreter..." << std::endl;
            
            TfLiteStatus status = impl_->interpreter_->ModifyGraphWithDelegate(impl_->neutron_delegate_);
            
            std::cout << "Delegate application status: " << status << " (0=success)" << std::endl;
            
            if (status == kTfLiteOk) {
                std::cout << " NEUTRON DELEGATE APPLIED SUCCESSFULLY " << std::endl;
                std::cout << " USING NPU FOR INFERENCE " << std::endl;
                use_neutron = true;
            } else {
                std::cerr << " Failed to apply Neutron delegate (status: " << status << ")" << std::endl;
                TfLiteExternalDelegateDelete(impl_->neutron_delegate_);
                impl_->neutron_delegate_ = nullptr;
                
                if (force_neutron) {
                    std::cerr << " Model requires Neutron delegate but it failed to load" << std::endl;
                    return false;
                }
            }
        } else {
            std::cerr << " Failed to create Neutron delegate" << std::endl;
            if (force_neutron) {
                std::cerr << " Model requires Neutron delegate but library not found" << std::endl;
                return false;
            }
        }
    }
    
    if (!use_neutron) {
        std::cout << " USING CPU INFERENCE " << std::endl;
        if (force_neutron) {
            std::cout << " WARNING: Model designed for NPU but running on CPU! " << std::endl;
        } else {
            std::cout << "Set USE_NEUTRON=1 to enable NPU" << std::endl;
        }
    }
    std::cout << "===================================" << std::endl;    

    impl_->interpreter_->SetNumThreads(1);
    std::cout << " Re-set CPU threads to 1 (after delegate)" << std::endl;

    // allocate tensor
    std::cout << "Allocating tensors..." << std::endl;
    if (impl_->interpreter_->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors" << std::endl;
        return false;
    }
    
    std::cout << " Tensors allocated" << std::endl;

    impl_->interpreter_->SetNumThreads(1);
    std::cout << " Final CPU threads set to 1 (after tensor allocation)" << std::endl;
    
    std::cout << "\n=== Model Information ===" << std::endl;
    std::cout << "Input tensors: " << impl_->interpreter_->inputs().size() << std::endl;
    for (int i : impl_->interpreter_->inputs()) {
        TfLiteTensor* tensor = impl_->interpreter_->tensor(i);
        std::cout << "  Input[" << i << "]: " << tensor->name 
                  << " shape: [";
        for (int j = 0; j < tensor->dims->size; j++) {
            std::cout << tensor->dims->data[j];
            if (j < tensor->dims->size - 1) std::cout << ", ";
        }
        std::cout << "] type: " << TfLiteTypeGetName(tensor->type) << std::endl;
    }
    
    std::cout << "Output tensors: " << impl_->interpreter_->outputs().size() << std::endl;
    for (int i : impl_->interpreter_->outputs()) {
        TfLiteTensor* tensor = impl_->interpreter_->tensor(i);
        std::cout << "  Output[" << i << "]: " << tensor->name 
                  << " shape: [";
        for (int j = 0; j < tensor->dims->size; j++) {
            std::cout << tensor->dims->data[j];
            if (j < tensor->dims->size - 1) std::cout << ", ";
        }
        std::cout << "] type: " << TfLiteTypeGetName(tensor->type) << std::endl;
    }
    std::cout << "=========================" << std::endl;
    
    return true;
}

void SSDDetectorNPU::setNumThreads(int num_threads) {
    impl_->num_threads_ = num_threads;
    if (impl_->interpreter_ && !impl_->neutron_delegate_) {
        impl_->interpreter_->SetNumThreads(num_threads);
    }
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::vector<Detection> SSDDetectorNPU::detect(const uint8_t* rgb_data, int width, int height) {
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    TfLiteTensor* input_tensor = impl_->interpreter_->input_tensor(0);
    int input_height = input_tensor->dims->data[1];
    int input_width = input_tensor->dims->data[2];
    
    if (width != input_width || height != input_height) {
        std::cerr << " Size mismatch: expected " << input_width << "x" << input_height 
                  << ", got " << width << "x" << height << std::endl;
        return {};
    }
    
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    
    input_tensor->data.data = const_cast<uint8_t*>(rgb_data);
    
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    
    auto inference_start = std::chrono::high_resolution_clock::now();
    
    if (impl_->interpreter_->Invoke() != kTfLiteOk) {
        std::cerr << "Inference failed" << std::endl;
        return {};
    }
    
    auto inference_end = std::chrono::high_resolution_clock::now();

    auto postprocess_start = std::chrono::high_resolution_clock::now();
    
    const float* box_encodings = impl_->interpreter_->typed_output_tensor<float>(0);
    const float* class_predictions = impl_->interpreter_->typed_output_tensor<float>(1);
    
    TfLiteTensor* box_tensor = impl_->interpreter_->output_tensor(0);
    TfLiteTensor* class_tensor = impl_->interpreter_->output_tensor(1);
    
    int num_boxes = box_tensor->dims->data[1];
    int num_classes = class_tensor->dims->data[2];
    
    auto decoded_boxes = impl_->anchor_generator_->decodeBoxes(box_encodings, num_boxes);
    
    if (decoded_boxes.empty()) {
        return {};
    }
    
    std::vector<std::array<float, 4>> all_boxes;
    std::vector<float> all_scores;
    std::vector<int> all_class_ids;
    
    all_boxes.reserve(100);
    all_scores.reserve(100);
    all_class_ids.reserve(100);
    
    int processed_count = 0;
    const float LOGIT_THRESHOLD = 0.0f;
    
    for (int i = 0; i < num_boxes; i++) {
        const float* class_logits = &class_predictions[i * num_classes];

        float max_logit = class_logits[1];
        int max_class_id = 1;
        
        for (int c = 2; c < num_classes; c++) {
            if (class_logits[c] > max_logit) {
                max_logit = class_logits[c];
                max_class_id = c;
            }
        }
        
        if (max_logit < LOGIT_THRESHOLD) {
            continue;
        }
        
        processed_count++;
        
        all_boxes.push_back(decoded_boxes[i]);
        all_scores.push_back(sigmoid(max_logit));
        all_class_ids.push_back(max_class_id);
        
        if (all_boxes.size() >= 100) break;
    }
    
    std::vector<int> keep_indices = impl_->nms(all_boxes, all_scores);
    
    std::vector<Detection> detections;
    for (int idx : keep_indices) {
        const auto& box = all_boxes[idx];
        int class_id = all_class_ids[idx];
        float score = all_scores[idx];
        
        BBox bbox(box[1], box[0], box[3], box[2]);
        
        std::string class_name = (class_id >= 0 && class_id < static_cast<int>(impl_->labels_.size()))
                                 ? impl_->labels_[class_id]
                                 : "unknown";
        if (class_id != 82 && class_id != 72) {
            detections.emplace_back(bbox, class_id, score, class_name);
        }
    }
    
    auto postprocess_end = std::chrono::high_resolution_clock::now();
    auto total_end = std::chrono::high_resolution_clock::now();
    
    double preprocess_ms = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
    double inference_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
    double postprocess_ms = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    
    static int frame_count = 0;
    static double sum_inf=0, sum_postprocess=0, sum_total=0;
    
    sum_inf += inference_ms;
    sum_postprocess += postprocess_ms;
    sum_total += total_ms;
    
    if (++frame_count % 300 == 0) {
        std::cout << "\n=======================================================" << std::endl;
        std::cout << "     NPU Detection Performance (avg 300 frames)         " << std::endl;
        std::cout << "=======================================================" << std::endl;
        printf("  NPU Inference: %6.2f ms  (%5.1f%%)  [NPU]            \n", 
               sum_inf/300, sum_inf/sum_total*100);
        printf("  Postprocess:   %6.2f ms  (%5.1f%%)                    \n", 
               sum_postprocess/300, sum_postprocess/sum_total*100);
        std::cout << "=======================================================" << std::endl;
        printf("  TOTAL:         %6.2f ms  (%.1f FPS)                  \n", 
               sum_total/300, 300000.0/sum_total);
        std::cout << "=======================================================" << std::endl;
        printf("  Detections:    %zu objects                             \n", 
               detections.size());
        std::cout << "=======================================================\n" << std::endl;
        
        sum_inf = sum_postprocess = sum_total = 0;
    }
    
    perf_stats_.preprocess_time_ms = preprocess_ms;
    perf_stats_.inference_time_ms = inference_ms;
    perf_stats_.postprocess_time_ms = postprocess_ms;
    perf_stats_.total_time_ms = total_ms;
    
    return detections;
}

} // namespace imx95

