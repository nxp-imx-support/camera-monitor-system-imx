/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 */

#pragma once

#include <vector>
#include <map>
#include <set>
#include <cstdint>
#include <chrono>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace imx95 {

// Visualization mode
enum class VisualizationMode {
    DEFECT_ONLY = 0,      // Only draw bounding boxes around detected defects (default)
    DEBUG_GRID = 1        // Draw grid lines + low-contrast cells + defect boxes (debug mode)
};

// Grid cell structure
struct GridCell {
    int grid_x;          // Grid X coordinate
    int grid_y;          // Grid Y coordinate
    int pixel_x;         // Pixel X coordinate (top-left)
    int pixel_y;         // Pixel Y coordinate (top-left)
    int width;           // Cell width in pixels
    int height;          // Cell height in pixels
    float contrast;      // Contrast value (0.0 - 1.0)
    bool is_low_contrast; // Whether this cell has low contrast
    
    GridCell() : grid_x(0), grid_y(0), pixel_x(0), pixel_y(0), 
                 width(0), height(0), contrast(0.0f), is_low_contrast(false) {}
};

// Contour point structure
struct ContourPoint {
    int x;
    int y;
    
    ContourPoint() : x(0), y(0) {}
    ContourPoint(int x_, int y_) : x(x_), y(y_) {}
};

// Dirty defect structure (continuous low-contrast region)
struct DirtyDefect {
    int x;              // Region top-left X coordinate
    int y;              // Region top-left Y coordinate
    int width;          // Region width
    int height;         // Region height
    float area;         // Region area (pixels)
    float confidence;   // Confidence score (0.0 - 1.0)
    int grid_count;     // Number of grids in this region
    std::vector<ContourPoint> contour;  // Contour points
    float avg_contrast; // Average contrast
    float avg_gradient; // Average gradient
    float avg_brightness; // Average brightness
    std::vector<GridCell> cells;  // Grid cells in this region
};

// Detection result structure
struct DirtyDetectionResult {
    bool valid;
    std::vector<DirtyDefect> defects;
    std::vector<GridCell> all_cells;  // All grid cells (for visualization)
    double processing_time_ms;
    int camera_id;
    int grid_width;   // Number of grids horizontally
    int grid_height;  // Number of grids vertically
    
    DirtyDetectionResult() : valid(false), processing_time_ms(0.0), 
                            camera_id(0), grid_width(0), grid_height(0) {}
};

struct RegionHistory {
    std::set<std::pair<int, int>> grid_cells;  // Grid cells in this region
    int stable_frames;                          // Number of consecutive stable frames
    uint64_t last_seen_frame;                   // Last frame number this region was seen
    
    RegionHistory() : stable_frames(0), last_seen_frame(0) {}
};

class DirtyDetector {
public:
    DirtyDetector();
    ~DirtyDetector();
    
    // Initialize detector
    bool initialize(int grid_size = 64, float contrast_threshold = 0.15f);
    
    // Detect from NV12 format
    DirtyDetectionResult detectFromNV12(const uint8_t* y_plane, 
                                        int width, int height, int stride);
    
    // Detect from DMA-BUF
    DirtyDetectionResult detectFromDMABuf(int y_fd, size_t y_size,
                                          int width, int height, int stride);
    
    // Draw detections on Y plane with specified visualization mode
    void drawDetections(const std::vector<DirtyDefect>& defects,
                       uint8_t* y_plane, int width, int height, int stride,
                       VisualizationMode mode = VisualizationMode::DEFECT_ONLY);
    
    // Parameter setters
    void setGridSize(int size);
    void setContrastThreshold(float threshold);
    void setMinGridCount(int count);
    void setConfidenceThreshold(float threshold_h, float threshold_l);
    void setVisualizationMode(VisualizationMode mode) { visualization_mode_ = mode; }
    void setGradientThreshold(float threshold) { gradient_threshold_ = threshold; }
    void setTemporalStability(int frames) { temporal_stability_frames_ = frames; }
    
    // NEW: Advanced filtering parameters
    void setBrightnessRange(float min_brightness, float max_brightness) {
        min_brightness_ = min_brightness;
        max_brightness_ = max_brightness;
    }
    void setMaxGridCount(int count) { max_grid_count_ = count; }
    void setEdgeMarginRatio(float ratio) { edge_margin_ratio_ = ratio; }
    void setMaxAspectRatio(float ratio) { max_aspect_ratio_ = ratio; }
    
    // Parameter getters
    int getGridSize() const { return grid_size_; }
    float getContrastThreshold() const { return contrast_threshold_; }
    int getMinGridCount() const { return min_grid_count_; }
    VisualizationMode getVisualizationMode() const { return visualization_mode_; }
    float getGradientThreshold() const { return gradient_threshold_; }

private:
    // OpenCL resources
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    cl_program program_;
    cl_kernel kernel_grid_contrast_;
    
    cl_mem buffer_input_;
    cl_mem buffer_grid_contrast_;
    cl_mem buffer_grid_gradient_;
    cl_mem buffer_grid_brightness_;
    
    // Detection parameters
    int grid_size_;
    float contrast_threshold_;
    int min_grid_count_;
    float confidence_threshold_h_;
    float confidence_threshold_l_;
    float gradient_threshold_;
    VisualizationMode visualization_mode_;
    
    // NEW: Advanced filtering parameters
    float min_brightness_;
    float max_brightness_;
    int max_grid_count_;
    float edge_margin_ratio_;
    float max_aspect_ratio_;
    int temporal_stability_frames_;
    
    // Image information
    int width_;
    int height_;
    int stride_;
    int grid_width_;
    int grid_height_;
    
    // Data cache
    std::vector<float> grid_contrast_data_;
    std::vector<float> grid_gradient_data_;
    std::vector<float> grid_brightness_data_;
    std::vector<uint8_t> grid_mask_data_;
    
    // Temporal stability tracking
    std::map<int, int> region_stability_counter_;

    // REPLACE: Temporal stability tracking
    std::map<int, RegionHistory> region_history_;
    uint64_t current_frame_number_;
    
    // NEW: Helper methods for temporal stability
    int findMatchingHistoryRegion(const std::set<std::pair<int, int>>& current_cells);
    float calculateRegionOverlap(const std::set<std::pair<int, int>>& cells1,
                                  const std::set<std::pair<int, int>>& cells2);
    void cleanupOldHistory();

    // Current detection result
    DirtyDetectionResult current_result_;
    
    bool initialized_;
    
    // Internal methods
    bool initializeOpenCL();
    bool buildKernels();
    bool allocateBuffers(int width, int height);
    void releaseBuffers();
    bool computeGridContrast(const uint8_t* y_plane);
    void generateLowContrastMask();
    std::vector<int> labelConnectedGrids(const std::vector<uint8_t>& mask);
    std::vector<DirtyDefect> extractRegions(const std::vector<int>& labels);
    
    // NEW: Helper for temporal stability
    int calculateRegionHash(int min_gx, int min_gy, int max_gx, int max_gy) const {
        return (min_gx << 24) | (min_gy << 16) | (max_gx << 8) | max_gy;
    }
    
    // Drawing helper functions
    void drawGridLines(uint8_t* y_plane, int width, int height, int stride, uint8_t color = 255);
    void drawLowContrastCells(uint8_t* y_plane, int width, int height, int stride);
    void drawDefectBoundingBoxes(const std::vector<DirtyDefect>& defects,
                                 uint8_t* y_plane, int width, int height, int stride);
    void drawDefectCellBorders(const std::vector<DirtyDefect>& defects,
                                uint8_t* y_plane, int width, int height, int stride);
    void drawRectangle(uint8_t* y_plane, int width, int height, int stride,
                      int x, int y, int w, int h, uint8_t color, int thickness = 2);
    void drawFilledRectangle(uint8_t* y_plane, int width, int height, int stride,
                            int x, int y, int w, int h, uint8_t color, float alpha = 0.5f);
    
    static uint8_t rgbToY(uint8_t r, uint8_t g, uint8_t b);
};

} // namespace imx95
