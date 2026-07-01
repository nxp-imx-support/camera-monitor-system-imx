/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 *
 * dirty_detector.cpp - OpenCL-based lens contamination detection
 */

#include "dirty_detector.hpp"
#include <iostream>
#include <iomanip>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>
#include <functional>

namespace imx95 {

// Complete OpenCL kernel with brightness calculation
static const char* KERNEL_SOURCE = R"CL(
__kernel void compute_grid_contrast(__global const uchar* input,
                                    __global float* grid_contrast,
                                    __global float* grid_gradient,
                                    __global float* grid_brightness,
                                    int width, int height, int stride,
                                    int grid_width, int grid_height,
                                    int cell_size)
{
    int grid_x = get_global_id(0);
    int grid_y = get_global_id(1);
    
    if (grid_x >= grid_width || grid_y >= grid_height) return;
    
    int start_x = grid_x * cell_size;
    int start_y = grid_y * cell_size;
    int end_x = min(start_x + cell_size, width);
    int end_y = min(start_y + cell_size, height);
    
    uchar min_val = 255;
    uchar max_val = 0;
    float gradient_sum = 0.0f;
    float brightness_sum = 0.0f;
    int pixel_count = 0;
    
    for (int y = start_y; y < end_y; y++) {
        for (int x = start_x; x < end_x; x++) {
            uchar val = input[y * stride + x];
            min_val = min(min_val, val);
            max_val = max(max_val, val);
            brightness_sum += val;
            
            if (x < end_x - 1 && y < end_y - 1) {
                uchar right = input[y * stride + x + 1];
                uchar down = input[(y + 1) * stride + x];
                
                int gx = abs((int)right - (int)val);
                int gy = abs((int)down - (int)val);
                
                float grad = (float)(gx + gy);
                gradient_sum += grad;
                pixel_count++;
            }
        }
    }
    
    float contrast = 0.0f;
    if (max_val + min_val > 0) {
        contrast = (float)(max_val - min_val) / (float)(max_val + min_val + 1);
    }
    
    float avg_gradient = (pixel_count > 0) ? (gradient_sum / pixel_count) : 0.0f;
    
    int total_pixels = (end_y - start_y) * (end_x - start_x);
    float avg_brightness = (total_pixels > 0) ? (brightness_sum / total_pixels) : 0.0f;
    
    int grid_idx = grid_y * grid_width + grid_x;
    grid_contrast[grid_idx] = contrast;
    grid_gradient[grid_idx] = avg_gradient;
    grid_brightness[grid_idx] = avg_brightness;
}
)CL";

DirtyDetector::DirtyDetector() 
    : platform_(nullptr), device_(nullptr), context_(nullptr), 
      queue_(nullptr), program_(nullptr),
      kernel_grid_contrast_(nullptr),
      buffer_input_(nullptr),
      buffer_grid_contrast_(nullptr),
      buffer_grid_gradient_(nullptr),
      buffer_grid_brightness_(nullptr),
      grid_size_(64), contrast_threshold_(0.15f),
      min_grid_count_(3), confidence_threshold_h_(0.5f),
      confidence_threshold_l_(0.01),
      gradient_threshold_(15.0f),
      visualization_mode_(VisualizationMode::DEFECT_ONLY),
      min_brightness_(40.0f), max_brightness_(90.0f),
      max_grid_count_(50), edge_margin_ratio_(0.15f),
      max_aspect_ratio_(3.0f), temporal_stability_frames_(5),
      width_(0), height_(0), stride_(0), 
      grid_width_(0), grid_height_(0),
      current_frame_number_(0), 
      initialized_(false)
{
}

DirtyDetector::~DirtyDetector()
{
    releaseBuffers();
    if (kernel_grid_contrast_) clReleaseKernel(kernel_grid_contrast_);
    if (program_) clReleaseProgram(program_);
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
}

bool DirtyDetector::initialize(int grid_size, float contrast_threshold)
{
    grid_size_ = grid_size;
    contrast_threshold_ = contrast_threshold;
    if (!initializeOpenCL()) {
        std::cerr << "Failed to initialize OpenCL" << std::endl;
        return false;
    }

    if (!buildKernels()) {
        std::cerr << "Failed to build OpenCL kernels" << std::endl;
        return false;
    }

    initialized_ = true;
    std::cout << "DirtyDetector initialized (grid_size=" << grid_size_ 
              << ", threshold=" << contrast_threshold_ << ")" << std::endl;
    return true;
}

bool DirtyDetector::initializeOpenCL()
{
    cl_int err;
    err = clGetPlatformIDs(1, &platform_, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "clGetPlatformIDs failed: " << err << std::endl;
        return false;
    }
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "GPU not available, trying CPU..." << std::endl;
        err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_CPU, 1, &device_, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "clGetDeviceIDs failed: " << err << std::endl;
            return false;
        }
    }
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "clCreateContext failed: " << err << std::endl;
        return false;
    }
    
    queue_ = clCreateCommandQueueWithProperties(context_, device_, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "clCreateCommandQueue failed: " << err << std::endl;
        return false;
    }
    
    return true;
}

bool DirtyDetector::buildKernels()
{
    cl_int err;
    
    program_ = clCreateProgramWithSource(context_, 1, &KERNEL_SOURCE, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "clCreateProgramWithSource failed: " << err << std::endl;
        return false;
    }
    
    err = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "clBuildProgram failed: " << err << std::endl;
        
        size_t log_size;
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Build log:\n" << log.data() << std::endl;
        
        return false;
    }
    
    kernel_grid_contrast_ = clCreateKernel(program_, "compute_grid_contrast", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "clCreateKernel failed: " << err << std::endl;
        return false;
    }
    
    return true;
}

bool DirtyDetector::allocateBuffers(int width, int height)
{
    if (width == width_ && height == height_ && buffer_input_ != nullptr) {
        return true;
    }
    
    releaseBuffers();
    
    width_ = width;
    height_ = height;
    
    grid_width_ = (width + grid_size_ - 1) / grid_size_;
    grid_height_ = (height + grid_size_ - 1) / grid_size_;
    int grid_total = grid_width_ * grid_height_;
    
    cl_int err;
    
    buffer_input_ = clCreateBuffer(context_, CL_MEM_READ_ONLY, 
                                   width * height * sizeof(uint8_t), nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create input buffer: " << err << std::endl;
        return false;
    }
    
    buffer_grid_contrast_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, 
                                          grid_total * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create contrast buffer: " << err << std::endl;
        return false;
    }
    
    buffer_grid_gradient_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, 
                                          grid_total * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create gradient buffer: " << err << std::endl;
        return false;
    }
    
    buffer_grid_brightness_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, 
                                            grid_total * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create brightness buffer: " << err << std::endl;
        return false;
    }
    
    grid_contrast_data_.resize(grid_total);
    grid_gradient_data_.resize(grid_total);
    grid_brightness_data_.resize(grid_total);
    grid_mask_data_.resize(grid_total);
    
    return true;
}

void DirtyDetector::releaseBuffers()
{
    if (buffer_input_) { 
        clReleaseMemObject(buffer_input_); 
        buffer_input_ = nullptr; 
    }
    if (buffer_grid_contrast_) { 
        clReleaseMemObject(buffer_grid_contrast_); 
        buffer_grid_contrast_ = nullptr; 
    }
    if (buffer_grid_gradient_) {
        clReleaseMemObject(buffer_grid_gradient_); 
        buffer_grid_gradient_ = nullptr; 
    }
    if (buffer_grid_brightness_) {
        clReleaseMemObject(buffer_grid_brightness_); 
        buffer_grid_brightness_ = nullptr; 
    }
}

// NEW: Calculate overlap ratio between two regions
float DirtyDetector::calculateRegionOverlap(const std::set<std::pair<int, int>>& cells1,
                                            const std::set<std::pair<int, int>>& cells2)
{
    if (cells1.empty() || cells2.empty()) {
        return 0.0f;
    }
    
    // Count intersection
    int intersection_count = 0;
    for (const auto& cell : cells1) {
        if (cells2.find(cell) != cells2.end()) {
            intersection_count++;
        }
    }
    
    // Calculate overlap ratio (based on smaller region)
    int min_size = std::min(cells1.size(), cells2.size());
    float overlap_ratio = static_cast<float>(intersection_count) / static_cast<float>(min_size);
    
    return overlap_ratio;
}

// NEW: Find matching region in history (60% overlap threshold)
int DirtyDetector::findMatchingHistoryRegion(const std::set<std::pair<int, int>>& current_cells)
{
    const float OVERLAP_THRESHOLD = 0.6f;  // 60% overlap required
    
    int best_match_id = -1;
    float best_overlap = 0.0f;
    
    for (auto& [region_id, history] : region_history_) {
        float overlap = calculateRegionOverlap(current_cells, history.grid_cells);
        
        if (overlap >= OVERLAP_THRESHOLD && overlap > best_overlap) {
            best_overlap = overlap;
            best_match_id = region_id;
        }
    }
    
    return best_match_id;
}

// NEW: Clean up old history entries
void DirtyDetector::cleanupOldHistory()
{
    const uint64_t MAX_FRAME_GAP = 10;  // Remove if not seen for 10 frames
    
    auto it = region_history_.begin();
    while (it != region_history_.end()) {
        if (current_frame_number_ - it->second.last_seen_frame > MAX_FRAME_GAP) {
            it = region_history_.erase(it);
        } else {
            ++it;
        }
    }
}

bool DirtyDetector::computeGridContrast(const uint8_t* y_plane)
{
    if (!y_plane) return false;
    
    cl_int err;
    
    std::vector<uint8_t> packed_data(width_ * height_);
    for (int y = 0; y < height_; y++) {
        std::memcpy(packed_data.data() + y * width_, 
                   y_plane + y * stride_, 
                   width_);
    }
    
    err = clEnqueueWriteBuffer(queue_, buffer_input_, CL_FALSE, 0, 
                               width_ * height_ * sizeof(uint8_t), 
                               packed_data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "clEnqueueWriteBuffer failed: " << err << std::endl;
        return false;
    }
    
    clSetKernelArg(kernel_grid_contrast_, 0, sizeof(cl_mem), &buffer_input_);
    clSetKernelArg(kernel_grid_contrast_, 1, sizeof(cl_mem), &buffer_grid_contrast_);
    clSetKernelArg(kernel_grid_contrast_, 2, sizeof(cl_mem), &buffer_grid_gradient_);
    clSetKernelArg(kernel_grid_contrast_, 3, sizeof(cl_mem), &buffer_grid_brightness_);
    clSetKernelArg(kernel_grid_contrast_, 4, sizeof(int), &width_);
    clSetKernelArg(kernel_grid_contrast_, 5, sizeof(int), &height_);
    clSetKernelArg(kernel_grid_contrast_, 6, sizeof(int), &width_);
    clSetKernelArg(kernel_grid_contrast_, 7, sizeof(int), &grid_width_);
    clSetKernelArg(kernel_grid_contrast_, 8, sizeof(int), &grid_height_);
    clSetKernelArg(kernel_grid_contrast_, 9, sizeof(int), &grid_size_);
    
    size_t global_work_size[2] = {(size_t)grid_width_, (size_t)grid_height_};
    err = clEnqueueNDRangeKernel(queue_, kernel_grid_contrast_, 2, nullptr, 
                                global_work_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "clEnqueueNDRangeKernel failed: " << err << std::endl;
        return false;
    }
    
    int grid_total = grid_width_ * grid_height_;
    err = clEnqueueReadBuffer(queue_, buffer_grid_contrast_, CL_TRUE, 0, 
                             grid_total * sizeof(float), 
                             grid_contrast_data_.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "clEnqueueReadBuffer (contrast) failed: " << err << std::endl;
        return false;
    }
    
    err = clEnqueueReadBuffer(queue_, buffer_grid_gradient_, CL_TRUE, 0, 
                             grid_total * sizeof(float), 
                             grid_gradient_data_.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "clEnqueueReadBuffer (gradient) failed: " << err << std::endl;
        return false;
    }
    
    err = clEnqueueReadBuffer(queue_, buffer_grid_brightness_, CL_TRUE, 0, 
                             grid_total * sizeof(float), 
                             grid_brightness_data_.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "clEnqueueReadBuffer (brightness) failed: " << err << std::endl;
        return false;
    }
    
    return true;
}

void DirtyDetector::generateLowContrastMask()
{
    int grid_total = grid_width_ * grid_height_;
    
    for (int i = 0; i < grid_total; i++) {
        grid_mask_data_[i] = (grid_contrast_data_[i] < contrast_threshold_) ? 1 : 0;
    }
}

std::vector<int> DirtyDetector::labelConnectedGrids(const std::vector<uint8_t>& mask)
{
    int grid_total = grid_width_ * grid_height_;
    std::vector<int> labels(grid_total, 0);
    std::vector<int> parent(grid_total);
    
    for (int i = 0; i < grid_total; i++) {
        parent[i] = i;
    }
    
    std::function<int(int)> find = [&](int x) -> int {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    };
    
    auto unite = [&](int x, int y) {
        int px = find(x);
        int py = find(y);
        if (px != py) {
            parent[px] = py;
        }
    };
    
    int next_label = 1;
    
    for (int gy = 0; gy < grid_height_; gy++) {
        for (int gx = 0; gx < grid_width_; gx++) {
            int idx = gy * grid_width_ + gx;
            
            if (mask[idx] == 0) continue;
            
            std::vector<int> neighbor_labels;
            
            if (gy > 0) {
                int top_idx = (gy - 1) * grid_width_ + gx;
                if (mask[top_idx] > 0 && labels[top_idx] > 0) {
                    neighbor_labels.push_back(labels[top_idx]);
                }
            }
            
            if (gx > 0) {
                int left_idx = gy * grid_width_ + (gx - 1);
                if (mask[left_idx] > 0 && labels[left_idx] > 0) {
                    neighbor_labels.push_back(labels[left_idx]);
                }
            }
            
            if (neighbor_labels.empty()) {
                labels[idx] = next_label++;
            } else {
                int min_label = *std::min_element(neighbor_labels.begin(), neighbor_labels.end());
                labels[idx] = min_label;
                
                for (int nl : neighbor_labels) {
                    if (nl != min_label) {
                        unite(min_label, nl);
                    }
                }
            }
        }
    }
    
    for (int i = 0; i < grid_total; i++) {
        if (labels[i] > 0) {
            labels[i] = find(labels[i]);
        }
    }
    
    return labels;
}

std::vector<DirtyDefect> DirtyDetector::extractRegions(const std::vector<int>& labels)
{
    std::vector<DirtyDefect> defects;
    std::map<int, std::vector<std::pair<int, int>>> regions;
    
    // Increment frame counter
    current_frame_number_++;
    
    for (int gy = 0; gy < grid_height_; gy++) {
        for (int gx = 0; gx < grid_width_; gx++) {
            int idx = gy * grid_width_ + gx;
            int label = labels[idx];
            if (label > 0) {
                regions[label].push_back({gx, gy});
            }
        }
    }
    
    static int extract_count = 0;
    bool should_debug = (++extract_count % 1 == 0);
    
    if (should_debug) {
        std::cout << "\n========================================================" << std::endl;
        printf("    Region Extraction Debug (Frame %d)             \n", extract_count);
        std::cout << "========================================================" << std::endl;
        printf("  Total regions found:     %5zu                         \n", regions.size());
        printf("  History regions:         %5zu                         \n", region_history_.size());
        printf("  Filter settings:                                      \n");
        printf("    Min grid count:        %5d                         \n", min_grid_count_);
        printf("    Max grid count:        %5d                         \n", max_grid_count_);
        printf("    Brightness range:      %.1f - %.1f                 \n", min_brightness_, max_brightness_);
        printf("    Gradient threshold:    %5.1f                       \n", gradient_threshold_);
        printf("    Edge margin ratio:     %5.2f                       \n", edge_margin_ratio_);
        printf("    Max aspect ratio:      %5.1f                       \n", max_aspect_ratio_);
        printf("    Temporal stability:    %5d frames (60%% overlap)   \n", temporal_stability_frames_);
        std::cout << "========================================================" << std::endl;
    }
    
    int filtered_by_size_min = 0;
    int filtered_by_size_max = 0;
    int filtered_by_brightness = 0;
    int filtered_by_gradient = 0;
    int filtered_by_edge = 0;
    int filtered_by_aspect = 0;
    int filtered_by_confidence = 0;
    int filtered_by_stability = 0;
    int filtered_by_wall_detection = 0;
    int accepted = 0;
    
    int margin_x = static_cast<int>(grid_width_ * edge_margin_ratio_);
    int margin_y = static_cast<int>(grid_height_ * edge_margin_ratio_);
    
    std::set<int> current_active_regions;  // Track which history regions are still active
    
    for (const auto& region : regions) {
        const auto& cells = region.second;
        
        if (should_debug) {
            printf("  Region %3d: %3zu grids", region.first, cells.size());
        }
        
        // Filter 1: Size (too small)
        if (cells.size() < (size_t)min_grid_count_) {
            filtered_by_size_min++;
            if (should_debug) {
                printf(" -> FILTERED (too small)         \n");
            }
            continue;
        }

        // Filter 2: Size (too large)
        if (cells.size() > (size_t)max_grid_count_) {
            filtered_by_size_max++;
            if (should_debug) {
                printf(" -> FILTERED (too large)         \n");
            }
            continue;
        }

        // Calculate bounding box
        int min_gx = grid_width_, min_gy = grid_height_;
        int max_gx = 0, max_gy = 0;
        
        for (const auto& cell : cells) {
            min_gx = std::min(min_gx, cell.first);
            max_gx = std::max(max_gx, cell.first);
            min_gy = std::min(min_gy, cell.second);
            max_gy = std::max(max_gy, cell.second);
        }
        
        // Calculate average metrics
        float avg_contrast = 0.0f;
        float avg_gradient = 0.0f;
        float avg_brightness = 0.0f;
        
        for (const auto& cell : cells) {
            int idx = cell.second * grid_width_ + cell.first;
            avg_contrast += grid_contrast_data_[idx];
            avg_gradient += grid_gradient_data_[idx];
            avg_brightness += grid_brightness_data_[idx];
        }
        avg_contrast /= cells.size();
        avg_gradient /= cells.size();
        avg_brightness /= cells.size();
        
        // Calculate shape metrics
        int bbox_width = max_gx - min_gx + 1;
        int bbox_height = max_gy - min_gy + 1;
        float aspect_ratio = static_cast<float>(std::max(bbox_width, bbox_height)) / 
                            static_cast<float>(std::min(bbox_width, bbox_height));
        
        // Check if touches edge
        bool touches_edge = (min_gx < margin_x) || 
                           (max_gx >= grid_width_ - margin_x) ||
                           (min_gy < margin_y) || 
                           (max_gy >= grid_height_ - margin_y);
        
        // Calculate confidence
        float confidence = 1.0f - (avg_contrast / contrast_threshold_);
        confidence = std::max(0.0f, std::min(1.0f, confidence));
        
        if (should_debug) {
            printf(", c=%.3f, g=%.1f, b=%.1f, ar=%.1f, edge=%d", 
                   avg_contrast, avg_gradient, avg_brightness, aspect_ratio, touches_edge ? 1 : 0);
        }
        
        // Filter 3: Wall/Ceiling Detection
        bool is_wall_type1 = (avg_brightness > 100.0f && avg_contrast < 0.08f);
        bool is_wall_type2 = (avg_brightness > 100.0f && avg_contrast < 0.11f && cells.size() > 60);
        bool is_wall_type3 = (avg_contrast < 0.06f);
        //bool is_wall_type4 = (avg_brightness < 60.0f && cells.size() < 40);
        bool is_wall_type5 = (avg_brightness > 95.0f && avg_brightness < 105.0f && 
                              avg_contrast < 0.07f && cells.size() > 50);
        
        if (is_wall_type1 || is_wall_type2 || is_wall_type3 || is_wall_type5) {
            filtered_by_wall_detection++;
            if (should_debug) {
                const char* reason = is_wall_type1 ? "wall-type1" : 
                                    is_wall_type2 ? "wall-type2" : 
                                    is_wall_type3 ? "wall-type3" :
                                    "wall-type5";
                printf(" -> FILTERED (%s)        \n", reason);
            }
            continue;
        }
        
        // Filter 4: Brightness
        if (avg_brightness < min_brightness_ || avg_brightness > max_brightness_) {
            filtered_by_brightness++;
            if (should_debug) {
                printf(" -> FILTERED (brightness)        \n");
            }
            continue;
        }
        
        // Filter 5: Gradient
        if (avg_gradient > gradient_threshold_) {
            filtered_by_gradient++;
            if (should_debug) {
                printf(" -> FILTERED (gradient)          \n");
            }
            continue;
        }
        
        // Filter 6: Edge position
        if (touches_edge) {
            filtered_by_edge++;
            if (should_debug) {
                printf(" -> FILTERED (edge)              \n");
            }
            continue;
        }
        
        // Filter 7: Aspect ratio
        if (aspect_ratio > max_aspect_ratio_) {
            filtered_by_aspect++;
            if (should_debug) {
                printf(" -> FILTERED (aspect)            \n");
            }
            continue;
        }
        
        // Filter 8: Confidence
        if (confidence < confidence_threshold_l_ || confidence > confidence_threshold_h_) {
            filtered_by_confidence++;
            if (should_debug) {
                printf(" -> FILTERED (confidence)  %f,%f,%f      \n",confidence,confidence_threshold_l_,confidence_threshold_h_);
            }
            continue;
        }
        printf(" -> 1FILTERED (confidence)  %f,%f,%f      \n",confidence,confidence_threshold_l_,confidence_threshold_h_);
        
        // Filter 9: Temporal stability (NEW ALGORITHM - 60% overlap)
        std::set<std::pair<int, int>> current_cells(cells.begin(), cells.end());
        int matching_region_id = findMatchingHistoryRegion(current_cells);
        
        int stable_frames = 0;
        float overlap_ratio = 0.0f;
        
        if (matching_region_id >= 0) {
            // Found matching region in history
            auto& history = region_history_[matching_region_id];
            history.grid_cells = current_cells;  // Update with current cells
            history.stable_frames++;
            history.last_seen_frame = current_frame_number_;
            stable_frames = history.stable_frames;
            overlap_ratio = calculateRegionOverlap(current_cells, history.grid_cells);
            current_active_regions.insert(matching_region_id);
        } else {
            // New region, create history entry
            static int next_region_id = 0;
            int new_region_id = next_region_id++;
            
            RegionHistory new_history;
            new_history.grid_cells = current_cells;
            new_history.stable_frames = 1;
            new_history.last_seen_frame = current_frame_number_;
            
            region_history_[new_region_id] = new_history;
            stable_frames = 1;
            current_active_regions.insert(new_region_id);
        }
        
        if (stable_frames < temporal_stability_frames_) {
            filtered_by_stability++;
            if (should_debug) {
                printf(" -> FILTERED (stability %d/%d, overlap=%.1f%%) \n", 
                       stable_frames, temporal_stability_frames_, overlap_ratio * 100);
            }
            continue;
        }
        
        accepted++;
        if (should_debug) {
            printf(" -> ACCEPTED  (stable %d, overlap=%.1f%%)   \n", 
                   stable_frames, overlap_ratio * 100);
        }
        
        // Create defect object
        int pixel_x = min_gx * grid_size_;
        int pixel_y = min_gy * grid_size_;
        int pixel_width = (max_gx - min_gx + 1) * grid_size_;
        int pixel_height = (max_gy - min_gy + 1) * grid_size_;
        
        pixel_width = std::min(pixel_width, width_ - pixel_x);
        pixel_height = std::min(pixel_height, height_ - pixel_y);
        
        DirtyDefect defect;
        defect.x = pixel_x;
        defect.y = pixel_y;
        defect.width = pixel_width;
        defect.height = pixel_height;
        defect.area = cells.size() * grid_size_ * grid_size_;
        defect.confidence = confidence;
        defect.grid_count = cells.size();
        defect.avg_contrast = avg_contrast;
        defect.avg_gradient = avg_gradient;
        defect.avg_brightness = avg_brightness;
        
        // Create contour
        defect.contour.push_back(ContourPoint(pixel_x, pixel_y));
        defect.contour.push_back(ContourPoint(pixel_x + pixel_width, pixel_y));
        defect.contour.push_back(ContourPoint(pixel_x + pixel_width, pixel_y + pixel_height));
        defect.contour.push_back(ContourPoint(pixel_x, pixel_y + pixel_height));
        
        // Save grid cells
        for (const auto& cell : cells) {
            GridCell grid_cell;
            grid_cell.grid_x = cell.first;
            grid_cell.grid_y = cell.second;
            grid_cell.pixel_x = cell.first * grid_size_;
            grid_cell.pixel_y = cell.second * grid_size_;
            grid_cell.width = grid_size_;
            grid_cell.height = grid_size_;
            
            int idx = cell.second * grid_width_ + cell.first;
            grid_cell.contrast = grid_contrast_data_[idx];
            grid_cell.is_low_contrast = true;
            
            defect.cells.push_back(grid_cell);
        }
        
        defects.push_back(defect);
    }
    
    // Clean up old history entries
    cleanupOldHistory();
    
    if (should_debug) {
        std::cout << "========================================================" << std::endl;
        printf("  Filter results:                                      \n");
        printf("    Too small:             %5d                         \n", filtered_by_size_min);
        printf("    Too large:             %5d                         \n", filtered_by_size_max);
        printf("    Wall detection:        %5d                         \n", filtered_by_wall_detection);
        printf("    Brightness:            %5d                         \n", filtered_by_brightness);
        printf("    Gradient:              %5d                         \n", filtered_by_gradient);
        printf("    Edge position:         %5d                         \n", filtered_by_edge);
        printf("    Aspect ratio:          %5d                         \n", filtered_by_aspect);
        printf("    Confidence:            %5d                         \n", filtered_by_confidence);
        printf("    Stability (60%% ovlp): %5d                         \n", filtered_by_stability);
        printf("    ACCEPTED:              %5d                         \n", accepted);
        std::cout << "========================================================\n" << std::endl;
    }
    
    return defects;
}


DirtyDetectionResult DirtyDetector::detectFromNV12(const uint8_t* y_plane, 
                                                    int width, int height, int stride)
{
    DirtyDetectionResult result;
    result.valid = false;
    
    if (!initialized_ || !y_plane) {
        std::cerr << "Detector not initialized or invalid input" << std::endl;
        return result;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    stride_ = stride;
    
    if (!allocateBuffers(width, height)) {
        std::cerr << "Failed to allocate buffers" << std::endl;
        return result;
    }
    
    // Use existing computeGridContrast (already handles memory copy internally)
    if (!computeGridContrast(y_plane)) {
        std::cerr << "Failed to compute grid contrast" << std::endl;
        return result;
    }
    
    generateLowContrastMask();
    std::vector<int> labels = labelConnectedGrids(grid_mask_data_);
    result.defects = extractRegions(labels);
    
    // save grid cells ...
    
    auto end = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    result.valid = true;
    result.grid_width = grid_width_;
    result.grid_height = grid_height_;
    
    current_result_ = result;
    
    return result;
}


DirtyDetectionResult DirtyDetector::detectFromDMABuf(int y_fd, size_t y_size,
                                                      int width, int height, int stride)
{
    DirtyDetectionResult result;
    result.valid = false;
    
    void* y_mapped = mmap(nullptr, y_size, PROT_READ, MAP_SHARED, y_fd, 0);
    if (y_mapped == MAP_FAILED) {
        std::cerr << "Failed to mmap Y plane: " << strerror(errno) << std::endl;
        return result;
    }
    
    const uint8_t* y_plane = static_cast<const uint8_t*>(y_mapped);
    
    result = detectFromNV12(y_plane, width, height, stride);
    
    munmap(y_mapped, y_size);
    
    return result;
}

void DirtyDetector::setGridSize(int size)
{
    if (size > 0 && size <= 256) {
        grid_size_ = size;
        releaseBuffers();
    }
}

void DirtyDetector::setContrastThreshold(float threshold)
{
    if (threshold > 0.0f && threshold < 1.0f) {
        contrast_threshold_ = threshold;
    }
}

void DirtyDetector::setMinGridCount(int count)
{
    if (count > 0) {
        min_grid_count_ = count;
    }
}

void DirtyDetector::setConfidenceThreshold(float threshold_h,float threshold_l)
{
    if (threshold_l >= 0.0f && threshold_h <= 1.0f) {
        confidence_threshold_h_ = threshold_h;
        confidence_threshold_l_ = threshold_l;
    }
}

// ===== Drawing Functions =====

uint8_t DirtyDetector::rgbToY(uint8_t r, uint8_t g, uint8_t b)
{
    int y = (66 * r + 129 * g + 25 * b + 128) / 256 + 16;
    return std::clamp(y, 16, 235);
}

void DirtyDetector::drawRectangle(uint8_t* y_plane, int width, int height, int stride,
                                  int x, int y, int w, int h, uint8_t color, int thickness)
{
    x = std::max(0, std::min(x, width - 1));
    y = std::max(0, std::min(y, height - 1));
    w = std::min(w, width - x);
    h = std::min(h, height - y);
    
    for (int t = 0; t < thickness; t++) {
        if (y + t < height) {
            for (int i = 0; i < w; i++) {
                if (x + i < width) {
                    y_plane[(y + t) * stride + x + i] = color;
                }
            }
        }
        if (y + h - 1 - t >= 0 && y + h - 1 - t < height) {
            for (int i = 0; i < w; i++) {
                if (x + i < width) {
                    y_plane[(y + h - 1 - t) * stride + x + i] = color;
                }
            }
        }
    }
    
    for (int t = 0; t < thickness; t++) {
        for (int i = 0; i < h; i++) {
            if (y + i < height) {
                if (x + t < width) {
                    y_plane[(y + i) * stride + x + t] = color;
                }
                if (x + w - 1 - t >= 0 && x + w - 1 - t < width) {
                    y_plane[(y + i) * stride + x + w - 1 - t] = color;
                }
            }
        }
    }
}

void DirtyDetector::drawFilledRectangle(uint8_t* y_plane, int width, int height, int stride,
                                        int x, int y, int w, int h, uint8_t color, float alpha)
{
    x = std::max(0, std::min(x, width - 1));
    y = std::max(0, std::min(y, height - 1));
    w = std::min(w, width - x);
    h = std::min(h, height - y);
    
    for (int dy = 0; dy < h; dy++) {
        for (int dx = 0; dx < w; dx++) {
            int px = x + dx;
            int py = y + dy;
            if (px < width && py < height) {
                uint8_t original = y_plane[py * stride + px];
                uint8_t blended = (uint8_t)(original * (1.0f - alpha) + color * alpha);
                y_plane[py * stride + px] = blended;
            }
        }
    }
}

void DirtyDetector::drawGridLines(uint8_t* y_plane, int width, int height, int stride,
                                  uint8_t color)
{
    for (int gx = 0; gx <= grid_width_; gx++) {
        int x = gx * grid_size_;
        if (x >= width) x = width - 1;
        
        for (int y = 0; y < height; y++) {
            y_plane[y * stride + x] = color;
        }
    }
    
    for (int gy = 0; gy <= grid_height_; gy++) {
        int y = gy * grid_size_;
        if (y >= height) y = height - 1;
        
        for (int x = 0; x < width; x++) {
            y_plane[y * stride + x] = color;
        }
    }
}

void DirtyDetector::drawLowContrastCells(uint8_t* y_plane, int width, int height, int stride)
{
    if (!current_result_.valid) return;
    
    for (const auto& cell : current_result_.all_cells) {
        if (cell.is_low_contrast) {
            float normalized_contrast = cell.contrast / contrast_threshold_;
            uint8_t brightness = (uint8_t)(250 - normalized_contrast * 50);
            
            drawFilledRectangle(y_plane, width, height, stride,
                              cell.pixel_x, cell.pixel_y, 
                              cell.width, cell.height,
                              brightness, 0.8f);
        }
    }
}

void DirtyDetector::drawDefectBoundingBoxes(const std::vector<DirtyDefect>& defects,
                                            uint8_t* y_plane, int width, int height, int stride)
{
    for (const auto& defect : defects) {
        uint8_t border_color;
        if (defect.confidence >= 0.8f) {
            border_color = rgbToY(255, 0, 0);
        } else if (defect.confidence >= 0.6f) {
            border_color = rgbToY(255, 165, 0);
        } else {
            border_color = rgbToY(255, 255, 0);
        }
        
        drawRectangle(y_plane, width, height, stride,
                     defect.x, defect.y, defect.width, defect.height,
                     border_color, 4);
    }
}

void DirtyDetector::drawDefectCellBorders(const std::vector<DirtyDefect>& defects,
                                          uint8_t* y_plane, int width, int height, int stride)
{
    static int debug_count = 0;
    if (++debug_count % 30 == 0) {
        std::cout << "[drawDefectCellBorders] Drawing " << defects.size() << " defects" << std::endl;
    }
    
    for (const auto& defect : defects) {
        uint8_t border_color;
        if (defect.confidence >= 0.8f) {
            border_color = rgbToY(255, 0, 0);
        } else if (defect.confidence >= 0.6f) {
            border_color = rgbToY(255, 165, 0);
        } else {
            border_color = rgbToY(255, 255, 0);
        }
        
        if (defect.cells.empty()) {
            drawRectangle(y_plane, width, height, stride,
                         defect.x, defect.y, defect.width, defect.height,
                         border_color, 2);
            continue;
        }
        
        int thickness = 2;
        
        for (const auto& cell : defect.cells) {
            int pixel_x = cell.pixel_x;
            int pixel_y = cell.pixel_y;
            int cell_w = std::min(grid_size_, width - pixel_x);
            int cell_h = std::min(grid_size_, height - pixel_y);
            
            // Top edge
            for (int t = 0; t < thickness; t++) {
                int py = pixel_y + t;
                if (py >= 0 && py < height) {
                    for (int px = pixel_x; px < pixel_x + cell_w && px < width; px++) {
                        y_plane[py * stride + px] = border_color;
                    }
                }
            }
            
            // Bottom edge
            for (int t = 0; t < thickness; t++) {
                int py = pixel_y + cell_h - 1 - t;
                if (py >= 0 && py < height) {
                    for (int px = pixel_x; px < pixel_x + cell_w && px < width; px++) {
                        y_plane[py * stride + px] = border_color;
                    }
                }
            }
            
            // Left edge
            for (int t = 0; t < thickness; t++) {
                int px = pixel_x + t;
                if (px >= 0 && px < width) {
                    for (int py = pixel_y; py < pixel_y + cell_h && py < height; py++) {
                        y_plane[py * stride + px] = border_color;
                    }
                }
            }
            
            // Right edge
            for (int t = 0; t < thickness; t++) {
                int px = pixel_x + cell_w - 1 - t;
                if (px >= 0 && px < width) {
                    for (int py = pixel_y; py < pixel_y + cell_h && py < height; py++) {
                        y_plane[py * stride + px] = border_color;
                    }
                }
            }
        }
    }
}

void DirtyDetector::drawDetections(const std::vector<DirtyDefect>& defects,
                                   uint8_t* y_plane, int width, int height, int stride,
                                   VisualizationMode mode)
{
    if (!y_plane) {
        std::cerr << "[drawDetections] y_plane is null!" << std::endl;
        return;
    }
    
    static int call_count = 0;
    if (++call_count % 30 == 0) {
        std::cout << "[drawDetections] Mode: " << (mode == VisualizationMode::DEBUG_GRID ? "DEBUG_GRID" : "DEFECT_ONLY")
                  << ", defects: " << defects.size() << std::endl;
    }
    
    if (mode == VisualizationMode::DEBUG_GRID) {
        drawLowContrastCells(y_plane, width, height, stride);
        drawGridLines(y_plane, width, height, stride, 235);
        drawDefectBoundingBoxes(defects, y_plane, width, height, stride);
    } else {
        drawDefectCellBorders(defects, y_plane, width, height, stride);
    }
}

} // namespace imx95

