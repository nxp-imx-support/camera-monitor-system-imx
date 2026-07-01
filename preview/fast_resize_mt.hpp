/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 *
 * fast_resize_mt.hpp - Multi-threaded fast image resizing utility.
 */
#ifndef FAST_RESIZE_MT_HPP
#define FAST_RESIZE_MT_HPP

#include <thread>
#include <vector>

class FastResizeMT {
private:
    std::vector<int> x_map;
    std::vector<int> y_map;
    int src_w, src_h, dst_w, dst_h, channels;
    int num_threads;

public:
    FastResizeMT(int threads = 0) {
        num_threads = threads > 0 ? threads : std::thread::hardware_concurrency();
        if (num_threads < 1) num_threads = 4;
    }
    
    void init(int src_width, int src_height, 
              int dst_width, int dst_height, int ch = 3) {
        src_w = src_width;
        src_h = src_height;
        dst_w = dst_width;
        dst_h = dst_height;
        channels = ch;
        
        x_map.resize(dst_width);
        y_map.resize(dst_height);
        
        for (int i = 0; i < dst_width; i++) {
            x_map[i] = (i * src_width / dst_width) * channels;
        }
        
        for (int i = 0; i < dst_height; i++) {
            y_map[i] = (i * src_height / dst_height) * src_width * channels;
        }
    }
    
    void resize(const uint8_t* src, uint8_t* dst) {
        std::vector<std::thread> threads;
        int rows_per_thread = (dst_h + num_threads - 1) / num_threads;
        
        for (int t = 0; t < num_threads; t++) {
            int start_row = t * rows_per_thread;
            int end_row = std::min(start_row + rows_per_thread, dst_h);
            
            if (start_row >= dst_h) break;
            
            threads.emplace_back([=]() {
                processRows(src, dst, start_row, end_row);
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
    }
    
private:
    void processRows(const uint8_t* src, uint8_t* dst, int start, int end) {
        for (int i = start; i < end; i++) {
            const uint8_t* src_row = src + y_map[i];
            uint8_t* dst_row = dst + i * dst_w * channels;
            
            if (channels == 3) {
                for (int j = 0; j < dst_w; j++) {
                    const uint8_t* sp = src_row + x_map[j];
                    uint8_t* dp = dst_row + j * 3;
                    dp[0] = sp[0];
                    dp[1] = sp[1];
                    dp[2] = sp[2];
                }
            } else if (channels == 4) {
                uint32_t* dst_ptr = (uint32_t*)dst_row;
                for (int j = 0; j < dst_w; j++) {
                    dst_ptr[j] = *((uint32_t*)(src_row + x_map[j]));
                }
            }
        }
    }
};

#endif
