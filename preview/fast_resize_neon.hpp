/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 *
 * fast_resize_neon.hpp - NEON-optimized fast image resizing with threading.
 */
#ifndef FAST_RESIZE_NEON_HPP
#define FAST_RESIZE_NEON_HPP

#include <arm_neon.h>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

class FastResizeNEON {
private:
    std::vector<int> x_map;
    std::vector<int> y_map;
    int src_w, src_h, dst_w, dst_h;
    
    // Thread pool
    std::thread worker_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> has_job_{false};
    std::atomic<bool> job_done_{false};
    
    const uint8_t* job_src_ = nullptr;
    uint8_t* job_dst_ = nullptr;
    int job_start_y_ = 0;
    int job_end_y_ = 0;

public:
    FastResizeNEON() {
        running_ = true;
        worker_thread_ = std::thread([this]() {
            while (running_.load(std::memory_order_acquire)) {
                while (!has_job_.load(std::memory_order_acquire)) {
                    if (!running_.load(std::memory_order_acquire)) return;
                    std::this_thread::yield();
                }

                processRows(job_src_, job_dst_, job_start_y_, job_end_y_);

                has_job_.store(false, std::memory_order_release);
                job_done_.store(true, std::memory_order_release);
            }
        });
    }
    
    ~FastResizeNEON() {
        running_.store(false, std::memory_order_release);
        has_job_.store(true, std::memory_order_release);
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
    
    void init(int src_width, int src_height, 
              int dst_width, int dst_height, int channels = 4) {
        src_w = src_width;
        src_h = src_height;
        dst_w = dst_width;
        dst_h = dst_height;
        
        x_map.resize(dst_width);
        y_map.resize(dst_height);
        
        for (int i = 0; i < dst_width; i++) {
            x_map[i] = (i * src_width / dst_width) * 4;
        }
        
        for (int i = 0; i < dst_height; i++) {
            y_map[i] = (i * src_height / dst_height) * src_width * 4;
        }
    }
    
    void processRows(const uint8_t* src, uint8_t* dst_rgb, int start_y, int end_y) {
        for (int y = start_y; y < end_y; y++) {
            const uint8_t* src_row = src + y_map[y];
            uint8_t* dst_row = dst_rgb + y * dst_w * 3;
            
            int x = 0;
            
            for (; x <= dst_w - 16; x += 16) {
                __builtin_prefetch(src_row + x_map[x + 16], 0, 0);
                
                const uint8_t* sp0 = src_row + x_map[x + 0];
                const uint8_t* sp1 = src_row + x_map[x + 1];
                const uint8_t* sp2 = src_row + x_map[x + 2];
                const uint8_t* sp3 = src_row + x_map[x + 3];
                const uint8_t* sp4 = src_row + x_map[x + 4];
                const uint8_t* sp5 = src_row + x_map[x + 5];
                const uint8_t* sp6 = src_row + x_map[x + 6];
                const uint8_t* sp7 = src_row + x_map[x + 7];
                const uint8_t* sp8 = src_row + x_map[x + 8];
                const uint8_t* sp9 = src_row + x_map[x + 9];
                const uint8_t* sp10 = src_row + x_map[x + 10];
                const uint8_t* sp11 = src_row + x_map[x + 11];
                const uint8_t* sp12 = src_row + x_map[x + 12];
                const uint8_t* sp13 = src_row + x_map[x + 13];
                const uint8_t* sp14 = src_row + x_map[x + 14];
                const uint8_t* sp15 = src_row + x_map[x + 15];
                
                uint8_t* dp = dst_row + x * 3;
                
                dp[0]  = sp0[2]; dp[1]  = sp0[1]; dp[2]  = sp0[0];
                dp[3]  = sp1[2]; dp[4]  = sp1[1]; dp[5]  = sp1[0];
                dp[6]  = sp2[2]; dp[7]  = sp2[1]; dp[8]  = sp2[0];
                dp[9]  = sp3[2]; dp[10] = sp3[1]; dp[11] = sp3[0];
                dp[12] = sp4[2]; dp[13] = sp4[1]; dp[14] = sp4[0];
                dp[15] = sp5[2]; dp[16] = sp5[1]; dp[17] = sp5[0];
                dp[18] = sp6[2]; dp[19] = sp6[1]; dp[20] = sp6[0];
                dp[21] = sp7[2]; dp[22] = sp7[1]; dp[23] = sp7[0];
                dp[24] = sp8[2]; dp[25] = sp8[1]; dp[26] = sp8[0];
                dp[27] = sp9[2]; dp[28] = sp9[1]; dp[29] = sp9[0];
                dp[30] = sp10[2]; dp[31] = sp10[1]; dp[32] = sp10[0];
                dp[33] = sp11[2]; dp[34] = sp11[1]; dp[35] = sp11[0];
                dp[36] = sp12[2]; dp[37] = sp12[1]; dp[38] = sp12[0];
                dp[39] = sp13[2]; dp[40] = sp13[1]; dp[41] = sp13[0];
                dp[42] = sp14[2]; dp[43] = sp14[1]; dp[44] = sp14[0];
                dp[45] = sp15[2]; dp[46] = sp15[1]; dp[47] = sp15[0];
            }
            
            for (; x < dst_w; x++) {
                const uint8_t* sp = src_row + x_map[x];
                uint8_t* dp = dst_row + x * 3;
                dp[0] = sp[2];
                dp[1] = sp[1];
                dp[2] = sp[0];
            }
        }
    }

    // Optimized dual-threading
    void resizeAndConvert(const uint8_t* src, uint8_t* dst_rgb) {
        int mid_y = dst_h / 2;

        job_src_ = src;
        job_dst_ = dst_rgb;
        job_start_y_ = mid_y;
        job_end_y_ = dst_h;
        job_done_.store(false, std::memory_order_release);
        has_job_.store(true, std::memory_order_release);

        processRows(src, dst_rgb, 0, mid_y);

        while (!job_done_.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }
};

#endif
