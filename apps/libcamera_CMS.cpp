/* SPDX-License-Identifier: BSD-2-Clause */
/*
 * Copyright 2025-2026 NXP
 * Copyright (C) 2020, Raspberry Pi (Trading) Ltd.
 *
 */

#include <signal.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <getopt.h>

#include "core/libcamera_app.hpp"

using namespace std::placeholders;

std::atomic<bool> g_should_exit(false);

void handle_sigint(int signal) {
    if (signal == SIGINT) {
        std::cout << "\n Ctrl+C received, exiting..." << std::endl;
        g_should_exit = true;
    }
}

struct CMSConfig {
    int camera_num = 1;
    int preview_type = 0;
    int detector_type = 0;
    bool enable_recording = false;
    std::string output_file = "output.mp4";
    std::string codec = "h264";
    int bitrate = 8000000;
    bool enable_scaling = false;
    int scale_width = 640;
    int scale_height = 480;
    bool verbose = false;
};

void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("\nCamera Options:\n");
    printf("  -n, --num-cameras <N>      Number of cameras (1 or 2, default: 1)\n");
    printf("  -p, --preview <TYPE>       Preview type (0=EGL, 1=DRM, default: 0)\n");
    printf("  -d, --detector <TYPE>      Detector type:\n");
    printf("                               0 = None (default)\n");
    printf("                               1 = CPU detection (SSD MobileNet)\n");
    printf("                               2 = NPU detection (SSDLite)\n");
    printf("                               3 = Dirty detection\n");
    printf("\nRecording Options:\n");
    printf("  -r, --record               Enable video recording\n");
    printf("  -o, --output <FILE>        Output file (default: output.mp4)\n");
    printf("                             For dual camera: output_cam1.mp4, output_cam2.mp4\n");
    printf("  -c, --codec <CODEC>        Video codec (h264/h265, default: h264)\n");
    printf("  -b, --bitrate <RATE>       Bitrate in bps (default: 8000000)\n");
    printf("\nOther Options:\n");
    printf("  -v, --verbose              Enable verbose output\n");
    printf("  --help                     Display this help message\n");
    printf("\nExamples:\n");
    printf("  # Single camera, EGL preview, no detection\n");
    printf("  %s -n 1 -p 0\n", program_name);
    printf("\n  # Single camera with CPU detection\n");
    printf("  %s -n 1 -d 1\n", program_name);
    printf("\n  # Dual camera with NPU detection\n");
    printf("  %s -n 2 -d 2\n", program_name);
    printf("\n  # Single camera with recording\n");
    printf("  %s -n 1 -r -o video.mp4\n", program_name);
    printf("\n  # Dirty detection on single camera\n");
    printf("  %s -n 1 -d 3\n", program_name);
}

bool parse_arguments(int argc, char *argv[], CMSConfig &config) {
    static struct option long_options[] = {
        {"num-cameras", required_argument, 0, 'n'},
        {"preview",     required_argument, 0, 'p'},
        {"detector",    required_argument, 0, 'd'},
        {"record",      no_argument,       0, 'r'},
        {"output",      required_argument, 0, 'o'},
        {"codec",       required_argument, 0, 'c'},
        {"bitrate",     required_argument, 0, 'b'},
        {"scale",       no_argument,       0, 's'},
        {"width",       required_argument, 0, 'w'},
        {"height",      required_argument, 0, 'H'},
        {"verbose",     no_argument,       0, 'v'},
        {"help",        no_argument,       0, '?'},
        {0, 0, 0, 0}
    };
    
    int option_index = 0;
    int c;
    
    while ((c = getopt_long(argc, argv, "n:p:d:ro:c:b:sw:H:v", long_options, &option_index)) != -1) {
        switch (c) {
            case 'n':
                config.camera_num = std::stoi(optarg);
                if (config.camera_num < 1 || config.camera_num > 2) {
                    fprintf(stderr, "ERROR: Number of cameras must be 1 or 2\n");
                    return false;
                }
                break;
                
            case 'p':
                config.preview_type = std::stoi(optarg);
                if (config.preview_type < 0 || config.preview_type > 1) {
                    fprintf(stderr, "ERROR: Preview type must be 0 (EGL) or 1 (DRM)\n");
                    return false;
                }
                break;
                
            case 'd':
                config.detector_type = std::stoi(optarg);
                if (config.detector_type < 0 || config.detector_type > 3) {
                    fprintf(stderr, "ERROR: Detector type must be 0-3\n");
                    return false;
                }
                break;
                
            case 'r':
                config.enable_recording = true;
                break;
                
            case 'o':
                config.output_file = optarg;
                break;
                
            case 'c':
                config.codec = optarg;
                if (config.codec != "h264" && config.codec != "h265") {
                    fprintf(stderr, "ERROR: Codec must be h264 or h265\n");
                    return false;
                }
                break;
                
            case 'b':
                config.bitrate = std::stoi(optarg);
                if (config.bitrate <= 0) {
                    fprintf(stderr, "ERROR: Bitrate must be positive\n");
                    return false;
                }
                break;
                
            case 's':
                config.enable_scaling = true;
                break;
                
            case 'w':
                config.scale_width = std::stoi(optarg);
                if (config.scale_width <= 0) {
                    fprintf(stderr, "ERROR: Width must be positive\n");
                    return false;
                }
                break;
                
            case 'H':
                config.scale_height = std::stoi(optarg);
                if (config.scale_height <= 0) {
                    fprintf(stderr, "ERROR: Height must be positive\n");
                    return false;
                }
                break;
                
            case 'v':
                config.verbose = true;
                break;
                
            case '?':
            default:
                return false;
        }
    }
    
    return true;
}

void print_config(const CMSConfig &config) {
    printf("\n========================================\n");
    printf("  CMS Configuration\n");
    printf("========================================\n");
    printf("  Cameras:       %d\n", config.camera_num);
    printf("  Preview:       %s\n", config.preview_type == 0 ? "EGL" : "DRM");
    printf("  Detector:      %s\n", 
           config.detector_type == 0 ? "None" : 
           (config.detector_type == 1 ? "SSD CPU" : 
           (config.detector_type == 2 ? "SSD NPU" : "Dirty Detection")));
    
    if (config.enable_recording) {
        printf("  Recording:     Enabled\n");
        printf("    Output:      %s\n", config.output_file.c_str());
        if (config.camera_num == 2) {
            size_t dot_pos = config.output_file.find_last_of('.');
            std::string base = config.output_file.substr(0, dot_pos);
            std::string ext = config.output_file.substr(dot_pos);
            printf("                 %s_cam1%s\n", base.c_str(), ext.c_str());
            printf("                 %s_cam2%s\n", base.c_str(), ext.c_str());
        }
        printf("    Codec:       %s\n", config.codec.c_str());
        printf("    Bitrate:     %d bps\n", config.bitrate);
    } else {
        printf("  Recording:     Disabled\n");
    }
    
    /*if (config.enable_scaling) {
        printf("  GPU Scaling:   Enabled\n");
        printf("    Resolution:  %dx%d\n", config.scale_width, config.scale_height);
    } else {
        printf("  GPU Scaling:   Disabled\n");
    }*/
    
    printf("  Verbose:       %s\n", config.verbose ? "Yes" : "No");
    printf("========================================\n\n");
}

// The main event loop for the application.
static void event_loop(LibcameraApp &app, const CMSConfig &config)
{
    app.setCameraNum(config.camera_num);
    app.OpenCamera(config.camera_num, config.preview_type, config.detector_type);
    app.ConfigureViewfinder();

    // Configure recording and scaling if enabled
    if (config.enable_recording) {
        app.enableVPUEncoder(config.codec, config.bitrate, config.output_file);
    }
    
    if (config.enable_scaling) {
        app.enableGPUScaler(config.scale_width, config.scale_height);
    }
    app.StartCamera();
    LOG(2, "Camera started with " << config.camera_num << " camera(s)");

    while (!g_should_exit)
    {
        LibcameraApp::Msg msg = app.Wait_camera1();
        if (msg.type == LibcameraApp::MsgType::Timeout)
        {
            LOG_ERROR("ERROR: Device timeout detected, attempting a restart!!!");
            app.StopCamera();
            app.StartCamera();
            continue;
        }
        if (msg.type == LibcameraApp::MsgType::Quit){
            return;
        }
        else if (msg.type == LibcameraApp::MsgType::RequestComplete1){
            // Camera 1 frame ready
        } else if(msg.type == LibcameraApp::MsgType::RequestComplete2){
            // Camera 2 frame ready
        } else {
            throw std::runtime_error("unrecognised message!");
        }
        if(config.camera_num > 1) {
            LibcameraApp::Msg msg2 = app.Wait_camera2();
            if (msg2.type == LibcameraApp::MsgType::Timeout)
            {
                LOG_ERROR("ERROR: Device timeout detected, attempting a restart!!!");
                app.StopCamera();
                app.StartCamera();
                continue;
            }
            if (msg2.type == LibcameraApp::MsgType::Quit){
                return;
            }
            else if (msg2.type == LibcameraApp::MsgType::RequestComplete1){
                // Camera 1 frame ready
            } else if(msg2.type == LibcameraApp::MsgType::RequestComplete2){
                // Camera 2 frame ready
            } else {
                throw std::runtime_error("unrecognised message!");
            }
            CompletedRequestPtr &completed_request = std::get<CompletedRequestPtr>(msg.payload);
            CompletedRequestPtr &completed_request2 = std::get<CompletedRequestPtr>(msg2.payload);

            app.ShowPreviewDual(completed_request, completed_request2, 
                              app.ViewfinderStream(), app.Viewfinder2Stream());
        } else {
            CompletedRequestPtr &completed_request = std::get<CompletedRequestPtr>(msg.payload);
            app.ShowPreview(completed_request, app.ViewfinderStream());
        }
    }

    app.StopCamera();
    app.Teardown();
    app.CloseCamera();
}

int main(int argc, char *argv[])
{
    struct sigaction sa;
    sa.sa_handler = handle_sigint;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, nullptr);

    CMSConfig config;
    
    // Parse command line arguments
    if (!parse_arguments(argc, argv, config)) {
        print_usage(argv[0]);
        return -1;
    }
    
    // Print configuration
    print_config(config);

    try
    {
        LibcameraApp app;
        
        if (config.verbose) {
            LibcameraApp::verbosity = 2;
        }
        
        event_loop(app, config);
    }
    catch (std::exception const &e)
    {
        printf("ERROR: *** %s ***\n", e.what());
        return -1;
    }
    
    return 0;
}
