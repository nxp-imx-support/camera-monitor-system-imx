#ifndef G2D_SCALER_HPP
#define G2D_SCALER_HPP

#include <cstddef>
#include <cstdint>

namespace imx95 {

class G2DScaler {
public:
    G2DScaler();
    ~G2DScaler();
    
    bool setup(int src_width, int src_height, int src_stride,
               int dst_width, int dst_height);
    
    bool scaleAndConvertNV12ToRGB_DMABuf(
        int y_dma_fd,
        int uv_dma_fd,
        size_t y_size,
        size_t uv_size,
        int dst_dma_fd,
        size_t dst_size,
        int camera_id);

    bool convertXRGB8888ToYUYV_DMABuf(
        int xrgb_dma_fd,
        int yuyv_dma_fd,
        int width,
        int height);

    bool isInitialized() const { return initialized_; }
    size_t getOutputSize() const { return dst_width_ * dst_height_ * 3; }
    
    void enableDebugDump(bool enable) { debug_dump_enabled_ = enable; }
    void setDumpInterval(int interval) { dump_interval_ = interval; }
    
    void* getG2DHandle() const { return g2d_handle_; }

private:
    void* g2d_handle_;
    int src_width_;
    int src_height_;
    int src_stride_;
    int dst_width_;
    int dst_height_;
    bool initialized_;
    
    bool debug_dump_enabled_ = false;
    int dump_interval_ = 30;
    int frame_count_ = 0;

    void* yuyv_buf_ = nullptr;

    void dumpNV12(const void* y_data, const void* uv_data, 
                  int width, int height, int stride, 
                  const char* filename);
    void dumpRGB888(const void* rgb_data, 
                    int width, int height, 
                    const char* filename);
};

} // namespace imx95

#endif // G2D_SCALER_HPP
