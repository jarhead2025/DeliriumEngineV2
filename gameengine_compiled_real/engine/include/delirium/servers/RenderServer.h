#pragma once
#include "delirium/core/RID.h"
#include <CL/cl.h>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace DeliRium {

// Godot-like RenderServer: owns OpenCL context (Intel-preferred) and frame FB.
// Core concept: sparse hits on equation surfaces → pattern fill (no tris as source).
class RenderServer {
public:
    static RenderServer& get_singleton();

    bool init();
    void finish();

    // Framebuffer size for host present path
    bool set_viewport_size(int width, int height);

    // Register a PSF-like body as raw coeff blob (host side for G2/G3)
    RID body_create();
    void body_set_coeffs(RID body, const float* data, size_t count);
    void body_free(RID body);

    // One frame: intersect → hit pixels → pattern average fill → readback RGBA8
    bool draw();
    const std::vector<uint32_t>& get_framebuffer() const { return fb_host_; }
    int width() const { return width_; }
    int height() const { return height_; }

    cl_context cl_context_() const { return ctx_; }
    cl_command_queue cl_queue_() const { return queue_; }

private:
    RenderServer() = default;
    bool init_opencl_intel();
    bool ensure_buffers();
    bool run_clear_and_demo_fill();

    bool ready_ = false;
    int width_ = 64;
    int height_ = 64;

    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context ctx_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_mem fb_cl_ = nullptr;

    std::vector<uint32_t> fb_host_;
    uint64_t next_id_ = 1;
    std::unordered_map<uint64_t, std::vector<float>> bodies_;
};

} // namespace DeliRium
