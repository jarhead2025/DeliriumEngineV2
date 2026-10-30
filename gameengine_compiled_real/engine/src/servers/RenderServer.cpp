// FILEPATH: engine/src/servers/RenderServer.cpp
// OpenCL RenderServer — Intel platform preferred; host demo fill for G1–G3.
#include "delirium/servers/RenderServer.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace DeliRium {

RenderServer& RenderServer::get_singleton() {
    static RenderServer s;
    return s;
}

bool RenderServer::init_opencl_intel() {
    cl_int err = CL_SUCCESS;
    cl_uint nplat = 0;
    err = clGetPlatformIDs(0, nullptr, &nplat);
    if (err != CL_SUCCESS || nplat == 0) return false;
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);

    platform_ = plats[0];
    for (cl_platform_id p : plats) {
        char name[256] = {};
        clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        if (std::strstr(name, "Intel") || std::strstr(name, "intel")) {
            platform_ = p;
            break;
        }
    }

    cl_uint ndev = 0;
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_, &ndev);
    if (err != CL_SUCCESS || ndev == 0) {
        err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, 1, &device_, &ndev);
        if (err != CL_SUCCESS || ndev == 0) return false;
    }

    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform_), 0};
    ctx_ = clCreateContext(props, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !ctx_) return false;

    queue_ = clCreateCommandQueue(ctx_, device_, 0, &err);
    if (err != CL_SUCCESS || !queue_) return false;
    return true;
}

bool RenderServer::init() {
    if (ready_) return true;
    if (!init_opencl_intel()) return false;
    if (!set_viewport_size(width_, height_)) return false;
    ready_ = true;
    return true;
}

void RenderServer::finish() {
    if (fb_cl_) {
        clReleaseMemObject(fb_cl_);
        fb_cl_ = nullptr;
    }
    if (queue_) {
        clReleaseCommandQueue(queue_);
        queue_ = nullptr;
    }
    if (ctx_) {
        clReleaseContext(ctx_);
        ctx_ = nullptr;
    }
    ready_ = false;
}

bool RenderServer::ensure_buffers() {
    if (!ctx_) return false;
    if (fb_cl_) {
        clReleaseMemObject(fb_cl_);
        fb_cl_ = nullptr;
    }
    cl_int err = CL_SUCCESS;
    const size_t bytes =
        static_cast<size_t>(width_) * static_cast<size_t>(height_) * sizeof(uint32_t);
    fb_cl_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, bytes, nullptr, &err);
    if (err != CL_SUCCESS || !fb_cl_) return false;
    fb_host_.assign(static_cast<size_t>(width_) * static_cast<size_t>(height_), 0u);
    return true;
}

bool RenderServer::set_viewport_size(int width, int height) {
    if (width < 1 || height < 1) return false;
    width_ = width;
    height_ = height;
    if (!ctx_) return true;
    return ensure_buffers();
}

RID RenderServer::body_create() {
    RID r;
    r.id = next_id_++;
    bodies_[r.id] = {};
    return r;
}

void RenderServer::body_set_coeffs(RID body, const float* data, size_t count) {
    if (!body.is_valid()) return;
    auto it = bodies_.find(body.id);
    if (it == bodies_.end()) return;
    it->second.assign(data, data + count);
}

void RenderServer::body_free(RID body) {
    bodies_.erase(body.id);
}

bool RenderServer::run_clear_and_demo_fill() {
    if (!queue_ || !fb_cl_) return false;
    // Core concept demo: treat non-zero coeffs as "surface energy" → tint fill.
    // Real path (later): OpenCL kernels intersect equation surfaces → sparse hits → pattern fill.
    float energy = 0.f;
    for (const auto& kv : bodies_) {
        for (float c : kv.second) energy += c * c;
    }
    const float e = energy > 0.f ? (energy < 4.f ? energy : 4.f) : 0.25f;
    const uint8_t r = static_cast<uint8_t>(40 + 40 * e);
    const uint8_t g = static_cast<uint8_t>(80 + 30 * e);
    const uint8_t b = static_cast<uint8_t>(160 + 20 * e);
    const uint32_t pixel = (0xFFu << 24) | (b << 16) | (g << 8) | r; // RGBA8888-ish host

    // Sparse "hits": stamp a few pixels (equation sample stand-in), then average neighborhood.
    std::fill(fb_host_.begin(), fb_host_.end(), 0xFF101010u);
    const int hits = 32;
    for (int i = 0; i < hits; ++i) {
        const int x = (width_ / 4 + (i * 17) % (width_ / 2));
        const int y = (height_ / 4 + (i * 13) % (height_ / 2));
        if (x >= 0 && x < width_ && y >= 0 && y < height_) {
            fb_host_[static_cast<size_t>(y * width_ + x)] = pixel;
        }
    }
    // Pattern fill: box average through neighbors (host stand-in for fill kernel)
    std::vector<uint32_t> out = fb_host_;
    for (int y = 1; y < height_ - 1; ++y) {
        for (int x = 1; x < width_ - 1; ++x) {
            unsigned ar = 0, ag = 0, ab = 0, n = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    uint32_t p = fb_host_[static_cast<size_t>((y + dy) * width_ + (x + dx))];
                    ar += p & 0xFF;
                    ag += (p >> 8) & 0xFF;
                    ab += (p >> 16) & 0xFF;
                    ++n;
                }
            }
            ar /= n;
            ag /= n;
            ab /= n;
            out[static_cast<size_t>(y * width_ + x)] =
                (0xFFu << 24) | (ab << 16) | (ag << 8) | ar;
        }
    }
    fb_host_.swap(out);

    cl_int err = clEnqueueWriteBuffer(
        queue_, fb_cl_, CL_TRUE, 0,
        fb_host_.size() * sizeof(uint32_t), fb_host_.data(), 0, nullptr, nullptr);
    return err == CL_SUCCESS;
}

bool RenderServer::draw() {
    if (!ready_ && !init()) return false;
    if (!ensure_buffers()) return false;
    return run_clear_and_demo_fill();
}

} // namespace DeliRium
