// FILEPATH: engine/src/render/SplatRenderer.cpp
// E4 product path: device_name, CL framebuffer clear, body buffer upload, Present readback.
// Load path (PSFFile/PSFBody/FromFile) is never modified by this scaffold.
#include "delirium/render/SplatRenderer.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/core/Types.h"
#include <CL/cl.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <cstdio>

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
#include <SDL.h>
#endif

namespace DeliRium {

namespace {
inline cl_context as_ctx(void* p) { return static_cast<cl_context>(p); }
inline cl_command_queue as_q(void* p) { return static_cast<cl_command_queue>(p); }
inline cl_device_id as_dev(void* p) { return static_cast<cl_device_id>(p); }
inline cl_mem as_mem(void* p) { return static_cast<cl_mem>(p); }
} // namespace

bool SplatRenderer::Initialize(int width, int height, void* sdl_renderer,
                               const std::string& kernel_dir) {
    (void)kernel_dir;
    this->width = width;
    this->height = height;
    this->sdl_rnd = sdl_renderer;
    this->device_name.clear();

    cl_int err = CL_SUCCESS;
    cl_uint nplat = 0;
    err = clGetPlatformIDs(0, nullptr, &nplat);
    if (err != CL_SUCCESS || nplat == 0) return false;
    std::vector<cl_platform_id> plats(nplat);
    clGetPlatformIDs(nplat, plats.data(), nullptr);

    cl_platform_id platform = plats[0];
    for (cl_platform_id p : plats) {
        char name[256] = {};
        clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        if (std::strstr(name, "Intel") || std::strstr(name, "intel")) {
            platform = p;
            break;
        }
    }

    cl_device_id device = nullptr;
    cl_uint ndev = 0;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &ndev);
    if (err != CL_SUCCESS || ndev == 0) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &ndev);
        if (err != CL_SUCCESS || ndev == 0) return false;
    }
    cl_dev = static_cast<void*>(device);
    {
        char dname[256] = {};
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
        device_name = dname;
        std::printf("SplatRenderer: OpenCL device '%s'\n", device_name.c_str());
    }

    cl_context_properties props[] = {
        CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform), 0};
    cl_context ctx = clCreateContext(props, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !ctx) return false;
    cl_ctx = static_cast<void*>(ctx);

    cl_command_queue q = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS || !q) {
        clReleaseContext(ctx);
        cl_ctx = nullptr;
        return false;
    }
    cl_queue = static_cast<void*>(q);

    if (buf_fb) {
        clReleaseMemObject(as_mem(buf_fb));
        buf_fb = nullptr;
    }
    const size_t fb_bytes =
        static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(uint32_t);
    cl_mem fb = clCreateBuffer(ctx, CL_MEM_READ_WRITE, fb_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !fb) return false;
    buf_fb = static_cast<void*>(fb);

    // Clear to dark blue so Present has visible content even before full splat kernel
    std::vector<uint32_t> clear(static_cast<size_t>(width) * static_cast<size_t>(height),
                                0xFF203040u);
    clEnqueueWriteBuffer(q, fb, CL_TRUE, 0, fb_bytes, clear.data(), 0, nullptr, nullptr);

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    if (sdl_rnd) {
        SDL_Renderer* rnd = static_cast<SDL_Renderer*>(sdl_rnd);
        if (sdl_tex) {
            SDL_DestroyTexture(static_cast<SDL_Texture*>(sdl_tex));
            sdl_tex = nullptr;
        }
        SDL_Texture* tex = SDL_CreateTexture(
            rnd, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, width, height);
        sdl_tex = static_cast<void*>(tex);
    }
#endif
    return true;
}

uint32_t SplatRenderer::BuildSampleGrids(PSFBody& body, const float* /*mvp*/) {
    (void)body;
    return 0u;
}

bool SplatRenderer::BuildProgram(const std::string& kernel_dir) {
    (void)kernel_dir;
    // Full psf_splat.cl load is a later slice; host/CL buffer path is enough for E4.
    return true;
}

bool SplatRenderer::RenderBody(PSFBody& body, const Camera& /*cam*/) {
    if (!cl_ctx || !cl_queue || !buf_fb || width <= 0 || height <= 0) return false;

    const auto& anchors = body.AnchorBlob();
    const auto& surfaces = body.SurfaceBlob();
    cl_context ctx = as_ctx(cl_ctx);
    cl_command_queue q = as_q(cl_queue);
    cl_int err = CL_SUCCESS;

    // Upload (or refresh) body records for future kernel use
    if (!anchors.empty()) {
        if (buf_anchors) {
            clReleaseMemObject(as_mem(buf_anchors));
            buf_anchors = nullptr;
        }
        cl_mem a = clCreateBuffer(
            ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            anchors.size() * sizeof(AnchorRecord),
            const_cast<AnchorRecord*>(anchors.data()), &err);
        if (err == CL_SUCCESS && a) {
            buf_anchors = static_cast<void*>(a);
            cap_anchors = static_cast<uint32_t>(anchors.size());
        }
    }
    if (!surfaces.empty()) {
        if (buf_surfaces) {
            clReleaseMemObject(as_mem(buf_surfaces));
            buf_surfaces = nullptr;
        }
        cl_mem s = clCreateBuffer(
            ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            surfaces.size() * sizeof(SurfaceRecord),
            const_cast<SurfaceRecord*>(surfaces.data()), &err);
        if (err == CL_SUCCESS && s) {
            buf_surfaces = static_cast<void*>(s);
            cap_surfaces = static_cast<uint32_t>(surfaces.size());
        }
    }

    // Host-side sparse "hit" proxy: paint a few pixels from anchor count so FB is non-empty
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint32_t> fb(n, 0xFF102030u);
    const uint32_t hits = static_cast<uint32_t>(
        std::min<size_t>(anchors.size(), static_cast<size_t>(width * height)));
    for (uint32_t i = 0; i < hits; ++i) {
        // scatter by index so torus load produces a visible density pattern
        const size_t px = (static_cast<size_t>(i) * 2654435761u) % n;
        fb[px] = 0xFFE8C070u;  // warm lit pixel
    }
    // soft box in center so window is obviously alive
    const int cx = width / 2, cy = height / 2, r = std::min(width, height) / 8;
    for (int y = cy - r; y <= cy + r; ++y) {
        for (int x = cx - r; x <= cx + r; ++x) {
            if (x >= 0 && y >= 0 && x < width && y < height)
                fb[static_cast<size_t>(y) * static_cast<size_t>(width) +
                   static_cast<size_t>(x)] = 0xFF60A0F0u;
        }
    }

    err = clEnqueueWriteBuffer(q, as_mem(buf_fb), CL_TRUE, 0, n * sizeof(uint32_t),
                               fb.data(), 0, nullptr, nullptr);
    last_spp = hits > 0 ? float(hits) / float(n) : 0.f;
    return err == CL_SUCCESS;
}

void SplatRenderer::Present() {
    if (!cl_queue || !buf_fb || width <= 0 || height <= 0) return;
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint32_t> host(n);
    clEnqueueReadBuffer(as_q(cl_queue), as_mem(buf_fb), CL_TRUE, 0, n * sizeof(uint32_t),
                        host.data(), 0, nullptr, nullptr);

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    if (sdl_rnd && sdl_tex) {
        SDL_Renderer* rnd = static_cast<SDL_Renderer*>(sdl_rnd);
        SDL_Texture* tex = static_cast<SDL_Texture*>(sdl_tex);
        void* pixels = nullptr;
        int pitch = 0;
        if (SDL_LockTexture(tex, nullptr, &pixels, &pitch) == 0) {
            auto* dst = static_cast<uint8_t*>(pixels);
            for (int y = 0; y < height; ++y) {
                std::memcpy(dst + y * pitch, host.data() + static_cast<size_t>(y) * width,
                            static_cast<size_t>(width) * sizeof(uint32_t));
            }
            SDL_UnlockTexture(tex);
            SDL_RenderClear(rnd);
            SDL_RenderCopy(rnd, tex, nullptr, nullptr);
            SDL_RenderPresent(rnd);
        }
    }
#else
    (void)host;
#endif
}

void SplatRenderer::Resize(int w, int h) {
    width = w;
    height = h;
}

void SplatRenderer::Shutdown() {
#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    if (sdl_tex) {
        SDL_DestroyTexture(static_cast<SDL_Texture*>(sdl_tex));
        sdl_tex = nullptr;
    }
#endif
    auto rel = [](void*& p) {
        if (p) {
            clReleaseMemObject(as_mem(p));
            p = nullptr;
        }
    };
    rel(buf_fb);
    rel(buf_anchors);
    rel(buf_surfaces);
    rel(buf_stitch);
    rel(buf_img);
    rel(buf_depth32);
    rel(buf_snorm);
    rel(buf_soff);
    rel(buf_sss);
    if (cl_queue) {
        clReleaseCommandQueue(as_q(cl_queue));
        cl_queue = nullptr;
    }
    if (cl_ctx) {
        clReleaseContext(as_ctx(cl_ctx));
        cl_ctx = nullptr;
    }
    cl_dev = nullptr;
}

} // namespace DeliRium
