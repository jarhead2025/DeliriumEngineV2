// FILEPATH: engine/src/render/SplatRenderer.cpp
// E5: real psf_splat.cl path — 64-byte AnchorRecord/SurfaceRecord blobs, no mesh.
// Matches kernels/psf_splat.cl record offsets and Types.h frozen layouts.
#define CL_TARGET_OPENCL_VERSION 300
#include "delirium/render/SplatRenderer.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/render/Camera.h"
#include "delirium/core/Types.h"
#include <CL/cl.h>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
#include <SDL.h>
#endif

namespace DeliRium {

// forward decls (E7 + lateral fill)
void average_infill(std::vector<uint32_t>& rgba, int w, int h, int radius, int passes);
bool lateral_plane_sweep_e7(
    const std::vector<AnchorRecord>& anchors,
    const std::vector<SurfaceRecord>& surfaces,
    Vec3 bbmin, Vec3 bbspan, float s_scale,
    const Camera& cam, int width, int height,
    std::vector<uint32_t>& out_rgba);


namespace {
inline cl_context as_ctx(void* p) { return static_cast<cl_context>(p); }
inline cl_command_queue as_q(void* p) { return static_cast<cl_command_queue>(p); }
inline cl_device_id as_dev(void* p) { return static_cast<cl_device_id>(p); }
inline cl_mem as_mem(void* p) { return static_cast<cl_mem>(p); }
inline cl_program as_prog(void* p) { return static_cast<cl_program>(p); }
inline cl_kernel as_k(void* p) { return static_cast<cl_kernel>(p); }

static std::string load_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}
} // namespace

bool SplatRenderer::BuildProgram(const std::string& kernel_dir) {
    if (!cl_ctx || !cl_dev) return false;
    std::string path = kernel_dir;
    if (!path.empty() && path.back() != '/') path += '/';
    path += "psf_splat.cl";
    std::string src = load_text_file(path);
    if (src.empty()) {
        // try relative from cwd
        src = load_text_file("kernels/psf_splat.cl");
    }
    if (src.empty()) {
        std::fprintf(stderr, "SplatRenderer: cannot read %s\n", path.c_str());
        return false;
    }

    cl_int err = CL_SUCCESS;
    const char* csrc = src.c_str();
    size_t slen = src.size();
    cl_program program = clCreateProgramWithSource(as_ctx(cl_ctx), 1, &csrc, &slen, &err);
    if (err != CL_SUCCESS || !program) return false;

    // Prefer 64-bit atomic path (Arc reports int64 atomics)
    const char* opts = "-cl-std=CL1.2 -DUSE_LONG_ATOMICS";
    cl_device_id dev_id = as_dev(cl_dev);
    err = clBuildProgram(program, 1, &dev_id, opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        opts = "-cl-std=CL1.2";
        err = clBuildProgram(program, 1, &dev_id, opts, nullptr, nullptr);
        has_long_atomics = false;
    } else {
        has_long_atomics = true;
    }
    if (err != CL_SUCCESS) {
        size_t log_sz = 0;
        clGetProgramBuildInfo(program, dev_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_sz);
        std::vector<char> log(log_sz + 1, 0);
        clGetProgramBuildInfo(program, dev_id, CL_PROGRAM_BUILD_LOG, log_sz, log.data(), nullptr);
        std::fprintf(stderr, "SplatRenderer build log:\n%s\n", log.data());
        clReleaseProgram(program);
        return false;
    }

    if (prog) {
        clReleaseProgram(as_prog(prog));
        prog = nullptr;
    }
    prog = static_cast<void*>(program);

    auto make_k = [&](const char* name, void*& slot) -> bool {
        if (slot) {
            clReleaseKernel(as_k(slot));
            slot = nullptr;
        }
        cl_kernel k = clCreateKernel(program, name, &err);
        if (err != CL_SUCCESS || !k) {
            std::fprintf(stderr, "SplatRenderer: missing kernel '%s'\n", name);
            return false;
        }
        slot = static_cast<void*>(k);
        return true;
    };
    if (!make_k("clear_fb", k_splat)) return false; // temp: store clear in k_splat then reassign
    // We need three kernels: clear, splat, resolve — use prog + recreate
    clReleaseKernel(as_k(k_splat));
    k_splat = nullptr;
    cl_kernel k_clear = clCreateKernel(program, "clear_fb", &err);
    cl_kernel k_ps = clCreateKernel(program, "psf_splat", &err);
    cl_kernel k_res = clCreateKernel(program, "resolve_fb", &err);
    if (!k_clear || !k_ps || !k_res) {
        std::fprintf(stderr, "SplatRenderer: kernel create failed\n");
        return false;
    }
    // Stash: k_splat = psf_splat, k_resolve = resolve; clear via name lookup each frame
    // Header only has k_splat and k_resolve — keep clear as local static in RenderBody via clCreateKernel from prog
    clReleaseKernel(k_clear);
    k_splat = static_cast<void*>(k_ps);
    k_resolve = static_cast<void*>(k_res);
    std::printf("SplatRenderer: program OK (%s atomics) from %s\n",
                has_long_atomics ? "long" : "32-bit", path.c_str());
    return true;
}

bool SplatRenderer::Initialize(int width, int height, void* sdl_renderer,
                               const std::string& kernel_dir) {
    this->width = width;
    this->height = height;
    this->sdl_rnd = sdl_renderer;
    this->device_name.clear();

    cl_int err = CL_SUCCESS;
    cl_uint nplat = 0;
    if (clGetPlatformIDs(0, nullptr, &nplat) != CL_SUCCESS || nplat == 0) return false;
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
    if (err != CL_SUCCESS || !q) return false;
    cl_queue = static_cast<void*>(q);

    const size_t n_px = static_cast<size_t>(width) * static_cast<size_t>(height);
    // Packed depth|color uses ulong per pixel when long atomics path is used
    cl_mem fb = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_px * sizeof(cl_ulong), nullptr, &err);
    if (err != CL_SUCCESS || !fb) return false;
    buf_fb = static_cast<void*>(fb);

    cl_mem img = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n_px * 4, nullptr, &err);
    if (err == CL_SUCCESS && img) buf_img = static_cast<void*>(img);

    if (!BuildProgram(kernel_dir.empty() ? "kernels" : kernel_dir)) {
        std::fprintf(stderr, "SplatRenderer: BuildProgram failed — host fallback only\n");
    }

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    if (sdl_rnd) {
        if (sdl_tex) {
            SDL_DestroyTexture(static_cast<SDL_Texture*>(sdl_tex));
            sdl_tex = nullptr;
        }
        SDL_Texture* tex = SDL_CreateTexture(
            static_cast<SDL_Renderer*>(sdl_rnd), SDL_PIXELFORMAT_ARGB8888,
            SDL_TEXTUREACCESS_STREAMING, width, height);
        sdl_tex = static_cast<void*>(tex);
    }
#endif
    return true;
}

uint32_t SplatRenderer::BuildSampleGrids(PSFBody& body, const float* mvp) {
    // E6: screen-space adaptive anisotropic grids — denser when closer / larger on screen.
    // sample_ss[s] = (ss_u << 16) | ss_v, matching kernels/psf_splat.cl.
    // Optional DELIRIUM_SS forces a fixed grid (debug). DELIRIUM_SS_MIN / MAX clamp.
    const auto& surfaces = body.SurfaceBlob();
    const auto& anchors = body.AnchorBlob();
    const uint32_t ns = static_cast<uint32_t>(surfaces.size());
    if (ns == 0) return 0;

    uint32_t ss_min = 4, ss_max = 48;
    if (const char* e = std::getenv("DELIRIUM_SS_MIN")) {
        unsigned v = 0;
        if (std::sscanf(e, "%u", &v) == 1 && v >= 2 && v <= 64) ss_min = v;
    }
    if (const char* e = std::getenv("DELIRIUM_SS_MAX")) {
        unsigned v = 0;
        if (std::sscanf(e, "%u", &v) == 1 && v >= ss_min && v <= 128) ss_max = v;
    }
    int force_ss = -1;
    if (const char* e = std::getenv("DELIRIUM_SS")) {
        unsigned v = 0;
        if (std::sscanf(e, "%u", &v) == 1 && v >= 4 && v <= 64) force_ss = int(v);
    }

    Vec3 bbmin = body.BBoxMin();
    Vec3 bbspan = body.BBoxSpan();
    if (bbspan.x == 0 && bbspan.y == 0 && bbspan.z == 0) {
        bbmin = {-1.f, -1.f, -1.f};
        bbspan = {2.f, 2.f, 2.f};
    }

    auto dequant = [&](const AnchorRecord& a) {
        const float ux = a.pos[0] / 65535.f;
        const float uy = a.pos[1] / 65535.f;
        const float uz = a.pos[2] / 65535.f;
        return Vec3{bbmin.x + ux * bbspan.x, bbmin.y + uy * bbspan.y,
                    bbmin.z + uz * bbspan.z};
    };

    // Project world point with row-major MVP -> NDC xy in pixels
    auto project = [&](const Vec3& p, float& out_x, float& out_y, float& out_w) -> bool {
        if (!mvp) return false;
        const float x = p.x, y = p.y, z = p.z;
        const float clip_x = mvp[0] * x + mvp[1] * y + mvp[2] * z + mvp[3];
        const float clip_y = mvp[4] * x + mvp[5] * y + mvp[6] * z + mvp[7];
        const float clip_w = mvp[12] * x + mvp[13] * y + mvp[14] * z + mvp[15];
        if (clip_w <= 1e-6f) return false;
        const float ndc_x = clip_x / clip_w;
        const float ndc_y = clip_y / clip_w;
        out_x = (ndc_x * 0.5f + 0.5f) * float(width);
        out_y = (1.0f - (ndc_y * 0.5f + 0.5f)) * float(height);
        out_w = clip_w;
        return true;
    };

    auto clamp_ss = [&](float edge_px) -> uint32_t {
        // ~1 sample per ~1.4 px of projected edge, +2 pad (kernel comment spirit)
        float s = edge_px / 1.4f + 2.f;
        if (s < float(ss_min)) s = float(ss_min);
        if (s > float(ss_max)) s = float(ss_max);
        return static_cast<uint32_t>(s + 0.5f);
    };

    std::vector<uint32_t> off(ns + 1, 0);
    std::vector<uint32_t> ss(ns, 0);
    uint64_t total_samples = 0;
    uint32_t min_seen = 999, max_seen = 0;

    for (uint32_t si = 0; si < ns; ++si) {
        uint32_t ssu = force_ss > 0 ? uint32_t(force_ss) : 12;
        uint32_t ssv = ssu;
        if (force_ss < 0 && mvp && width > 0 && height > 0) {
            const SurfaceRecord& srec = surfaces[si];
            Vec3 P[4];
            bool ok = true;
            for (int k = 0; k < 4; ++k) {
                const uint16_t ai = srec.anchor_idx[k];
                if (ai >= anchors.size()) {
                    ok = false;
                    break;
                }
                P[k] = dequant(anchors[ai]);
            }
            if (ok) {
                float sx[4], sy[4], sw[4];
                int nproj = 0;
                for (int k = 0; k < 4; ++k) {
                    if (project(P[k], sx[k], sy[k], sw[k])) ++nproj;
                    else sx[k] = sy[k] = 0.f;
                }
                if (nproj >= 2) {
                    // edge lengths in pixels: u-dir avg of (0-1) and (3-2), v-dir (0-3)/(1-2)
                    auto elen = [&](int a, int b) {
                        const float dx = sx[a] - sx[b], dy = sy[a] - sy[b];
                        return std::sqrt(dx * dx + dy * dy);
                    };
                    const float eu = 0.5f * (elen(0, 1) + elen(3, 2));
                    const float ev = 0.5f * (elen(0, 3) + elen(1, 2));
                    ssu = clamp_ss(eu);
                    ssv = clamp_ss(ev);
                    // closer patches (smaller |w|) get a mild boost
                    float wavg = 0.f;
                    int nw = 0;
                    for (int k = 0; k < 4; ++k) {
                        if (sw[k] > 1e-6f) {
                            wavg += sw[k];
                            ++nw;
                        }
                    }
                    if (nw > 0) {
                        wavg /= float(nw);
                        // w smaller => closer in typical perspective; boost up to 1.5x
                        const float boost = std::min(1.5f, std::max(1.0f, 2.5f / std::max(wavg, 0.5f)));
                        ssu = std::min(ss_max, std::max(ss_min, uint32_t(ssu * boost + 0.5f)));
                        ssv = std::min(ss_max, std::max(ss_min, uint32_t(ssv * boost + 0.5f)));
                    }
                }
            }
        }
        if (ssu < 2) ssu = 2;
        if (ssv < 2) ssv = 2;
        ss[si] = (ssu << 16) | ssv;
        const uint32_t per = ssu * ssv;
        off[si + 1] = off[si] + per;
        total_samples += per;
        min_seen = std::min(min_seen, std::min(ssu, ssv));
        max_seen = std::max(max_seen, std::max(ssu, ssv));
    }

    static int s_log_once = 0;
    if ((s_log_once++ % 120) == 0) {
        std::printf("E6 grids: surfaces=%u samples=%llu ss_range=[%u,%u]%s\n",
                    ns, (unsigned long long)total_samples, min_seen, max_seen,
                    force_ss > 0 ? " (forced DELIRIUM_SS)" : " (adaptive)");
    }

    cl_context ctx = as_ctx(cl_ctx);
    cl_int err = CL_SUCCESS;
    if (buf_soff) {
        clReleaseMemObject(as_mem(buf_soff));
        buf_soff = nullptr;
    }
    if (buf_sss) {
        clReleaseMemObject(as_mem(buf_sss));
        buf_sss = nullptr;
    }
    cl_mem moff = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 off.size() * sizeof(uint32_t), off.data(), &err);
    cl_mem mss = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                ss.size() * sizeof(uint32_t), ss.data(), &err);
    if (moff) buf_soff = static_cast<void*>(moff);
    if (mss) buf_sss = static_cast<void*>(mss);
    cap_soff = ns;
    return off.back();
}

bool SplatRenderer::RenderBody(PSFBody& body, const Camera& cam) {
    if (!cl_ctx || !cl_queue || !buf_fb || width <= 0 || height <= 0) return false;

    // E7 alternative pipeline: frustum lateral plane sweep (set DELIRIUM_LATERAL=0 to use pure CL splat)
    bool use_e7 = true;
    if (const char* e = std::getenv("DELIRIUM_LATERAL")) {
        if (e[0] == '0' && e[1] == '\0') use_e7 = false;
    }
    if (use_e7) {
        Vec3 bbmin = body.BBoxMin();
        Vec3 bbspan = body.BBoxSpan();
        if (bbspan.x == 0 && bbspan.y == 0 && bbspan.z == 0) {
            bbmin = {-1.f, -1.f, -1.f};
            bbspan = {2.f, 2.f, 2.f};
        }
        std::vector<uint32_t> host;
        if (lateral_plane_sweep_e7(body.AnchorBlob(), body.SurfaceBlob(), bbmin, bbspan,
                                   body.StitchScale(), cam, width, height, host)) {
            if (buf_img) {
                clEnqueueWriteBuffer(as_q(cl_queue), as_mem(buf_img), CL_TRUE, 0,
                                     host.size() * 4, host.data(), 0, nullptr, nullptr);
            }
            last_spp = float(host.size() > 0 ? 1.f : 0.f);
            return true;
        }
    }

    const auto& anchors = body.AnchorBlob();
    const auto& surfaces = body.SurfaceBlob();
    if (anchors.empty() || surfaces.empty()) return false;

    cl_context ctx = as_ctx(cl_ctx);
    cl_command_queue q = as_q(cl_queue);
    cl_int err = CL_SUCCESS;

    // Upload raw 64-byte records (device kernel uses pointer math, not C++ structs)
    if (buf_anchors) {
        clReleaseMemObject(as_mem(buf_anchors));
        buf_anchors = nullptr;
    }
    if (buf_surfaces) {
        clReleaseMemObject(as_mem(buf_surfaces));
        buf_surfaces = nullptr;
    }
    cl_mem ba = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               anchors.size() * 64, const_cast<AnchorRecord*>(anchors.data()), &err);
    cl_mem bs = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               surfaces.size() * 64, const_cast<SurfaceRecord*>(surfaces.data()), &err);
    if (!ba || !bs) return false;
    buf_anchors = static_cast<void*>(ba);
    buf_surfaces = static_cast<void*>(bs);
    cap_anchors = static_cast<uint32_t>(anchors.size());
    cap_surfaces = static_cast<uint32_t>(surfaces.size());

    // stitch: 8 int8 per surface (zeros if body has none)
    std::vector<char> stitch(surfaces.size() * 8, 0);
    const auto& st = body.StitchBlob();
    if (!st.empty()) {
        const size_t n = std::min(st.size(), stitch.size());
        std::memcpy(stitch.data(), st.data(), n);
    }
    if (buf_stitch) {
        clReleaseMemObject(as_mem(buf_stitch));
        buf_stitch = nullptr;
    }
    cl_mem bst = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                stitch.size(), stitch.data(), &err);
    if (bst) buf_stitch = static_cast<void*>(bst);

    // snorm: float4 per anchor (simple up if missing)
    std::vector<cl_float4> snorm(anchors.size());
    for (size_t i = 0; i < anchors.size(); ++i)
        snorm[i] = {{0.f, 0.f, 1.f, 0.f}};
    if (buf_snorm) {
        clReleaseMemObject(as_mem(buf_snorm));
        buf_snorm = nullptr;
    }
    cl_mem bsn = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                snorm.size() * sizeof(cl_float4), snorm.data(), &err);
    if (bsn) buf_snorm = static_cast<void*>(bsn);

    float mvp[16];
    cam.BuildMVP(mvp, float(width) / float(height));
    const uint32_t total = BuildSampleGrids(body, mvp);
    if (total == 0 || !buf_soff || !buf_sss) return false;

    Vec3 bbmin = body.BBoxMin();
    Vec3 bbspan = body.BBoxSpan();
    // If bbox is zero (some v2 files), use unit box so dequant still runs
    if (bbspan.x == 0 && bbspan.y == 0 && bbspan.z == 0) {
        bbmin = {-1.f, -1.f, -1.f};
        bbspan = {2.f, 2.f, 2.f};
    }
    const float s_scale = body.StitchScale() > 0.f ? body.StitchScale() : 1.f;

    // ---- GPU path when program built ----
    if (prog && k_splat && k_resolve && has_long_atomics) {
        cl_mem m_fb = as_mem(buf_fb);
        cl_mem m_anch = as_mem(buf_anchors);
        cl_mem m_sn = as_mem(buf_snorm);
        cl_mem m_surf = as_mem(buf_surfaces);
        cl_mem m_st = as_mem(buf_stitch);
        cl_mem m_off = as_mem(buf_soff);
        cl_mem m_ss = as_mem(buf_sss);
        cl_mem m_img = buf_img ? as_mem(buf_img) : nullptr;

        cl_kernel k_clear = clCreateKernel(as_prog(prog), "clear_fb", &err);
        const cl_uint npx = static_cast<cl_uint>(width * height);
        if (k_clear) {
            clSetKernelArg(k_clear, 0, sizeof(cl_mem), &m_fb);
            clSetKernelArg(k_clear, 1, sizeof(cl_uint), &npx);
            size_t g = npx;
            clEnqueueNDRangeKernel(q, k_clear, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
            clReleaseKernel(k_clear);
        }

        cl_kernel ks = as_k(k_splat);
        cl_uint nsurf = cap_surfaces;
        cl_float3 bbmin3 = {{bbmin.x, bbmin.y, bbmin.z}};
        cl_float3 bbspan3 = {{bbspan.x, bbspan.y, bbspan.z}};
        cl_float16 mvp16;
        std::memcpy(&mvp16, mvp, sizeof(mvp));
        cl_float3 light = {{-0.4f, -0.7f, -0.5f}};
        cl_float3 eye = {{cam.position.x, cam.position.y, cam.position.z}};
        cl_uint w = static_cast<cl_uint>(width), h = static_cast<cl_uint>(height);

        int a = 0;
        clSetKernelArg(ks, a++, sizeof(cl_mem), &m_anch);
        clSetKernelArg(ks, a++, sizeof(cl_mem), &m_sn);
        clSetKernelArg(ks, a++, sizeof(cl_mem), &m_surf);
        clSetKernelArg(ks, a++, sizeof(cl_mem), &m_st);
        clSetKernelArg(ks, a++, sizeof(cl_mem), &m_off);
        clSetKernelArg(ks, a++, sizeof(cl_mem), &m_ss);
        clSetKernelArg(ks, a++, sizeof(cl_uint), &nsurf);
        clSetKernelArg(ks, a++, sizeof(cl_float3), &bbmin3);
        clSetKernelArg(ks, a++, sizeof(cl_float3), &bbspan3);
        clSetKernelArg(ks, a++, sizeof(cl_float), &s_scale);
        clSetKernelArg(ks, a++, sizeof(cl_float16), &mvp16);
        clSetKernelArg(ks, a++, sizeof(cl_float3), &light);
        clSetKernelArg(ks, a++, sizeof(cl_float3), &eye);
        clSetKernelArg(ks, a++, sizeof(cl_uint), &w);
        clSetKernelArg(ks, a++, sizeof(cl_uint), &h);
        clSetKernelArg(ks, a++, sizeof(cl_mem), &m_fb);

        size_t g = total;
        err = clEnqueueNDRangeKernel(q, ks, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::fprintf(stderr, "psf_splat enqueue err=%d\n", err);
            return false;
        }

        if (m_img && k_resolve) {
            cl_kernel kr = as_k(k_resolve);
            clSetKernelArg(kr, 0, sizeof(cl_mem), &m_fb);
            clSetKernelArg(kr, 1, sizeof(cl_uint), &npx);
            clSetKernelArg(kr, 2, sizeof(cl_mem), &m_img);
            size_t gr = npx;
            clEnqueueNDRangeKernel(q, kr, 1, nullptr, &gr, nullptr, 0, nullptr, nullptr);
        }
        clFinish(q);
        last_spp = float(total) / float(width * height);
        return true;
    }

    // ---- Host fallback: evaluate is expensive; mark that kernel path missing ----
    std::fprintf(stderr, "SplatRenderer: GPU kernel path unavailable (prog=%p)\n", prog);
    return false;
}


namespace {
// Average-overwrite infill: empty/background pixels pull mean of nearby lit pixels.
void average_infill(std::vector<uint32_t>& rgba, int w, int h, int radius, int passes) {
    // Lateral-march fill (not morphological dilate):
    // each sweep finds consecutive lit hits on a scanline and linearly
    // interpolates RGB through the empty span between them ("section" average).
    // Alternating H then V sweeps match "every sweep infills averages between dots".
    (void)radius; // kept for API / env compatibility; span fill uses full gap
    auto is_bg = [](uint32_t c) {
        const uint32_t r = c & 255u;
        const uint32_t g = (c >> 8) & 255u;
        const uint32_t b = (c >> 16) & 255u;
        return (r <= 28 && g <= 28 && b <= 36) || ((c & 0x00FFFFFFu) == 0u);
    };
    auto unpack = [](uint32_t c, float& r, float& g, float& b) {
        r = float(c & 255u);
        g = float((c >> 8) & 255u);
        b = float((c >> 16) & 255u);
    };
    auto pack = [](float r, float g, float b) -> uint32_t {
        auto q = [](float x) -> uint32_t {
            if (x < 0.f) x = 0.f;
            if (x > 255.f) x = 255.f;
            return uint32_t(x + 0.5f);
        };
        return (255u << 24) | (q(b) << 16) | (q(g) << 8) | q(r);
    };

    // Horizontal lateral march: between two lit pixels on a row, lerp the gap
    auto sweep_h = [&](std::vector<uint32_t>& img) {
        for (int y = 0; y < h; ++y) {
            const size_t row = size_t(y) * size_t(w);
            int prev = -1;
            for (int x = 0; x < w; ++x) {
                if (is_bg(img[row + size_t(x)])) continue;
                if (prev >= 0 && x - prev > 1) {
                    float r0, g0, b0, r1, g1, b1;
                    unpack(img[row + size_t(prev)], r0, g0, b0);
                    unpack(img[row + size_t(x)], r1, g1, b1);
                    const float denom = float(x - prev);
                    for (int t = prev + 1; t < x; ++t) {
                        if (!is_bg(img[row + size_t(t)])) continue; // keep existing hits
                        const float u = float(t - prev) / denom;
                        img[row + size_t(t)] = pack(
                            r0 + (r1 - r0) * u,
                            g0 + (g1 - g0) * u,
                            b0 + (b1 - b0) * u);
                    }
                }
                prev = x;
            }
        }
    };
    // Vertical lateral march
    auto sweep_v = [&](std::vector<uint32_t>& img) {
        for (int x = 0; x < w; ++x) {
            int prev = -1;
            for (int y = 0; y < h; ++y) {
                const size_t i = size_t(y) * size_t(w) + size_t(x);
                if (is_bg(img[i])) continue;
                if (prev >= 0 && y - prev > 1) {
                    float r0, g0, b0, r1, g1, b1;
                    unpack(img[size_t(prev) * size_t(w) + size_t(x)], r0, g0, b0);
                    unpack(img[i], r1, g1, b1);
                    const float denom = float(y - prev);
                    for (int t = prev + 1; t < y; ++t) {
                        const size_t j = size_t(t) * size_t(w) + size_t(x);
                        if (!is_bg(img[j])) continue;
                        const float u = float(t - prev) / denom;
                        img[j] = pack(
                            r0 + (r1 - r0) * u,
                            g0 + (g1 - g0) * u,
                            b0 + (b1 - b0) * u);
                    }
                }
                prev = y;
            }
        }
    };

    const int npass = passes > 0 ? passes : 2;
    for (int p = 0; p < npass; ++p) {
        sweep_h(rgba);
        sweep_v(rgba);
    }
}

namespace {
// --- E7: frustum lateral plane sweep against PSF equations (host foundation) ---
inline float deform_h(const float c[8], float u, float v) {
    const float bu = u * (1.f - u), bv = v * (1.f - v);
    const float au = 2.f * u - 1.f, av = 2.f * v - 1.f;
    const float w = 1.f - v, x = 1.f - u;
    return fma(c[0], bu * w,
           fma(c[1], bu * v,
           fma(c[2], bv * x,
           fma(c[3], bv * u,
           fma(c[4], bu * au * w,
           fma(c[5], bu * au * v,
           fma(c[6], bv * av * x, c[7] * bv * av * u)))))));
}

// Parallel plane stack through the frustum; denser sampling nearer the eye.
// Each plane: lateral rays (grid) impact bilinear+deform surfaces; average into FB.
bool lateral_plane_sweep_e7(
    const std::vector<AnchorRecord>& anchors,
    const std::vector<SurfaceRecord>& surfaces,
    Vec3 bbmin, Vec3 bbspan, float s_scale,
    const Camera& cam, int width, int height,
    std::vector<uint32_t>& out_rgba) {
    (void)s_scale;
    if (anchors.empty() || surfaces.empty() || width <= 0 || height <= 0) return false;

    float mvp[16];
    cam.BuildMVP(mvp, float(width) / float(height));

    auto dequant = [&](const AnchorRecord& a) {
        return Vec3{
            bbmin.x + (a.pos[0] / 65535.f) * bbspan.x,
            bbmin.y + (a.pos[1] / 65535.f) * bbspan.y,
            bbmin.z + (a.pos[2] / 65535.f) * bbspan.z};
    };

    // Precompute patch corners in world space
    struct Patch {
        Vec3 P[4];
        float coef[8];
        uint32_t rgba;
    };
    std::vector<Patch> patches;
    patches.reserve(surfaces.size());
    for (const auto& s : surfaces) {
        Patch p{};
        bool ok = true;
        for (int k = 0; k < 4; ++k) {
            if (s.anchor_idx[k] >= anchors.size()) {
                ok = false;
                break;
            }
            p.P[k] = dequant(anchors[s.anchor_idx[k]]);
        }
        if (!ok) continue;
        for (int k = 0; k < 8; ++k) p.coef[k] = s.coef[k];
        // cheap color from coef energy + index (pattern full later)
        float e = 0.f;
        for (int k = 0; k < 8; ++k) e += std::fabs(p.coef[k]);
        const uint32_t r = uint32_t(std::min(255.f, 80.f + e * 40.f));
        const uint32_t g = uint32_t(std::min(255.f, 120.f + float(patches.size() % 97)));
        const uint32_t b = uint32_t(std::min(255.f, 160.f + float(patches.size() % 53)));
        p.rgba = (255u << 24) | (b << 16) | (g << 8) | r;
        patches.push_back(p);
    }
    if (patches.empty()) return false;

    auto eval_patch = [&](const Patch& p, float u, float v) {
        const float um = 1.f - u, vm = 1.f - v;
        Vec3 bil{
            um * vm * p.P[0].x + u * vm * p.P[1].x + u * v * p.P[2].x + um * v * p.P[3].x,
            um * vm * p.P[0].y + u * vm * p.P[1].y + u * v * p.P[2].y + um * v * p.P[3].y,
            um * vm * p.P[0].z + u * vm * p.P[1].z + u * v * p.P[2].z + um * v * p.P[3].z};
        // normal approx from corners for displacement
        Vec3 e1{p.P[1].x - p.P[0].x, p.P[1].y - p.P[0].y, p.P[1].z - p.P[0].z};
        Vec3 e2{p.P[3].x - p.P[0].x, p.P[3].y - p.P[0].y, p.P[3].z - p.P[0].z};
        Vec3 n{e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x};
        float nl = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        if (nl > 1e-8f) {
            n.x /= nl;
            n.y /= nl;
            n.z /= nl;
        }
        const float h = deform_h(p.coef, u, v);
        return Vec3{bil.x + n.x * h, bil.y + n.y * h, bil.z + n.z * h};
    };

    auto project = [&](const Vec3& p, float& sx, float& sy, float& depth) -> bool {
        const float x = p.x, y = p.y, z = p.z;
        const float cx = mvp[0] * x + mvp[1] * y + mvp[2] * z + mvp[3];
        const float cy = mvp[4] * x + mvp[5] * y + mvp[6] * z + mvp[7];
        const float cw = mvp[12] * x + mvp[13] * y + mvp[14] * z + mvp[15];
        if (cw <= 1e-5f) return false;
        const float ndc_x = cx / cw, ndc_y = cy / cw;
        sx = (ndc_x * 0.5f + 0.5f) * float(width);
        sy = (1.f - (ndc_y * 0.5f + 0.5f)) * float(height);
        depth = cw;
        return sx >= -1 && sy >= -1 && sx < width + 1 && sy < height + 1;
    };

    // Accumulation buffers for per-pixel average
    const size_t n = size_t(width) * size_t(height);
    std::vector<float> ar(n, 0.f), ag(n, 0.f), ab(n, 0.f), wsum(n, 0.f);
    std::vector<float> zbuf(n, 1e30f);

    // Plane count: up to half vertical resolution (user target)
    int n_planes = height / 2;
    if (n_planes < 16) n_planes = 16;
    if (n_planes > 400) n_planes = 400;
    if (const char* e = std::getenv("DELIRIUM_E7_PLANES")) {
        unsigned v = 0;
        if (std::sscanf(e, "%u", &v) == 1 && v >= 8 && v <= 800) n_planes = int(v);
    }

    // Lateral samples across FOV on each plane: denser for nearer planes
    const int rays_near = std::min(width, 320);
    const int rays_far = std::max(32, width / 8);

    // Camera basis
    Vec3 f{cam.target.x - cam.position.x, cam.target.y - cam.position.y,
           cam.target.z - cam.position.z};
    float fl = std::sqrt(f.x * f.x + f.y * f.y + f.z * f.z);
    if (fl < 1e-6f) return false;
    f.x /= fl;
    f.y /= fl;
    f.z /= fl;
    Vec3 r{f.y * cam.up.z - f.z * cam.up.y, f.z * cam.up.x - f.x * cam.up.z,
           f.x * cam.up.y - f.y * cam.up.x};
    float rl = std::sqrt(r.x * r.x + r.y * r.y + r.z * r.z);
    r.x /= rl;
    r.y /= rl;
    r.z /= rl;
    Vec3 u{r.y * f.z - r.z * f.y, r.z * f.x - r.x * f.z, r.x * f.y - r.y * f.x};

    const float tan_half = std::tan(cam.fov_deg * 3.14159265f / 360.f);
    const float aspect = float(width) / float(height);
    const float near_d = std::max(cam.near_p, 0.05f);
    const float far_d = cam.far_p;

    // UV samples per patch on each plane impact (equation domain)
    const int uvn = 4; // 4x4 domain samples per patch per plane contribution

    for (int pi = 0; pi < n_planes; ++pi) {
        // nearer planes first; density high near camera
        const float t = (float(pi) + 0.5f) / float(n_planes);
        const float depth = near_d * std::pow(far_d / near_d, t); // log depth spacing
        const float dens = 1.f - t; // 1 at near, 0 at far
        const int n_rays = int(rays_far + dens * float(rays_near - rays_far));
        const float half_h = depth * tan_half;
        const float half_w = half_h * aspect;

        // Center of plane along view ray
        Vec3 C{cam.position.x + f.x * depth, cam.position.y + f.y * depth,
               cam.position.z + f.z * depth};

        // Parallel lateral samples across the plane rectangle
        for (int j = 0; j < n_rays; ++j) {
            const float sj = (n_rays == 1) ? 0.f : (float(j) / float(n_rays - 1)) * 2.f - 1.f;
            // one "row" of the plane; also sweep a few vertical offsets within section
            const int v_steps = 2 + int(dens * 4);
            for (int vi = 0; vi < v_steps; ++vi) {
                const float sv =
                    (v_steps == 1) ? 0.f : (float(vi) / float(v_steps - 1)) * 2.f - 1.f;
                Vec3 P{C.x + r.x * (sj * half_w) + u.x * (sv * half_h),
                       C.y + r.y * (sj * half_w) + u.y * (sv * half_h),
                       C.z + r.z * (sj * half_w) + u.z * (sv * half_h)};

                // Impact equations: sample patches in domain; keep hits near this plane
                for (const auto& patch : patches) {
                    for (int iu = 0; iu < uvn; ++iu) {
                        for (int iv = 0; iv < uvn; ++iv) {
                            const float uu = (float(iu) + 0.5f) / float(uvn);
                            const float vv = (float(iv) + 0.5f) / float(uvn);
                            Vec3 S = eval_patch(patch, uu, vv);
                            // lateral plane affinity: point near plane depth along view
                            const float along = (S.x - cam.position.x) * f.x +
                                                (S.y - cam.position.y) * f.y +
                                                (S.z - cam.position.z) * f.z;
                            if (std::fabs(along - depth) > half_h * 0.35f) continue;

                            float sx, sy, zd;
                            if (!project(S, sx, sy, zd)) continue;
                            const int x = int(sx + 0.5f);
                            const int y = int(sy + 0.5f);
                            if (x < 0 || y < 0 || x >= width || y >= height) continue;
                            const size_t idx = size_t(y) * size_t(width) + size_t(x);
                            // depth-aware average (near wins more weight)
                            if (zd > zbuf[idx] * 1.02f) continue;
                            const float wt = 1.f / (zd + 1e-3f);
                            const float pr = float(patch.rgba & 255u);
                            const float pg = float((patch.rgba >> 8) & 255u);
                            const float pb = float((patch.rgba >> 16) & 255u);
                            if (zd < zbuf[idx]) {
                                zbuf[idx] = zd;
                                ar[idx] = pr * wt;
                                ag[idx] = pg * wt;
                                ab[idx] = pb * wt;
                                wsum[idx] = wt;
                            } else {
                                ar[idx] += pr * wt;
                                ag[idx] += pg * wt;
                                ab[idx] += pb * wt;
                                wsum[idx] += wt;
                            }
                        }
                    }
                }
            }
        }
    }

    out_rgba.assign(n, 0xFF121218u);
    for (size_t i = 0; i < n; ++i) {
        if (wsum[i] <= 0.f) continue;
        const float inv = 1.f / wsum[i];
        auto q = [](float x) {
            if (x < 0.f) x = 0.f;
            if (x > 255.f) x = 255.f;
            return uint32_t(x + 0.5f);
        };
        out_rgba[i] = (255u << 24) | (q(ab[i] * inv) << 16) | (q(ag[i] * inv) << 8) | q(ar[i] * inv);
    }
    // Finish with a couple of lateral span sweeps for continuity along sections
    average_infill(out_rgba, width, height, 1, 2);
    std::printf("E7 lateral planes=%d patches=%zu\n", n_planes, patches.size());
    return true;
}

void SplatRenderer::Present() {
    if (!cl_queue || width <= 0 || height <= 0) return;
    const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<uint32_t> rgba(n, 0xFF102030u);

    if (buf_img) {
        clEnqueueReadBuffer(as_q(cl_queue), as_mem(buf_img), CL_TRUE, 0, n * 4, rgba.data(), 0,
                            nullptr, nullptr);
    } else if (buf_fb && has_long_atomics) {
        // unpack ulong fb host-side: low 32 = RGBA
        std::vector<cl_ulong> packed(n);
        clEnqueueReadBuffer(as_q(cl_queue), as_mem(buf_fb), CL_TRUE, 0, n * sizeof(cl_ulong),
                            packed.data(), 0, nullptr, nullptr);
        for (size_t i = 0; i < n; ++i)
            rgba[i] = static_cast<uint32_t>(packed[i] & 0xffffffffu);
    }

    // Average-overwrite infill (host). DELIRIUM_INFILL=0 disables.
    // DELIRIUM_INFILL_R / DELIRIUM_INFILL_P override radius and passes.
    int infill_r = 1, infill_p = 3;  // r unused; p = H/V sweep pairs
    if (const char* e = std::getenv("DELIRIUM_INFILL")) {
        if (e[0] == '0' && e[1] == '\0') infill_r = 0;
    }
    if (const char* e = std::getenv("DELIRIUM_INFILL_R")) {
        unsigned v = 0;
        if (std::sscanf(e, "%u", &v) == 1 && v >= 1 && v <= 16) infill_r = int(v);
    }
    if (const char* e = std::getenv("DELIRIUM_INFILL_P")) {
        unsigned v = 0;
        if (std::sscanf(e, "%u", &v) == 1 && v >= 1 && v <= 16) infill_p = int(v);
    }
    if (infill_r > 0) {
        size_t lit0 = 0;
        for (uint32_t c : rgba) {
            const uint32_t r = c & 255u, g = (c >> 8) & 255u, b = (c >> 16) & 255u;
            if (!(r <= 28 && g <= 28 && b <= 36)) ++lit0;
        }
        average_infill(rgba, width, height, infill_r, infill_p);
        size_t lit1 = 0;
        for (uint32_t c : rgba) {
            const uint32_t r = c & 255u, g = (c >> 8) & 255u, b = (c >> 16) & 255u;
            if (!(r <= 28 && g <= 28 && b <= 36)) ++lit1;
        }
        static int s_infill_log = 0;
        if ((s_infill_log++ % 120) == 0)
            std::printf("lateral-infill: sweeps=%d lit %zu -> %zu / %zu\n",
                        infill_p, lit0, lit1, rgba.size());
    }

#if defined(DELIRIUM_HAS_SDL) && DELIRIUM_HAS_SDL
    if (sdl_rnd && sdl_tex) {
        SDL_Renderer* rnd = static_cast<SDL_Renderer*>(sdl_rnd);
        SDL_Texture* tex = static_cast<SDL_Texture*>(sdl_tex);
        void* pixels = nullptr;
        int pitch = 0;
        if (SDL_LockTexture(tex, nullptr, &pixels, &pitch) == 0) {
            auto* dst = static_cast<uint8_t*>(pixels);
            for (int y = 0; y < height; ++y)
                std::memcpy(dst + y * pitch, rgba.data() + size_t(y) * size_t(width),
                            size_t(width) * 4);
            SDL_UnlockTexture(tex);
            SDL_RenderClear(rnd);
            SDL_RenderCopy(rnd, tex, nullptr, nullptr);
            SDL_RenderPresent(rnd);
        }
    }
#else
    (void)rgba;
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
    auto relm = [](void*& p) {
        if (p) {
            clReleaseMemObject(as_mem(p));
            p = nullptr;
        }
    };
    auto relk = [](void*& p) {
        if (p) {
            clReleaseKernel(as_k(p));
            p = nullptr;
        }
    };
    relk(k_splat);
    relk(k_resolve);
    if (prog) {
        clReleaseProgram(as_prog(prog));
        prog = nullptr;
    }
    relm(buf_fb);
    relm(buf_img);
    relm(buf_anchors);
    relm(buf_surfaces);
    relm(buf_stitch);
    relm(buf_snorm);
    relm(buf_soff);
    relm(buf_sss);
    relm(buf_depth32);
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
