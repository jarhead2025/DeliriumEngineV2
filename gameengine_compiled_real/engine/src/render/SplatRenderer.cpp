#include "delirium/render/SplatRenderer.h"
#include "delirium/core/Engine.h"
#include "delirium/core/Transform.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/render/Camera.h"
#include "delirium/render/EquationAccel.h"
#include "delirium/render/LateralRayMarch.h"
#include <CL/cl.h>
#include <SDL2/SDL.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstdlib>

namespace DeliRium {

static void* k_infill = nullptr;
static void* prog_infill = nullptr;
static void* buf_img_b = nullptr;
static void* k_raymarch = nullptr;
static void* prog_raymarch = nullptr;
static void* k_obj_stitch = nullptr;
static void* prog_obj_stitch = nullptr;
static void* k_lod_stitch = nullptr;
static void* prog_lod_stitch = nullptr;
static void* buf_img2 = nullptr;
static void* buf_visible = nullptr;
static void* buf_uvn = nullptr;
static void* buf_patch_aabb = nullptr;
static void* buf_frustum = nullptr;
static size_t g_aabb_count = 0;
static EquationAccel g_accel;
static size_t g_accel_anchors = 0;

// --- Layered lateral march (kernels/surfel_march.cl) ---
// The user's method: per-pixel ray marched far->near through depth bands,
// each band drawn+filled into its OWN buffer, then composed by min-depth
// (frontmost overwrites the one behind). 6 kernels.
static void* prog_march = nullptr;
static void* k_march_clear_grid = nullptr, *k_march_build_grid = nullptr;
static void* k_march_layer_clear = nullptr, *k_march_draw = nullptr;
static void* k_march_layer_fill = nullptr, *k_march_layer_compose = nullptr;
static void* k_march_bake = nullptr;
static void* k_march_compose_fb = nullptr;
// THE LAYERED BACK-TO-FRONT PATH: one band drawn per distance plane, each
// into its OWN buffer, filled on its own, then stacked far->near.
static void* k_march_band_draw = nullptr;
static void* buf_surfels = nullptr;
static void* buf_grid = nullptr;
// single march buffer + ping-pong for the 2 fill passes
static void* buf_mcolA = nullptr, *buf_mcolB = nullptr;
static void* buf_mdepA = nullptr, *buf_mdepB = nullptr;
static uint32_t cap_surfels = 0, cap_grid = 0;
// GPU-bake bookkeeping: raw records are cached in buf_anchors/buf_surfaces/
// buf_snorm (shared with the splat path); re-upload only when the body is
// dirty. m_bake_* track what was last uploaded.
static bool   m_bake_anchors_dirty = true, m_bake_surfaces_dirty = true, m_bake_snorm_dirty = true;
static uint32_t m_bake_ns = 0, m_bake_nsn = 0;


#define CLC(call) do { cl_int _e = (call); if (_e != CL_SUCCESS) { \
    std::fprintf(stderr, "CL error %d at %s:%d\n", _e, __FILE__, __LINE__); \
    return false; } } while (0)

const char* SplatRenderer::PathName() const {
    switch (path_) {
        case Path::March:        return "march";
        case Path::Raymarch:     return "raymarch";
        case Path::SplatClassic: return "splat";
        case Path::SplatLayered: return "splat-layered";
        case Path::SplatArea:    return "splat-area";
    }
    return "unknown";
}

// The path switch announces itself: a path change that is invisible in
// the log is indistinguishable from a path change that did not happen --
// the same failure class as a silently disabled kernel, where the scene
// renders through the fallback and still looks plausible.
void SplatRenderer::SetPath(Path p) {
    path_ = p;
    // An explicit choice SUPERSEDES the env-var seed, even if the GPU is
    // initialised afterwards.
    path_explicit_ = true;
    std::printf("render path: %s\n", PathName());
}

bool SplatRenderer::Initialize(int w, int h, void* sdl_renderer,
                               const std::string& kernel_dir) {
    width = w; height = h; sdl_rnd = sdl_renderer;

    // SEED the render path from the env vars. The env vars are consulted
    // ONCE, here; from then on `path_` is the source of truth (read every
    // frame), so the editor can switch paths live. Precedence, most
    // specific first:
    //   DELIRIUM_SPLAT_LAYERS=1  -> Path::SplatLayered (the user's method)
    //   DELIRIUM_RAYMARCH=1      -> Path::Raymarch
    //   DELIRIUM_MARCH=0         -> Path::SplatClassic
    //   otherwise                -> Path::March
    {
        const char* e = std::getenv("DELIRIUM_SPLAT_LAYERS");
        const bool want_layered = e && e[0] == '1';
        const char* r = std::getenv("DELIRIUM_RAYMARCH");
        const bool want_ray = r && r[0] == '1';
        const char* m = std::getenv("DELIRIUM_MARCH");
        const bool march_off = m && m[0] == '0';
        /* The env vars SEED the path -- they do not OVERRIDE an explicit
         * choice.
         *
         * WHY: Initialize() can run AFTER a caller has already selected a
         * path (an editor that sets the path, then the engine initialises
         * the GPU). An unconditional assignment here silently discarded
         * that choice, and the symptom was the same as a path switch that
         * did not happen: the scene rendered through a path the caller
         * never asked for, and looked plausible.
         *
         * A path is only seeded if the caller has not already chosen one
         * -- `path_explicit_` is set by SetPath(). */
        if (!path_explicit_) {
            if (want_layered)          path_ = Path::SplatLayered;
            else if (want_ray)         path_ = Path::Raymarch;
            else if (march_off)        path_ = Path::SplatClassic;
            else                       path_ = Path::March;
        }
        std::printf("render path: %s\n", PathName());
    }

    cl_platform_id plats[8]; cl_uint np = 0;
    CLC(clGetPlatformIDs(8, plats, &np));
    cl_device_id dev = nullptr;
    for (cl_uint p = 0; p < np && !dev; ++p) {
        cl_device_id d; cl_uint nd = 0;
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 1, &d, &nd)
            == CL_SUCCESS && nd) dev = d;
    }
    if (!dev) {                       // CPU OpenCL still validates the path
        for (cl_uint p = 0; p < np && !dev; ++p) {
            cl_device_id d; cl_uint nd = 0;
            if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 1, &d, &nd)
                == CL_SUCCESS && nd) dev = d;
        }
    }
    if (!dev) { std::fprintf(stderr, "no OpenCL device\n"); return false; }
    cl_dev = dev;

    char info[2048] = {};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(info), info, nullptr);
    device_name = info;
    clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sizeof(info), info, nullptr);
    has_long_atomics = std::strstr(info, "cl_khr_int64_extended_atomics")
                    && std::strstr(info, "cl_khr_int64_base_atomics");
    char driver[256] = {};
    clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(driver), driver, nullptr);
    std::printf("OpenCL device: %s (driver: %s, 64-bit atomics: %s)\n",
                device_name.c_str(), driver, has_long_atomics ? "yes" : "no, two-pass");

    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CLC(err); cl_ctx = ctx;
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    CLC(err); cl_queue = q;

    if (!BuildProgram(kernel_dir)) return false;

    buf_fb  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size_t(w)*h*8, nullptr, &err); CLC(err);
    buf_img = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, size_t(w)*h*4, nullptr, &err); CLC(err);
    // Depth-tie diagnostic counter (SPLAT_TIE_DIAG). One uint, zeroed once;
    // the kernel atomic_adds a hit whenever a biased (fragment) sample lands
    // within 2e-5 NDC of the pixel's current winner. Read back per frame and
    // printed when DELIRIUM_TIE_DIAG=1. 0 = no coplanar contention.
    buf_tie_cnt = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(cl_uint),
                                 nullptr, &err); CLC(err);
    { cl_uint z = 0;
      CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_tie_cnt, CL_TRUE, 0,
                               sizeof(cl_uint), &z, 0, nullptr, nullptr)); }
    if (!has_long_atomics) {
        buf_depth32 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size_t(w)*h*4,
                                     nullptr, &err); CLC(err);
    }
    // Headless mode: no SDL renderer -> no texture. The framebuffer still
    // lives on the GPU and SaveFramePPM works, so batch/gallery renders
    // (and Engine's headless branch, which passes nullptr) work without a
    // window or a display.
    if (sdl_rnd) {
        sdl_tex = SDL_CreateTexture((SDL_Renderer*)sdl_rnd,
                                    SDL_PIXELFORMAT_RGBA32,
                                    SDL_TEXTUREACCESS_STREAMING, w, h);
        if (!sdl_tex) return false;
        // resolve_fb uses alpha=0 for background as an internal hit mask.
        // Present must copy RGB, not blend that background over old frames.
        if (SDL_SetTextureBlendMode((SDL_Texture*)sdl_tex, SDL_BLENDMODE_NONE) != 0)
            std::fprintf(stderr, "SDL texture blend setup: %s\n", SDL_GetError());
    }
    return true;
}

bool SplatRenderer::BuildProgram(const std::string& dir) {
    std::ifstream f(dir + "/psf_splat.cl");
    if (!f) { std::fprintf(stderr, "kernel file not found in %s\n", dir.c_str());
              return false; }
    std::stringstream ss; ss << f.rdbuf();
    const std::string src = ss.str();
    const char* p = src.c_str(); const size_t n = src.size();
    cl_int err = CL_SUCCESS;
    cl_program pr = clCreateProgramWithSource((cl_context)cl_ctx, 1, &p, &n, &err);
    CLC(err);
    const char* opts = has_long_atomics
        ? "-cl-fast-relaxed-math -DUSE_LONG_ATOMICS" : "-cl-fast-relaxed-math";
    if (clBuildProgram(pr, 1, (cl_device_id*)&cl_dev, opts, nullptr, nullptr)
        != CL_SUCCESS) {
        char log[16384] = {};
        clGetProgramBuildInfo(pr, (cl_device_id)cl_dev, CL_PROGRAM_BUILD_LOG,
                              sizeof(log), log, nullptr);
        std::fprintf(stderr, "kernel build log:\n%s\n", log);
        return false;
    }
    prog = pr;
    k_splat   = clCreateKernel(pr, "psf_splat", &err);  CLC(err);
    k_resolve = clCreateKernel(pr, "resolve_fb", &err); CLC(err);
    // The LAYERED distance-plane splat (the user's method): the same
    // per-sample workload, but the impacts are bucketed into distance
    // shells (bands), averaged WITHIN each shell, and the shells are
    // stacked back-to-front. See kernels/psf_splat.cl for the algorithm.
    k_splat_layered = clCreateKernel(pr, "psf_splat_layered", &err);
    k_layer_fill_band = clCreateKernel(pr, "layer_fill_band", &err);
    k_layer_stack_band = clCreateKernel(pr, "layer_stack_band", &err);
    // The AREA-WEIGHTED splat: the pixel-footprint pyramid. Each surface
    // sample is distributed over the pixels its footprint covers, with
    // the area as the weight, and the pixel's colour is the
    // area-weighted mean of everything that touched it. See
    // kernels/psf_splat.cl for the algorithm and the derivation.
    k_splat_area = clCreateKernel(pr, "psf_splat_area", &err);
    k_area_resolve = clCreateKernel(pr, "area_resolve", &err);
    if (err != CL_SUCCESS) {
        k_splat_layered = k_layer_fill_band = k_layer_stack_band = nullptr;
        k_splat_area = k_area_resolve = nullptr;
    }

    // One-time signature audit for the SPLAT kernel too (same drift class of
    // bug that broke march with CL_INVALID_VALUE -51): the host sets a fixed
    // arg count per build config; if the .cl signature drifted, catch it once
    // with a clear message instead of failing every frame.
    // psf_splat arg order (must match kernels/psf_splat.cl):
    //  0 anchors, 1 snorm, 2 surfaces, 3 stitch, 4 sample_off, 5 sample_ss,
    //  6 n_surfaces, 7 bbmin, 8 bbspan, 9 s_scale, 10 mvp, 11 light_dir,
    // 12 eye, 13 model_r0, 14 model_r1, 15 model_r2, 16 model_r3,
    // 17 width, 18 height, 19 depth_bias, 20 fb, 21 tie_cnt,
    //  [!USE_LONG_ATOMICS] 22 depth32, 23 pass
    //
    // The four model_r* args carry the object's LOCAL->WORLD placement. The
    // 64-byte records are FROZEN in local space, so moving an object is a
    // GPU-side transform -- no re-quantization, no anchor re-upload.
    {
        const cl_uint expect_splat = has_long_atomics ? 22 : 24;
        cl_uint n = 0;
        cl_int e3 = clGetKernelInfo((cl_kernel)k_splat, CL_KERNEL_NUM_ARGS,
                                    sizeof(n), &n, nullptr);
        if (e3 == CL_SUCCESS && n != expect_splat) {
            std::fprintf(stderr,
                "SplatRenderer: psf_splat arg mismatch (kernel=%u host=%u) -- "
                "kernel/host drift; disabling splat path.\n", n, expect_splat);
            k_splat = nullptr;
        }
    }

    // The same audit for the LAYERED splat kernels. A drift here would
    // make clSetKernelArg fail with CL_INVALID_VALUE on every frame, so
    // catch it once and disable the layered path (the classic path keeps
    // working).
    {
        struct { const char* name; cl_kernel k; cl_uint expect; } checks[] = {
            // psf_splat_layered: 0 anchors, 1 snorm, 2 surfaces, 3 stitch,
            // 4 sample_off, 5 sample_ss, 6 n_surfaces, 7 bbmin, 8 bbspan,
            // 9 s_scale, 10 mvp, 11 light_dir, 12 eye, 13 view_fwd,
            // 14-17 model_r*, 18 width, 19 height, 20 depth_bias,
            // 21 d_near, 22 d_far, 23 nbands, 24 band_key
            {"psf_splat_layered", (cl_kernel)k_splat_layered, 25},
            // layer_fill_band: 0 kin, 1 kout, 2 W, 3 H, 4 slot, 5 INF
            {"layer_fill_band",   (cl_kernel)k_layer_fill_band, 6},
            // layer_stack_band: 0 keys, 1 W, 2 H, 3 slot, 4 INF, 5 fb,
            // 6 depth32
            {"layer_stack_band",  (cl_kernel)k_layer_stack_band, 7},
        };
        bool ok = true;
        for (auto& c : checks) {
            if (!c.k) continue;
            cl_uint n = 0;
            cl_int e3 = clGetKernelInfo(c.k, CL_KERNEL_NUM_ARGS, sizeof(n), &n, nullptr);
            if (e3 != CL_SUCCESS || n != c.expect) {
                std::fprintf(stderr,
                    "SplatRenderer: %s arg mismatch (kernel=%u host=%u) -- "
                    "kernel/host drift; disabling the layered splat path.\n",
                    c.name, n, c.expect);
                ok = false;
            }
        }
        if (!ok) k_splat_layered = k_layer_fill_band = k_layer_stack_band = nullptr;
    }

    // The same audit for the AREA-WEIGHTED splat. A SEPARATE block so a
    // drift here disables only the area path -- the classic and layered
    // paths keep working.
    //
    // psf_splat_area: 0 anchors, 1 snorm, 2 surfaces, 3 stitch,
    // 4 sample_off, 5 sample_ss, 6 n_surfaces, 7 bbmin, 8 bbspan,
    // 9 s_scale, 10 mvp, 11 light_dir, 12 eye, 13 view_fwd,
    // 14-17 model_r*, 18 width, 19 height, 20 depth_bias, 21 tan_half,
    // 22 area_csum, 23 area_wsum, 24 area_key
    // area_resolve: 0 area_csum, 1 area_wsum, 2 area_key, 3 W, 4 H, 5 image
    {
        struct { const char* name; cl_kernel k; cl_uint expect; } checks[] = {
            {"psf_splat_area", (cl_kernel)k_splat_area, 25},
            {"area_resolve",    (cl_kernel)k_area_resolve, 6},
        };
        bool ok = true;
        for (auto& c : checks) {
            if (!c.k) continue;
            cl_uint n = 0;
            cl_int e3 = clGetKernelInfo(c.k, CL_KERNEL_NUM_ARGS, sizeof(n), &n, nullptr);
            if (e3 != CL_SUCCESS || n != c.expect) {
                std::fprintf(stderr,
                    "SplatRenderer: %s arg mismatch (kernel=%u host=%u) -- "
                    "kernel/host drift; disabling the area-weighted splat path.\n",
                    c.name, n, c.expect);
                ok = false;
            }
        }
        if (!ok) k_splat_area = k_area_resolve = nullptr;
    }

    // GPU hole-fill kernel from fb_infill.cl (separate program)
    {
        std::ifstream fi(dir + "/fb_infill.cl");
        if (!fi) fi.open("kernels/fb_infill.cl");
        if (fi) {
            std::stringstream ss; ss << fi.rdbuf();
            const std::string isrc = ss.str();
            const char* ip = isrc.c_str(); const size_t in = isrc.size();
            cl_program ipr = clCreateProgramWithSource((cl_context)cl_ctx, 1, &ip, &in, &err);
            if (err == CL_SUCCESS &&
                clBuildProgram(ipr, 1, (cl_device_id*)&cl_dev, "-cl-fast-relaxed-math",
                               nullptr, nullptr) == CL_SUCCESS) {
                k_infill = clCreateKernel(ipr, "fb_infill_pass", &err);
                if (err != CL_SUCCESS) k_infill = nullptr;
                // keep program alive via prog_infill opaque
                prog_infill = ipr;
                std::printf("GPU fb_infill: ready\n");
            } else {
                std::fprintf(stderr, "fb_infill build failed (optional)\n");
                if (ipr) clReleaseProgram(ipr);
            }
        }
    }
    
    // GPU equation ray-march
    {
        std::ifstream fi(dir + "/psf_raymarch.cl");
        if (!fi) fi.open("kernels/psf_raymarch.cl");
        if (fi) {
            std::stringstream ss; ss << fi.rdbuf();
            const std::string rsrc = ss.str();
            const char* rp = rsrc.c_str(); const size_t rn = rsrc.size();
            cl_int e2 = CL_SUCCESS;
            cl_program rpr = clCreateProgramWithSource((cl_context)cl_ctx, 1, &rp, &rn, &e2);
            if (e2 == CL_SUCCESS &&
                clBuildProgram(rpr, 1, (cl_device_id*)&cl_dev, "-cl-fast-relaxed-math",
                               nullptr, nullptr) == CL_SUCCESS) {
                k_raymarch = clCreateKernel(rpr, "psf_raymarch", &e2);
                if (e2 != CL_SUCCESS) k_raymarch = nullptr;
                prog_raymarch = rpr;
                std::printf("GPU psf_raymarch: ready\n");
            } else {
                char log[8192]={};
                if (rpr) clGetProgramBuildInfo(rpr, (cl_device_id)cl_dev, CL_PROGRAM_BUILD_LOG,
                    sizeof(log), log, nullptr);
                std::fprintf(stderr, "psf_raymarch build failed:\n%s\n", log);
                if (rpr) clReleaseProgram(rpr);
            }
        }
    }

    // Object-masked lateral stitch (same-alpha average only)
    {
        std::ifstream fi(dir + "/fb_object_stitch.cl");
        if (!fi) fi.open("kernels/fb_object_stitch.cl");
        if (fi) {
            std::stringstream ss; ss << fi.rdbuf();
            const std::string src = ss.str();
            const char* sp = src.c_str(); const size_t sn = src.size();
            cl_int e2 = CL_SUCCESS;
            cl_program pr = clCreateProgramWithSource((cl_context)cl_ctx, 1, &sp, &sn, &e2);
            if (e2 == CL_SUCCESS &&
                clBuildProgram(pr, 1, (cl_device_id*)&cl_dev, "-cl-fast-relaxed-math", nullptr, nullptr) == CL_SUCCESS) {
                k_obj_stitch = clCreateKernel(pr, "fb_object_stitch", &e2);
                if (e2 != CL_SUCCESS) k_obj_stitch = nullptr;
                prog_obj_stitch = pr;
                std::printf("GPU object-stitch: ready\n");
            } else {
                if (pr) clReleaseProgram(pr);
            }
        }
    }

    // LOD / distance-aware surface stitch (variable radius average of equation impacts)
    {
        std::ifstream fi(dir + "/fb_lod_stitch.cl");
        if (!fi) fi.open("kernels/fb_lod_stitch.cl");
        if (fi) {
            std::stringstream ss; ss << fi.rdbuf();
            const std::string src = ss.str();
            const char* sp = src.c_str(); const size_t sn = src.size();
            cl_int e2 = CL_SUCCESS;
            cl_program pr = clCreateProgramWithSource((cl_context)cl_ctx, 1, &sp, &sn, &e2);
            if (e2 == CL_SUCCESS &&
                clBuildProgram(pr, 1, (cl_device_id*)&cl_dev, "-cl-fast-relaxed-math", nullptr, nullptr) == CL_SUCCESS) {
                k_lod_stitch = clCreateKernel(pr, "fb_lod_stitch", &e2);
                if (e2 != CL_SUCCESS) k_lod_stitch = nullptr;
                prog_lod_stitch = pr;
                std::printf("GPU lod-stitch: ready\n");
            } else {
                if (pr) clReleaseProgram(pr);
            }
        }
    }

    // Layered lateral march (kernels/surfel_march.cl) — the user's method.
    {
        std::ifstream fi(dir + "/surfel_march.cl");
        if (!fi) fi.open("kernels/surfel_march.cl");
        if (fi) {
            std::stringstream ss; ss << fi.rdbuf();
            const std::string src = ss.str();
            const char* sp = src.c_str(); const size_t sn = src.size();
            cl_int e2 = CL_SUCCESS;
            cl_program pr = clCreateProgramWithSource((cl_context)cl_ctx, 1, &sp, &sn, &e2);
            // march_compose_fb uses atom_min on the 64-bit depth|rgba key when
            // the device has cl_khr_int64_extended_atomics (march is the
            // primary renderer for ALL bodies, so it must interleave by depth
            // exactly like psf_splat). Same options as the splat program.
            const char* mopts = has_long_atomics
                ? "-cl-fast-relaxed-math -DUSE_LONG_ATOMICS" : "-cl-fast-relaxed-math";
            if (e2 == CL_SUCCESS &&
                clBuildProgram(pr, 1, (cl_device_id*)&cl_dev, mopts, nullptr, nullptr) == CL_SUCCESS) {
                k_march_clear_grid  = clCreateKernel(pr, "clear_grid", &e2);
                k_march_bake        = clCreateKernel(pr, "bake_surfels", &e2);
                k_march_build_grid  = clCreateKernel(pr, "build_grid", &e2);
                k_march_layer_clear = clCreateKernel(pr, "layer_clear", &e2);
                k_march_draw        = clCreateKernel(pr, "march_draw", &e2);
                k_march_layer_fill  = clCreateKernel(pr, "layer_fill", &e2);
                k_march_layer_compose = clCreateKernel(pr, "layer_compose", &e2);
                k_march_band_draw     = clCreateKernel(pr, "layer_draw_band", &e2);
                k_march_compose_fb  = clCreateKernel(pr, "march_compose_fb", &e2);
                if (e2 != CL_SUCCESS) {
                    k_march_clear_grid = k_march_bake = k_march_build_grid = k_march_layer_clear = nullptr;
                    k_march_draw = k_march_layer_fill = k_march_layer_compose = nullptr;
                    k_march_compose_fb = nullptr;
                }
                // One-time signature audit: the host sets a FIXED number of
                // args per kernel (see the clSetKernelArg blocks in the march
                // path). If the .cl signature drifted (a param added/removed
                // without updating the host), clSetKernelArg fails with
                // CL_INVALID_VALUE (-51) on EVERY frame. Catch it once here
                // with a clear message and disable the path instead.
                // march_draw arg order (must match kernels/surfel_march.cl):
                //  0 S, 1 grid, 2 gx, 3 gy, 4 gz, 5 org, 6 cell, 7 eye,
                //  8 fwd, 9 right, 10 up, 11 tanfov, 12 W, 13 H, 14 d_near,
                // 15 d_far, 16 steps, 17 foot, 18 light, 19-22 mvp_s0123/
                // 4567/89ab/scdef, 23 color, 24 depth, 25 use_slab,
                // 26 accumulate
                {
                    struct { const char* name; cl_kernel k; cl_uint expect; } checks[] = {
                        {"clear_grid",       (cl_kernel)k_march_clear_grid, 2},
                        {"bake_surfels",     (cl_kernel)k_march_bake,      13},
                        {"build_grid",       (cl_kernel)k_march_build_grid, 8},
                        {"layer_clear",      (cl_kernel)k_march_layer_clear,4},
                        {"march_draw",       (cl_kernel)k_march_draw,       27},
                        // layer_draw_band arg order (kernels/surfel_march.cl):
                        //  0 S, 1 grid, 2 gx, 3 gy, 4 gz, 5 org, 6 cell,
                        //  7 eye, 8 fwd, 9 right, 10 up, 11 tanfov, 12 aspect,
                        //  13 W, 14 H, 15 d_far, 16 d_near, 17 bsteps,
                        //  18 foot, 19 light, 20-23 mvp_s0123/4567/89ab/scdef,
                        //  24 color, 25 depth
                        //
                        // NOTE: the band kernel takes NO model-matrix rows --
                        // the surfels it reads are ALREADY in world space,
                        // because bake_surfels applies the model matrix when
                        // it writes them. A previous revision of this audit
                        // expected 28 args (as if the rows were kernel args),
                        // which MISMATCHED and silently disabled the ENTIRE
                        // march path -- "path 1" then rendered as splat and
                        // looked plausible. The audit count must equal what
                        // the host actually sets: 26.
                        {"layer_draw_band",  (cl_kernel)k_march_band_draw,  26},
                        {"layer_fill",       (cl_kernel)k_march_layer_fill, 9},
                        {"march_compose_fb", (cl_kernel)k_march_compose_fb, 9},
                    };
                    bool sig_ok = true;
                    for (auto& c : checks) {
                        if (!c.k) continue;
                        cl_uint n = 0;
                        cl_int e3 = clGetKernelInfo(c.k, CL_KERNEL_NUM_ARGS,
                            sizeof(n), &n, nullptr);
                        if (e3 != CL_SUCCESS || n != c.expect) {
                            std::fprintf(stderr,
                                "surfel_march signature mismatch: kernel '%s' has %u args, "
                                "host expects %u (e3=%d). Check kernels/surfel_march.cl "
                                "matches the host clSetKernelArg order. Disabling march path.\n",
                                c.name, n, c.expect, (int)e3);
                            sig_ok = false;
                        }
                    }
                    if (!sig_ok) {
                        k_march_clear_grid = k_march_bake = k_march_build_grid = k_march_layer_clear = nullptr;
                        k_march_draw = k_march_layer_fill = k_march_layer_compose = nullptr;
                        k_march_compose_fb = nullptr;
                    }
                }
                prog_march = pr;
                std::printf("GPU surfel_march (single-pass lateral): ready\n");
            } else {
                char log[8192] = {};
                if (pr) clGetProgramBuildInfo(pr, (cl_device_id)cl_dev, CL_PROGRAM_BUILD_LOG,
                    sizeof(log), log, nullptr);
                std::fprintf(stderr, "surfel_march build failed:\n%s\n", log);
                if (pr) clReleaseProgram(pr);
            }
        }
    }
return true;
}

uint32_t SplatRenderer::BuildSampleGrids(PSFBody& body, const float mvp[16]) {
    const auto& S = body.SurfaceBlob();
    const uint32_t ns = (uint32_t)S.size();
    h_soff.assign(ns + 1, 0);
    h_sss.assign(ns, 0);
    const auto& Ablob = body.AnchorBlob();
    Vec3 bbmin = body.BBoxMin(), bbspan = body.BBoxSpan();
    if (bbspan.x == 0.f && bbspan.y == 0.f && bbspan.z == 0.f) {
        bbmin = {-1.f, -1.f, -1.f};
        bbspan = {2.f, 2.f, 2.f};
    }
    // Byte-exact dequant (offset 32, ushort[3]) — same as kernels/psf_splat.cl
    auto dequant = [&](uint16_t ai) -> Vec3 {
        if (ai >= Ablob.size()) return Vec3{0,0,0};
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&Ablob[ai]);
        uint16_t px, py, pz;
        std::memcpy(&px, bytes + 32, 2);
        std::memcpy(&py, bytes + 34, 2);
        std::memcpy(&pz, bytes + 36, 2);
        return Vec3{
            bbmin.x + (px / 65535.f) * bbspan.x,
            bbmin.y + (py / 65535.f) * bbspan.y,
            bbmin.z + (pz / 65535.f) * bbspan.z};
    };
    auto project = [&](Vec3 P, float& x, float& y) {
        const float px = mvp[0]*P.x + mvp[1]*P.y + mvp[2]*P.z + mvp[3];
        const float py = mvp[4]*P.x + mvp[5]*P.y + mvp[6]*P.z + mvp[7];
        const float pw = mvp[12]*P.x + mvp[13]*P.y + mvp[14]*P.z + mvp[15];
        const float w = pw > 1e-6f ? pw : 1e-6f;
        x = (px/w*0.5f + 0.5f) * (width  - 1);
        y = (0.5f - py/w*0.5f) * (height - 1);
    };
    float density = 2.0f; // quality baseline; close-up costs grow ~quadratically
    if (const char* e = std::getenv("DELIRIUM_SPLAT_DENSITY")) {
        float v = 0.f;
        if (std::sscanf(e, "%f", &v) == 1 && v >= 0.75f && v <= 2.5f)
            density = v;
    }
    // Diagnostic only: reduce close-up samples without changing the normal
    // quality path. Compare work counts AND frame time; a speedup is not assumed.
    uint32_t diagnostic_cap = 320;
    if (const char* e = std::getenv("DELIRIUM_SPLAT_AXIS_CAP")) {
        unsigned v = 0; char tail = 0;
        if (std::sscanf(e, "%u%c", &v, &tail) == 1 && v >= 10 && v <= 320)
            diagnostic_cap = v;
    }
    uint64_t total = 0;
    for (uint32_t s = 0; s < ns; ++s) {
        float x[4], y[4];
        for (int k = 0; k < 4; ++k) {
            Vec3 p = dequant(S[s].anchor_idx[k]);
            project(p, x[k], y[k]);
        }
        auto elen = [&](int a, int b) {
            return std::sqrt((x[a]-x[b])*(x[a]-x[b]) + (y[a]-y[b])*(y[a]-y[b]));
        };
        // u runs A0->A3 (and A1->A2); v runs A0->A1 (and A3->A2)
        const float eu = std::fmax(elen(0,3), elen(1,2));
        const float ev = std::fmax(elen(0,1), elen(3,2));
        // Projected size grows as 1/distance; evaluating a 2D grid makes
        // sample work grow roughly as 1/distance^2. A camera/FOV illusion
        // cannot change that for the same on-screen detail. Keep full
        // projected-edge coverage by default (including close-up contours).
        // A lower quality mode can trade detail for speed, deliberately.
        auto clampg = [&](float e) {
            const int g = int(e * density) + 3;
            const int hi = int(std::fmax(eu, ev) * (density + 0.05f)) + 6;
            const int cap = std::max(12, std::min(320, hi));
            return (uint32_t)std::min<int>(diagnostic_cap, std::max(10, std::min(cap, g)));
        };
        const uint32_t su = clampg(eu), sv = clampg(ev);
        h_sss[s] = (su << 16) | sv;
        h_soff[s] = (uint32_t)total;
        total += uint64_t(su) * sv;
    }
    h_soff[ns] = (uint32_t)total;
    return (uint32_t)total;
}

bool SplatRenderer::RenderBody(PSFBody& body, const Camera& cam, bool clear, bool final,
                               float depth_bias, const float model_in[16]) {
    cl_context ctx = (cl_context)cl_ctx;
    cl_command_queue q = (cl_command_queue)cl_queue;
    cl_int err = CL_SUCCESS;

    // MODEL MATRIX (local -> world). The 64-byte records hold the body in
    // its own local space; the placement lives outside them so that moving
    // an object never re-quantizes an anchor. Identity when not supplied.
    float model[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    if (model_in) std::memcpy(model, model_in, sizeof(model));
    const cl_float4 mr0 = {{model[0], model[1], model[2], model[3]}};
    const cl_float4 mr1 = {{model[4], model[5], model[6], model[7]}};
    const cl_float4 mr2 = {{model[8], model[9], model[10], model[11]}};
    const cl_float4 mr3 = {{model[12], model[13], model[14], model[15]}};

    // The AREA-WEIGHTED path resolves into the SHARED framebuffer (the
    // same (depth|rgba) surface every other path uses), so the shared
    // resolve_fb runs for it exactly as it does for the classic path.
    // No per-path flag is needed.

    // =====================================================================
    // LAYERED LATERAL MARCH (kernels/surfel_march.cl) — the user's method,
    // and (2026-10-28) THE PRIMARY RENDERER for every body:
    //   small per-ray impacts (foot) -> fill ONLY between impact points
    //   (layer_fill) -> stack front-to-back, frontmost overwrites the back
    //   (atom_min on the depth|rgba key in the shared buf_fb) -> resolve to
    //   the image. Splat is no longer the main path; it remains only as the
    //   DELIRIUM_MARCH=0 fallback. `clear` (body 0) now only controls the
    //   frame reset (buf_fb clear); every body composes on top of it.
    // =====================================================================
    // The RENDER PATH. The env vars are read ONCE (a `static const int`
    // initialiser), so they cannot express a runtime switch -- the editor
    // needs to change the path live. `path_` is read EVERY FRAME; the env
    // vars only SEED it, in Initialize().
    const bool march_on = (path_ == Path::March);
    const bool ray_on   = (path_ == Path::Raymarch);
    const bool layered_on = (path_ == Path::SplatLayered);
    if (march_on && buf_img && buf_fb &&
        k_march_clear_grid && k_march_bake && k_march_build_grid && k_march_layer_clear &&
        k_march_draw && k_march_layer_fill && k_march_compose_fb && k_resolve) {
            // fill_min: min non-black neighbors for a hole pixel to be filled
            // (the user's "grab 3 pixels that aren't all zero, average").
            // fill_passes: each pass propagates the fill 1px outward, so N
            // passes bridge gaps up to ~N px. The single-pixel impacts are
            // spaced ~3-4px apart, so the old default (2 passes) left 1px
            // holes; in overlap areas the surface BEHIND shows through those
            // holes -> the zebra/screen-door. 4 passes bridges the gaps while
            // keeping fill_min=3 (the user's spec). Each pass is a cheap
            // full-screen W*H read/avg/write.
            int uvn = 10, bsteps = 40, fill_min = 3, fill_passes = 4;
            if (const char* e = std::getenv("DELIRIUM_MARCH_UVN")) { int v=0; if (std::sscanf(e,"%d",&v)==1 && v>=2 && v<=16) uvn=v; }
            if (const char* e = std::getenv("DELIRIUM_MARCH_STEPS")) { int v=0; if (std::sscanf(e,"%d",&v)==1 && v>=1 && v<=256) bsteps=v; }
            if (const char* e = std::getenv("DELIRIUM_MARCH_FILL")) { int v=0; if (std::sscanf(e,"%d",&v)==1 && v>=1 && v<=8) fill_min=v; }
            if (const char* e = std::getenv("DELIRIUM_MARCH_PASSES")) { int v=0; if (std::sscanf(e,"%d",&v)==1 && v>=1 && v<=16) fill_passes=v; }

            // ---- frame reset: clear the SHARED framebuffer -----------------
            // ONLY body 0 (clear=true) resets the frame. Every body now
            // marches, so this MUST stay gated on `clear`: if it ran per
            // body, each body would wipe the previous bodies' accumulated
            // image and only the last body would be visible. Clearing to the
            // far key lets the first body store cleanly and every later body
            // atom_min ON TOP of it by depth (frontmost wins).
            if (clear && buf_fb) {
                const cl_ulong far_key = ~cl_ulong(0);
                CLC(clEnqueueFillBuffer(q, (cl_mem)buf_fb, &far_key, 8, 0,
                                        size_t(width)*height*8, 0, nullptr, nullptr));
                if (!has_long_atomics && buf_depth32) {
                    const cl_uint far_d = ~cl_uint(0);
                    CLC(clEnqueueFillBuffer(q, (cl_mem)buf_depth32, &far_d, 4, 0,
                                            size_t(width)*height*4, 0, nullptr, nullptr));
                }
            }

            Vec3 bmn = body.BBoxMin(), bsp = body.BBoxSpan();
            if (bsp.x < 1e-8f && bsp.y < 1e-8f && bsp.z < 1e-8f) {
                bmn = Vec3{-1.f,-1.f,-1.f}; bsp = Vec3{2.f,2.f,2.f};
            }
            // LOCAL bounds (bmn/bsp) dequantize the anchors inside
            // bake_surfels -> they MUST stay in the body's local space.
            //
            // WORLD bounds (wmn/wsp) are the GRID's bounds: bake_surfels
            // transforms every surfel to world space with the MODEL MATRIX,
            // so build_grid (which indexes the surfels) and march_draw (which
            // marches that grid) must both work in world space. For an
            // identity transform they are the same box.
            //
            // The deformed-surface union below grows the LOCAL box; the world
            // box is derived from it AFTER that union (so a deformed surface
            // that escapes its anchor bbox is still covered).
            Vec3 wmn = bmn, wsp = bsp;
            auto refresh_world_bounds = [&]() {
                if (!model_in) { wmn = bmn; wsp = bsp; return; }
                wmn = Vec3{1e30f,1e30f,1e30f};
                Vec3 wmx{-1e30f,-1e30f,-1e30f};
                for (int c = 0; c < 8; ++c) {
                    const Vec3 corner{ bmn.x + ((c&1)?bsp.x:0.f),
                                      bmn.y + ((c&2)?bsp.y:0.f),
                                      bmn.z + ((c&4)?bsp.z:0.f) };
                    const Vec3 w = Mat4_TransformPoint(*reinterpret_cast<const Mat4*>(model), corner);
                    wmn.x=std::min(wmn.x,w.x); wmn.y=std::min(wmn.y,w.y); wmn.z=std::min(wmn.z,w.z);
                    wmx.x=std::max(wmx.x,w.x); wmx.y=std::max(wmx.y,w.y); wmx.z=std::max(wmx.z,w.z);
                }
                wsp = { wmx.x-wmn.x, wmx.y-wmn.y, wmx.z-wmn.z };
            };
            refresh_world_bounds();
            const int GRID = 256;
            // The grid indexes WORLD-space surfels, so cell/diag/foot are
            // world-space quantities. For an identity transform world ==
            // local and this is byte-identical to the pre-transform code.
            float diag = std::sqrt(wsp.x*wsp.x + wsp.y*wsp.y + wsp.z*wsp.z);
            float cell = diag / float(GRID);
            // Hit radius = per-surfel DISK size on screen (radius ~ foot*focal/z).
            //
            // foot = max(2*cell, 1.6*spacing)  -- the PROVEN Oct-8 formula.
            //
            // Why 2*cell: the march's grid stores ONE surfel index per cell, so
            // 34.4% of the 54,400 baked surfels are silently dropped by cell
            // collisions on the torus (measured: 54,400 surfels -> 35,696
            // occupied cells, max 4 colliding in one cell). The surviving
            // surfels are therefore IRREGULARLY SPACED -- the effective local
            // spacing can be several times the nominal `spacing`. A hit radius
            // sized for the NOMINAL spacing (0.5*spacing) is far smaller than
            // the real gaps, so rays slip between surfels and every hit is a
            // single isolated surfel -> salt-and-pepper shading noise (measured
            // 52% edge density, 846 surface holes on scene A).
            //
            // foot = 2*cell covers the grid's own quantization: a surfel within
            // `foot` of the sample point is at most floor(foot/cell) cells away,
            // and the kernel's adaptive neighbourhood radius nr = 1+floor(foot/cell)
            // is capped at 2 -> the 5x5x5 search spans exactly 2*cell, matching
            // the hit radius. (The hard invariant: foot <= nr_max*cell, else the
            // search radius is smaller than the hit test and hits are missed.)
            //
            // The 1.6*spacing term is a secondary floor for the case where the
            // surfel grid is finer than the grid cell (small bodies, high uvn).
            //
            // Measured on the torus (640x480, scene A):
            //   foot=max(0.5*cell,0.5*spacing) -> 846 holes, 52.2% edge density
            //   foot=max(2.0*cell,1.6*spacing) ->  20 holes,  7.0% edge density
            // The smaller formula was introduced on 2026-10-26 to avoid "blobby"
            // overlapping disks, but it ignored the grid's own quantization loss.
            //
            // Tune live with DELIRIUM_MARCH_FOOT (multiplier on the computed foot).
            const auto& Sref = body.SurfaceBlob();
            float patch_size = diag / std::sqrt(std::max<size_t>(1, Sref.size()));
            float spacing = patch_size / float(uvn);
            float foot = std::max(2.0f * cell, 1.6f * spacing);
            // Live tune: DELIRIUM_MARCH_FOOT=<mult> scales the per-surfel disk.
            // <1 = smaller impacts (more fill work, finer surface); >1 = bigger
            // disks (coarser, more overlap). 0.5 = half-size, 2.0 = old blobby.
            if (const char* e = std::getenv("DELIRIUM_MARCH_FOOT")) {
                float m = 0.f;
                if (std::sscanf(e, "%f", &m) == 1 && m > 0.05f && m < 8.f)
                    foot *= m;
            }

            // ---- deformed-surface bounds (grid + march range) -------------
            // The anchor bbox does NOT bound the deformed surface: a patch's
            // 8 coefficients displace its interior along the anchor normals,
            // and a curved body (e.g. an ellipsoid dome) can rise far above
            // its flat anchor plane. The old grid/range used the anchor bbox,
            // so march_draw's cell lookup (cx<0||cx>=gx...) skipped every
            // sample of a surface that escaped the box -> 0 lit pixels at
            // the window camera (the "black screen"). The raymarch path
            // already does this via EquationAccel's deform margin; do the
            // same here: evaluate each patch at 5 points (4 corners +
            // center) with the exact bake_surfels math (bilinear anchors +
            // deform_h along the bilinear anchor normal) and take the
            // min/max. ~5*ns evals/frame is negligible (ns ~ 900).
            {
                const auto& Ab = body.AnchorBlob();
                const auto& Sb = body.SurfaceBlob();
                auto dq = [&](uint16_t ai) -> Vec3 {
                    if (ai >= Ab.size()) return Vec3{0,0,0};
                    const uint8_t* by = reinterpret_cast<const uint8_t*>(&Ab[ai]);
                    uint16_t px, py, pz;
                    std::memcpy(&px, by + 32, 2);
                    std::memcpy(&py, by + 34, 2);
                    std::memcpy(&pz, by + 36, 2);
                    return Vec3{ bmn.x + (px / 65535.f) * bsp.x,
                                 bmn.y + (py / 65535.f) * bsp.y,
                                 bmn.z + (pz / 65535.f) * bsp.z };
                };
                auto dn = [&](uint16_t ai) -> Vec3 {
                    if (ai >= Ab.size()) return Vec3{0,1,0};
                    const uint8_t* by = reinterpret_cast<const uint8_t*>(&Ab[ai]);
                    float nx = by[44] * (2.f/255.f) - 1.f;
                    float ny = by[45] * (2.f/255.f) - 1.f;
                    float nz = by[46] * (2.f/255.f) - 1.f;
                    float l = std::sqrt(nx*nx + ny*ny + nz*nz);
                    if (l < 1e-6f) return Vec3{0,1,0};
                    return Vec3{nx/l, ny/l, nz/l};
                };
                auto dh = [&](const float c[8], float u, float v) -> float {
                    const float bu = u*(1.f-u), bv = v*(1.f-v);
                    const float au = 2.f*u-1.f, av = 2.f*v-1.f;
                    return c[0]*bu*(1.f-v) + c[1]*bu*v + c[2]*bv*(1.f-u) + c[3]*bv*u
                         + c[4]*bu*au*(1.f-v) + c[5]*bu*au*v
                         + c[6]*bv*av*(1.f-u) + c[7]*bv*av*u;
                };
                float dmnx=1e30f, dmny=1e30f, dmnz=1e30f,
                      dmxn=-1e30f, dmxn2=-1e30f, dmxz=-1e30f;
                const float PUV[5][2] = {{0.f,0.f},{1.f,0.f},{0.f,1.f},{1.f,1.f},{0.5f,0.5f}};
                for (const auto& s : Sb) {
                    const Vec3 A0 = dq(s.anchor_idx[0]), A1 = dq(s.anchor_idx[1]);
                    const Vec3 A2 = dq(s.anchor_idx[2]), A3 = dq(s.anchor_idx[3]);
                    const Vec3 N0 = dn(s.anchor_idx[0]), N1 = dn(s.anchor_idx[1]);
                    const Vec3 N2 = dn(s.anchor_idx[2]), N3 = dn(s.anchor_idx[3]);
                    for (int k = 0; k < 5; ++k) {
                        const float uu = PUV[k][0], vv = PUV[k][1];
                        // bilinear anchors + bilinear normal (bake_surfels order)
                        const Vec3 P{ (1-vv)*((1-uu)*A0.x + uu*A1.x) + vv*((1-uu)*A3.x + uu*A2.x),
                                      (1-vv)*((1-uu)*A0.y + uu*A1.y) + vv*((1-uu)*A3.y + uu*A2.y),
                                      (1-vv)*((1-uu)*A0.z + uu*A1.z) + vv*((1-uu)*A3.z + uu*A2.z) };
                        Vec3 Nd{ (1-vv)*((1-uu)*N0.x + uu*N1.x) + vv*((1-uu)*N3.x + uu*N2.x),
                                 (1-vv)*((1-uu)*N0.y + uu*N1.y) + vv*((1-uu)*N3.y + uu*N2.y),
                                 (1-vv)*((1-uu)*N0.z + uu*N1.z) + vv*((1-uu)*N3.z + uu*N2.z) };
                        const float nl = std::sqrt(Nd.x*Nd.x + Nd.y*Nd.y + Nd.z*Nd.z);
                        if (nl < 1e-6f) { Nd.x=0; Nd.y=1; Nd.z=0; }
                        else { Nd.x/=nl; Nd.y/=nl; Nd.z/=nl; }
                        const float h = dh(s.coef, uu, vv);
                        const float wx = P.x + Nd.x*h, wy = P.y + Nd.y*h, wz = P.z + Nd.z*h;
                        dmnx=std::min(dmnx,wx); dmny=std::min(dmny,wy); dmnz=std::min(dmnz,wz);
                        dmxn=std::max(dmxn,wx); dmxn2=std::max(dmxn2,wy); dmxz=std::max(dmxz,wz);
                    }
                }
                if (dmnx <= dmxn && dmny <= dmxn2 && dmnz <= dmxz) {
                    // union with the anchor bbox so nothing regresses
                    bmn = Vec3{ std::min(bmn.x, dmnx), std::min(bmn.y, dmny), std::min(bmn.z, dmnz) };
                    bsp = Vec3{ std::max(bsp.x, dmxn - bmn.x),
                                std::max(bsp.y, dmxn2 - bmn.y),
                                std::max(bsp.z, dmxz - bmn.z) };
                    // The LOCAL box grew -> the WORLD grid box must follow.
                    refresh_world_bounds();
                    // The grid cell and the hit radius are WORLD-space
                    // quantities (the grid indexes world-space surfels), so
                    // they are derived from the WORLD diagonal. For an
                    // identity transform world == local (unchanged behavior).
                    const float wdiag = std::sqrt(wsp.x*wsp.x + wsp.y*wsp.y + wsp.z*wsp.z);
                    diag = wdiag;
                    cell = wdiag / float(GRID);
                    // Track the (possibly grown) cell at the SAME small ratio as
                    // the initial foot -- do NOT re-clamp to 2*cell. That old
                    // 2*cell floor is what made every surfel paint a disk ~2
                    // grid-cells wide (the "round surfels too big" bug): it
                    // negated any attempt to shrink foot. 0.5*cell keeps the
                    // disk just under one cell; the 5x5x5 march_draw
                    // neighborhood (radius ~2.5*cell) still comfortably covers
                    // it, so no hit is missed.
                    foot = std::max(foot, 0.5f * cell);
                }
            }

            // ---- GPU surfel bake (kernels/surfel_march.cl::bake_surfels) ----
            // Replaces the per-frame CPU BakeSurfels (54k surfel evals) +
            // ~2.2 MB upload: the kernel reads the raw PSF records already on
            // the GPU and bakes the uvn x uvn grid per patch there. Raw records
            // are cached in buf_anchors/buf_surfaces/buf_snorm and re-uploaded
            // ONLY when the body is dirty (the same flags the splat path uses)
            // -- a deformed frame still costs one anchor-blob upload, but the
            // 54k-surfel CPU bake and its 2.2 MB upload are gone. The bake
            // writes a FIXED ns*uvn*uvn grid, so build_grid's work-size is
            // known up front. Buffer management mirrors the splat path exactly
            // so the shared buffers stay consistent across both paths.
            const auto& A  = body.AnchorBlob();
            const auto& S  = body.SurfaceBlob();
            const auto& St = body.StitchBlob();
            const uint32_t ns = (uint32_t)S.size();
            if (A.empty() || S.empty()) return false;

            // CROSS-BODY OWNERSHIP: the shared GPU buffers hold the
            // PREVIOUS body's records until this body re-uploads.
            //
            // The check MUST NOT be a pointer comparison. A body is often
            // destroyed and a new one allocated at the SAME heap address
            // (e.g. two `unique_ptr<PSFBody>` locals in sequence), so
            // `&body == last_body` can be true for a COMPLETELY DIFFERENT
            // body. The renderer then marches the previous body's geometry:
            // the new body silently renders as the old one.
            //
            // That bug was measured: a 4-sheet gearbox followed by a 2-sheet
            // sphere at the same address rendered the sphere with the
            // GEARBOX's anchors (43,218 background pixels -> 22,909, roughly
            // half the silhouette, because the sphere's own 4,608 anchors
            // were never uploaded).
            //
            // The identity used here is CONTENT, not address: the record
            // COUNT plus the body's own dirty flag. A body that has been
            // deformed in place keeps its count but sets AnchorsDirty(), so
            // it still re-uploads.
            const bool body_changed = (last_body != &body) ||
                                      (last_anchor_count != A.size()) ||
                                      (last_surface_count != S.size());
            const bool grew = cap_anchors < A.size() || cap_surfaces < S.size();
            if (grew) {
                auto rel = [](void* m) { if (m) clReleaseMemObject((cl_mem)m); };
                rel(buf_anchors); rel(buf_surfaces); rel(buf_stitch);
                buf_anchors  = clCreateBuffer(ctx, CL_MEM_READ_ONLY, A.size()*64, nullptr, &err); CLC(err);
                buf_surfaces = clCreateBuffer(ctx, CL_MEM_READ_ONLY, S.size()*64, nullptr, &err); CLC(err);
                // OpenCL rejects size==0; torus.psf often has an empty stitch blob
                const size_t stitch_bytes = St.empty() ? size_t(ns) * 8u : St.size();
                buf_stitch   = clCreateBuffer(ctx, CL_MEM_READ_ONLY, stitch_bytes, nullptr, &err); CLC(err);
                cap_anchors = (uint32_t)A.size(); cap_surfaces = (uint32_t)S.size();
            }
            const bool full_upload = grew || body_changed || body.AnchorsDirty();
            if (full_upload) {
                CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_anchors, CL_FALSE, 0, A.size()*64, A.data(), 0, nullptr, nullptr));
                {
                    const size_t stitch_bytes = St.empty() ? size_t(ns) * 8u : St.size();
                    if (St.empty()) {
                        const cl_char zero = 0;
                        CLC(clEnqueueFillBuffer(q, (cl_mem)buf_stitch, &zero,
                                                sizeof(zero), 0, stitch_bytes, 0, nullptr, nullptr));
                    } else {
                        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_stitch, CL_FALSE, 0, St.size(), St.data(), 0, nullptr, nullptr));
                    }
                }
                // smooth per-anchor shading normals (parallel buffer, not the
                // anchor record's displacement slot)
                const std::vector<Vec3>& sn = body.ComputeSmoothNormals();
                std::vector<cl_float4> snf(std::max(sn.size(), A.size()));
                for (size_t i = 0; i < snf.size(); ++i) {
                    if (i < sn.size())
                        snf[i] = cl_float4{{sn[i].x, sn[i].y, sn[i].z, 0.0f}};
                    else
                        snf[i] = cl_float4{{0.f, 1.f, 0.f, 0.0f}};
                }
                if (cap_snorm < snf.size() || !buf_snorm) {
                    if (buf_snorm) clReleaseMemObject((cl_mem)buf_snorm);
                    buf_snorm = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                    snf.size()*sizeof(cl_float4), nullptr, &err); CLC(err);
                    cap_snorm = (uint32_t)snf.size();
                }
                // Local staging must survive a nonblocking copy; wait for this
                // transfer before snf leaves scope.
                CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_snorm, CL_TRUE, 0,
                                snf.size()*sizeof(cl_float4), snf.data(), 0, nullptr, nullptr));
                body.MarkAnchorsClean();
            }
            if (grew || body_changed) {
                // body_changed: the shared buffer still holds the PREVIOUS
                // body's surface records -> full re-upload, not a partial flush.
                CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_surfaces, CL_FALSE, 0, S.size()*64, S.data(), 0, nullptr, nullptr));
                body.ClearDirty();
            } else if (body.AnyDirty()) {
                // partial upload: only dirty 64-byte records
                PSFFlushState fs; fs.queue = q;
                fs.anchors_buf = (cl_mem)buf_anchors; fs.surfaces_buf = (cl_mem)buf_surfaces;
                fs.anchors_bytes = A.size()*64; fs.surfaces_bytes = S.size()*64;
                body.FlushToGPU(&fs);
            }
            last_body = &body;
            last_anchor_count = A.size(); last_surface_count = S.size();

            // grid buffer (256^3 ints)
            const uint32_t ncells = (uint32_t)GRID * GRID * GRID;
            if (cap_grid < ncells || !buf_grid) {
                if (buf_grid) clReleaseMemObject((cl_mem)buf_grid);
                buf_grid = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)ncells * 4, nullptr, &err); CLC(err);
                cap_grid = ncells;
            }
            // surfel buffer: fixed ns*uvn*uvn grid (the GPU bake writes every slot)
            const uint32_t nsurf = ns * (uint32_t)uvn * (uint32_t)uvn;
            if (cap_surfels < nsurf || !buf_surfels) {
                if (buf_surfels) clReleaseMemObject((cl_mem)buf_surfels);
                buf_surfels = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)nsurf * sizeof(Surfel), nullptr, &err); CLC(err);
                cap_surfels = nsurf;
            }
            // bake the surfels on the GPU (reads raw records, writes the grid)
            {
                cl_uint aa = 0;
                cl_float3 bbmin3  = {{bmn.x, bmn.y, bmn.z}};
                cl_float3 bbspan3 = {{bsp.x, bsp.y, bsp.z}};
                CLC(clSetKernelArg((cl_kernel)k_march_bake, aa++, sizeof(cl_mem),   &buf_anchors));
                CLC(clSetKernelArg((cl_kernel)k_march_bake, aa++, sizeof(cl_mem),   &buf_surfaces));
                CLC(clSetKernelArg((cl_kernel)k_march_bake, aa++, sizeof(cl_mem),   &buf_snorm));
                CLC(clSetKernelArg((cl_kernel)k_march_bake, aa++, sizeof(uint32_t), &ns));
                CLC(clSetKernelArg((cl_kernel)k_march_bake, aa++, sizeof(int),      &uvn));
                CLC(clSetKernelArg((cl_kernel)k_march_bake, aa++, sizeof(cl_float3),&bbmin3));
                CLC(clSetKernelArg((cl_kernel)k_march_bake, aa++, sizeof(cl_float3),&bbspan3));
                CLC(clSetKernelArg((cl_kernel)k_march_bake, aa++, sizeof(float),    &foot));
                // MODEL MATRIX (local -> world): applied to every baked
                // surfel position + normal. The anchors stay in local space
                // (bbmin3/bbspan3 dequantize them); only the final surfel
                // position/normal is taken to world space.
                CLC(clSetKernelArg((cl_kernel)k_march_bake, aa++, sizeof(cl_float4),&mr0));
                CLC(clSetKernelArg((cl_kernel)k_march_bake, aa++, sizeof(cl_float4),&mr1));
                CLC(clSetKernelArg((cl_kernel)k_march_bake, aa++, sizeof(cl_float4),&mr2));
                CLC(clSetKernelArg((cl_kernel)k_march_bake, aa++, sizeof(cl_float4),&mr3));
                CLC(clSetKernelArg((cl_kernel)k_march_bake, aa++, sizeof(cl_mem),   &buf_surfels));
                size_t g = nsurf;
                CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_march_bake, 1, nullptr, &g, nullptr, 0, nullptr, nullptr));
            }
            // single march buffer + ping-pong for the 2 fill passes
            // Sized to the CURRENT render resolution. Recreated when null
            // (first frame, or after a Resize()).
            const uint32_t npx = (uint32_t)width * (uint32_t)height;
            const uint32_t layer_bytes = npx * 4;
            if (!buf_mcolA || !buf_mcolB || !buf_mdepA || !buf_mdepB) {
                if (buf_mcolA) clReleaseMemObject((cl_mem)buf_mcolA);
                if (buf_mcolB) clReleaseMemObject((cl_mem)buf_mcolB);
                if (buf_mdepA) clReleaseMemObject((cl_mem)buf_mdepA);
                if (buf_mdepB) clReleaseMemObject((cl_mem)buf_mdepB);
                buf_mcolA = clCreateBuffer(ctx, CL_MEM_READ_WRITE, layer_bytes, nullptr, &err); CLC(err);
                buf_mcolB = clCreateBuffer(ctx, CL_MEM_READ_WRITE, layer_bytes, nullptr, &err); CLC(err);
                buf_mdepA = clCreateBuffer(ctx, CL_MEM_READ_WRITE, layer_bytes, nullptr, &err); CLC(err);
                buf_mdepB = clCreateBuffer(ctx, CL_MEM_READ_WRITE, layer_bytes, nullptr, &err); CLC(err);
            }

            // camera basis (matches psf_raymarch.cl)
            Vec3 f{cam.target.x-cam.position.x, cam.target.y-cam.position.y, cam.target.z-cam.position.z};
            float fl = std::sqrt(f.x*f.x+f.y*f.y+f.z*f.z); if (fl<1e-6f) fl=1.f;
            f.x/=fl; f.y/=fl; f.z/=fl;
            Vec3 rr{f.y*cam.up.z-f.z*cam.up.y, f.z*cam.up.x-f.x*cam.up.z, f.x*cam.up.y-f.y*cam.up.x};
            float rl=std::sqrt(rr.x*rr.x+rr.y*rr.y+rr.z*rr.z); rr.x/=rl; rr.y/=rl; rr.z/=rl;
            Vec3 uu{rr.y*f.z-rr.z*f.y, rr.z*f.x-rr.x*f.z, rr.x*f.y-rr.y*f.x};
            float tan_half = std::tan(cam.fov_deg * 3.14159265f / 360.f);
            float aspect = float(width)/float(height);
            cl_float3 eye = {{cam.position.x,cam.position.y,cam.position.z}};
            cl_float3 fwd = {{f.x,f.y,f.z}};
            cl_float3 right = {{rr.x,rr.y,rr.z}};
            cl_float3 upv = {{uu.x,uu.y,uu.z}};
            // Grid origin = WORLD-space min of the transformed body. The
            // surfels are in world space (the model matrix is applied in
            // bake_surfels), so the grid must be too.
            cl_float3 org = {{wmn.x,wmn.y,wmn.z}};
            // key light: toward-light from upper-front (camera side). Kernel shades with
// dot(N, light) where light points TOWARD the light source, so pass +toward.
            float tlx = 0.55f, tly = -0.45f, tlz = 0.70f;
            float lf = 1.f / std::sqrt(tlx*tlx+tly*tly+tlz*tlz);
            cl_float3 light = {{tlx*lf, tly*lf, tlz*lf}};
            // MVP for NDC depth in march_draw (same matrix as the splat path)
            float mvp_march[16];
            cam.BuildMVP(mvp_march, aspect);
            cl_float4 mvp0 = {{mvp_march[0],mvp_march[1],mvp_march[2],mvp_march[3]}};
            cl_float4 mvp1 = {{mvp_march[4],mvp_march[5],mvp_march[6],mvp_march[7]}};
            cl_float4 mvp2 = {{mvp_march[8],mvp_march[9],mvp_march[10],mvp_march[11]}};
            cl_float4 mvp3 = {{mvp_march[12],mvp_march[13],mvp_march[14],mvp_march[15]}};
            // march depth range: camera to far side of the bbox
            float dmin = 1e30f, dmax = -1e30f;
            for (int ox=0;ox<2;++ox) for (int oy=0;oy<2;++oy) for (int oz=0;oz<2;++oz) {
                float cx = bmn.x + (ox?bsp.x:0.f), cy = bmn.y + (oy?bsp.y:0.f), cz = bmn.z + (oz?bsp.z:0.f);
                float dx=cx-cam.position.x, dy=cy-cam.position.y, dz=cz-cam.position.z;
                float d = std::sqrt(dx*dx+dy*dy+dz*dz);
                dmin = std::min(dmin,d); dmax = std::max(dmax,d);
            }
            float near_d = std::max(0.05f, dmin - 0.2f*diag);
            float far_d  = dmax + 0.2f*diag;
            const float INF = 1e30f;

            // ---- adaptive step count --------------------------------------
            // The old fixed 40 steps broke at camera distance: step =
            // (far-near)/steps can exceed the hit radius `foot`, so a ray
            // jumps OVER the thin surface between samples and misses
            // (0 lit pixels at the window camera; fine up close where the
            // range is short).
            //
            // GUARANTEE: step <= foot (>=2 samples per head-on hit-sphere
            // crossing). A ray is within `foot` of a surfel for a chord of
            // 2*foot head-on, so step<=foot guarantees >=2 samples land in
            // that chord. (The old 0.5*foot guarantee was 4x tighter than
            // needed; it was paired with the OLD large foot=2*cell, where
            // (far-near)/(0.5*foot) ~= 258 -> the 256 cap. Now foot is
            // ~0.5*spacing ~= 3.6x smaller -- the user's "smaller surfel
            // disks" request -- so the same camera range needs
            // (far-near)/foot ~= 469 steps; the cap is raised 256 -> 512 to
            // keep the guarantee honest (below ~469, head-on rays near the
            // far end can miss). The 5x5x5 neighborhood (radius 2*cell >=
            // foot) plus the 2-pass layer_fill catch the only exposed case:
            // a very-grazing near-silhouette ray whose chord is < foot. The
            // slab cull below recovers most of the off-center cost. Cost:
            // ~1.8x the old march time at the benchmark camera (~21 -> ~12
            // FPS single-body); DELIRIUM_MARCH_STEPS can force a lower count
            // if speed beats the far-end guarantee.
            //
            // HISTORY (2026-10-28): an EARLIER "loosening" to step<=foot was
            // tried while foot was STILL 2*cell and caused a front-face-miss
            // glitch -- but that was actually the per-ray slab cull (added the
            // same session) clipping the march range, NOT the step loosening
            // (the slab AABB is a superset of the body, so it cannot clip a
            // front-face hit; verified). The slab cull is now gated behind
            // DELIRIUM_MARCH_SLAB for A/B. The step<=foot guarantee is correct
            // on its own.
            {
                // Step cap: the guarantee is step <= foot (>=2 samples per
                // head-on hit-sphere crossing). The natural count is
                // need = (far-near)/foot. The old hard cap of 512 was BELOW
                // need (~812 at the window camera), so step > foot and front
                // faces were missed (checkered surface + z-fighting in holes).
                // The cap is now env-tunable (DELIRIUM_MARCH_MAXSTEPS, default
                // 1024) so denser sampling holds the guarantee; raise it if you
                // zoom far out, lower it to trade far-end coverage for FPS.
                int maxsteps = 1024;
                if (const char* e = std::getenv("DELIRIUM_MARCH_MAXSTEPS")) {
                    int v = 0; if (std::sscanf(e, "%d", &v) == 1 && v >= 1 && v <= 4096) maxsteps = v;
                }
                const float need = (far_d - near_d) / std::max(1e-6f, foot) + 1.0f;
                if (need > (float)bsteps)
                    bsteps = (int)std::min((float)maxsteps, std::ceil(need));
            }

            // 1) clear + build the 3D grid
            {
                cl_uint aa=0;
                CLC(clSetKernelArg((cl_kernel)k_march_clear_grid, aa++, sizeof(cl_mem), &buf_grid));
                CLC(clSetKernelArg((cl_kernel)k_march_clear_grid, aa++, sizeof(uint32_t), &ncells));
                size_t g = ncells;
                CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_march_clear_grid, 1, nullptr, &g, nullptr, 0, nullptr, nullptr));
            }
            {
                cl_uint aa=0;
                CLC(clSetKernelArg((cl_kernel)k_march_build_grid, aa++, sizeof(cl_mem), &buf_surfels));
                CLC(clSetKernelArg((cl_kernel)k_march_build_grid, aa++, sizeof(uint32_t), &nsurf));
                CLC(clSetKernelArg((cl_kernel)k_march_build_grid, aa++, sizeof(cl_mem), &buf_grid));
                cl_int gx=GRID, gy=GRID, gz=GRID;
                CLC(clSetKernelArg((cl_kernel)k_march_build_grid, aa++, sizeof(cl_int), &gx));
                CLC(clSetKernelArg((cl_kernel)k_march_build_grid, aa++, sizeof(cl_int), &gy));
                CLC(clSetKernelArg((cl_kernel)k_march_build_grid, aa++, sizeof(cl_int), &gz));
                CLC(clSetKernelArg((cl_kernel)k_march_build_grid, aa++, sizeof(cl_float3), &org));
                CLC(clSetKernelArg((cl_kernel)k_march_build_grid, aa++, sizeof(float), &cell));
                size_t g = nsurf;
                CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_march_build_grid, 1, nullptr, &g, nullptr, 0, nullptr, nullptr));
            }

            // 2) single full-range march (NEAR->FAR, first hit wins) -> buffer A
            size_t g2[2] = {size_t(width), size_t(height)};

            // THE LAYERED BACK-TO-FRONT PATH (the user's method).
            //
            // The single-pass march_draw below resolves ONE hit per pixel.
            // The layered path instead sweeps the ray through DISTANCE
            // PLANES (bands), each band drawn into its OWN buffer, each
            // band filled on its own, and the bands stacked FAR -> NEAR so
            // the nearest overwrites what is behind it. The image is only
            // produced after the last band.
            //
            // Selected with Path::SplatLayered. It is the path the user described
            // and the one the golden reference implements
            // (registry/golden/kernels/surfel_march.cl).
            const bool use_layers = layered_on;
            // The number of distance planes. Each band spans 1/nlayers of
            // the ray's depth range and is sampled with bsteps/nlayers
            // steps, so the TOTAL step count is unchanged -- the work is
            // redistributed, not multiplied.
            int nlayers = 24;
            if (const char* e = std::getenv("DELIRIUM_MARCH_BANDS")) {
                int v=0; if (std::sscanf(e,"%d",&v)==1 && v>=2 && v<=96) nlayers=v;
            }

            if (use_layers && k_march_band_draw) {
                // Per band: clear its own buffer, draw into it, fill it on
                // its own, then compose it into the shared framebuffer.
                // The compose is an atom_min on (depth|rgba), so the NEAREST
                // band wins regardless of the order they are launched in --
                // but we still launch far -> near, which is the user's
                // stated order and keeps the depth values monotonic.
                const float span = far_d - near_d;
                if (span > 1e-6f) {
                    for (int li = 0; li < nlayers; ++li) {
                        // Band li covers [far - li*span/n, far - (li+1)*span/n]
                        // -> band 0 is the FARTHEST.
                        const float band_far  = far_d - (float(li)    / float(nlayers)) * span;
                        const float band_near = far_d - (float(li + 1) / float(nlayers)) * span;
                        // Each band gets an equal share of the steps, at
                        // least 1.
                        int bsteps_band = bsteps / nlayers;
                        if (bsteps_band < 1) bsteps_band = 1;

                        // clear this band's own buffer
                        {
                            cl_uint aa=0;
                            CLC(clSetKernelArg((cl_kernel)k_march_layer_clear, aa++, sizeof(cl_mem), &buf_mcolA));
                            CLC(clSetKernelArg((cl_kernel)k_march_layer_clear, aa++, sizeof(cl_mem), &buf_mdepA));
                            CLC(clSetKernelArg((cl_kernel)k_march_layer_clear, aa++, sizeof(uint32_t), &npx));
                            CLC(clSetKernelArg((cl_kernel)k_march_layer_clear, aa++, sizeof(float), &INF));
                            size_t g1 = npx;
                            CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_march_layer_clear, 1, nullptr, &g1, nullptr, 0, nullptr, nullptr));
                        }
                        // draw this band into its own buffer
                        {
                            cl_uint W=(uint32_t)width, H=(uint32_t)height;
                            cl_uint aa=0;
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(cl_mem), &buf_surfels));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(cl_mem), &buf_grid));
                            cl_int gx=GRID, gy=GRID, gz=GRID;
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(cl_int), &gx));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(cl_int), &gy));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(cl_int), &gz));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(cl_float3), &org));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(float), &cell));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(cl_float3), &eye));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(cl_float3), &fwd));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(cl_float3), &right));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(cl_float3), &upv));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(float), &tan_half));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(float), &aspect));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(uint32_t), &W));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(uint32_t), &H));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(float), &band_far));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(float), &band_near));
                            cl_int bs = bsteps_band;
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(int), &bs));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(float), &foot));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(cl_float3), &light));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(cl_float4), &mvp0));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(cl_float4), &mvp1));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(cl_float4), &mvp2));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(cl_float4), &mvp3));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(cl_mem), &buf_mcolA));
                            CLC(clSetKernelArg((cl_kernel)k_march_band_draw, aa++, sizeof(cl_mem), &buf_mdepA));
                            CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_march_band_draw, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
                        }
                        // fill this band's holes ON ITS OWN (no other band
                        // interferes -- this is the structural difference
                        // from the single-pass path).
                        {
                            cl_mem a_c=(cl_mem)buf_mcolA, a_d=(cl_mem)buf_mdepA;
                            cl_mem b_c=(cl_mem)buf_mcolB, b_d=(cl_mem)buf_mdepB;
                            cl_uint W=(uint32_t)width, H=(uint32_t)height;
                            for (int p=0;p<fill_passes;++p) {
                                cl_uint aa=0;
                                CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(cl_mem), &a_c));
                                CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(cl_mem), &a_d));
                                CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(cl_mem), &b_c));
                                CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(cl_mem), &b_d));
                                CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(uint32_t), &W));
                                CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(uint32_t), &H));
                                CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(float), &INF));
                                // The band's depth: its NEAR edge, so a filled
                                // pixel is still behind the band's real hits.
                                CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(float), &band_near));
                                cl_int fm = fill_min;
                                CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(int), &fm));
                                CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_march_layer_fill, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
                                cl_mem tc=a_c; a_c=b_c; b_c=tc;
                                cl_mem td=a_d; a_d=b_d; b_d=td;
                            }
                            // stack this band into the shared framebuffer
                            if (k_march_compose_fb && buf_fb) {
                                cl_uint aa=0;
                                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(cl_mem), &a_c));
                                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(cl_mem), &a_d));
                                cl_uint WW=(uint32_t)width, HH=(uint32_t)height;
                                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(uint32_t), &WW));
                                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(uint32_t), &HH));
                                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(float), &INF));
                                const float m_bias = depth_bias;
                                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(float), &m_bias));
                                cl_uint pass0=0;
                                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(uint32_t), &pass0));
                                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(cl_mem), &buf_fb));
                                cl_mem d32 = (cl_mem)buf_depth32;
                                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(cl_mem), &d32));
                                if (has_long_atomics) {
                                    CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_march_compose_fb, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
                                } else if (buf_depth32) {
                                    cl_uint pass1=1;
                                    CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_march_compose_fb, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
                                    CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, 6, sizeof(uint32_t), &pass1));
                                    CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_march_compose_fb, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
                                }
                            }
                        }
                    }
                }
            } else {
            {
                cl_uint aa=0;
                CLC(clSetKernelArg((cl_kernel)k_march_layer_clear, aa++, sizeof(cl_mem), &buf_mcolA));
                CLC(clSetKernelArg((cl_kernel)k_march_layer_clear, aa++, sizeof(cl_mem), &buf_mdepA));
                CLC(clSetKernelArg((cl_kernel)k_march_layer_clear, aa++, sizeof(uint32_t), &npx));
                CLC(clSetKernelArg((cl_kernel)k_march_layer_clear, aa++, sizeof(float), &INF));
                size_t g1 = npx;
                CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_march_layer_clear, 1, nullptr, &g1, nullptr, 0, nullptr, nullptr));
            }
            {
                cl_uint W=(uint32_t)width, H=(uint32_t)height;
                cl_uint aa=0;
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(cl_mem), &buf_surfels));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(cl_mem), &buf_grid));
                cl_int gx=GRID, gy=GRID, gz=GRID;
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(cl_int), &gx));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(cl_int), &gy));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(cl_int), &gz));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(cl_float3), &org));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(float), &cell));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(cl_float3), &eye));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(cl_float3), &fwd));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(cl_float3), &right));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(cl_float3), &upv));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(float), &tan_half));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(uint32_t), &W));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(uint32_t), &H));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(float), &near_d));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(float), &far_d));
                cl_int bs = bsteps;
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(int), &bs));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(float), &foot));
                // Per-ray slab cull toggle (default ON). Set
                // DELIRIUM_MARCH_SLAB=0 to march the full [near,far] range
                // (pre-2026-10-28 behavior) for A/B testing the glitch.
                static const int use_slab =
                    []{ const char* e = std::getenv("DELIRIUM_MARCH_SLAB");
                        return (e && e[0]=='0') ? 0 : 1; }();
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(cl_float3), &light));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(cl_float4), &mvp0));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(cl_float4), &mvp1));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(cl_float4), &mvp2));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(cl_float4), &mvp3));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(cl_mem), &buf_mcolA));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(cl_mem), &buf_mdepA));
                // use_slab (index 25) and accumulate (index 26) are the LAST
                // two kernel params — must be set after color/depth buffers
                // to match the .cl signature order. accumulate=1 (default) =
                // frontmost-dominant layer accumulation (the user's method);
                // DELIRIUM_MARCH_ACCUM=0 = legacy early-exit for A/B.
                static const int use_accum =
                    []{ const char* e = std::getenv("DELIRIUM_MARCH_ACCUM");
                        return (e && e[0]=='0') ? 0 : 1; }();
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(int), &use_slab));
                CLC(clSetKernelArg((cl_kernel)k_march_draw, aa++, sizeof(int), &use_accum));
                CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_march_draw, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
            }   // end: single-pass march
            }   // end: else (use_layers == false)

            // 3) fill holes (N passes, ping-pong A<->B) — "groups of 3, averaged"
            //
            // In the LAYERED path each band was already filled on its own and
            // already composed into buf_fb, so there is nothing left to fill
            // here. The flag says so.
            cl_mem final_fill_color=(cl_mem)buf_mcolA, final_fill_depth=(cl_mem)buf_mdepA;
            if (use_layers) {
                // The layered path has already stacked every band into
                // buf_fb. Nothing to fill or compose here.
                final_fill_color = nullptr;
                final_fill_depth = nullptr;
            } else {
                cl_mem a_c=(cl_mem)buf_mcolA, a_d=(cl_mem)buf_mdepA;
                cl_mem b_c=(cl_mem)buf_mcolB, b_d=(cl_mem)buf_mdepB;
                cl_uint W=(uint32_t)width, H=(uint32_t)height;
                for (int p=0;p<fill_passes;++p) {
                    cl_uint aa=0;
                    CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(cl_mem), &a_c));
                    CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(cl_mem), &a_d));
                    CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(cl_mem), &b_c));
                    CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(cl_mem), &b_d));
                    CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(uint32_t), &W));
                    CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(uint32_t), &H));
                    CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(float), &INF));
                    CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(float), &far_d));
                    cl_int fm = fill_min;
                    CLC(clSetKernelArg((cl_kernel)k_march_layer_fill, aa++, sizeof(int), &fm));
                    CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_march_layer_fill, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
                    cl_mem tc=a_c; a_c=b_c; b_c=tc;
                    cl_mem td=a_d; a_d=b_d; b_d=td;
                }
                // result buffer after N swaps: A if even, B if odd
                final_fill_color = a_c;
                final_fill_depth = a_d;
            }
            // 4) compose the marched layer into the SHARED framebuffer buf_fb
            //    (FRONTMOST WINS: atom_min on the depth|rgba key, so every
            //    body's marched layer interleaves with the others by depth),
            //    then resolve buf_fb -> buf_img. Every body does this now
            //    (march is the primary renderer); the per-body resolve is a
            //    cheap W*H pass and keeps Present() caller-agnostic.
            //
            // In the LAYERED path each band was ALREADY composed into buf_fb
            // (that is the back-to-front stacking), so this step is skipped.
            if (!use_layers && k_march_compose_fb && buf_fb) {
                cl_uint aa=0;
                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(cl_mem), &final_fill_color));
                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(cl_mem), &final_fill_depth));
                cl_uint W=(uint32_t)width, H=(uint32_t)height;
                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(uint32_t), &W));
                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(uint32_t), &H));
                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(float), &INF));
                const float m_bias = depth_bias;
                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(float), &m_bias));
                cl_uint pass0=0, pass1=1;
                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(uint32_t), &pass0));
                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(cl_mem), &buf_fb));
                cl_mem d32 = (cl_mem)buf_depth32;
                CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, aa++, sizeof(cl_mem), &d32));
                if (has_long_atomics) {
                    CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_march_compose_fb, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
                } else if (buf_depth32) {
                    // two-pass fallback (same as psf_splat): pass 0 claims the
                    // depth with a 32-bit atomic_min, pass 1 stores the color
                    // only for the pixels this body won.
                    CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_march_compose_fb, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
                    CLC(clSetKernelArg((cl_kernel)k_march_compose_fb, 6, sizeof(uint32_t), &pass1));
                    CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_march_compose_fb, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
                }
            }
            // resolve buf_fb -> buf_img (the image is only produced AFTER the
            // frontmost-wins composite, per the user's spec). DEFERRED to the
            // last body of the frame (final=true): the resolve is a full-screen
            // pass and clFinish is a full queue sync, so doing both once per
            // body (5x/frame in recon_window) wasted ~4 full-screen resolves
            // and 4 GPU syncs per frame. Non-final bodies just compose into
            // buf_fb and return; the final body resolves + finishes once.
            if (final) {
                if (k_resolve && buf_img) {
                    cl_uint npx2 = (uint32_t)width * (uint32_t)height;
                    CLC(clSetKernelArg((cl_kernel)k_resolve, 0, sizeof(cl_mem), &buf_fb));
                    CLC(clSetKernelArg((cl_kernel)k_resolve, 1, sizeof(cl_uint), &npx2));
                    CLC(clSetKernelArg((cl_kernel)k_resolve, 2, sizeof(cl_mem), &buf_img));
                    size_t gr = (size_t(npx2) + 255) & ~size_t(255);
                    CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_resolve, 1, nullptr, &gr, nullptr, 0, nullptr, nullptr));
                }
                CLC(clFinish(q));
            }
            static int mlog = 0;
            if ((mlog++ % 60) == 0)
                std::printf("march: %ux%u surfels=%u steps=%d fill>=%d cell=%.4f (primary, frontmost-wins)\n",
                            (uint32_t)width, (uint32_t)height, nsurf, bsteps, fill_min, cell);
            last_spp = 1.f;
            return true;
    }

    // True lateral / view-ray equation path (raw math). CPU builds hits; GPU only holds FB + optional AA.
    // Path::Raymarch selects it; Path::SplatClassic/SplatLayered fall back to the splat.
    // GPU equation ray-march (OpenCL).
    const bool use_ray = ray_on;
    if (use_ray && k_raymarch && buf_img) {
        const auto& A = body.AnchorBlob();
        const auto& S = body.SurfaceBlob();
        const uint32_t ns = (uint32_t)S.size();
        Vec3 bmn = body.BBoxMin(), bsp = body.BBoxSpan();
        if (bsp.x < 1e-8f && bsp.y < 1e-8f && bsp.z < 1e-8f) {
            bmn = Vec3{-1.f,-1.f,-1.f}; bsp = Vec3{2.f,2.f,2.f};
        }
        // ensure buffers. CROSS-BODY OWNERSHIP: the shared buffers hold the
        // PREVIOUS body's records until this body re-uploads; a smaller/clean
        // body must force a full upload (body_changed) or it renders the
        // previous body's geometry.
        const bool body_changed = (last_body != &body);
        const bool grew = cap_anchors < A.size() || cap_surfaces < S.size();
        if (grew || body_changed || !buf_anchors) {
            auto rel = [](void* m) { if (m) clReleaseMemObject((cl_mem)m); };
            rel(buf_anchors); rel(buf_surfaces);
            buf_anchors  = clCreateBuffer(ctx, CL_MEM_READ_ONLY, A.size()*64, nullptr, &err); CLC(err);
            buf_surfaces = clCreateBuffer(ctx, CL_MEM_READ_ONLY, S.size()*64, nullptr, &err); CLC(err);
            cap_anchors = (uint32_t)A.size(); cap_surfaces = (uint32_t)S.size();
            CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_anchors, CL_FALSE, 0, A.size()*64, A.data(), 0, nullptr, nullptr));
            CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_surfaces, CL_FALSE, 0, S.size()*64, S.data(), 0, nullptr, nullptr));
            last_body = &body;
            last_anchor_count = A.size(); last_surface_count = S.size();
        } else {
            // partial upload: only dirty 64-byte records (PSFBody::FlushToGPU)
            PSFFlushState fs; fs.queue = q;
            fs.anchors_buf = (cl_mem)buf_anchors; fs.surfaces_buf = (cl_mem)buf_surfaces;
            fs.anchors_bytes = A.size()*64; fs.surfaces_bytes = S.size()*64;
            body.FlushToGPU(&fs);
        }
        Vec3 f{cam.target.x-cam.position.x, cam.target.y-cam.position.y, cam.target.z-cam.position.z};
        float fl = std::sqrt(f.x*f.x+f.y*f.y+f.z*f.z); if (fl<1e-6f) fl=1.f;
        f.x/=fl; f.y/=fl; f.z/=fl;
        Vec3 rr{f.y*cam.up.z-f.z*cam.up.y, f.z*cam.up.x-f.x*cam.up.z, f.x*cam.up.y-f.y*cam.up.x};
        float rl=std::sqrt(rr.x*rr.x+rr.y*rr.y+rr.z*rr.z); rr.x/=rl; rr.y/=rl; rr.z/=rl;
        Vec3 uu{rr.y*f.z-rr.z*f.y, rr.z*f.x-rr.x*f.z, rr.x*f.y-rr.y*f.x};
        float tan_half = std::tan(cam.fov_deg * 3.14159265f / 360.f);
        float aspect = float(width)/float(height);
        static float tsec=0.f; tsec += 1.f/60.f;
        float emissive = 0.45f;
        if (const char* e = std::getenv("DELIRIUM_GLOW")) { float g=0; if (std::sscanf(e,"%f",&g)==1) emissive=g; }
        int uvn = 4;
        if (const char* e = std::getenv("DELIRIUM_RAY_UV")) { unsigned v=0; if (std::sscanf(e,"%u",&v)==1 && v>=2 && v<=12) uvn=int(v); }
        float hit_r = 0.028f;  // smaller equation hit tube (was ~0.06)
        if (const char* e = std::getenv("DELIRIUM_HIT_RADIUS")) {
            float hr = 0.f; if (std::sscanf(e, "%f", &hr) == 1 && hr > 0.001f && hr < 0.5f) hit_r = hr;
        }
        cl_float3 eye = {{cam.position.x,cam.position.y,cam.position.z}};
        cl_float3 fwd = {{f.x,f.y,f.z}};
        cl_float3 right = {{rr.x,rr.y,rr.z}};
        cl_float3 upv = {{uu.x,uu.y,uu.z}};
        cl_float3 bbmin = {{bmn.x,bmn.y,bmn.z}};
        cl_float3 bbspan = {{bsp.x,bsp.y,bsp.z}};
                        // CPU coordinator only: camera + frustum planes + enqueue. GPU does cull + math.
        if (g_accel_anchors != A.size() || g_accel.patches.empty()) {
            g_accel.Build(body);
            g_accel_anchors = A.size();
            // Upload patch AABBs once (math bounds from anchors + deform margin)
            std::vector<float> aabb(g_accel.patches.size() * 6);
            for (size_t pi = 0; pi < g_accel.patches.size(); ++pi) {
                const auto& pa = g_accel.patches[pi];
                // patches[] may be subset if bad anchors skipped — index by surface_index
                (void)pa;
            }
            // dense array indexed by surface index
            aabb.assign(size_t(ns) * 6u, 0.f);
            for (const auto& pa : g_accel.patches) {
                if (pa.surface_index >= ns) continue;
                float* d = &aabb[pa.surface_index * 6];
                d[0]=pa.bmin.x; d[1]=pa.bmin.y; d[2]=pa.bmin.z;
                d[3]=pa.bmax.x; d[4]=pa.bmax.y; d[5]=pa.bmax.z;
            }
            if (buf_patch_aabb) clReleaseMemObject((cl_mem)buf_patch_aabb);
            cl_int e2 = CL_SUCCESS;
            buf_patch_aabb = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                aabb.size() * sizeof(float), aabb.data(), &e2); CLC(e2);
            g_aabb_count = ns;
        }
        float mvp_rm[16];
        cam.BuildMVP(mvp_rm, float(width) / float(height));
        Frustum fr = EquationAccel::FrustumFromMVP(mvp_rm);
        float planes[24];
        for (int pi = 0; pi < 6; ++pi) {
            planes[pi*4+0] = fr.plane[pi][0];
            planes[pi*4+1] = fr.plane[pi][1];
            planes[pi*4+2] = fr.plane[pi][2];
            planes[pi*4+3] = fr.plane[pi][3];
        }
        if (!buf_frustum) {
            cl_int e2 = CL_SUCCESS;
            buf_frustum = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                6 * 4 * sizeof(float), nullptr, &e2); CLC(e2);
        }
        // planes is stack storage: finish this copy before returning from scope.
        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_frustum, CL_TRUE, 0, sizeof(planes), planes, 0, nullptr, nullptr));

        int uvn_min = 2, uvn_max = 5;
        if (const char* e = std::getenv("DELIRIUM_RAY_UV")) {
            unsigned v=0; if (std::sscanf(e,"%u",&v)==1 && v>=2 && v<=12) { uvn_min=int(v); uvn_max=int(v); }
        }
        if (const char* e = std::getenv("DELIRIUM_RAY_UV_MAX")) {
            unsigned v=0; if (std::sscanf(e,"%u",&v)==1 && v>=2 && v<=12) uvn_max=int(v);
        }
        cl_uint object_id = 1u;
        if (const char* e = std::getenv("DELIRIUM_OBJECT_ID")) {
            unsigned v=0; if (std::sscanf(e,"%u",&v)==1 && v>=1 && v<=255) object_id=v;
        }

        cl_kernel kr = (cl_kernel)k_raymarch;
        cl_uint a = 0;
        CLC(clSetKernelArg(kr, a++, sizeof(cl_mem), &buf_anchors));
        CLC(clSetKernelArg(kr, a++, sizeof(cl_mem), &buf_surfaces));
        CLC(clSetKernelArg(kr, a++, sizeof(cl_mem), &buf_patch_aabb));
        CLC(clSetKernelArg(kr, a++, sizeof(cl_uint), &ns));
        CLC(clSetKernelArg(kr, a++, sizeof(cl_mem), &buf_frustum));
        CLC(clSetKernelArg(kr, a++, sizeof(cl_float3), &bbmin));
        CLC(clSetKernelArg(kr, a++, sizeof(cl_float3), &bbspan));
        CLC(clSetKernelArg(kr, a++, sizeof(cl_float3), &eye));
        CLC(clSetKernelArg(kr, a++, sizeof(cl_float3), &fwd));
        CLC(clSetKernelArg(kr, a++, sizeof(cl_float3), &right));
        CLC(clSetKernelArg(kr, a++, sizeof(cl_float3), &upv));
        CLC(clSetKernelArg(kr, a++, sizeof(float), &tan_half));
        CLC(clSetKernelArg(kr, a++, sizeof(float), &aspect));
        CLC(clSetKernelArg(kr, a++, sizeof(float), &tsec));
        CLC(clSetKernelArg(kr, a++, sizeof(float), &emissive));
        CLC(clSetKernelArg(kr, a++, sizeof(float), &hit_r));
        CLC(clSetKernelArg(kr, a++, sizeof(int), &uvn_min));
        CLC(clSetKernelArg(kr, a++, sizeof(int), &uvn_max));
        CLC(clSetKernelArg(kr, a++, sizeof(cl_uint), &object_id));
        cl_uint WW = width, HH = height;
        CLC(clSetKernelArg(kr, a++, sizeof(cl_uint), &WW));
        CLC(clSetKernelArg(kr, a++, sizeof(cl_uint), &HH));
        CLC(clSetKernelArg(kr, a++, sizeof(cl_mem), &buf_img));
        size_t g2[2] = {size_t(width), size_t(height)};
        CLC(clEnqueueNDRangeKernel(q, kr, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
        // no clFinish yet — stitch can chain; finish once at end

        // Lateral object-masked stitch on GPU
        int stitch_passes = 3;
        int stitch_radius = 2;
        if (const char* e = std::getenv("DELIRIUM_STITCH_PASSES")) {
            int v=0; if (std::sscanf(e,"%d",&v)==1 && v>=0 && v<=12) stitch_passes=v;
        }
        if (const char* e = std::getenv("DELIRIUM_STITCH_RADIUS")) {
            int v=0; if (std::sscanf(e,"%d",&v)==1 && v>=1 && v<=6) stitch_radius=v;
        }
        if ((k_lod_stitch || k_obj_stitch) && stitch_passes > 0) {
            if (!buf_img2) {
                cl_int e2=CL_SUCCESS;
                buf_img2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size_t(width)*height*4, nullptr, &e2); CLC(e2);
            }
            cl_kernel ks = (cl_kernel)k_obj_stitch;
            void* a_src = buf_img; void* a_dst = buf_img2;
            for (int p=0;p<stitch_passes;++p) {
                cl_uint aa=0;
                CLC(clSetKernelArg(ks, aa++, sizeof(cl_mem), &a_src));
                CLC(clSetKernelArg(ks, aa++, sizeof(cl_mem), &a_dst));
                CLC(clSetKernelArg(ks, aa++, sizeof(cl_uint), &WW));
                CLC(clSetKernelArg(ks, aa++, sizeof(cl_uint), &HH));
                CLC(clSetKernelArg(ks, aa++, sizeof(int), &stitch_radius));
                CLC(clEnqueueNDRangeKernel(q, ks, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
                std::swap(a_src, a_dst);
            }
            if (a_src != buf_img)
                CLC(clEnqueueCopyBuffer(q,(cl_mem)a_src,(cl_mem)buf_img,0,0,size_t(width)*height*4,0,nullptr,nullptr));
        }
        CLC(clFinish(q)); // single sync for present

        static int slog=0;
        if ((slog++ % 60)==0)
            std::printf("GPU scene: %ux%u surfaces=%u uvn=[%d,%d] stitch=%d (CPU=coord only)\n",
                        WW, HH, ns, uvn_min, uvn_max, stitch_passes);
        last_spp = 1.f;
        return true;
    }

    const auto& A  = body.AnchorBlob();
    const auto& S  = body.SurfaceBlob();
    const auto& St = body.StitchBlob();
    const uint32_t ns = (uint32_t)S.size();

    // (re)create on growth (live splits!), upload on dirty flags.
    // CROSS-BODY OWNERSHIP: the buffers are SHARED across every body in the
    // frame. A body that is smaller than the previous one (so `grew` is
    // false) and not dirty (static, or already uploaded earlier) would skip
    // its upload and render with the PREVIOUS body's anchors/surfaces/snorm
    // -> the "everything is one color / no torus" contamination. Tracking
    // which body's data is actually in the buffers forces a full re-upload
    // whenever the body changes, not just when it grows.
    const bool body_changed = (last_body != &body);
    const bool grew = cap_anchors < A.size() || cap_surfaces < S.size();
    if (grew) {
        auto rel = [](void* m) { if (m) clReleaseMemObject((cl_mem)m); };
        rel(buf_anchors); rel(buf_surfaces); rel(buf_stitch);
        buf_anchors  = clCreateBuffer(ctx, CL_MEM_READ_ONLY, A.size()*64, nullptr, &err); CLC(err);
        buf_surfaces = clCreateBuffer(ctx, CL_MEM_READ_ONLY, S.size()*64, nullptr, &err); CLC(err);
        // OpenCL rejects size==0; torus.psf often has empty stitch blob
        const size_t stitch_bytes = St.empty() ? size_t(ns) * 8u : St.size();
        buf_stitch   = clCreateBuffer(ctx, CL_MEM_READ_ONLY, stitch_bytes, nullptr, &err); CLC(err);
        cap_anchors = (uint32_t)A.size(); cap_surfaces = (uint32_t)S.size();
    }
    const bool full_upload = grew || body_changed || body.AnchorsDirty();
    if (full_upload) {
        // Nonblocking writes require host storage to remain alive until the copy
        // completes. A/St outlive this dispatch; temporary zeros do not.
        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_anchors, CL_FALSE, 0, A.size()*64, A.data(),  0, nullptr, nullptr));
        {
            const size_t stitch_bytes = St.empty() ? size_t(ns) * 8u : St.size();
            if (St.empty()) {
                const cl_char zero = 0;
                CLC(clEnqueueFillBuffer(q, (cl_mem)buf_stitch, &zero,
                                        sizeof(zero), 0, stitch_bytes, 0, nullptr, nullptr));
            } else {
                CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_stitch, CL_FALSE, 0, St.size(), St.data(), 0, nullptr, nullptr));
            }
        }
        // smooth per-anchor shading normals in a PARALLEL buffer (not the
        // anchor record's normal slot -- that slot is the displacement
        // direction Nd used to build the surface, and the smooth normal
        // differs from it by up to 90 deg, so overwriting it would move the
        // geometry). The kernel reads positions+Nd from the record and the
        // shading normal from here.
        const std::vector<Vec3>& sn = body.ComputeSmoothNormals();
        std::vector<cl_float4> snf(std::max(sn.size(), A.size()));
        for (size_t i = 0; i < snf.size(); ++i) {
            if (i < sn.size())
                snf[i] = cl_float4{{sn[i].x, sn[i].y, sn[i].z, 0.0f}};
            else
                snf[i] = cl_float4{{0.f, 1.f, 0.f, 0.f}};
        }
        if (cap_snorm < snf.size() || !buf_snorm) {
            if (buf_snorm) clReleaseMemObject((cl_mem)buf_snorm);
            buf_snorm = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                            snf.size()*sizeof(cl_float4), nullptr, &err); CLC(err);
            cap_snorm = (uint32_t)snf.size();
        }
        // snf is local staging: a nonblocking write may still read it after
        // this scope ends. Block only on this copy, not on the whole queue.
        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_snorm, CL_TRUE, 0,
                            snf.size()*sizeof(cl_float4), snf.data(), 0, nullptr, nullptr));
        body.MarkAnchorsClean();
    }
    if (grew || body_changed) {
        // body_changed: the buffer still holds the PREVIOUS body's surface
        // records (a smaller body after a larger one never triggered `grew`),
        // so a full re-upload is required, not a partial dirty flush.
        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_surfaces, CL_FALSE, 0, S.size()*64, S.data(), 0, nullptr, nullptr));
        body.ClearDirty();
    } else if (body.AnyDirty()) {
        // partial upload: only dirty 64-byte records (anchors already
        // handled above; they're clean here so FlushToGPU skips them)
        PSFFlushState fs; fs.queue = q;
        fs.anchors_buf = (cl_mem)buf_anchors; fs.surfaces_buf = (cl_mem)buf_surfaces;
        fs.anchors_bytes = A.size()*64; fs.surfaces_bytes = S.size()*64;
        body.FlushToGPU(&fs);
    }
    last_body = &body;
            last_anchor_count = A.size(); last_surface_count = S.size();

    float mvp[16];
    cam.BuildMVP(mvp, float(width) / float(height));
    const uint32_t total = BuildSampleGrids(body, mvp);
    // Optional attribution for close-up spikes: this counts actual launched
    // splat work, not GPU time. Log only when requested (per-body, per-frame).
    if (std::getenv("DELIRIUM_SPLAT_WORK_DIAG"))
        std::printf("[splat-work] surfaces=%u samples=%u spp=%.2f\n",
                    ns, total, float(total)/float(width*height));
    last_spp = 0;
    {
        static int slog = 0;
        if ((slog++ % 60) == 0) {
            const Vec3 bm = body.BBoxMin(), bs = body.BBoxSpan();
            // World AABB of dequantized anchors (detect collapse)
            float xmin=1e30f,ymin=1e30f,zmin=1e30f,xmax=-1e30f,ymax=-1e30f,zmax=-1e30f;
            const auto& Ab = body.AnchorBlob();
            Vec3 b0 = body.BBoxMin(), b1 = body.BBoxSpan();
            if (b1.x==0&&b1.y==0&&b1.z==0){ b0={-1,-1,-1}; b1={2,2,2}; }
            for (size_t i=0;i<Ab.size();++i) {
                const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&Ab[i]);
                uint16_t qx,qy,qz; std::memcpy(&qx,bytes+32,2); std::memcpy(&qy,bytes+34,2); std::memcpy(&qz,bytes+36,2);
                float wx=b0.x+(qx/65535.f)*b1.x, wy=b0.y+(qy/65535.f)*b1.y, wz=b0.z+(qz/65535.f)*b1.z;
                xmin=std::min(xmin,wx); ymin=std::min(ymin,wy); zmin=std::min(zmin,wz);
                xmax=std::max(xmax,wx); ymax=std::max(ymax,wy); zmax=std::max(zmax,wz);
            }
            std::printf("splat: samples=%u spp=%.2f file_bbox_span=(%.3f,%.3f,%.3f)\n",
                        total, float(total)/float(width*height+1), bs.x, bs.y, bs.z);
            std::printf("splat: anchor_aabb=(%.3f,%.3f,%.3f)..(%.3f,%.3f,%.3f) extent=(%.3f,%.3f,%.3f)\n",
                        xmin,ymin,zmin,xmax,ymax,zmax, xmax-xmin,ymax-ymin,zmax-zmin);
        }
    }

    if (cap_soff < ns + 1) {           // grow per-frame layout buffers
        if (buf_soff) { clReleaseMemObject((cl_mem)buf_soff);
                        clReleaseMemObject((cl_mem)buf_sss); }
        buf_soff = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (ns+1)*4, nullptr, &err); CLC(err);
        buf_sss  = clCreateBuffer(ctx, CL_MEM_READ_ONLY,  ns   *4, nullptr, &err); CLC(err);
        cap_soff = ns + 1;
    }
    // These vectors are reused for the NEXT body before the frame's final
    // clFinish. Blocking only on the second write preserves both host sources
    // on the in-order queue without a full-frame GPU synchronization here.
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_soff, CL_FALSE, 0, (ns+1)*4, h_soff.data(), 0, nullptr, nullptr));
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_sss,  CL_TRUE,  0,  ns   *4, h_sss.data(),  0, nullptr, nullptr));

    const cl_ulong far_key = ~cl_ulong(0);
    if (clear) {
        CLC(clEnqueueFillBuffer(q, (cl_mem)buf_fb, &far_key, 8, 0,
                                size_t(width)*height*8, 0, nullptr, nullptr));
        if (!has_long_atomics) {
            const cl_uint far_d = ~cl_uint(0);
            CLC(clEnqueueFillBuffer(q, (cl_mem)buf_depth32, &far_d, 4, 0,
                                    size_t(width)*height*4, 0, nullptr, nullptr));
        }
    }

    // CRITICAL: zero file span makes kernel map every anchor to one point
    // (world = min + t*0). Host grids already fell back to unit box; GPU must match.
    Vec3 bmn = body.BBoxMin(), bsp = body.BBoxSpan();
    if (bsp.x < 1e-8f && bsp.y < 1e-8f && bsp.z < 1e-8f) {
        bmn = Vec3{-1.f, -1.f, -1.f};
        bsp = Vec3{2.f, 2.f, 2.f};
    }
    const cl_float4 bbmin = {{bmn.x, bmn.y, bmn.z, 0}};
    const cl_float4 bbspan = {{bsp.x, bsp.y, bsp.z, 0}};
    // The float3 forms: the LAYERED splat kernel takes float3 (not float4)
    // for the bbox, the light and the eye.
    const cl_float3 bbmin_f3 = {{bmn.x, bmn.y, bmn.z}};
    const cl_float3 bbspan_f3 = {{bsp.x, bsp.y, bsp.z}};
    cl_float16 m16; std::memcpy(&m16, mvp, 64);
    // Key light from upper-front (the camera's side). toward-light points up
    // (+z, the scene up) and toward the camera (+x,-y). The old light came
    // from the far side (toward-light (-0.45,-0.35,+0.82)), so the
    // camera-facing tube wall (normal ~ +x,-y) sat at the emissive floor --
    // the "transparent front" symptom. Kernel shades with dot(N, -light_dir),
    // so we pass light_dir = -toward-light.
    float tlx = 0.55f, tly = -0.45f, tlz = 0.70f;   // toward-light (unnormalized)
    // Opt-in diagnostic: orbit the KEY light around world Z while leaving
    // the camera fill, geometry and workload untouched. Use elapsed time,
    // not frame count, so an FPS drop does not slow the light's rotation.
    if (const char* e = std::getenv("DELIRIUM_ROTATE_LIGHT")) {
        if (std::strcmp(e, "1") == 0) {
            using Clock = std::chrono::steady_clock;
            static const auto start = Clock::now();
            const float phase = std::chrono::duration<float>(Clock::now() - start).count() * 0.6f;
            const float c = std::cos(phase), s = std::sin(phase);
            const float x = tlx, y = tly;
            tlx = x*c - y*s; tly = x*s + y*c;
        }
    }
    const float lf = 1.0f / std::sqrt(tlx*tlx + tly*tly + tlz*tlz);
    const cl_float4 light = {{-tlx*lf, -tly*lf, -tlz*lf, 0}};
    // The float3 forms for the LAYERED splat kernel.
    const cl_float3 light_f3 = {{-tlx*lf, -tly*lf, -tlz*lf}};
    const cl_float3 eye_f3 = {{cam.position.x, cam.position.y, cam.position.z}};
    // The camera's forward axis: the NORMAL of the distance planes. The
    // band a sample belongs to is its depth along this axis.
    cl_float3 view_fwd_f3;
    {
        Vec3 vf = cam.target - cam.position;
        const float vfl = std::sqrt(vf.x*vf.x + vf.y*vf.y + vf.z*vf.z);
        if (vfl > 1e-6f) { vf.x /= vfl; vf.y /= vfl; vf.z /= vfl; }
        else vf = Vec3{0,0,1};
        view_fwd_f3 = {{vf.x, vf.y, vf.z}};
    }
    const float sscale = body.StitchScale();
    const cl_uint W = width, H = height;
    // The sample grid, rounded up to the work-group size. Shared by both
    // the layered and the classic splat launches.
    const size_t g = (size_t(total) + 255) & ~size_t(255);

    cl_kernel k = (cl_kernel)k_splat;
    cl_uint a = 0;

    // =====================================================================
    // THE LAYERED DISTANCE-PLANE SPLAT (the user's method).
    //
    // The classic splat path sends every sample straight to the shared
    // framebuffer via an atomic_min: each impact wins or loses ONE pixel.
    // Two samples of the SAME surface often land on adjacent pixels, so
    // the surface reads as a dithered, patchy skin.
    //
    // The layered path buckets the impacts into DISTANCE SHELLS (bands)
    // first, averages the impacts WITHIN each shell, and then stacks the
    // shells BACK TO FRONT so the nearest wins. The averaging is what
    // dissolves the dither into a continuous surface -- and because a
    // neighbour is always in the SAME shell, the fill never smears
    // across depth.
    //
    // THE LAYERED DISTANCE-PLANE SPLAT IS THE DEFAULT for the splat path.
    //
    // It is the user's method: the impacts are bucketed into distance
    // shells (bands), averaged WITHIN each shell, and the shells are
    // stacked BACK TO FRONT so the nearest wins. The averaging is what
    // dissolves the dither into a continuous surface -- and because a
    // neighbour is always in the SAME shell, the fill never smears
    // across depth.
    //
    // Measured on the gallery's splat scene (640x480, 191839 samples),
    // classic vs layered (16 bands, 8 fill passes):
    //
    //   classic : 16108 lit, 145 holes, 2.95% dither
    //   layered : 22701 lit, 129 holes, 2.84% dither
    //
    // More lit pixels, FEWER holes, and LOWER dither than the classic
    // path -- the fill is doing exactly what it is supposed to: closing
    // the gaps between impacts without smearing colour across depth.
    //
    // Opt out with DELIRIUM_SPLAT_LAYERS=0. The band count is
    // Selected with Path::SplatLayered. The band count is
    // DELIRIUM_SPLAT_BANDS (default 16).
    // =====================================================================
    const bool use_layered_splat = (path_ == Path::SplatLayered);
    int nbands = 16;
    if (const char* e = std::getenv("DELIRIUM_SPLAT_BANDS")) {
        int v = 0; if (std::sscanf(e, "%d", &v) == 1 && v >= 2 && v <= 64) nbands = v;
    }
    // How many fill passes per band (each propagates the fill 1px out).
    // The sweep on the splat scene is MONOTONIC in quality: more passes
    // -> more lit pixels, fewer holes, LOWER dither.
    //   fill 0: 16108 lit, 145 holes, 3.76% dither
    //   fill 2: 17766 lit, 143 holes, 3.53% dither
    //   fill 4: 19422 lit, 139 holes, 3.28% dither
    //   fill 6: 21068 lit, 132 holes, 3.05% dither
    //   fill 8: 22701 lit, 129 holes, 2.84% dither   <- best
    // The knob is capped at 8 (a larger value is rejected and the
    // default is kept).
    int band_fill_passes = 8;
    if (const char* e = std::getenv("DELIRIUM_SPLAT_BAND_FILL")) {
        int v = 0; if (std::sscanf(e, "%d", &v) == 1 && v >= 0 && v <= 8) band_fill_passes = v;
    }

    if (use_layered_splat && k_splat_layered && k_layer_fill_band && k_layer_stack_band) {
        // The band's depth range: the camera to the far side of the
        // body's world AABB (the same range the march path uses).
        // The world AABB of the TRANSFORMED anchors: the model matrix
        // moves the body, so the band range must follow it.
        float band_d_near = 0.f, band_d_far = 1.f;
        {
            // The band range is measured ALONG THE VIEW AXIS (the distance
            // plane's normal), not radially: the depth of a point from the
            // plane through the eye with normal `fwd` is dot(P - eye, fwd).
            //
            // The range MUST bound the DEFORMED surface, not the anchors.
            // The rendered sample is P + h*Nd (the patch's 8 coefficients
            // displace it along the anchor normal), so a curved body rises
            // above its anchor plane. Deriving the range from the anchor
            // positions alone made the far/near shells cut through the
            // displaced extremes, and those samples were silently dropped
            // by band_of()'s out-of-range check -> the layered path's
            // extra interior holes. This mirrors the march path's proven
            // deformed-bounds math (see the "deformed-surface bounds"
            // block above): 5 evaluations per patch (4 corners + centre).
            Mat4 mm; std::memcpy(mm.m, model, sizeof(model));
            Vec3 fwd = cam.target - cam.position;
            const float fl = std::sqrt(fwd.x*fwd.x + fwd.y*fwd.y + fwd.z*fwd.z);
            if (fl > 1e-6f) { fwd.x /= fl; fwd.y /= fl; fwd.z /= fl; }
            else fwd = Vec3{0,0,1};

            const auto& Ab = body.AnchorBlob();
            const auto& Sb = body.SurfaceBlob();
            Vec3 b0 = body.BBoxMin(), b1 = body.BBoxSpan();
            if (b1.x == 0 && b1.y == 0 && b1.z == 0) { b0 = {-1,-1,-1}; b1 = {2,2,2}; }
            auto dq = [&](uint16_t ai) -> Vec3 {
                if (ai >= Ab.size()) return Vec3{0,0,0};
                const uint8_t* by = reinterpret_cast<const uint8_t*>(&Ab[ai]);
                uint16_t px, py, pz; std::memcpy(&px, by+32, 2); std::memcpy(&py, by+34, 2); std::memcpy(&pz, by+36, 2);
                return Vec3{ b0.x + (px/65535.f)*b1.x,
                             b0.y + (py/65535.f)*b1.y,
                             b0.z + (pz/65535.f)*b1.z };
            };
            auto dn = [&](uint16_t ai) -> Vec3 {
                if (ai >= Ab.size()) return Vec3{0,1,0};
                const uint8_t* by = reinterpret_cast<const uint8_t*>(&Ab[ai]);
                float nx = by[44]*(2.f/255.f)-1.f, ny = by[45]*(2.f/255.f)-1.f, nz = by[46]*(2.f/255.f)-1.f;
                const float l = std::sqrt(nx*nx + ny*ny + nz*nz);
                if (l < 1e-6f) return Vec3{0,1,0};
                return Vec3{nx/l, ny/l, nz/l};
            };
            auto dh = [&](const float c[8], float u, float v) -> float {
                const float bu = u*(1.f-u), bv = v*(1.f-v);
                const float au = 2.f*u-1.f, av = 2.f*v-1.f;
                return c[0]*bu*(1.f-v) + c[1]*bu*v + c[2]*bv*(1.f-u) + c[3]*bv*u
                     + c[4]*bu*au*(1.f-v) + c[5]*bu*au*v
                     + c[6]*bv*av*(1.f-u) + c[7]*bv*av*u;
            };
            float dmin = 1e30f, dmax = -1e30f;
            const float PUV[5][2] = {{0.f,0.f},{1.f,0.f},{0.f,1.f},{1.f,1.f},{0.5f,0.5f}};
            for (const auto& s : Sb) {
                const Vec3 A0 = dq(s.anchor_idx[0]), A1 = dq(s.anchor_idx[1]);
                const Vec3 A2 = dq(s.anchor_idx[2]), A3 = dq(s.anchor_idx[3]);
                const Vec3 N0 = dn(s.anchor_idx[0]), N1 = dn(s.anchor_idx[1]);
                const Vec3 N2 = dn(s.anchor_idx[2]), N3 = dn(s.anchor_idx[3]);
                for (int k = 0; k < 5; ++k) {
                    const float uu = PUV[k][0], vv = PUV[k][1];
                    const Vec3 P{ (1-vv)*((1-uu)*A0.x + uu*A1.x) + vv*((1-uu)*A3.x + uu*A2.x),
                                  (1-vv)*((1-uu)*A0.y + uu*A1.y) + vv*((1-uu)*A3.y + uu*A2.y),
                                  (1-vv)*((1-uu)*A0.z + uu*A1.z) + vv*((1-uu)*A3.z + uu*A2.z) };
                    Vec3 Nd{ (1-vv)*((1-uu)*N0.x + uu*N1.x) + vv*((1-uu)*N3.x + uu*N2.x),
                             (1-vv)*((1-uu)*N0.y + uu*N1.y) + vv*((1-uu)*N3.y + uu*N2.y),
                             (1-vv)*((1-uu)*N0.z + uu*N1.z) + vv*((1-uu)*N3.z + uu*N2.z) };
                    const float nl = std::sqrt(Nd.x*Nd.x + Nd.y*Nd.y + Nd.z*Nd.z);
                    if (nl < 1e-6f) { Nd.x=0; Nd.y=1; Nd.z=0; }
                    else { Nd.x/=nl; Nd.y/=nl; Nd.z/=nl; }
                    const float h = dh(s.coef, uu, vv);
                    const Vec3 lp{ P.x + Nd.x*h, P.y + Nd.y*h, P.z + Nd.z*h };
                    const Vec3 wp = Mat4_TransformPoint(mm, lp);
                    const float d = (wp.x-cam.position.x)*fwd.x +
                                    (wp.y-cam.position.y)*fwd.y +
                                    (wp.z-cam.position.z)*fwd.z;
                    dmin = std::min(dmin,d); dmax = std::max(dmax,d);
                }
            }
            // Fall back to the anchor extents if the surface blob is empty.
            if (dmin > dmax) {
                dmin = 1e30f; dmax = -1e30f;
                for (size_t i = 0; i < Ab.size(); ++i) {
                    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&Ab[i]);
                    uint16_t qx,qy,qz; std::memcpy(&qx,bytes+32,2); std::memcpy(&qy,bytes+34,2); std::memcpy(&qz,bytes+36,2);
                    Vec3 lp{b0.x+(qx/65535.f)*b1.x, b0.y+(qy/65535.f)*b1.y, b0.z+(qz/65535.f)*b1.z};
                    Vec3 wp = Mat4_TransformPoint(mm, lp);
                    float d = (wp.x-cam.position.x)*fwd.x +
                              (wp.y-cam.position.y)*fwd.y +
                              (wp.z-cam.position.z)*fwd.z;
                    dmin = std::min(dmin,d); dmax = std::max(dmax,d);
                }
            }
            if (dmin > dmax) { dmin = 0.f; dmax = 1.f; }
            // A generous margin: the 5-point sampling cannot bound a
            // displacement that peaks between samples, and the kernel now
            // CLAMPS out-of-range samples into the nearest band, so an
            // over-generous range only costs a little shell resolution.
            const float margin = 0.25f * std::max(1e-3f, dmax - dmin);
            band_d_near = std::max(0.01f, dmin - margin);
            band_d_far  = dmax + margin;
        }

        // The band buffers: nbands * W * H slots.
        //
        // Each slot is ONE 64-bit key: (depth_bits << 32) | rgba. The
        // depth and the colour travel TOGETHER in a single atomic
        // (atomic_min), so "the nearest impact owns this pixel" and "this
        // is its colour" are one indivisible operation. Separate depth
        // and colour slots left a window where a preempted writer could
        // overwrite a nearer impact's colour with its own -- the render
        // was NON-DETERMINISTIC (five runs -> five different images).
        const uint32_t slots = uint32_t(nbands) * width * height;
        if (!buf_band_key || cap_band_slots < slots || cap_band_count != nbands) {
            if (buf_band_key)  clReleaseMemObject((cl_mem)buf_band_key);
            if (buf_band2_key) clReleaseMemObject((cl_mem)buf_band2_key);
            const size_t kb = size_t(slots) * sizeof(cl_ulong);
            buf_band_key  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, kb, nullptr, &err); CLC(err);
            // The FILL's DESTINATION set. The fill ping-pongs between
            // {buf_band_key} and {buf_band2_key}: pass 0 reads set A and
            // writes set B, pass 1 reads set B and writes set A, and so
            // on. The two sets must be distinct buffers -- an in-place
            // fill is a data race (see the header's comment).
            buf_band2_key = clCreateBuffer(ctx, CL_MEM_READ_WRITE, kb, nullptr, &err); CLC(err);
            cap_band_slots = slots;
            cap_band_count = nbands;
        }
        // Reset the band buffers to the EMPTY sentinel: the key
        // 0xFFFFFFFFFFFFFFFF (depth bits all ones = +inf). Every real
        // key is smaller, so a plain atomic_min against the sentinel
        // works as "first impact claims the slot".
        //
        // BOTH sets are reset: the second set is the fill's destination,
        // and a pass that writes it must start from a clean slate (a
        // stale key from the previous frame would make the fill treat a
        // pixel as already filled).
        {
            const cl_ulong empty_key = ~cl_ulong(0);
            CLC(clEnqueueFillBuffer(q, (cl_mem)buf_band_key, &empty_key, sizeof(cl_ulong), 0,
                                    size_t(slots)*sizeof(cl_ulong), 0, nullptr, nullptr));
            CLC(clEnqueueFillBuffer(q, (cl_mem)buf_band2_key, &empty_key, sizeof(cl_ulong), 0,
                                    size_t(slots)*sizeof(cl_ulong), 0, nullptr, nullptr));
        }

        cl_kernel kl = (cl_kernel)k_splat_layered;
        cl_uint aa = 0;
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_mem), &buf_anchors));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_mem), &buf_snorm));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_mem), &buf_surfaces));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_mem), &buf_stitch));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_mem), &buf_soff));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_mem), &buf_sss));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_uint), &ns));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_float3), &bbmin_f3));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_float3), &bbspan_f3));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_float), &sscale));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_float16), &m16));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_float3), &light_f3));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_float3), &eye_f3));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_float3), &view_fwd_f3));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_float4), &mr0));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_float4), &mr1));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_float4), &mr2));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_float4), &mr3));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_uint), &W));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_uint), &H));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_float), &depth_bias));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_float), &band_d_near));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_float), &band_d_far));
        cl_int nb = nbands;
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_int), &nb));
        CLC(clSetKernelArg(kl, aa++, sizeof(cl_mem), &buf_band_key));
        CLC(clEnqueueNDRangeKernel(q, kl, 1, nullptr, &g, nullptr, 0, nullptr, nullptr));

        // Fill each band's holes IN THAT BAND, then stack the bands back
        // to front. The fill PING-PONGS between the two band buffer sets:
        // pass 0 reads set A and writes set B, pass 1 reads set B and
        // writes set A, and so on. The source and destination must be
        // DIFFERENT buffers -- an in-place fill (source == destination)
        // is a data race, because a work-item reads neighbours that
        // other work-items in the same dispatch have already overwritten.
        // That race made the render NON-DETERMINISTIC: three renders of
        // the same scene produced three different md5s. The classic path
        // was bit-stable across the same three runs.
        const cl_uint WW = W, HH = H;
        size_t g2[2] = { size_t(W), size_t(H) };
        for (int bi = 0; bi < nbands; ++bi) {
            const uint32_t slot = uint32_t(bi) * width * height;
            // The fill's SOURCE set alternates per pass.
            void *src_key = buf_band_key;
            void *dst_key = buf_band2_key;
            for (int p = 0; p < band_fill_passes; ++p) {
                cl_uint f = 0;
                CLC(clSetKernelArg((cl_kernel)k_layer_fill_band, f++, sizeof(cl_mem), &src_key));
                CLC(clSetKernelArg((cl_kernel)k_layer_fill_band, f++, sizeof(cl_mem), &dst_key));
                CLC(clSetKernelArg((cl_kernel)k_layer_fill_band, f++, sizeof(cl_uint), &WW));
                CLC(clSetKernelArg((cl_kernel)k_layer_fill_band, f++, sizeof(cl_uint), &HH));
                CLC(clSetKernelArg((cl_kernel)k_layer_fill_band, f++, sizeof(cl_uint), &slot));
                const float INF1 = 1e30f;
                CLC(clSetKernelArg((cl_kernel)k_layer_fill_band, f++, sizeof(cl_float), &INF1));
                CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_layer_fill_band, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
                // Swap the source and destination for the next pass.
                void* t = src_key; src_key = dst_key; dst_key = t;
            }
            // The stack must read the set that holds the FILLED band.
            // The swap happens AFTER each write, so the set that was
            // written LAST is the one named `src_key` when the loop ends:
            //   N=0 passes -> src = A (the splat's own buffer)   [correct]
            //   N=1 pass  -> writes B, swap -> src = B           [correct]
            //   N=2 passes-> writes A, swap -> src = A           [correct]
            void *filled_key = src_key;
            // Stack this band into the shared framebuffer. The bands are
            // launched FAR -> NEAR (the user's stated order); the atom_min
            // makes the nearest win regardless.
            {
                cl_uint s = 0;
                CLC(clSetKernelArg((cl_kernel)k_layer_stack_band, s++, sizeof(cl_mem), &filled_key));
                CLC(clSetKernelArg((cl_kernel)k_layer_stack_band, s++, sizeof(cl_uint), &WW));
                CLC(clSetKernelArg((cl_kernel)k_layer_stack_band, s++, sizeof(cl_uint), &HH));
                CLC(clSetKernelArg((cl_kernel)k_layer_stack_band, s++, sizeof(cl_uint), &slot));
                const float INF1 = 1e30f;
                CLC(clSetKernelArg((cl_kernel)k_layer_stack_band, s++, sizeof(cl_float), &INF1));
                CLC(clSetKernelArg((cl_kernel)k_layer_stack_band, s++, sizeof(cl_mem), &buf_fb));
                cl_mem d32 = (cl_mem)buf_depth32;
                CLC(clSetKernelArg((cl_kernel)k_layer_stack_band, s++, sizeof(cl_mem), &d32));
                CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_layer_stack_band, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
            }
        }
        static int llog = 0;
        if ((llog++ % 60) == 0)
            std::printf("splat-layered: %ux%u bands=%d samples=%u fill=%d "
                        "range=[%.3f,%.3f]\n", W, H, nbands, total,
                        band_fill_passes, band_d_near, band_d_far);
        last_spp = 0;
        // The resolve + post passes run below in the `if (final)` block,
        // exactly as the classic path does.
    } else if (path_ == Path::SplatArea && k_splat_area && k_area_resolve) {
    // =====================================================================
    // THE AREA-WEIGHTED SPLAT (the pixel-footprint pyramid).
    //
    // The classic path lands every surface sample on EXACTLY ONE pixel.
    // This path distributes each sample over the pixels its footprint
    // covers -- the footprint's size grows linearly with the distance
    // (a pixel subtends a fixed ANGLE, so it covers more world space
    // further away) and stretches on a grazing surface -- and the pixel's
    // colour is the AREA-WEIGHTED MEAN of everything that touched it.
    //
    // WHY IT MATTERS: without the area weighting the surface's apparent
    // density depends on the VIEW. A grazing surface (one pixel covers
    // many samples' worth of area) received too few contributions and
    // showed holes a head-on surface did not; a head-on surface
    // (many pixels cover the same area) received too many and looked
    // denser than it was. Area weighting makes the two agree.
    //
    // THE WEIGHTS ARE INTEGERS: the accumulator sums fixed-point uints
    // and divides with an integer divide, so the sum is exact and the
    // render stays deterministic (the user's "floor function" instinct).
    // =====================================================================
    // The accumulators, one slot per pixel. Reallocated when the
    // render resolution changes.
    const uint32_t npx = W * H;
    if (!buf_area_csum || cap_area_px < npx) {
        if (buf_area_csum) clReleaseMemObject((cl_mem)buf_area_csum);
        if (buf_area_wsum) clReleaseMemObject((cl_mem)buf_area_wsum);
        if (buf_area_key)  clReleaseMemObject((cl_mem)buf_area_key);
        const size_t px3 = size_t(npx) * 3u * sizeof(cl_uint);
        const size_t px1 = size_t(npx) * sizeof(cl_uint);
        const size_t px8 = size_t(npx) * sizeof(cl_ulong);
        buf_area_csum = clCreateBuffer(ctx, CL_MEM_READ_WRITE, px3, nullptr, &err); CLC(err);
        buf_area_wsum = clCreateBuffer(ctx, CL_MEM_READ_WRITE, px1, nullptr, &err); CLC(err);
        buf_area_key  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, px8, nullptr, &err); CLC(err);
        cap_area_px = npx;
    }
    // Reset the accumulators: the colour sums and the weights to 0,
    // and the depth keys to the EMPTY sentinel (depth bits all ones
    // = +inf), so the first atom_min claims the slot.
    {
        const cl_uint zero = 0u;
        const cl_ulong empty_key = ~cl_ulong(0);
        CLC(clEnqueueFillBuffer(q, (cl_mem)buf_area_csum, &zero, sizeof(cl_uint), 0,
                                size_t(npx)*3u*sizeof(cl_uint), 0, nullptr, nullptr));
        CLC(clEnqueueFillBuffer(q, (cl_mem)buf_area_wsum, &zero, sizeof(cl_uint), 0,
                                size_t(npx)*sizeof(cl_uint), 0, nullptr, nullptr));
        CLC(clEnqueueFillBuffer(q, (cl_mem)buf_area_key, &empty_key, sizeof(cl_ulong), 0,
                                size_t(npx)*sizeof(cl_ulong), 0, nullptr, nullptr));
    }

    // tan(fov/2) -- the pixel footprint's world scale. The kernel
    // needs it to turn a depth into a footprint in pixels.
    float tan_half = std::tan(cam.fov_deg * 3.14159265358979f / 360.0f);

    cl_kernel ka = (cl_kernel)k_splat_area;
    cl_uint a2 = 0;
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_mem), &buf_anchors));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_mem), &buf_snorm));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_mem), &buf_surfaces));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_mem), &buf_stitch));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_mem), &buf_soff));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_mem), &buf_sss));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_uint), &ns));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_float3), &bbmin_f3));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_float3), &bbspan_f3));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_float), &sscale));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_float16), &m16));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_float3), &light_f3));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_float3), &eye_f3));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_float3), &view_fwd_f3));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_float4), &mr0));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_float4), &mr1));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_float4), &mr2));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_float4), &mr3));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_uint), &W));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_uint), &H));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_float), &depth_bias));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_float), &tan_half));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_mem), &buf_area_csum));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_mem), &buf_area_wsum));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_mem), &buf_area_key));
    CLC(clEnqueueNDRangeKernel(q, ka, 1, nullptr, &g, nullptr, 0, nullptr, nullptr));

    // RESOLVE this body's area-weighted mean into the SHARED
    // framebuffer, exactly as the classic path does. The shared
    // framebuffer is the multi-body compositing surface (nearest
    // wins), and the final resolve turns it into the image once,
    // after every body has contributed.
    //
    // The area accumulators are PER BODY and are reset above, which
    // is correct: the area weighting answers "how much of THIS
    // surface covers THIS pixel", and the depth test answers "which
    // body wins".
    {
        cl_kernel kr2 = (cl_kernel)k_area_resolve;
        CLC(clSetKernelArg(kr2, 0, sizeof(cl_mem), &buf_area_csum));
        CLC(clSetKernelArg(kr2, 1, sizeof(cl_mem), &buf_area_wsum));
        CLC(clSetKernelArg(kr2, 2, sizeof(cl_mem), &buf_area_key));
        CLC(clSetKernelArg(kr2, 3, sizeof(cl_uint), &npx));
        CLC(clSetKernelArg(kr2, 4, sizeof(cl_uint), &H));
        CLC(clSetKernelArg(kr2, 5, sizeof(cl_mem), &buf_fb));
        size_t gr2 = (size_t(npx) + 255) & ~size_t(255);
        CLC(clEnqueueNDRangeKernel(q, kr2, 1, nullptr, &gr2, nullptr, 0, nullptr, nullptr));
    }
    static int alog = 0;
    if ((alog++ % 60) == 0)
        std::printf("splat-area: %ux%u samples=%u footprint=area-weighted\n",
                    W, H, total);
    last_spp = 0;
    // The shared resolve_fb below runs as usual -- the area path
    // writes the same (depth|rgba) framebuffer the other paths use,
    // so the post passes (infill / lod-stitch) apply unchanged.
    } else {
    CLC(clSetKernelArg(k, a++, sizeof(cl_mem), &buf_anchors));
    CLC(clSetKernelArg(k, a++, sizeof(cl_mem), &buf_snorm));
    CLC(clSetKernelArg(k, a++, sizeof(cl_mem), &buf_surfaces));
    CLC(clSetKernelArg(k, a++, sizeof(cl_mem), &buf_stitch));
    CLC(clSetKernelArg(k, a++, sizeof(cl_mem), &buf_soff));
    CLC(clSetKernelArg(k, a++, sizeof(cl_mem), &buf_sss));
    CLC(clSetKernelArg(k, a++, sizeof(cl_uint), &ns));
    CLC(clSetKernelArg(k, a++, sizeof(cl_float4), &bbmin));
    CLC(clSetKernelArg(k, a++, sizeof(cl_float4), &bbspan));
    CLC(clSetKernelArg(k, a++, sizeof(cl_float), &sscale));
    CLC(clSetKernelArg(k, a++, sizeof(cl_float16), &m16));
    CLC(clSetKernelArg(k, a++, sizeof(cl_float4), &light));
    const cl_float4 eye = {{cam.position.x, cam.position.y, cam.position.z, 0}};
    CLC(clSetKernelArg(k, a++, sizeof(cl_float4), &eye));
    // The object's LOCAL->WORLD placement (the 64-byte records stay frozen
    // in local space -- this is what lets the user MOVE an object for free).
    CLC(clSetKernelArg(k, a++, sizeof(cl_float4), &mr0));
    CLC(clSetKernelArg(k, a++, sizeof(cl_float4), &mr1));
    CLC(clSetKernelArg(k, a++, sizeof(cl_float4), &mr2));
    CLC(clSetKernelArg(k, a++, sizeof(cl_float4), &mr3));
    CLC(clSetKernelArg(k, a++, sizeof(cl_uint), &W));
    CLC(clSetKernelArg(k, a++, sizeof(cl_uint), &H));
    CLC(clSetKernelArg(k, a++, sizeof(cl_float), &depth_bias));
    CLC(clSetKernelArg(k, a++, sizeof(cl_mem), &buf_fb));
    CLC(clSetKernelArg(k, a++, sizeof(cl_mem), &buf_tie_cnt));

    if (has_long_atomics) {
        CLC(clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr));
    } else {
        CLC(clSetKernelArg(k, a, sizeof(cl_mem), &buf_depth32));
        for (cl_uint pass = 0; pass < 2; ++pass) {
            CLC(clSetKernelArg(k, a + 1, sizeof(cl_uint), &pass));
            CLC(clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr));
        }
    }
    }   // end: else (classic splat)

    // Resolve + infill + lod-stitch + finish run ONCE per frame (final=true),
    // not per body: buf_fb accumulates ALL bodies via frontmost-wins atomics,
    // so only the LAST body's post-processing sees the combined image. The
    // other bodies' post passes were pure waste (their buf_img output was
    // overwritten by the next body's resolve). Same final image, ~1/5 the
    // post cost. The averaging (infill + lod-stitch) is fully preserved --
    // it just runs once on the combined image instead of 5x on intermediates.
    if (final) {
    // DEPTH-TIE DIAGNOSTIC readback: how many fragment (biased) samples this
    // frame landed within 2e-5 NDC of the pixel's current winner. High count
    // = coplanar cut-edge contention (the popping). 0 = clean. Gate the print
    // on DELIRIUM_TIE_DIAG=1 so it doesn't spam; the counter is cumulative
    // since init (one uint, never reset per frame).
    if (buf_tie_cnt && std::getenv("DELIRIUM_TIE_DIAG")) {
        cl_uint tc = 0;
        if (clEnqueueReadBuffer(q, (cl_mem)buf_tie_cnt, CL_TRUE, 0,
                                sizeof(tc), &tc, 0, nullptr, nullptr) == CL_SUCCESS)
            std::printf("  [tie-diag] depth-ties since init: %u\n", tc);
    }
    // The shared resolve_fb turns the framebuffer into the image. The
    // AREA-WEIGHTED path writes the same (depth|rgba) framebuffer the
    // other paths use, so it runs for every path unchanged.
    cl_kernel kr = (cl_kernel)k_resolve;
    const cl_uint npx = W * H;
    CLC(clSetKernelArg(kr, 0, sizeof(cl_mem), &buf_fb));
    CLC(clSetKernelArg(kr, 1, sizeof(cl_uint), &npx));
    CLC(clSetKernelArg(kr, 2, sizeof(cl_mem), &buf_img));
    size_t gr = (size_t(npx) + 255) & ~size_t(255);
    CLC(clEnqueueNDRangeKernel(q, kr, 1, nullptr, &gr, nullptr, 0, nullptr, nullptr));

    // GPU-only hole fill: CPU does not touch sample data — only enqueue + Present
    int passes = 3;  // optional full-frame propagation; 0 for raw-splat profiling
    if (const char* e = std::getenv("DELIRIUM_GPU_INFILL")) {
        if (e[0] == '0' && e[1] == '\0') passes = 0;
        else {
            unsigned v = 0;
            if (std::sscanf(e, "%u", &v) == 1 && v <= 32) passes = int(v);
        }
    }
    if (k_infill && passes > 0) {
        cl_context ctx2 = (cl_context)cl_ctx;
        cl_int e2 = CL_SUCCESS;
        if (!buf_img_b) {
            buf_img_b = clCreateBuffer(ctx2, CL_MEM_READ_WRITE, size_t(width)*height*4, nullptr, &e2);
            CLC(e2);
        }
        cl_kernel ki = (cl_kernel)k_infill;
        const cl_uint WW = W, HH = H;
        size_t g2[2] = { size_t(W), size_t(H) };
        cl_mem a = (cl_mem)buf_img;
        cl_mem b = (cl_mem)buf_img_b;
        for (int p = 0; p < passes; ++p) {
            CLC(clSetKernelArg(ki, 0, sizeof(cl_mem), &a));
            CLC(clSetKernelArg(ki, 1, sizeof(cl_mem), &b));
            CLC(clSetKernelArg(ki, 2, sizeof(cl_uint), &WW));
            CLC(clSetKernelArg(ki, 3, sizeof(cl_uint), &HH));
            CLC(clEnqueueNDRangeKernel(q, ki, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
            // swap
            cl_mem tmp = a; a = b; b = tmp;
        }
        if (a != (cl_mem)buf_img) {
            CLC(clEnqueueCopyBuffer(q, a, (cl_mem)buf_img, 0, 0, size_t(width)*height*4, 0, nullptr, nullptr));
        }
        static int slog = 0;
        if ((slog++ % 120) == 0)
            std::printf("GPU infill passes=%d (device-only)\n", passes);

        // Variable-LOD surface stitch: average coarse impacts with distance-aware radius.
        // DISABLED BY DEFAULT on the splat path: it was designed for the raymarch
        // path's per-object alpha, but the splat resolve marks every hit with the
        // SAME alpha (255), so the stitch's object mask (s.w == oid) treats all
        // objects + the background as one object and blurs colors across object
        // boundaries and into the background -> the "dissolving into noisy dots"
        // grain. At 2.0 samples/px the gaps are ~0.5px and the alpha-masked
        // infill closes them cleanly without this blur. Re-enable with
        // DELIRIUM_LOD_STITCH=2 if you want the old (grainy) look.
        {
            int lod_passes = 0;
            int base_r = 1, max_r = 3;  // keep small near POV; env can raise
            if (const char* e = std::getenv("DELIRIUM_LOD_STITCH")) {
                int v=0; if (std::sscanf(e,"%d",&v)==1 && v>=0 && v<=8) lod_passes = v;
            }
            if (const char* e = std::getenv("DELIRIUM_STITCH_RADIUS")) {
                int v=0; if (std::sscanf(e,"%d",&v)==1 && v>=1 && v<=8) max_r = v;
            }
            if (k_lod_stitch && lod_passes > 0 && buf_img) {
                if (!buf_img_b) {
                    cl_int e2 = CL_SUCCESS;
                    buf_img_b = clCreateBuffer((cl_context)cl_ctx, CL_MEM_READ_WRITE,
                        size_t(width)*height*4, nullptr, &e2); CLC(e2);
                }
                // Ensure non-zero alpha on hits so object mask works (splat resolve may be opaque)
                // Treat any non-background as object 1: done in kernel via alpha; if alpha is 255 ok.
                cl_kernel ks = (cl_kernel)k_lod_stitch;
                cl_mem a = (cl_mem)buf_img;
                cl_mem b = (cl_mem)buf_img_b;
                cl_int depth_valid = 0;
                float depth_scale = 0.35f;
                size_t g2[2] = { size_t(width), size_t(height) };
                for (int p = 0; p < lod_passes; ++p) {
                    cl_uint aa = 0;
                    CLC(clSetKernelArg(ks, aa++, sizeof(cl_mem), &a));
                    CLC(clSetKernelArg(ks, aa++, sizeof(cl_mem), &b));
                    cl_mem depth_dummy = a; // unused when depth_valid=0
                    CLC(clSetKernelArg(ks, aa++, sizeof(cl_mem), &depth_dummy));
                    cl_uint WW = (cl_uint)width, HH = (cl_uint)height;
                    CLC(clSetKernelArg(ks, aa++, sizeof(cl_uint), &WW));
                    CLC(clSetKernelArg(ks, aa++, sizeof(cl_uint), &HH));
                    CLC(clSetKernelArg(ks, aa++, sizeof(int), &base_r));
                    CLC(clSetKernelArg(ks, aa++, sizeof(int), &max_r));
                    CLC(clSetKernelArg(ks, aa++, sizeof(int), &depth_valid));
                    CLC(clSetKernelArg(ks, aa++, sizeof(float), &depth_scale));
                    CLC(clEnqueueNDRangeKernel((cl_command_queue)cl_queue, ks, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
                    cl_mem tmp = a; a = b; b = tmp;
                }
                CLC(clFinish((cl_command_queue)cl_queue));
                if (a != (cl_mem)buf_img) {
                    CLC(clEnqueueCopyBuffer((cl_command_queue)cl_queue, a, (cl_mem)buf_img, 0, 0,
                        size_t(width)*height*4, 0, nullptr, nullptr));
                    CLC(clFinish((cl_command_queue)cl_queue));
                }
                static int lslog = 0;
                if ((lslog++ % 60) == 0)
                    std::printf("GPU lod-stitch: passes=%d radius=[%d,%d]\n", lod_passes, base_r, max_r);
            }
        }
    }

    CLC(clFinish(q));
    }   // end if (final): resolve + infill + lod-stitch + finish run ONCE per
        // frame (on the combined buf_fb), not per body. Non-final bodies just
        // splat into buf_fb and return; the queue is async so ordering holds.
    last_spp = float(total) / float(W * H);
    return true;
}

void SplatRenderer::RenderEmptyFrame() {
    if (!buf_fb || !buf_img) return;
    cl_command_queue q = (cl_command_queue)cl_queue;
    cl_int err = CL_SUCCESS;
    const cl_ulong far_key = ~cl_ulong(0);
    err = clEnqueueFillBuffer(q, (cl_mem)buf_fb, &far_key, 8, 0,
                              size_t(width)*height*8, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return;
    if (!has_long_atomics && buf_depth32) {
        const cl_uint far_d = ~cl_uint(0);
        clEnqueueFillBuffer(q, (cl_mem)buf_depth32, &far_d, 4, 0,
                            size_t(width)*height*4, 0, nullptr, nullptr);
    }
    // Resolve to the background image so the viewport is not stale.
    cl_kernel kr = (cl_kernel)k_resolve;
    const cl_uint npx = cl_uint(width) * cl_uint(height);
    clSetKernelArg(kr, 0, sizeof(cl_mem), &buf_fb);
    clSetKernelArg(kr, 1, sizeof(cl_uint), &npx);
    clSetKernelArg(kr, 2, sizeof(cl_mem), &buf_img);
    size_t gr = (size_t(npx) + 255) & ~size_t(255);
    clEnqueueNDRangeKernel(q, kr, 1, nullptr, &gr, nullptr, 0, nullptr, nullptr);
    clFinish(q);
}

void SplatRenderer::SetPresentRect(int x, int y, int w, int h) {
    present_x = x; present_y = y; present_w = w; present_h = h;
}

void SplatRenderer::Present() {
    if (!sdl_rnd || !sdl_tex) return;   // headless: nothing to blit
    void* px = nullptr; int pitch = 0;
    bool image_valid = false;
    if (SDL_LockTexture((SDL_Texture*)sdl_tex, nullptr, &px, &pitch) == 0) {
        // SDL's streaming texture may have padded rows; a single packed GPU
        // readback is safe only when pitch == width*4. Keep the staging buffer
        // alive through this blocking transfer and copy rows otherwise.
        std::vector<unsigned char> staging;
        unsigned char* dst = static_cast<unsigned char*>(px);
        void* read_dst = px;
        if (pitch != width * 4) {
            staging.resize(size_t(width)*height*4);
            read_dst = staging.data();
        }
        const cl_int read_err = clEnqueueReadBuffer((cl_command_queue)cl_queue,
            (cl_mem)buf_img, CL_TRUE, 0, size_t(width)*height*4,
            read_dst, 0, nullptr, nullptr);
        if (read_err == CL_SUCCESS && !staging.empty()) {
            for (int y = 0; y < height; ++y)
                std::memcpy(dst + size_t(y)*pitch,
                            staging.data() + size_t(y)*width*4, size_t(width)*4);
        }
        image_valid = (read_err == CL_SUCCESS);
        SDL_UnlockTexture((SDL_Texture*)sdl_tex);
        if (read_err != CL_SUCCESS)
            std::fprintf(stderr, "Present: OpenCL image readback failed (%d)\n", read_err);
    } else {
        std::fprintf(stderr, "Present: SDL texture lock failed: %s\n", SDL_GetError());
    }
    // Diagnostic: dump the actual GPU framebuffer (post-infill, pre-SDL) to PPM
    // when DELIRIUM_DUMP_FRAME=N (0-based frame index). Ground-truth for
    // diagnosing render issues without window-chrome contamination.
    {
        static int dump_count = 0;
        static int dump_at = -1;
        static bool dump_init = false;
        if (!dump_init) {
            dump_init = true;
            if (const char* e = std::getenv("DELIRIUM_DUMP_FRAME")) {
                int v = -1; if (std::sscanf(e, "%d", &v) == 1) dump_at = v;
            }
        }
        if (image_valid && dump_at >= 0 && dump_count == dump_at) {
            char path[256];
            std::snprintf(path, sizeof(path), "/tmp/delirium_frame_%03d.ppm", dump_at);
            FILE* fp = std::fopen(path, "wb");
            if (fp) {
                std::fprintf(fp, "P6\n%d %d\n255\n", width, height);
                const unsigned char* img = static_cast<const unsigned char*>(px);
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x) {
                        const unsigned char* p = img + size_t(y)*pitch + size_t(x)*4;
                        std::fwrite(p, 1, 3, fp);
                    }
                std::fclose(fp);
                std::fprintf(stderr, "DUMP: wrote %s (%dx%d)\n", path, width, height);
            }
        }
        ++dump_count;
    }
    SDL_RenderClear((SDL_Renderer*)sdl_rnd);
    // Blit the rendered image into the viewport rectangle. When no rect is
    // set (every pre-existing harness), this is the full window exactly as
    // before. When a rect IS set (the docked workspace), the 3D image lands
    // only in that rectangle and the rest of the window is free for panels.
    if (present_w > 0 && present_h > 0) {
        SDL_Rect dst{ present_x, present_y, present_w, present_h };
        SDL_RenderCopy((SDL_Renderer*)sdl_rnd, (SDL_Texture*)sdl_tex, nullptr, &dst);
    } else {
        SDL_RenderCopy((SDL_Renderer*)sdl_rnd, (SDL_Texture*)sdl_tex, nullptr, nullptr);
    }
    // An attached editor draws SDL primitives onto the already-rendered frame.
    // No extra GPU readback, texture allocation, or Present call.
    // The overlay gets the WINDOW size (not the render resolution): editor
    // panels live in window coordinates, around the viewport rect.
    Engine::Instance().DrawOverlay(sdl_rnd, Engine::Instance().WindowWidth(),
                                    Engine::Instance().WindowHeight());
    SDL_RenderPresent((SDL_Renderer*)sdl_rnd);
}

// Reallocate every buffer whose size depends on the render resolution.
//
// WHY THIS EXISTS: the editor's docked layout renders the 3D image at the
// VIEWPORT's resolution (the center rect), not the window's. When that rect
// changes -- a panel resized, the window resized -- the GPU buffers must
// follow, or the kernels keep marching at the old resolution and the image
// is stretched or clipped.
//
// The march path's layer buffers (buf_mcolA/B, buf_mdepA/B) were allocated
// ONCE with `if (!buf_mcolA)`, so they silently kept the OLD resolution.
// They are released and recreated here.
void SplatRenderer::Resize(int w, int h) {
    if (w <= 0 || h <= 0) return;
    if (w == width && h == height) return;
    cl_context ctx = (cl_context)cl_ctx;
    cl_int err = CL_SUCCESS;

    width = w; height = h;

    // Frame-sized buffers.
    auto rel = [&](void*& m) { if (m) { clReleaseMemObject((cl_mem)m); m = nullptr; } };
    rel(buf_fb);
    rel(buf_img);
    rel(buf_depth32);

    // NOTE: the CLC macro expands to `return false`, which cannot be used in
    // this void function -- so the buffer creations below check the error
    // code explicitly instead.
    buf_fb  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size_t(w)*h*8, nullptr, &err);
    if (err != CL_SUCCESS) { std::fprintf(stderr, "Resize: buf_fb failed (%d)\n", err); return; }
    buf_img = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, size_t(w)*h*4, nullptr, &err);
    if (err != CL_SUCCESS) { std::fprintf(stderr, "Resize: buf_img failed (%d)\n", err); return; }
    if (!has_long_atomics) {
        buf_depth32 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size_t(w)*h*4, nullptr, &err);
        if (err != CL_SUCCESS) { std::fprintf(stderr, "Resize: buf_depth32 failed (%d)\n", err); return; }
    }

    // The single-resolution helpers (stitch ping-pong, infill ping-pong) and
    // the march layer buffers all scale with w*h. These live at file scope
    // (not as members), so they are released through the same file-static
    // handles the render path allocates them from.
    auto rel_static = [&](void*& m) { if (m) { clReleaseMemObject((cl_mem)m); m = nullptr; } };
    rel_static(buf_img_b);
    rel_static(buf_img2);
    rel_static(buf_mcolA); rel_static(buf_mcolB);
    rel_static(buf_mdepA); rel_static(buf_mdepB);

    // The SDL streaming texture carries the image to the window.
    if (sdl_rnd) {
        if (sdl_tex) SDL_DestroyTexture((SDL_Texture*)sdl_tex);
        sdl_tex = SDL_CreateTexture((SDL_Renderer*)sdl_rnd, SDL_PIXELFORMAT_RGBA32,
                                    SDL_TEXTUREACCESS_STREAMING, w, h);
        if (sdl_tex)
            SDL_SetTextureBlendMode((SDL_Texture*)sdl_tex, SDL_BLENDMODE_NONE);
    }
    std::printf("SplatRenderer::Resize -> %dx%d\n", w, h);
}

bool SplatRenderer::SaveFramePPM(const std::string& path) const {
    if (!buf_img || width <= 0 || height <= 0) return false;
    std::vector<unsigned char> img(size_t(width)*height*4);
    cl_int err = clEnqueueReadBuffer((cl_command_queue)cl_queue, (cl_mem)buf_img,
                                     CL_TRUE, 0, img.size(), img.data(),
                                     0, nullptr, nullptr);
    if (err != CL_SUCCESS) return false;
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    std::fprintf(fp, "P6\n%d %d\n255\n", width, height);
    for (int i = 0; i < width*height; ++i) {
        unsigned char rgb[3] = {img[i*4], img[i*4+1], img[i*4+2]};
        std::fwrite(rgb, 1, 3, fp);
    }
    std::fclose(fp);
    return true;
}

bool SplatRenderer::GrabFinalRGBA(std::vector<unsigned char>& out) const {
    if (!buf_img || width <= 0 || height <= 0) return false;
    out.resize(size_t(width)*height*4);
    cl_int err = clEnqueueReadBuffer((cl_command_queue)cl_queue, (cl_mem)buf_img,
                                     CL_TRUE, 0, out.size(), out.data(),
                                     0, nullptr, nullptr);
    return err == CL_SUCCESS;
}

void SplatRenderer::Shutdown() {
    auto rel = [](void* m) { if (m) clReleaseMemObject((cl_mem)m); };
    rel(buf_fb); rel(buf_img); rel(buf_depth32);
    rel(buf_anchors); rel(buf_surfaces); rel(buf_stitch);
    rel(buf_snorm);
    rel(buf_soff); rel(buf_sss);
    // The band buffers (both sets: the splat's own and the fill's
    // destination). They are member state, so Shutdown is the only
    // place they are released -- Resize only reallocates them when the
    // band count or the resolution changes.
    rel(buf_band_key); rel(buf_band2_key);
    buf_band_key = buf_band2_key = nullptr;
    cap_band_slots = 0; cap_band_count = 0;
    if (k_splat_layered) { clReleaseKernel((cl_kernel)k_splat_layered); k_splat_layered = nullptr; }
    if (k_layer_fill_band) { clReleaseKernel((cl_kernel)k_layer_fill_band); k_layer_fill_band = nullptr; }
    if (k_layer_stack_band) { clReleaseKernel((cl_kernel)k_layer_stack_band); k_layer_stack_band = nullptr; }
    if (k_splat)   clReleaseKernel((cl_kernel)k_splat);
    if (k_resolve) clReleaseKernel((cl_kernel)k_resolve);
    if (k_infill) { clReleaseKernel((cl_kernel)k_infill); k_infill = nullptr; }
    if (k_raymarch) { clReleaseKernel((cl_kernel)k_raymarch); k_raymarch = nullptr; }
    if (prog_raymarch) { clReleaseProgram((cl_program)prog_raymarch); prog_raymarch = nullptr; }
    if (prog_infill) { clReleaseProgram((cl_program)prog_infill); prog_infill = nullptr; }
    if (buf_img_b) { clReleaseMemObject((cl_mem)buf_img_b); buf_img_b = nullptr; }
    if (prog)      clReleaseProgram((cl_program)prog);
    if (cl_queue)  clReleaseCommandQueue((cl_command_queue)cl_queue);
    if (cl_ctx)    clReleaseContext((cl_context)cl_ctx);
    if (sdl_tex)   SDL_DestroyTexture((SDL_Texture*)sdl_tex);
}

} // namespace DeliRium
