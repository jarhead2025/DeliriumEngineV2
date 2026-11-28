#include "delirium/render/SplatRenderer.h"
#include "delirium/core/Engine.h"
#include "delirium/core/Transform.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/render/Camera.h"
#include "delirium/render/EquationAccel.h"
#include "delirium/render/LateralRayMarch.h"
#include <CL/cl.h>
#include <SDL.h>
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
// --- Multi-start layered equation renderer (kernels/equation_layered.cl) ---
// Per-pixel multi-start Newton inverse: each pixel tries N initial guesses
// spread across the patch domain and averages converged hits. Renders directly
// to buf_img (RGBA), like the raymarch path.
static void* prog_equation = nullptr;
static void* k_equation_layered = nullptr;
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
        case Path::EquationLayered: return "equation-layered";
        case Path::TriInfill:    return "tri-infill";
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
    //   DELIRIUM_TRI=1           -> Path::TriInfill  (structured-tri GPU raster)
    //   DELIRIUM_EQ=1            -> Path::EquationLayered (multi-start inverse)
    //   DELIRIUM_SPLAT_LAYERS=1  -> Path::SplatLayered (the user's method)
    //   DELIRIUM_RAYMARCH=1      -> Path::Raymarch
    //   DELIRIUM_MARCH=0         -> Path::SplatClassic
    //   otherwise                -> Path::March
    {
        const char* ti = std::getenv("DELIRIUM_TRI");
        const bool want_tri = ti && ti[0] == '1';
        const char* eq = std::getenv("DELIRIUM_EQ");
        const bool want_eq = eq && eq[0] == '1';
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
            if (want_tri)            path_ = Path::TriInfill;
            else if (want_eq)        path_ = Path::EquationLayered;
            else if (want_layered)   path_ = Path::SplatLayered;
            else if (want_ray)       path_ = Path::Raymarch;
            else if (march_off)      path_ = Path::SplatClassic;
            else                     path_ = Path::March;
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

    // Register the GPU reflow backend so PSFBody::ReflowAnchors uses it first
    // (fast path), falling back to CPU if unavailable. This wires RefineWhere's
    // overflow-safe fallback to the GPU without PSFBody depending on the renderer.
    PSFBody::RegisterGPUReflow([this](PSFBody& body, float a, float d, int it) {
        return ReflowBodyGPU(body, a, d, it);
    });

    // Register the GPU AMR (split) backend so PSFBody::SplitBodyGPU uses it
    // first for batches of legacy patches, falling back to the CPU SplitPatch
    // loop if unavailable. This wires RefineWhere's hot split path to the GPU.
    PSFBody::RegisterGPUSplit([this](PSFBody& body, const std::vector<uint32_t>& list) {
        return SplitBodyGPU(body, list);
    });

    // --- M2: allocate coarse occlusion depth tile buffer (16x16 floats) ---
    // Stores the nearest depth already rendered this frame per tile. Cleared
    // each frame by RenderEmptyFrame() / first RenderBody(clear=true).
    {
        cl_int e = CL_SUCCESS;
        cap_occlusion_tiles = occlusion_tiles_x * occlusion_tiles_y;
        buf_occlusion_depth = clCreateBuffer((cl_context)cl_ctx, CL_MEM_READ_WRITE,
                                             cap_occlusion_tiles * sizeof(float), nullptr, &e);
        if (e != CL_SUCCESS) {
            std::fprintf(stderr, "SplatRenderer: occlusion buffer alloc failed (%d); occlusion culling disabled.\n", e);
            buf_occlusion_depth = nullptr; cap_occlusion_tiles = 0;
        }
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
    // The TriInfill GPU path (the optimized renderer): psf_sample evaluates
    // PSF math surfaces on a structured su×sv grid per patch, tri_build builds
    // a structured triangle list (2 triangles per cell) with indexed output,
    // tri_raster rasterizes triangles with Gouraud shading + early-z rejection.
    // psf_sample and tri_build live in psf_splat.cl; tri_raster lives in
    // tri_infill.cl (separate program). See kernels/psf_splat.cl and
    // kernels/tri_infill.cl for the algorithms.
    k_psf_sample = clCreateKernel(pr, "psf_sample", &err);
    k_tri_build  = clCreateKernel(pr, "tri_build", &err);
    // Build and load the tri_raster kernel from its own source file.
    {
        std::ifstream tf(dir + "/tri_infill.cl");
        if (!tf) {
            std::fprintf(stderr, "tri_infill.cl not found; TriInfill path disabled.\n");
            k_psf_sample = k_tri_build = k_tri_raster = nullptr;
        } else {
            std::stringstream ts; ts << tf.rdbuf();
            const std::string tsrc = ts.str();
            const char* tp = tsrc.c_str(); const size_t tn = tsrc.size();
            cl_int terr = CL_SUCCESS;
            cl_program tpr = clCreateProgramWithSource((cl_context)cl_ctx, 1, &tp, &tn, &terr);
            if (terr != CL_SUCCESS ||
                clBuildProgram(tpr, 1, (cl_device_id*)&cl_dev, opts, nullptr, nullptr)
                != CL_SUCCESS) {
                char tlog[16384] = {};
                if (terr == CL_SUCCESS)
                    clGetProgramBuildInfo(tpr, (cl_device_id)cl_dev, CL_PROGRAM_BUILD_LOG,
                                          sizeof(tlog), tlog, nullptr);
                else
                    std::snprintf(tlog, sizeof(tlog), "clCreateProgramWithSource failed (%d)", terr);
                std::fprintf(stderr, "tri_infill.cl build failed:\n%s\n", tlog);
                k_psf_sample = k_tri_build = k_tri_raster = nullptr;
            } else {
                k_tri_raster = clCreateKernel(tpr, "tri_raster", &err);
                if (err != CL_SUCCESS) {
                    std::fprintf(stderr, "tri_raster kernel create failed (%d); TriInfill path disabled.\n", err);
                    k_psf_sample = k_tri_build = k_tri_raster = nullptr;
                }
            }
        }
    }

    // Anchor reflow GPU path kernels (anchor_reflow.cl): snapshot_target,
    // compute_error_indicators, compute_anchor_displacement,
    // update_anchor_positions, refit_coefficients. Shape-preserving — the
    // snapshot + refit reproduce the original surface over redistributed
    // anchors so repeated calls do not drift the shape.
    {
        std::ifstream rf(dir + "/anchor_reflow.cl");
        if (!rf) {
            std::fprintf(stderr, "anchor_reflow.cl not found; GPU reflow path disabled.\n");
        } else {
            std::stringstream rfs; rfs << rf.rdbuf();
            const std::string rsrc = rfs.str();
            const char* rrp = rsrc.c_str(); const size_t rsn = rsrc.size();
            cl_int rerr = CL_SUCCESS;
            cl_program rpr = clCreateProgramWithSource((cl_context)cl_ctx, 1, &rrp, &rsn, &rerr);
            if (rerr != CL_SUCCESS ||
                clBuildProgram(rpr, 1, (cl_device_id*)&cl_dev, opts, nullptr, nullptr)
                != CL_SUCCESS) {
                char rlog[16384] = {};
                if (rerr == CL_SUCCESS)
                    clGetProgramBuildInfo(rpr, (cl_device_id)cl_dev, CL_PROGRAM_BUILD_LOG,
                                          sizeof(rlog), rlog, nullptr);
                else
                    std::snprintf(rlog, sizeof(rlog), "clCreateProgramWithSource failed (%d)", rerr);
                std::fprintf(stderr, "anchor_reflow.cl build failed:\n%s\n", rlog);
            } else {
                k_reflow_accum  = clCreateKernel(rpr, "snapshot_error_accumulate", &err);
                k_reflow_disp   = clCreateKernel(rpr, "compute_anchor_displacement", &err);
                k_reflow_update = clCreateKernel(rpr, "update_anchor_positions", &err);
                k_reflow_refit  = clCreateKernel(rpr, "refit_coefficients", &err);
                if (err != CL_SUCCESS) {
                    std::fprintf(stderr, "anchor_reflow kernel create failed (%d); GPU reflow path disabled.\n", err);
                    k_reflow_accum = k_reflow_disp = k_reflow_update = k_reflow_refit = nullptr;
                }
            }
        }
    }

    // GPU AMR (live subdivision) kernel from psf_refine.cl: split_patches
    // evaluates the 5 hanging anchors on the parent equation, fits each child
    // by LSQ over a 9x9 grid, and appends records via an atomic free-list.
    // Disabled gracefully if the kernel file is missing or fails to build.
    {
        std::ifstream rf(dir + "/psf_refine.cl");
        if (!rf) {
            std::fprintf(stderr, "psf_refine.cl not found; GPU AMR path disabled.\n");
        } else {
            std::stringstream rfs; rfs << rf.rdbuf();
            const std::string rsrc = rfs.str();
            const char* rrp = rsrc.c_str(); const size_t rsn = rsrc.size();
            cl_int rerr = CL_SUCCESS;
            cl_program rpr = clCreateProgramWithSource((cl_context)cl_ctx, 1, &rrp, &rsn, &rerr);
            if (rerr != CL_SUCCESS ||
                clBuildProgram(rpr, 1, (cl_device_id*)&cl_dev, opts, nullptr, nullptr)
                != CL_SUCCESS) {
                char rlog[16384] = {};
                if (rerr == CL_SUCCESS)
                    clGetProgramBuildInfo(rpr, (cl_device_id)cl_dev, CL_PROGRAM_BUILD_LOG,
                                          sizeof(rlog), rlog, nullptr);
                else
                    std::snprintf(rlog, sizeof(rlog), "clCreateProgramWithSource failed (%d)", rerr);
                std::fprintf(stderr, "psf_refine.cl build failed:\n%s\n", rlog);
            } else {
                k_split = clCreateKernel(rpr, "split_patches", &err);
                if (err != CL_SUCCESS) {
                    std::fprintf(stderr, "psf_refine split_patches create failed (%d); GPU AMR path disabled.\n", err);
                    k_split = nullptr;
                } else {
                    // Signature audit: split_patches arg order must match kernel:
                    //  0 surfaces, 1 anchors, 2 split_list, 3 alloc_heads,
                    //  4 n_splits, 5 bbmin, 6 bbspan, 7 stitch, 8 s_scale
                    cl_uint n = 0;
                    cl_int e3 = clGetKernelInfo((cl_kernel)k_split, CL_KERNEL_NUM_ARGS,
                                                sizeof(n), &n, nullptr);
                    if (e3 == CL_SUCCESS && n != 9) {
                        std::fprintf(stderr,
                            "SplatRenderer: split_patches arg mismatch (kernel=%u host=9) -- "
                            "kernel/host drift; disabling GPU AMR path.\n", n);
                        k_split = nullptr;
                    }
                }
                clReleaseProgram(rpr);
            }
        }
    }

    // --- M2 GPU modal writeback: load apply_modes kernel from psf_deform.cl ---
    // apply_modes applies modal amplitudes K to coef[8] directly on the GPU,
    // eliminating the CPU WriteBack per frame. Disabled gracefully if the
    // kernel file is missing or fails to build.
    {
        std::string psf_deform_path = dir + "/psf_deform.cl";
        std::ifstream pf(psf_deform_path);
        if (pf) {
            std::stringstream pss; pss << pf.rdbuf();
            std::string psrc = pss.str();
            if (!psrc.empty() && cl_ctx) {
                const char* psp = psrc.c_str(); size_t psn = psrc.size();
                cl_int prerr = CL_SUCCESS;
                cl_program ppr = clCreateProgramWithSource((cl_context)cl_ctx, 1, &psp, &psn, &prerr);
                if (prerr == CL_SUCCESS) {
                    const char* popts = "-cl-fast-relaxed-math";
                    if (clBuildProgram(ppr, 1, (cl_device_id*)&cl_dev, popts, nullptr, nullptr) == CL_SUCCESS) {
                        k_apply_modes = clCreateKernel(ppr, "apply_modes", &err);
                        if (err != CL_SUCCESS) {
                            std::fprintf(stderr, "psf_deform.cl apply_modes create failed (%d); GPU modal writeback disabled.\n", err);
                            k_apply_modes = nullptr;
                        }
                    } else {
                        char plog[8192] = {};
                        clGetProgramBuildInfo(ppr, (cl_device_id)cl_dev, CL_PROGRAM_BUILD_LOG, sizeof(plog), plog, nullptr);
                        std::fprintf(stderr, "psf_deform.cl build failed:\n%s\n", plog);
                    }
                    clReleaseProgram(ppr);
                } else {
                    std::fprintf(stderr, "psf_deform.cl create program failed (%d); GPU modal writeback disabled.\n", prerr);
                }
            }
        }
    }

    // --- GPU-FIRST modal writeback kernels (psf_deform.cl) ---
    // modal_writeback_anchors / modal_writeback_surfaces compute ABSOLUTE
    // positions/coef from REST buffers + K each frame (no drift), writing to
    // working buffers resident on the device. This eliminates the per-frame
    // anchor/surface/snorm uploads (~85KB/frame). Enabled by DELIRIUM_GPU_MODAL=1.
    gpu_modal_on = false;
    if (const char* e = std::getenv("DELIRIUM_GPU_MODAL")) gpu_modal_on = (e[0] == '1');
    gpu_modal_diag = false;
    if (const char* e = std::getenv("DELIRIUM_GPU_MODAL_DIAG")) gpu_modal_diag = (e[0] == '1');
    if (gpu_modal_on) {
        std::string psf_deform_path = dir + "/psf_deform.cl";
        std::ifstream pf(psf_deform_path);
        if (pf) {
            std::stringstream pss; pss << pf.rdbuf();
            std::string psrc = pss.str();
            if (!psrc.empty() && cl_ctx) {
                const char* psp = psrc.c_str(); size_t psn = psrc.size();
                cl_int prerr = CL_SUCCESS;
                cl_program ppr = clCreateProgramWithSource((cl_context)cl_ctx, 1, &psp, &psn, &prerr);
                if (prerr == CL_SUCCESS) {
                    const char* popts = "-cl-fast-relaxed-math";
                    if (clBuildProgram(ppr, 1, (cl_device_id*)&cl_dev, popts, nullptr, nullptr) == CL_SUCCESS) {
                        k_modal_writeback_anchors = clCreateKernel(ppr, "modal_writeback_anchors", &err);
                        k_modal_writeback_surfaces = clCreateKernel(ppr, "modal_writeback_surfaces", &err);
                        k_modal_snorm_pass1 = clCreateKernel(ppr, "modal_snorm_pass1", &err);
                        k_modal_snorm_pass2 = clCreateKernel(ppr, "modal_snorm_pass2", &err);
                        if (err != CL_SUCCESS || !k_modal_writeback_anchors || !k_modal_writeback_surfaces
                            || !k_modal_snorm_pass1 || !k_modal_snorm_pass2) {
                            std::fprintf(stderr, "psf_deform.cl modal_writeback/snorm create failed (%d); GPU modal writeback disabled.\n", err);
                            k_modal_writeback_anchors = k_modal_writeback_surfaces = nullptr;
                            k_modal_snorm_pass1 = k_modal_snorm_pass2 = nullptr;
                            gpu_modal_on = false;
                        }
                    } else {
                        char plog[8192] = {};
                        clGetProgramBuildInfo(ppr, (cl_device_id)cl_dev, CL_PROGRAM_BUILD_LOG, sizeof(plog), plog, nullptr);
                        std::fprintf(stderr, "psf_deform.cl build failed:\n%s\n", plog);
                        gpu_modal_on = false;
                    }
                    clReleaseProgram(ppr);
                }
            }
        }
        if (gpu_modal_on)
            std::fprintf(stderr, "[gpu-modal] GPU modal writeback enabled\n");
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
            // layer_fill_band: 0 kin, 1 kout, 2 W, 3 H, 4 slot, 5 INF,
            // 6 depth_tol
            {"layer_fill_band",   (cl_kernel)k_layer_fill_band, 7},
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
    // 22 pass (0 = depth winners, 1 = colour), 23 area_csum, 24 area_wsum,
    // 25 area_key
    // area_resolve: 0 area_csum, 1 area_wsum, 2 area_key, 3 W, 4 H, 5 image
    {
        struct { const char* name; cl_kernel k; cl_uint expect; } checks[] = {
            {"psf_splat_area", (cl_kernel)k_splat_area, 26},
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

    // Signature audit for the TriInfill GPU path kernels. A drift here would
    // make clSetKernelArg fail with CL_INVALID_VALUE on every frame, so catch
    // it once and disable the TriInfill path (the other paths keep working).
    // psf_sample arg order (must match kernels/psf_splat.cl):
    //  0 anchors, 1 surfaces, 2 stitch, 3 patch_surf, 4 n_patches, 5 uv_n,
    //  6 bbmin, 7 bbspan, 8 s_scale, 9 mvp, 10 W, 11 H, 12 depth_bias,
    // 13 verts_out, 14 norms_out, 15 pos_out, 16 sample_off, 17 patch_uvn,
    // 18 lod_mode, 19 branch_trace
    // tri_build arg order (must match kernels/psf_splat.cl):
    //  0 verts_in, 1 norms_in, 2 n_patches, 3 uv_n, 4 indices_out, 5 n_tri_out,
    //  6 patch_uvn, 7 patch_cell_off, 8 sample_off, 9 lod_mode
    // tri_raster arg order (must match kernels/tri_infill.cl):
    //  0 indices, 1 n_tri, 2 W, 3 H, 4 INF, 5 depth_bias, 6 verts_grid,
    //  7 norms_grid, 8 pos_grid, 9 light_dir, 10 eye, 11 fb, 12 depth32
    {
        struct { const char* name; cl_kernel k; cl_uint expect; } checks[] = {
            {"psf_sample", (cl_kernel)k_psf_sample, 20},
            {"tri_build",  (cl_kernel)k_tri_build,  10},
            {"tri_raster", (cl_kernel)k_tri_raster, 13},
        };
        bool ok = true;
        for (auto& c : checks) {
            if (!c.k) continue;
            cl_uint n = 0;
            cl_int e3 = clGetKernelInfo(c.k, CL_KERNEL_NUM_ARGS, sizeof(n), &n, nullptr);
            if (e3 != CL_SUCCESS || n != c.expect) {
                std::fprintf(stderr,
                    "SplatRenderer: %s arg mismatch (kernel=%u host=%u) -- "
                    "kernel/host drift; disabling the TriInfill GPU path.\n",
                    c.name, n, c.expect);
                ok = false;
            }
        }
        if (!ok) k_psf_sample = k_tri_build = k_tri_raster = nullptr;
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
                        // march_compose_fb arg order (kernels/surfel_march.cl):
                        //  0 color, 1 depth, 2 W, 3 H, 4 INF, 5 depth_bias,
                        //  6 pass, 7 fb, 8 depth32
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

    // --- Multi-start layered equation renderer (kernels/equation_layered.cl) ---
    // Per-pixel multi-start Newton inverse: each pixel tries N initial guesses
    // spread across the patch domain and averages converged hits. Renders
    // directly to buf_img (RGBA). No surfel bake, no grid — evaluates the
    // parametric surface equation LIVE per pixel.
    {
        std::ifstream fi(dir + "/equation_layered.cl");
        if (!fi) fi.open("kernels/equation_layered.cl");
        if (fi) {
            std::stringstream ss; ss << fi.rdbuf();
            const std::string src = ss.str();
            const char* sp = src.c_str(); const size_t sn = src.size();
            cl_int e2 = CL_SUCCESS;
            cl_program pr = clCreateProgramWithSource((cl_context)cl_ctx, 1, &sp, &sn, &e2);
            if (e2 == CL_SUCCESS &&
                clBuildProgram(pr, 1, (cl_device_id*)&cl_dev, "-cl-fast-relaxed-math", nullptr, nullptr) == CL_SUCCESS) {
                k_equation_layered = clCreateKernel(pr, "equation_layered", &e2);
                if (e2 != CL_SUCCESS) {
                    k_equation_layered = nullptr;
                } else {
                    // Signature audit: equation_layered has 16 args.
                    struct { const char* name; cl_kernel k; cl_uint expect; } checks[] = {
                        {"equation_layered", (cl_kernel)k_equation_layered, 16},
                    };
                    bool sig_ok = true;
                    for (auto& c : checks) {
                        if (!c.k) continue;
                        cl_uint n = 0;
                        cl_int e3 = clGetKernelInfo(c.k, CL_KERNEL_NUM_ARGS, sizeof(n), &n, nullptr);
                        if (e3 != CL_SUCCESS || n != c.expect) {
                            std::fprintf(stderr,
                                "equation_layered signature mismatch: kernel '%s' has %u args, "
                                "host expects %u (e3=%d). Disabling equation path.\n",
                                c.name, n, c.expect, (int)e3);
                            sig_ok = false;
                        }
                    }
                    if (!sig_ok) k_equation_layered = nullptr;
                }
                prog_equation = pr;
                std::printf("GPU equation_layered (multi-start inverse): ready\n");
            } else {
                char log[8192] = {};
                if (pr) clGetProgramBuildInfo(pr, (cl_device_id)cl_dev, CL_PROGRAM_BUILD_LOG,
                    sizeof(log), log, nullptr);
                std::fprintf(stderr, "equation_layered build failed:\n%s\n", log);
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

    // STAGE TIMING: DELIRIUM_RENDER_STAGES=1 prints per-stage CPU timings inside
    // RenderBody to pinpoint where the ~26ms stall lives (GPU work is only ~2.5ms).
    const bool render_stages = std::getenv("DELIRIUM_RENDER_STAGES") != nullptr;
    using StageClock = std::chrono::steady_clock;
    auto stage_t = StageClock::now();
    auto stage_print = [&](const char* label) {
        if (!render_stages) return;
        double dt = std::chrono::duration<double, std::milli>(StageClock::now() - stage_t).count();
        stage_t = StageClock::now();
        std::fprintf(stderr, "[render-stage] %-22s %.2f ms\n", label, dt);
    };

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
    stage_print("path-flags-set");
    if (march_on && buf_img && buf_fb &&
        k_march_clear_grid && k_march_bake && k_march_build_grid && k_march_layer_clear &&
        k_march_draw && k_march_layer_fill && k_march_compose_fb && k_resolve) {
            // [march-diag] stage timing for the march path (guarded by DELIRIUM_MARCH_DIAG)
            bool march_diag = std::getenv("DELIRIUM_MARCH_DIAG") != nullptr;
            auto md0 = std::chrono::steady_clock::now();
            auto md_stage = md0;
            auto md_check = [&](const char* label) {
                if (!march_diag) return;
                double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - md_stage).count() * 1000.0;
                md_stage = std::chrono::steady_clock::now();
                std::printf("[march-diag] %-16s %.2f ms\n", label, dt);
            };
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

            // =================================================================
            // GPU-FIRST MODAL WRITEBACK PATH (eliminates per-frame uploads).
            // Modal bodies derive deformed anchors/surfaces/snorm from REST + K
            // entirely on the GPU each frame. Physics still runs CPU WriteBack
            // (for collision/FEM), but render is independent — no host round-trip,
            // no ~85KB/frame upload that stalled the Intel Arc driver (~27ms).
            // REST state uploaded ONCE at first bind; each frame only the tiny
            // K vector uploads, kernels write directly into buf_anchors/buf_surfaces/
            // buf_snorm that bake_surfels reads. Non-modal bodies use the path below.
            // =================================================================
            bool modal_gpu_path = false;
            ModalDynamics* md = body.ModalDynamicsPtr();
            if (md && gpu_modal_on) {
                if (body_changed) {
                    // First bind of a modal body (or topology change): upload REST once.
                    if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] frame: body_changed, calling SetupGPUModal\n");
                    if (SetupGPUModal(body, *md)) {
                        modal_gpu_path = true;
                        if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] frame: SetupGPUModal OK, gpu path active\n");
                    } else {
                        if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] frame: SetupGPUModal FAILED, falling back to upload\n");
                    }
                } else if (modal_body == &body) {
                    // Subsequent frames for the already-bound modal body.
                    modal_gpu_path = true;
                } else {
                    if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] frame: modal body not bound yet (body_changed=0)\n");
                }
                if (modal_gpu_path && md) {
                    // Per-frame: upload only K, run writeback kernels into render buffers.
                    const float* K = md->Amplitudes();
                    if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] frame: calling GPUModalWriteback, K=%p modes=%d\n",
                            (const void*)K, md->ModeCount());
                    if (K && GPUModalWriteback(body, *md, K, (uint32_t)md->ModeCount())) {
                        // Skip the host upload block below — bake_surfels reads the
                        // GPU buffers the kernels just wrote. Update ownership tracking.
                        last_body = &body;
                        last_anchor_count = A.size();
                        last_surface_count = S.size();
                        body.MarkAnchorsClean();
                        body.MarkSurfacesClean();
                        if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] frame: writeback OK\n");
                    } else {
                        modal_gpu_path = false;  // GPU path failed, fall back to upload
                        if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] frame: writeback FAILED, falling back to upload\n");
                    }
                }
            } else if (md && !gpu_modal_on) {
                if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] frame: modal body present but gpu_modal_on=0\n");
            }

            const bool full_upload = !modal_gpu_path && (grew || body_changed || body.AnchorsDirty());
            auto up_t = std::chrono::steady_clock::now();
            if (full_upload) {
                // NON-BLOCKING writes (CL_FALSE) for the per-frame anchor/stitch
                // uploads. The earlier code used CL_TRUE, which on the Intel Arc
                // driver forces a full queue sync per write -- measured ~27-31ms
                // per frame for a ~0.7 MB transfer (anchors+stitch+snorm). The
                // GPU kernels themselves cost <0.1ms, so the blocking sync was
                // the ENTIRE FPS bottleneck (22-25 FPS).
                //
                // WHY CL_FALSE is now SAFE here (the old CL_TRUE rationale no
                // longer applies to these two writes):
                //   1. A / St are PSFBody MEMBERS that persist for the whole
                //      frame. They are written in scene->Update() BEFORE this
                //      RenderBody call, and are NOT touched again until the
                //      NEXT frame's Update().
                //   2. The queue is FIFO-ordered: the kernels that read
                //      buf_anchors/buf_stitch are enqueued AFTER these writes,
                //      so they execute only after the DMA completes.
                //   3. The clFinish(q) at the end of this path (line ~1791)
                //      waits for ALL queued work before RenderBody returns, so
                //      the DMA is guaranteed complete before the next frame's
                //      Update() can modify A/St. No reallocation overlap.
                //
                // The old comment's hazard ("a live AMR split does push_back
                // which REALLOCATES") applies to a MODIFICATION of the blob
                // BETWEEN enqueue and completion. That cannot happen here:
                // the blob is only modified in Update(), which runs strictly
                // after the previous frame's clFinish. The partial-upload path
                // (body.FlushToGPU below) still uses CL_FALSE with the same
                // safety argument.
                //
                // NOTE: the snorm upload below keeps CL_TRUE because its
                // source `snf` is a LOCAL staging vector destroyed at the end
                // of this block -- that one genuinely needs the write to
                // complete before the vector goes out of scope.
                CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_anchors, CL_FALSE, 0, A.size()*64, A.data(), 0, nullptr, nullptr));
                {
                    const size_t stitch_bytes = St.empty() ? size_t(ns) * 8u : St.size();
                    if (St.empty()) {
                        const cl_char zero = 0;
                        CLC(clEnqueueFillBuffer(q, (cl_mem)buf_stitch, &zero,
                                                sizeof(zero), 0, stitch_bytes, 0, nullptr, nullptr));
                    } else {
                        // NON-BLOCKING (CL_FALSE): St is a PSFBody member that
                        // persists for the frame; see the buf_anchors comment
                        // above for the safety argument.
                        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_stitch, CL_FALSE, 0, St.size(), St.data(), 0, nullptr, nullptr));
                    }
                }
                // smooth per-anchor shading normals (parallel buffer, not the
                // anchor record's displacement slot)
                const std::vector<Vec3>& sn = body.ComputeSmoothNormals();
                if (march_diag) {
                    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - up_t).count() * 1000.0;
                    std::printf("[march-diag] uploads-CPU-setup %.2f ms (includes ComputeSmoothNormals)\n", dt);
                }
                const size_t sn_n = std::max(sn.size(), A.size());
                // PERSISTENT staging (renderer member) instead of a local vector,
                // so the upload can be NON-BLOCKING (CL_FALSE). A local staging
                // vector would be destroyed before the DMA completes; the member
                // survives until the next clFinish(q) at the end of this path,
                // which guarantees the transfer finishes before the staging is
                // reused next frame. This is the KEY change: the previous code
                // used CL_TRUE here, which forced a full queue sync of ALL prior
                // queued work (~27-31ms on the Intel Arc driver) -- the ENTIRE
                // FPS bottleneck. With CL_FALSE the uploads stage drops to ~1ms.
                if (cap_snorm_staging < sn_n) {
                    h_snorm_staging.resize(sn_n);
                    cap_snorm_staging = (uint32_t)sn_n;
                }
                for (size_t i = 0; i < sn_n; ++i) {
                    if (i < sn.size())
                        h_snorm_staging[i] = cl_float4{{sn[i].x, sn[i].y, sn[i].z, 0.0f}};
                    else
                        h_snorm_staging[i] = cl_float4{{0.f, 1.f, 0.f, 0.0f}};
                }
                if (cap_snorm < sn_n || !buf_snorm) {
                    if (buf_snorm) clReleaseMemObject((cl_mem)buf_snorm);
                    buf_snorm = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                    sn_n*sizeof(cl_float4), nullptr, &err); CLC(err);
                    cap_snorm = (uint32_t)sn_n;
                }
                // NON-BLOCKING (CL_FALSE): the persistent staging survives until
                // the clFinish(q) at the end of this path; see the comment above.
                CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_snorm, CL_FALSE, 0,
                                sn_n*sizeof(cl_float4), h_snorm_staging.data(), 0, nullptr, nullptr));
                body.MarkAnchorsClean();
            }
            if (grew || body_changed) {
                // body_changed: the shared buffer still holds the PREVIOUS
                // body's surface records -> full re-upload, not a partial flush.
                // BLOCKING (CL_TRUE): the source is the body's own vector,
                // which a live split can reallocate (see the anchor upload
                // above for the full argument).
                CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_surfaces, CL_TRUE, 0, S.size()*64, S.data(), 0, nullptr, nullptr));
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
            md_check("uploads");
            if (march_diag) {
                double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - up_t).count() * 1000.0;
                std::printf("[march-diag] uploads-total %.2f ms (full_upload path)\n", dt);
            }

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
            md_check("bake_kernel");
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
            // DIAGNOSTIC: attach the key light to the camera's azimuth.
            //
            // WHY: the default key light is FIXED IN WORLD SPACE. As the
            // camera orbits, the set of front-lit surfaces changes, so the
            // render changes -- which is physically correct, but it makes
            // "is the geometry symmetric?" unanswerable from the image
            // alone: a dim surface and a missing surface look identical.
            //
            // With the light attached to the camera, every azimuth sees the
            // same lighting, so any remaining variation in the coverage
            // curve is GEOMETRY. This is the A/B that separates "the shape
            // is wrong" from "the light moved".
            if (const char* e = std::getenv("DELIRIUM_CAMERA_LIGHT")) {
                if (std::strcmp(e, "1") == 0) {
                    // The camera's right vector, in world space: with up=+Y,
                    // cross(up, fwd) = (fwd.z, 0, -fwd.x). The light's
                    // horizontal component is placed along it, so the light
                    // stays at the camera's upper-left and turns WITH the
                    // camera.
                    Vec3 fwd = cam.target - cam.position;
                    const float fl = std::sqrt(fwd.x*fwd.x + fwd.y*fwd.y + fwd.z*fwd.z);
                    if (fl > 1e-6f) { fwd.x /= fl; fwd.y /= fl; fwd.z /= fl; }
                    Vec3 r{fwd.z, 0.f, -fwd.x};
                    const float rl = std::sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
                    if (rl > 1e-6f) { r.x /= rl; r.y /= rl; r.z /= rl; }
                    // toward-light = right*0.55 + up*0.45 + fwd*0.70
                    tlx = r.x * 0.55f + fwd.x * 0.70f;
                    tly = r.y * 0.55f + 0.45f + fwd.y * 0.70f;
                    tlz = r.z * 0.55f + fwd.z * 0.70f;
                }
            }
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
            md_check("build_grid_kernel");

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
                                // NOTE: march_compose_fb has exactly 9 args (0..8);
                                // the kernel does NOT take a tie-break epsilon
                                // (unlike psf_splat). Setting a 10th arg here
                                // caused CL_INVALID_KERNEL_ARGS (-49) every frame
                                // and produced an empty scene. Removed.
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
            md_check("march_draw_kernel");

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
            md_check("fill_passes");
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
                // NOTE: march_compose_fb has exactly 9 args (0..8); the kernel
                // does NOT take a tie-break epsilon (unlike psf_splat). Setting
                // a 10th arg here caused CL_INVALID_KERNEL_ARGS (-49) every
                // frame and produced an empty scene. Removed.
                if (has_long_atomics) {
                    CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_march_compose_fb, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
                } else if (buf_depth32) {
                    // two-pass fallback (same as psf_splat): pass 0 claims the
                    // depth with a 32-bit atomic_min, pass 1 stores the color
                    // only for the pixels this body won.
                    md_check("compose_fb_kernel");
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
                md_check("resolve_kernel");
                CLC(clFinish(q));
            }
            md_check("clFinish(sync)");
            {
                double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - md0).count() * 1000.0;
                std::printf("[march-diag] TOTAL renderbody %.2f ms (steps=%d nsurf=%u)\n", total, bsteps, nsurf);
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

    // --- MULTI-START LAYERED EQUATION RENDERER (kernels/equation_layered.cl) ---
    // Per-pixel multi-start Newton inverse: each pixel tries N initial guesses
    // spread across the patch domain and averages converged hits. Renders
    // directly to buf_img (RGBA), like raymarch. No surfel bake, no grid —
    // evaluates the parametric surface equation LIVE per pixel. Validated on
    // CPU (validate_render_path_holes): 16 layers -> ~13% lit coverage on the
    // torus, competitive with GPU sampling paths.
    const bool use_eq = (path_ == Path::EquationLayered);
    if (use_eq && k_equation_layered && buf_img) {
        const auto& A = body.AnchorBlob();
        const auto& S = body.SurfaceBlob();
        const uint32_t ns = (uint32_t)S.size();
        Vec3 bmn = body.BBoxMin(), bsp = body.BBoxSpan();
        if (bsp.x < 1e-8f && bsp.y < 1e-8f && bsp.z < 1e-8f) {
            bmn = Vec3{-1.f,-1.f,-1.f}; bsp = Vec3{2.f,2.f,2.f};
        }
        // Upload raw records (shared buffers, same ownership rule as raymarch).
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
        } else if (body.AnyDirty()) {
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
        cl_float3 eye = {{cam.position.x,cam.position.y,cam.position.z}};
        cl_float3 fwd = {{f.x,f.y,f.z}};
        cl_float3 right = {{rr.x,rr.y,rr.z}};
        cl_float3 upv = {{uu.x,uu.y,uu.z}};
        cl_float3 bbmin = {{bmn.x,bmn.y,bmn.z}};
        cl_float3 bbspan = {{bsp.x,bsp.y,bsp.z}};
        cl_float3 light = {{0.4f, 0.7f, 0.5f}};
        int n_layers = 16;
        if (const char* e = std::getenv("DELIRIUM_EQ_LAYERS")) {
            int v=0; if (std::sscanf(e,"%d",&v)==1 && v>=1 && v<=64) n_layers=v;
        }
        cl_int nl = n_layers;
        cl_uint W=(uint32_t)width, H=(uint32_t)height;
        cl_uint aa=0;
        CLC(clSetKernelArg((cl_kernel)k_equation_layered, aa++, sizeof(cl_mem), &buf_anchors));
        CLC(clSetKernelArg((cl_kernel)k_equation_layered, aa++, sizeof(cl_mem), &buf_surfaces));
        CLC(clSetKernelArg((cl_kernel)k_equation_layered, aa++, sizeof(uint32_t), &ns));
        CLC(clSetKernelArg((cl_kernel)k_equation_layered, aa++, sizeof(cl_float3), &bbmin));
        CLC(clSetKernelArg((cl_kernel)k_equation_layered, aa++, sizeof(cl_float3), &bbspan));
        CLC(clSetKernelArg((cl_kernel)k_equation_layered, aa++, sizeof(cl_float3), &eye));
        CLC(clSetKernelArg((cl_kernel)k_equation_layered, aa++, sizeof(cl_float3), &fwd));
        CLC(clSetKernelArg((cl_kernel)k_equation_layered, aa++, sizeof(cl_float3), &right));
        CLC(clSetKernelArg((cl_kernel)k_equation_layered, aa++, sizeof(cl_float3), &upv));
        CLC(clSetKernelArg((cl_kernel)k_equation_layered, aa++, sizeof(float), &tan_half));
        CLC(clSetKernelArg((cl_kernel)k_equation_layered, aa++, sizeof(float), &aspect));
        CLC(clSetKernelArg((cl_kernel)k_equation_layered, aa++, sizeof(uint32_t), &W));
        CLC(clSetKernelArg((cl_kernel)k_equation_layered, aa++, sizeof(uint32_t), &H));
        CLC(clSetKernelArg((cl_kernel)k_equation_layered, aa++, sizeof(int), &nl));
        CLC(clSetKernelArg((cl_kernel)k_equation_layered, aa++, sizeof(cl_float3), &light));
        CLC(clSetKernelArg((cl_kernel)k_equation_layered, aa++, sizeof(cl_mem), &buf_img));
        size_t g2[2] = {size_t(width), size_t(height)};
        CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_equation_layered, 2, nullptr, g2, nullptr, 0, nullptr, nullptr));
        CLC(clFinish(q));
        static int elog = 0;
        if ((elog++ % 60) == 0)
            std::printf("equation-layered: %ux%u surfaces=%u layers=%d (multi-start inverse)\n",
                        W, H, ns, n_layers);
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
    stage_print("uncond-block-entry");
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
        stage_print("snorm-cltrue-write");
        body.MarkAnchorsClean();
    }
    stage_print("after-full-upload");
    // GPU-FIRST modal bodies: skip the unconditional host upload block below.
    // Modal dynamics modifies host blobs every frame, so without this gate the
    // FlushToGPU loop uploads dirty surface records every frame (hundreds of
    // clEnqueueWriteBuffer calls that serialize on the Intel Arc in-order queue
    // -> ~32ms stall). The tri-infill path's GPU modal writeback writes
    // buf_anchors/buf_surfaces/buf_snorm directly, so the host upload is
    // redundant. Detected early here because this block runs BEFORE the path
    // dispatch (line 2897).
    ModalDynamics* md = body.ModalDynamicsPtr();
    const bool is_modal_body = (md != nullptr) && gpu_modal_on;
    if (!is_modal_body) {
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
    }
    last_body = &body;
            last_anchor_count = A.size(); last_surface_count = S.size();

    float mvp[16];
    cam.BuildMVP(mvp, float(width) / float(height));
    stage_print("mvp-built");
    const uint32_t total = BuildSampleGrids(body, mvp);
    stage_print("build-grids");
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
    // Both vectors are written before any kernel reads them in the same frame,
    // and the final clFinish(q) ensures ordering. CL_FALSE avoids a full GPU
    // queue sync (~27ms stall on Intel Arc) that CL_TRUE forced every frame.
    stage_print("pre-sss-writes");
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_soff, CL_FALSE, 0, (ns+1)*4, h_soff.data(), 0, nullptr, nullptr));
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_sss,  CL_FALSE, 0,  ns   *4, h_sss.data(),  0, nullptr, nullptr));
    stage_print("sss-write");

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

    // THE DEPTH TOLERANCE for the band fill's neighbour vote.
    //
    // WHY IT EXISTS: the fill averages a missing pixel's filled NEIGHBOURS.
    // "Same band" is not "same surface" -- a band is a depth shell, and a
    // shell can hold several surfaces (at a grazing angle a torus's near
    // rim and far rim project onto the same pixels at very different
    // depths). Averaging across them blends two surfaces into a colour that
    // exists nowhere in the geometry.
    //
    // MEASURED (gallery scenes U and W -- a 1x render compared against a 4x
    // render downsampled as the reference; the classic path is the baseline
    // at 0.75% interior error):
    //
    //   gate off               20808 lit   3.52% interior error  (scene U)
    //   gate on, tol = 0       16776 lit   1.76% interior error  (scene U)
    //   gate on, tol >= 0.002  20808 lit   3.52% interior error  (gate off!)
    //
    // So the fill was inventing INTERIOR pixels that do not match the
    // surface, and the gate HALVES that error. The knee is sharp and it is
    // at ZERO because the NDC depth is compressed toward 1 with distance:
    // a RELATIVE window is WIDE in world units exactly where the
    // compression is steep, so the other surface falls inside it. The gate
    // is therefore an ABSOLUTE window on the NDC depth.
    //
    // WHY IT IS A KNOB: the host sweeps it and MEASURES which value gives
    // the best fidelity -- the measurement picks the value, not a guess.
    // A NEGATIVE value disables the gate (the pre-gate behaviour), which
    // has to stay expressible so the gate can be measured against the
    // thing it replaced.
    float band_depth_tol = 0.0f;
    if (const char* e = std::getenv("DELIRIUM_SPLAT_DEPTH_TOL")) {
        float v = 0.f;
        if (std::sscanf(e, "%f", &v) == 1 && v >= -1.0f && v <= 0.05f)
            band_depth_tol = v;
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
                CLC(clSetKernelArg((cl_kernel)k_layer_fill_band, f++, sizeof(cl_float), &band_depth_tol));
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

    // THE AREA PATH RUNS IN TWO PASSES.
    //
    // Pass 0: every sample offers its depth to the pixels its footprint
    //         covers, and the NEAREST wins each pixel. No colour.
    // Pass 1: the same samples offer their colour, but only to pixels
    //         whose winner is at the same depth -- so the area-weighted
    //         mean never mixes the near wall of a torus with its far wall.
    //
    // WHY TWO PASSES AND NOT ONE: the winner is only known after EVERY
    // sample has been offered. An in-kernel gate cannot work -- a far
    // sample that arrives FIRST sees no winner and passes, so the gate
    // leaks exactly the samples it is meant to reject. (MEASURED: an
    // in-kernel gate made the interior dither slightly WORSE, 4.437 ->
    // 4.522, because the far samples that arrived first still counted.)
    //
    // The two passes share the same buffer, so pass 1 reads the winners
    // pass 0 wrote. The queue is in-order, so no fence is needed.
    for (uint32_t area_pass = 0; area_pass < 2; ++area_pass) {
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
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_uint), &area_pass));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_mem), &buf_area_csum));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_mem), &buf_area_wsum));
    CLC(clSetKernelArg(ka, a2++, sizeof(cl_mem), &buf_area_key));
    CLC(clEnqueueNDRangeKernel(q, ka, 1, nullptr, &g, nullptr, 0, nullptr, nullptr));
    } // end of the two-pass loop (area_pass 0 = depths, 1 = colour)

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
    } else if (path_ == Path::TriInfill && k_psf_sample && k_tri_build && k_tri_raster) {
    stage_print("tri-infill-block-start");
    // =====================================================================
    // THE TRI-INFILL GPU PATH (the optimized renderer).
    // Local 2D vector type (Types.h only defines Vec3 locally).
    struct Vec2 { float x, y; };
    // --- SWITCHABLE DEBUG MODE: DELIRIUM_TRI_DEBUG=1 enables all diagnostic
    // readbacks and printf tracing. Default OFF = zero overhead, clean log. ---
    const bool tri_debug = std::getenv("DELIRIUM_TRI_DEBUG") != nullptr;
    //
    // FRAMEBUFFER CLEAR: the tri-infill path writes the SAME depth-key format
    // as every other path into the shared buf_fb, so it MUST reset the frame
    // on body 0 (clear=true) exactly like the march/splat paths. Without this
    // the buffer retains garbage depths from VRAM: atom_min picks up garbage
    // (rainbow bands), and early-z sees garbage depths so it never skips
    // pixels (raster spikes to 100+ ms). Gated on `clear` so multi-body
    // frames keep earlier bodies' contributions.
    if (clear && buf_fb) {
        const cl_ulong far_key = ~cl_ulong(0);
        CLC(clEnqueueFillBuffer(q, (cl_mem)buf_fb, &far_key, 8, 0,
                                size_t(width)*height*8, 0, nullptr, nullptr));
        // buf_depth32 MUST be cleared to +inf on EVERY frame regardless of
        // atomic support: tri_raster reads it unconditionally for early-z
        // rejection BEFORE the atom_min write. If it holds VRAM garbage,
        // early-z skips valid fragments, those pixels keep garbage keys in
        // buf_fb (not the far-key sentinel), and resolve_fb extracts garbage
        // rgba -> black strips instead of background color. The atomics guard
        // only gates whether the depth buffer is USED for the non-atomic
        // double-pass path; the early-z read happens either way.
        if (buf_depth32) {
            const cl_uint far_d = ~cl_uint(0);
            CLC(clEnqueueFillBuffer(q, (cl_mem)buf_depth32, &far_d, 4, 0,
                                    size_t(width)*height*4, 0, nullptr, nullptr));
        }
    }
    //
    // Three kernels compose into the shared framebuffer buf_fb:
    //   psf_sample : one work-item per structured sample point. Evaluates the
    //                PSF math surface equation at that (patch, u, v) using the
    //                same bilinear + corner-vanishing-polynomial convention as
    //                the CPU Patch::EvaluateDerivs, projects to screen space,
    //                writes the sample's screen position + packed base colour
    //                into verts_grid and its world-space normal into norms_grid.
    //   tri_build  : one work-item per triangle (2 per grid cell). Reads the
    //                structured grid via indexed output and writes a uint[3]
    //                triangle index list into buf_indices (indexed consumption
    //                avoids the redundant vertex writes of the flat-output
    //                version: ~4-5x write reduction).
    //   tri_raster : one work-item per triangle. Edge-function rasterization
    //                with Gouraud shading (lighting scalar interpolated across
    //                the triangle) and early-z rejection (skip occluded pixels).
    //                Writes the SAME depth-key format as the march path so the
    //                paths compose identically into buf_fb:
    //                  key = (as_uint(depth) << 32) | rgba ; atom_min keeps nearest.
    // =====================================================================
    const auto& A = body.AnchorBlob();
    const auto& S = body.SurfaceBlob();
    const uint32_t ns = (uint32_t)S.size();
    Vec3 bmn = body.BBoxMin(), bsp = body.BBoxSpan();
    if (bsp.x < 1e-8f && bsp.y < 1e-8f && bsp.z < 1e-8f) {
        bmn = Vec3{-1.f,-1.f,-1.f}; bsp = Vec3{2.f,2.f,2.f};
    }

    // --- SURFACE RECORD COLOR SOURCE DIAGNOSTIC ---
    // Trace the color data at its SOURCE: the surface records themselves.
    // channel_color reads coefficients from offset 44 + ch*6 in each record.
    // If entire patches have zero coefficients, that's why color-diag shows
    // clustered zeros (not scattered pattern-eval zeros). This pinpoints
    // whether the bug is upstream (file/loader) or downstream (kernel).
    if (tri_debug) {
        int patches_zero_color = 0, patches_full_color = 0, patches_partial = 0;
        float total_abs_K = 0, total_signed_K = 0;
        int patches_signed_zero = 0;
        for (uint32_t s = 0; s < ns; ++s) {
            const auto& rec = S[s];
            const uint8_t* by = reinterpret_cast<const uint8_t*>(&rec);
            int ch_zeros = 0, ch_signed_zeros = 0;
            for (int ch = 0; ch < 3; ++ch) {
                // channel_color layout: offset 44+ch*6 = 2 bytes indices + 4 bytes K (signed char)
                const uint8_t* c = by + 44 + ch*6;
                const int8_t* K = reinterpret_cast<const int8_t*>(c + 2);
                float abs_sum = std::fabs((float)K[0]) + std::fabs((float)K[1])
                              + std::fabs((float)K[2]) + std::fabs((float)K[3]);
                float signed_sum = (float)K[0] + (float)K[1] + (float)K[2] + (float)K[3];
                total_abs_K += abs_sum;
                total_signed_K += std::fabs(signed_sum);
                if (abs_sum < 1e-4f) ++ch_zeros; // all four K coefficients are zero
                // Signed cancellation: K={+31,-31,0,0} has abs_sum=62 but signed_sum=0
                // -> color is zero everywhere even though coefficients look healthy.
                if (std::fabs(signed_sum) < 1e-4f) ++ch_signed_zeros;
            }
            if (ch_zeros == 3) ++patches_zero_color;
            else if (ch_zeros == 0) ++patches_full_color;
            else ++patches_partial;
            if (ch_signed_zeros == 3) ++patches_signed_zero;
        }
        std::printf("  [src-diag] color source: patches_zero=%d full=%d partial=%d "
                    "avg|K|=%.3f avg|signed_K|=%.3f signed_zero_patches=%d\n",
                    patches_zero_color, patches_full_color, patches_partial,
                    total_abs_K / (ns * 3), total_signed_K / (ns * 3), patches_signed_zero);
    }

    // Upload raw records (shared buffers, same ownership rule as the splat path).
    const bool body_changed = (last_body != &body);
    const bool grew = cap_anchors < A.size() || cap_surfaces < S.size();
    // =================================================================
    // GPU-FIRST modal writeback for tri-infill path (mirrors march path).
    // Modal bodies modify host blobs every frame (ModalDynamics::WriteBack),
    // so without this the path below uploads ~85KB/frame every frame ->
    // Intel Arc driver stall ~27-31ms. REST state uploaded ONCE at first
    // bind; each frame only the tiny K vector uploads, kernels write directly
    // into buf_anchors/buf_surfaces/buf_snorm that psf_sample reads.
    // Non-modal bodies use the host upload path below.
    // =================================================================
    stage_print("pre-modal-block");
    bool modal_gpu_path = false;
    ModalDynamics* md = body.ModalDynamicsPtr();
    if (md && gpu_modal_on) {
        // Use modal_body (not body_changed) as the binding indicator. body_changed
        // depends on last_body which may already equal &body from a prior frame,
        // so it never triggers SetupGPUModal. modal_body != &body is the robust
        // signal that the GPU modal path needs (re)setup for this body.
        if (modal_body != &body) {
            if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] tri-frame: calling SetupGPUModal (modal_body=%p body=%p)\n",
                    (const void*)modal_body, (const void*)&body);
            if (SetupGPUModal(body, *md)) {
                modal_gpu_path = true;
                if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] tri-frame: SetupGPUModal OK, gpu path active\n");
            } else {
                if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] tri-frame: SetupGPUModal FAILED, falling back to upload\n");
            }
        } else {
            // Subsequent frames for the already-bound modal body.
            modal_gpu_path = true;
        }
        if (modal_gpu_path && md) {
            // Per-frame: upload only K, run writeback kernels into render buffers.
            const float* K = md->Amplitudes();
            if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] tri-frame: calling GPUModalWriteback, K=%p modes=%d\n",
                    (const void*)K, md->ModeCount());
            if (K && GPUModalWriteback(body, *md, K, (uint32_t)md->ModeCount())) {
                // Skip the host upload block below — psf_sample reads the
                // GPU buffers the kernels just wrote. Update ownership tracking.
                last_body = &body;
                last_anchor_count = A.size();
                last_surface_count = S.size();
                body.MarkAnchorsClean();
                body.MarkSurfacesClean();
                if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] tri-frame: writeback OK\n");
            } else {
                modal_gpu_path = false;  // GPU path failed, fall back to upload
                if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] tri-frame: writeback FAILED, falling back to upload\n");
            }
        }
    } else if (md && !gpu_modal_on) {
        if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] tri-frame: modal body present but gpu_modal_on=0\n");
    }
    stage_print("modal-writeback/setup");

    if (!modal_gpu_path && (grew || body_changed || !buf_anchors)) {
        using Clock = std::chrono::steady_clock;
        auto t0 = Clock::now();
        auto rel = [](void* m) { if (m) clReleaseMemObject((cl_mem)m); };
        rel(buf_anchors); rel(buf_surfaces); rel(buf_stitch); rel(buf_snorm);
        buf_anchors  = clCreateBuffer(ctx, CL_MEM_READ_ONLY, A.size()*64, nullptr, &err); CLC(err);
        buf_surfaces = clCreateBuffer(ctx, CL_MEM_READ_ONLY, S.size()*64, nullptr, &err); CLC(err);
        buf_stitch   = clCreateBuffer(ctx, CL_MEM_READ_ONLY, S.size()*8,  nullptr, &err); CLC(err);
        buf_snorm    = clCreateBuffer(ctx, CL_MEM_READ_WRITE, A.size()*12, nullptr, &err); CLC(err);
        cap_anchors = (uint32_t)A.size(); cap_surfaces = (uint32_t)S.size();
        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_anchors, CL_FALSE, 0, A.size()*64, A.data(), 0, nullptr, nullptr));
        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_surfaces, CL_FALSE, 0, S.size()*64, S.data(), 0, nullptr, nullptr));
        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_stitch, CL_FALSE, 0, S.size()*8, body.StitchBlob().data(), 0, nullptr, nullptr));
        // --- GPU READBACK DIAGNOSTIC: verify what kind is actually in VRAM ---
        // The host vector S shows kind=1, but the kernel reads from buf_surfaces.
        // Read the buffer back and print the kind byte to confirm VRAM matches host.
        // If VRAM kind != host kind, the upload is corrupt/wrong buffer.
        if (tri_debug) {
            std::vector<uint8_t> h_surf_back(S.size() * 64);
            CLC(clEnqueueReadBuffer(q, (cl_mem)buf_surfaces, CL_TRUE, 0,
                                    S.size() * 64, h_surf_back.data(), 0, nullptr, nullptr));
            std::printf("  [tri-track-gpu-rb] GPU surface buffer readback:\n");
            for (uint32_t s = 0; s < ns; ++s) {
                const uint8_t* rec = h_surf_back.data() + s * 64;
                uint32_t kind_gpu = 0;
                std::memcpy(&kind_gpu, rec + 8, 4);
                float p0, p1, p2;
                std::memcpy(&p0, rec + 12, 4);
                std::memcpy(&p1, rec + 16, 4);
                std::memcpy(&p2, rec + 20, 4);
                std::printf("    gpu_surf[%u]: kind=%u anchor_idx=[%u,%u,%u,%u] params=(%g,%g,%g)\n",
                            s, kind_gpu,
                            *(const uint16_t*)(rec+0), *(const uint16_t*)(rec+2),
                            *(const uint16_t*)(rec+4), *(const uint16_t*)(rec+6),
                            (double)p0, (double)p1, (double)p2);
            }
            // --- TRACKING: print exactly what surface kind is uploaded to GPU ---
            auto kname = [](uint32_t k) -> const char* {
                switch (k) {
                    case 0: return "PLANE"; case 1: return "ELLIPSOID"; case 2: return "TORUS";
                    case 3: return "BOX"; case 4: return "GAUSSIAN"; case 5: return "WAVEFIELD";
                    case 6: return "CYLINDER"; case 7: return "COMPOUND"; case 8: return "LEGACY";
                    default: return "UNKNOWN";
                }
            };
            std::printf("  [tri-track-upload] surface_blob: ns=%u\n", ns);
            for (uint32_t s = 0; s < ns; ++s) {
                const uint32_t kind = S[s].surface_def;
                std::printf("    surface[%u]: kind=%u (%s) anchor_idx=[%u,%u,%u,%u] coef=[%g,%g,%g,%g,%g,%g,%g,%g]\n",
                            s, kind, kname(kind),
                            S[s].anchor_idx[0], S[s].anchor_idx[1], S[s].anchor_idx[2], S[s].anchor_idx[3],
                            (double)S[s].coef[0], (double)S[s].coef[1], (double)S[s].coef[2], (double)S[s].coef[3],
                            (double)S[s].coef[4], (double)S[s].coef[5], (double)S[s].coef[6], (double)S[s].coef[7]);
            }
            // Also print anchor positions (dequantized) to verify they're not all coplanar.
            std::printf("  [tri-track-upload] anchors: bbox_min=(%g,%g,%g) bbox_span=(%g,%g,%g)\n",
                        (double)bmn.x, (double)bmn.y, (double)bmn.z,
                        (double)bsp.x, (double)bsp.y, (double)bsp.z);
        }
        // Smooth normals: CPU computes them once per anchor (average of adjacent patch normals).
        {
            std::vector<cl_float3> snorm(A.size(), {{0,0,0}});
            for (const auto& s : S) {
                for (int k = 0; k < 4; ++k) {
                    const uint16_t ai = s.anchor_idx[k];
                    if (ai >= A.size()) continue;
                    const uint8_t* by = reinterpret_cast<const uint8_t*>(&A[ai]);
                    float nx = by[44]*(2.f/255.f)-1.f, ny = by[45]*(2.f/255.f)-1.f, nz = by[46]*(2.f/255.f)-1.f;
                    snorm[ai].s[0] += nx; snorm[ai].s[1] += ny; snorm[ai].s[2] += nz;
                }
            }
            std::vector<uint8_t> snorm_q(A.size()*12);
            for (size_t i = 0; i < A.size(); ++i) {
                float l = std::sqrt(snorm[i].s[0]*snorm[i].s[0] + snorm[i].s[1]*snorm[i].s[1] + snorm[i].s[2]*snorm[i].s[2]);
                if (l < 1e-6f) { snorm[i].s[0]=0; snorm[i].s[1]=1; snorm[i].s[2]=0; l=1.f; }
                else { snorm[i].s[0]/=l; snorm[i].s[1]/=l; snorm[i].s[2]/=l; }
                snorm_q[i*12+0] = (uint8_t)(snorm[i].s[0]*0.5f+0.5f)*255;
                snorm_q[i*12+1] = (uint8_t)(snorm[i].s[1]*0.5f+0.5f)*255;
                snorm_q[i*12+2] = (uint8_t)(snorm[i].s[2]*0.5f+0.5f)*255;
            }
            CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_snorm, CL_FALSE, 0, A.size()*12, snorm_q.data(), 0, nullptr, nullptr));
        }
        last_body = &body;
        last_anchor_count = A.size(); last_surface_count = S.size();
        last_tri_upload_ms = std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
    } else if (!modal_gpu_path && body.AnyDirty()) {
        // Host upload path for non-modal bodies (or modal fallback). Modal bodies
        // skip this — GPUModalWriteback already wrote the GPU buffers.
        using Clock = std::chrono::steady_clock;
        auto t0 = Clock::now();
        PSFFlushState fs; fs.queue = q;
        fs.anchors_buf = (cl_mem)buf_anchors; fs.surfaces_buf = (cl_mem)buf_surfaces;
        fs.anchors_bytes = A.size()*64; fs.surfaces_bytes = S.size()*64;
        body.FlushToGPU(&fs);
        last_tri_upload_ms = std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
    }
    stage_print("host-upload/skip");

    // Build the structured sample grid per patch (adaptive UVN from projected
    // size). psf_sample reads patch_uvn (packed su<<16|sv) and sample_off
    // (prefix sums of sample counts) to know each patch's grid resolution.
    // The grid is sized so near-camera patches get high UVN (quality) and far
    // patches get low UVN (fewer triangles), cutting upload + raster cost
    // where it does not matter visually.
    //
    // DEFAULT ADAPTIVE LOD ON (2026-11-18 fix): without adaptive LOD every
    // STRUCTURAL FIX (matches working March path sampling): use UNIFORM dense
    // sampling per patch like March's bake_surfels (uvn=10), NOT adaptive LOD.
    // Adaptive LOD sends some patches to very low UVN (count=4 samples), which
    // makes the bilinear height-field approximation too coarse -> flat faces and
    // the validity check clipping valid samples. Uniform sampling keeps every
    // patch dense enough that the approximation is accurate and no clipping is
    // needed — exactly how March works (dense surfels, no clipping). Override
    // with DELIRIUM_TRI_UVN (2..256). Default raised to 30 for parametric
    // surfaces: a single parametric patch needs enough samples around BOTH
    // the major circle and tube cross-section to look round instead of square.
    // uvn=10 → ~10 samples per circle → polygonal tube; uvn=30 → smooth.
    int base_uvn = 30;
    if (const char* e = std::getenv("DELIRIUM_TRI_UVN")) {
        int v = 0; if (std::sscanf(e, "%d", &v) == 1 && v >= 2 && v <= 256) base_uvn = v;
    }
    // Disable adaptive LOD by default to match March's uniform sampling.
    // Set DELIRIUM_TRI_LOD=1 to re-enable adaptive per-patch UVN.
    const bool lod_mode = (std::getenv("DELIRIUM_TRI_LOD") != nullptr &&
                           std::strcmp(std::getenv("DELIRIUM_TRI_LOD"), "1") == 0);
    // Adaptive LOD bounds (only used when lod_mode=true). Matches prior range.
    int lod_min = 6, lod_max = 200;
    if (const char* e = std::getenv("DELIRIUM_TRI_LOD_RANGE")) {
        if (std::sscanf(e, "%d %d", &lod_min, &lod_max) != 2) { lod_min = 6; lod_max = 200; }
        if (lod_min < 2) lod_min = 2; if (lod_max > 256) lod_max = 256;
    }

    // Compute per-patch UVN and sample offsets.
    std::vector<uint32_t> h_patch_uvn(ns), h_sample_off(ns+1), h_patch_cell_off(ns+1);
    uint32_t total_samples = 0, total_cells = 0;
    if (lod_mode) {
        // Adaptive: project each patch's corner anchors to screen space and
        // derive UVN from the projected edge length in pixels.
        Mat4 mm; std::memcpy(mm.m, model, sizeof(model));
        float tan_half = std::tan(cam.fov_deg * 3.14159265358979f / 360.0f);
        float aspect = float(W) / float(H);
        Vec3 fwd{cam.target.x-cam.position.x, cam.target.y-cam.position.y, cam.target.z-cam.position.z};
        float fl = std::sqrt(fwd.x*fwd.x+fwd.y*fwd.y+fwd.z*fwd.z); if (fl<1e-6f) fl=1.f;
        fwd.x/=fl; fwd.y/=fl; fwd.z/=fl;
        Vec3 rr{fwd.y*cam.up.z-fwd.z*cam.up.y, fwd.z*cam.up.x-fwd.x*cam.up.z, fwd.x*cam.up.y-fwd.y*cam.up.x};
        float rl=std::sqrt(rr.x*rr.x+rr.y*rr.y+rr.z*rr.z); rr.x/=rl; rr.y/=rl; rr.z/=rl;
        Vec3 uu{rr.y*fwd.z-rr.z*fwd.y, rr.z*fwd.x-rr.x*fwd.z, rr.x*fwd.y-rr.y*fwd.x};
        auto dq = [&](uint16_t ai) -> Vec3 {
            if (ai >= A.size()) return Vec3{0,0,0};
            const uint8_t* by = reinterpret_cast<const uint8_t*>(&A[ai]);
            uint16_t px, py, pz; std::memcpy(&px, by+32, 2); std::memcpy(&py, by+34, 2); std::memcpy(&pz, by+36, 2);
            return Vec3{ bmn.x + (px/65535.f)*bsp.x, bmn.y + (py/65535.f)*bsp.y, bmn.z + (pz/65535.f)*bsp.z };
        };
        auto project = [&](const Vec3& P) -> Vec2 {
            Vec3 wp = Mat4_TransformPoint(mm, P);
            float d = (wp.x-cam.position.x)*fwd.x + (wp.y-cam.position.y)*fwd.y + (wp.z-cam.position.z)*fwd.z;
            if (d < 1e-4f) d = 1e-4f;
            float rx = ((wp.x-cam.position.x)*rr.x + (wp.y-cam.position.y)*rr.y + (wp.z-cam.position.z)*rr.z) / d;
            float ry = ((wp.x-cam.position.x)*uu.x + (wp.y-cam.position.y)*uu.y + (wp.z-cam.position.z)*uu.z) / d;
            return Vec2{ (rx*tan_half*aspect*0.5f + 0.5f) * float(W),
                         (0.5f - ry*tan_half*0.5f) * float(H) };
        };
        for (uint32_t s = 0; s < ns; ++s) {
            const auto& patch = S[s];
            const Vec3 A0 = dq(patch.anchor_idx[0]), A1 = dq(patch.anchor_idx[1]);
            const Vec3 A2 = dq(patch.anchor_idx[2]), A3 = dq(patch.anchor_idx[3]);
            Vec2 P0 = project(A0), P1 = project(A1), P2 = project(A2), P3 = project(A3);
            float e01 = std::sqrt((P1.x-P0.x)*(P1.x-P0.x)+(P1.y-P0.y)*(P1.y-P0.y));
            float e23 = std::sqrt((P3.x-P2.x)*(P3.x-P2.x)+(P3.y-P2.y)*(P3.y-P2.y));
            float e02 = std::sqrt((P2.x-P0.x)*(P2.x-P0.x)+(P2.y-P0.y)*(P2.y-P0.y));
            float e13 = std::sqrt((P3.x-P1.x)*(P3.x-P1.x)+(P3.y-P1.y)*(P3.y-P1.y));
            float max_edge = std::max(std::max(e01, e23), std::max(e02, e13));
            // UVN scales with projected edge length: more samples where the
            // patch covers more pixels. Clamp to [lod_min, lod_max].
            int uvn = (int)(max_edge * 0.5f + 0.5f);
            if (uvn < lod_min) uvn = lod_min; if (uvn > lod_max) uvn = lod_max;
            if (uvn < 2) uvn = 2;
            h_patch_uvn[s] = (uint32_t(uvn) << 16) | uint32_t(uvn);
            uint32_t su = uvn, sv = uvn;
            uint32_t samples = su * sv;
            uint32_t cells = (su > 1 && sv > 1) ? (su-1)*(sv-1) : 0;
            h_sample_off[s] = total_samples; total_samples += samples;
            h_patch_cell_off[s] = total_cells; total_cells += cells;
        }
    } else {
        // Uniform UVN: every patch gets base_uvn x base_uvn samples.
        for (uint32_t s = 0; s < ns; ++s) {
            uint32_t su = base_uvn, sv = base_uvn;
            uint32_t samples = su * sv;
            uint32_t cells = (su > 1 && sv > 1) ? (su-1)*(sv-1) : 0;
            h_patch_uvn[s] = (uint32_t(su) << 16) | uint32_t(sv);
            h_sample_off[s] = total_samples; total_samples += samples;
            h_patch_cell_off[s] = total_cells; total_cells += cells;
        }
    }
    h_sample_off[ns] = total_samples;
    h_patch_cell_off[ns] = total_cells;
    if (total_cells == 0) {
        // Degenerate: no cells to rasterize. Skip this body cleanly.
        last_spp = 0;
    } else {
        // Allocate grid buffers lazily (frame-sized, reallocated on growth).
        if (cap_tri_samples < total_samples) {
            auto rel = [](void* m) { if (m) clReleaseMemObject((cl_mem)m); };
            rel(buf_verts_grid); rel(buf_norms_grid); rel(buf_pos_grid); rel(buf_branch_trace);
            buf_verts_grid = clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_samples*sizeof(cl_float4), nullptr, &err); CLC(err);
            buf_norms_grid = clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_samples*sizeof(cl_float3), nullptr, &err); CLC(err);
            buf_pos_grid   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_samples*sizeof(cl_float3), nullptr, &err); CLC(err);
            buf_branch_trace = clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_samples*sizeof(uint32_t), nullptr, &err); CLC(err);
            cap_tri_samples = total_samples;
        }
        if (cap_tri_tris < total_cells*2) {
            auto rel = [](void* m) { if (m) clReleaseMemObject((cl_mem)m); };
            rel(buf_indices); rel(buf_n_tri);
            buf_indices = clCreateBuffer(ctx, CL_MEM_READ_WRITE, total_cells*2*3*sizeof(uint32_t), nullptr, &err); CLC(err);
            buf_n_tri   = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(uint32_t), nullptr, &err); CLC(err);
            uint32_t zero = 0;
            CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_n_tri, CL_TRUE, 0, sizeof(uint32_t), &zero, 0, nullptr, nullptr));
            cap_tri_tris = total_cells*2;
        }
        if (cap_tri_patches < ns) {
            auto rel = [](void* m) { if (m) clReleaseMemObject((cl_mem)m); };
            rel(buf_sample_off); rel(buf_patch_uvn); rel(buf_patch_cell_off); rel(buf_patch_surf);
            buf_sample_off   = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (ns+1)*sizeof(uint32_t), nullptr, &err); CLC(err);
            buf_patch_uvn    = clCreateBuffer(ctx, CL_MEM_READ_ONLY, ns*sizeof(uint32_t), nullptr, &err); CLC(err);
            buf_patch_cell_off = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (ns+1)*sizeof(uint32_t), nullptr, &err); CLC(err);
            buf_patch_surf   = clCreateBuffer(ctx, CL_MEM_READ_ONLY, ns*sizeof(uint32_t), nullptr, &err); CLC(err);
            cap_tri_patches = ns;
            // patch_surf: which surface owns each patch — here it's identity (one
            // patch per surface). Written ONCE at allocation (CL_TRUE) because it
            // never changes per frame. Writing it every frame with CL_TRUE forces
            // a full queue sync on Intel Arc (~9-10ms each) — the entire FPS stall.
            {
                std::vector<uint32_t> h_patch_surf(ns);
                for (uint32_t s = 0; s < ns; ++s) h_patch_surf[s] = s;
                CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_patch_surf, CL_TRUE, 0, ns*sizeof(uint32_t), h_patch_surf.data(), 0, nullptr, nullptr));
            }
        }
        // Upload per-patch tables. All CL_FALSE — the kernels run AFTER these
        // writes on the same in-order queue, so they're guaranteed to see the
        // data without an explicit sync. CL_TRUE here stalls the queue (~9-10ms).
        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_sample_off,  CL_FALSE, 0, (ns+1)*sizeof(uint32_t), h_sample_off.data(),  0, nullptr, nullptr));
        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_patch_uvn,   CL_FALSE, 0, ns*sizeof(uint32_t),      h_patch_uvn.data(),  0, nullptr, nullptr));
        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_patch_cell_off, CL_FALSE, 0, (ns+1)*sizeof(uint32_t), h_patch_cell_off.data(), 0, nullptr, nullptr));
        // buf_patch_surf already written once at allocation — skip per-frame write.
        stage_print("grid-buffers-setup");

        // MVP for projection (row-major float16) — same convention as the
        // raymarch path (cam.BuildMVP) so psf_sample.cl projects identically.
        // TRACKING: show whether the model matrix is being applied.
        if (tri_debug) {
            const bool has_model = (model_in != nullptr);
            std::printf("  [tri-track-mvp] model_in=%s (model=%s)\n",
                        has_model ? "SET" : "NULL",
                        has_model ? "will compose model×view×proj" : "camera-only MVP (NO transform)");
        }
        float mvp[16];
        cam.BuildMVP(mvp, float(W) / float(H));
        // Compose model matrix into the MVP so parametric surfaces render at
        // their simulated positions instead of stacked at the origin.
        // model is local->world; mvp is view×proj. We want view×proj×model,
        // i.e. transform point by model first, then project.
        if (model_in) {
            Mat4 model_m; std::memcpy(model_m.m, model, sizeof(model));
            Mat4 viewproj_m; std::memcpy(viewproj_m.m, mvp, sizeof(mvp));
            Mat4 composed = Mat4_Multiply(viewproj_m, model_m);
            std::memcpy(mvp, composed.m, sizeof(mvp));
            if (tri_debug) std::printf("  [tri-track-mvp] composed MVP = viewproj × model\n");
        }
        const cl_float16 m16 = {{mvp[0],mvp[1],mvp[2],mvp[3],mvp[4],mvp[5],mvp[6],mvp[7],
                                 mvp[8],mvp[9],mvp[10],mvp[11],mvp[12],mvp[13],mvp[14],mvp[15]}};
        const cl_float3 bbmin_f3 = {{bmn.x, bmn.y, bmn.z}};
        const cl_float3 bbspan_f3 = {{bsp.x, bsp.y, bsp.z}};
        const float sscale = body.StitchScale();
        const cl_float3 light_dir = {{0.4f, 0.7f, 0.5f}}; // normalized-ish world-space light
        const cl_float3 eye = {{cam.position.x, cam.position.y, cam.position.z}}; // world-space camera position
        const float INF = 1e30f;
        stage_print("mvp-setup");

        // Clear grid buffers to SENTINEL CLIPPED-VERTEX values before psf_sample
        // runs. The buffer capacity (cap_tri_samples) may exceed this frame's
        // total_samples (lazy growth), leaving slots [total_samples, cap) with
        // stale VRAM garbage. Even though psf_sample writes [0, total_samples),
        // any read of an unwritten slot would feed garbage rgba_base=0 -> black
        // triangles (the vertical black band, confirmed RED by the diagnostic).
        // Sentinel z=1e30 makes tri_build skip the cell (any vertex z>=1e29f is
        // clipped), so stale slots never become triangles. The packed color in
        // the w component doesn't matter since the sentinel z clips the triangle.
        {
            const cl_float4 sentinel_v = {{0.f, 0.f, INF, 0.f}}; // z=INF clips
            CLC(clEnqueueFillBuffer(q, (cl_mem)buf_verts_grid, &sentinel_v, sizeof(sentinel_v), 0,
                                    cap_tri_samples * sizeof(cl_float4), 0, nullptr, nullptr));
            const cl_float3 sentinel_n = {{0.f, 1.f, 0.f}}; // harmless default normal
            CLC(clEnqueueFillBuffer(q, (cl_mem)buf_norms_grid, &sentinel_n, sizeof(sentinel_n), 0,
                                    cap_tri_samples * sizeof(cl_float3), 0, nullptr, nullptr));
            const cl_float3 sentinel_p = {{0.f, 0.f, 0.f}}; // harmless default world pos
            CLC(clEnqueueFillBuffer(q, (cl_mem)buf_pos_grid, &sentinel_p, sizeof(sentinel_p), 0,
                                    cap_tri_samples * sizeof(cl_float3), 0, nullptr, nullptr));
            const cl_uint zero_idx = 0u;
            CLC(clEnqueueFillBuffer(q, (cl_mem)buf_indices, &zero_idx, sizeof(zero_idx), 0,
                                    cap_tri_tris * 3 * sizeof(uint32_t), 0, nullptr, nullptr));
        }
        stage_print("grid-sentinel-fill");

        // --- GPU READBACK before kernel 1 dispatch (DEBUG ONLY) ---
        // Print what kind/params are actually in VRAM for buf_surfaces.
        // Runs REGARDLESS of whether the upload block executed above.
        // This isolates upload-corruption vs kernel-read-offset bugs.
        if (tri_debug) {
            if (buf_surfaces && ns > 0) {
                std::vector<uint8_t> rb(S.size() * 64);
                CLC(clEnqueueReadBuffer(q, (cl_mem)buf_surfaces, CL_TRUE, 0,
                                        S.size() * 64, rb.data(), 0, nullptr, nullptr));
                std::printf("  [tri-kernel-rb] buf_surfaces readback (ns=%u):\n", ns);
                for (uint32_t s = 0; s < ns; ++s) {
                    const uint8_t* rec = rb.data() + s * 64;
                    uint32_t kind = 0; std::memcpy(&kind, rec + 8, 4);
                    float p0, p1, p2;
                    std::memcpy(&p0, rec + 12, 4);
                    std::memcpy(&p1, rec + 16, 4);
                    std::memcpy(&p2, rec + 20, 4);
                    std::printf("    rb_surf[%u]: kind=%u anchor=[%u,%u,%u,%u] params=(%g,%g,%g)\n",
                                s, kind,
                                *(const uint16_t*)(rec+0), *(const uint16_t*)(rec+2),
                                *(const uint16_t*)(rec+4), *(const uint16_t*)(rec+6),
                                (double)p0, (double)p1, (double)p2);
                }
            } else {
                std::printf("  [tri-kernel-rb] buf_surfaces NULL or ns=0 — no surface buffer to read!\n");
            }
        }

        // --- KERNEL 1: psf_sample ---
        {
            using Clock = std::chrono::steady_clock;
            auto t0 = Clock::now();
            cl_kernel k1 = (cl_kernel)k_psf_sample;
            cl_uint a = 0;
            CLC(clSetKernelArg(k1, a++, sizeof(cl_mem), &buf_anchors));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_mem), &buf_surfaces));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_mem), &buf_stitch));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_mem), &buf_patch_surf));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_uint), &ns));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_uint), &base_uvn));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_float3), &bbmin_f3));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_float3), &bbspan_f3));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_float), &sscale));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_float16), &m16));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_uint), &W));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_uint), &H));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_float), &depth_bias));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_mem), &buf_verts_grid));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_mem), &buf_norms_grid));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_mem), &buf_pos_grid));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_mem), &buf_sample_off));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_mem), &buf_patch_uvn));
            cl_uint lod = lod_mode ? 1u : 0u;
            CLC(clSetKernelArg(k1, a++, sizeof(cl_uint), &lod));
            CLC(clSetKernelArg(k1, a++, sizeof(cl_mem), &buf_branch_trace));
            size_t g1 = (size_t(total_samples) + 255) & ~size_t(255);
            CLC(clEnqueueNDRangeKernel(q, k1, 1, nullptr, &g1, nullptr, 0, nullptr, nullptr));
            last_tri_sample_ms = std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
            stage_print("kernel1-psf_sample-dispatch");
            // BRANCH TRACE DIAGNOSTIC: read back the kernel-recorded surface kind
            // for each sample. This tells us EXACTLY what value the kernel saw in
            // VRAM for each sample's patch — independent of host-side claims.
            // If all samples show kind=1 but output is flat EvalPlane, the kernel
            // switch is broken for SK_ELLIPSOID. If samples show kind=8 (LEGACY),
            // the upload wrote the wrong value despite host-side kind=1.
            if (tri_debug) {
                CLC(clFinish(q));  // sync only when reading back for diagnostics
                std::vector<uint32_t> h_branch(total_samples);
                CLC(clEnqueueReadBuffer(q, (cl_mem)buf_branch_trace, CL_TRUE, 0,
                                        total_samples * sizeof(uint32_t), h_branch.data(),
                                        0, nullptr, nullptr));
                auto kname = [](uint32_t k) -> const char* {
                    switch (k) {
                        case 0: return "PLANE"; case 1: return "ELLIPSOID"; case 2: return "TORUS";
                        case 3: return "BOX"; case 4: return "GAUSSIAN"; case 5: return "WAVEFIELD";
                        case 6: return "CYLINDER"; case 7: return "COMPOUND"; case 8: return "LEGACY";
                        default: return "UNKNOWN";
                    }
                };
                uint32_t packed0 = h_branch[0];
                uint32_t s0 = packed0 >> 16;
                uint32_t kind0 = packed0 & 0xFFFFu;
                uint32_t packedN = h_branch[total_samples-1];
                uint32_t sN = packedN >> 16;
                uint32_t kindN = packedN & 0xFFFFu;
                std::printf("  [branch-trace] kernel-recorded surface_index=%u kind=%u (%s) "
                            "for sample[0], surface_index=%u kind=%u (%s) for sample[%u]\n",
                            s0, kind0, kname(kind0), sN, kindN, kname(kindN), total_samples-1);
            // DIAGNOSTIC: read back grid, report screen-space bbox of valid samples.
            // If bbox spans ~[0,W]x[0,H] geometry is placed correctly (bug in raster/color).
            // If bbox is garbage/tiny/all-zero, psf_sample projection is broken.
            {
                std::vector<cl_float4> h_grid(total_samples);
                CLC(clEnqueueReadBuffer(q, (cl_mem)buf_verts_grid, CL_TRUE, 0,
                                        total_samples * sizeof(cl_float4), h_grid.data(),
                                        0, nullptr, nullptr));
                float minx = 1e30f, maxx = -1e30f, miny = 1e30f, maxy = -1e30f;
                int valid = 0, sentinel = 0;
                for (uint32_t i = 0; i < total_samples; ++i) {
                    float z = h_grid[i].s[2];
                    if (z >= 1e29f) { ++sentinel; continue; }
                    float sx = h_grid[i].s[0], sy = h_grid[i].s[1];
                    if (sx < minx) minx = sx;
                    if (sx > maxx) maxx = sx;
                    if (sy < miny) miny = sy;
                    if (sy > maxy) maxy = sy;
                    ++valid;
                }
                // Depth histogram from verts_grid (before raster): bucket screen-space
                // depth z into 20 bins. If all valid samples collapse to one bin, the
                // projection depth is broken (constant z) -> atom_min can't separate near/far.
                int z_hist[20] = {0};
                for (uint32_t i = 0; i < total_samples; ++i) {
                    float z = h_grid[i].s[2];
                    if (z >= 1e29f) continue;
                    float zf = (z < 0.0f) ? 0.0f : (z > 1.0f ? 1.0f : z);
                    int zb = (int)(zf * 19.999f);
                    if (zb < 0) zb = 0; if (zb > 19) zb = 19;
                    z_hist[zb]++;
                }
                std::printf("  [grid-diag] samples=%u valid=%u sentinel=%u "
                            "bbox_sx=[%.1f,%.1f] bbox_sy=[%.1f,%.1f] (W=%u H=%u)\n",
                            total_samples, valid, sentinel, minx, maxx, miny, maxy, W, H);
                std::printf("  [grid-diag] depth-z hist (verts_grid, before raster):");
                for (int b = 0; b < 20; ++b) if (z_hist[b])
                    std::printf(" z=[%.2f..%.2f)=%d", b*0.05f, (b+1)*0.05f, z_hist[b]);
                std::printf("\n");
                // --- BASE COLOR STATISTICS (unlit color written by psf_sample) ---
                // If base colors are dim here, the collapse is in psf_sample's
                // channel_color(), not the lighting. If they're bright but final
                // pixels are dim, the collapse is in tri_raster's lighting/packing.
                {
                    float cr_sum = 0, cg_sum = 0, cb_sum = 0;
                    float cr_min = 1e30f, cr_max = -1e30f;
                    float cg_min = 1e30f, cg_max = -1e30f;
                    float cb_min = 1e30f, cb_max = -1e30f;
                    int cr_zero = 0, cg_zero = 0, cb_zero = 0;
                    int all_black = 0; // all 3 channels == 0
                    // Per-patch clustering: count patches that contain any all-black sample.
                    std::vector<int> patch_has_black(ns, 0);
                    for (uint32_t i = 0; i < total_samples; ++i) {
                        float z = h_grid[i].s[2];
                        if (z >= 1e29f) continue; // sentinel
                        uint32_t rgba; std::memcpy(&rgba, &h_grid[i].s[3], 4);
                        float cr = (float)((rgba >> 24) & 0xFF);
                        float cg = (float)((rgba >> 16) & 0xFF);
                        float cb = (float)((rgba >> 8) & 0xFF);
                        cr_sum += cr; cg_sum += cg; cb_sum += cb;
                        if (cr < cr_min) cr_min = cr; if (cr > cr_max) cr_max = cr;
                        if (cg < cg_min) cg_min = cg; if (cg > cg_max) cg_max = cg;
                        if (cb < cb_min) cb_min = cb; if (cb > cb_max) cb_max = cb;
                        if (cr == 0) ++cr_zero;
                        if (cg == 0) ++cg_zero;
                        if (cb == 0) ++cb_zero;
                        if (cr == 0 && cg == 0 && cb == 0) {
                            ++all_black;
                            // Map sample i to its patch via sample_off (binary search).
                            // sample_off is ascending; find patch p with sample_off[p] <= i < sample_off[p+1].
                            uint32_t lo = 0, hi = ns;
                            while (hi - lo > 1u) {
                                uint32_t mid = (lo + hi) >> 1;
                                if (h_sample_off[mid] <= i) lo = mid; else hi = mid;
                            }
                            patch_has_black[lo] = 1;
                        }
                    }
                    int valid_c = valid;
                    if (valid_c == 0) valid_c = 1;
                    int patches_with_black = 0;
                    for (uint32_t p = 0; p < ns; ++p) if (patch_has_black[p]) ++patches_with_black;
                    std::printf("  [color-diag] BASE(unlit) R=[%.0f..%.0f] avg=%.1f(zeros=%d%%) "
                                "G=[%.0f..%.0f] avg=%.1f(zeros=%d%%) "
                                "B=[%.0f..%.0f] avg=%.1f(zeros=%d%%)\n",
                                cr_min, cr_max, cr_sum/valid_c, 100*cr_zero/valid_c,
                                cg_min, cg_max, cg_sum/valid_c, 100*cg_zero/valid_c,
                                cb_min, cb_max, cb_sum/valid_c, 100*cb_zero/valid_c);
                    std::printf("  [color-diag] all-black samples=%d (%.1f%%) "
                                "patches_with_black=%d/%u\n",
                                all_black, 100.0f*all_black/valid_c, patches_with_black, ns);
                }
                // --- NORMAL STATISTICS ---
                // Check normals are valid unit vectors. Degenerate normals ->
                // lighting dot products collapse -> dark render.
                {
                    std::vector<cl_float3> h_norms(total_samples);
                    CLC(clEnqueueReadBuffer(q, (cl_mem)buf_norms_grid, CL_TRUE, 0,
                                            total_samples * sizeof(cl_float3), h_norms.data(),
                                            0, nullptr, nullptr));
                    float nx_sum = 0, ny_sum = 0, nz_sum = 0;
                    float nlen_min = 1e30f, nlen_max = -1e30f;
                    int n_zero = 0, n_bad = 0;
                    for (uint32_t i = 0; i < total_samples; ++i) {
                        float z = h_grid[i].s[2];
                        if (z >= 1e29f) continue;
                        float nx = h_norms[i].s[0], ny = h_norms[i].s[1], nz = h_norms[i].s[2];
                        float len = std::sqrt(nx*nx + ny*ny + nz*nz);
                        if (len < 1e-6f) { ++n_zero; continue; }
                        if (len > 1e6f || len != len) { ++n_bad; continue; }
                        nx_sum += nx; ny_sum += ny; nz_sum += nz;
                        if (len < nlen_min) nlen_min = len;
                        if (len > nlen_max) nlen_max = len;
                    }
                    std::printf("  [norm-diag] valid normals: n_zero=%d n_bad=%d "
                                "len=[%.3f..%.3f] avg_dir=(%.2f,%.2f,%.2f)\n",
                                n_zero, n_bad, nlen_min, nlen_max,
                                nx_sum/valid, ny_sum/valid, nz_sum/valid);
                }
                // --- WORLD POSITION STATISTICS (pos_grid written by psf_sample) ---
                // tri_raster computes view = P - eye. If pos_grid is all zeros,
                // sentinel, or NaN, view direction is wrong -> fillL collapses ->
                // dark render even with bright base colors + good normals.
                {
                    std::vector<cl_float3> h_pos(total_samples);
                    CLC(clEnqueueReadBuffer(q, (cl_mem)buf_pos_grid, CL_TRUE, 0,
                                            total_samples * sizeof(cl_float3), h_pos.data(),
                                            0, nullptr, nullptr));
                    float px_sum = 0, py_sum = 0, pz_sum = 0;
                    float px_min = 1e30f, px_max = -1e30f;
                    float py_min = 1e30f, py_max = -1e30f;
                    float pz_min = 1e30f, pz_max = -1e30f;
                    int p_zero = 0, p_bad = 0;
                    // Full world-position histogram (bucket by 0.1 units in each axis).
                    int px_hist[30] = {0}, py_hist[30] = {0}, pz_hist[30] = {0}; // range -1.5..1.5
                    auto bucket = [](float v) { int b = (int)std::floor(v*10.0f + 15.0f); return (b>=0 && b<30) ? b : -1; };
                    // Per-patch z-range: does each patch actually span z in [-0.35,0.35]?
                    std::vector<float> patch_pz_min(ns, 1e30f), patch_pz_max(ns, -1e30f);
                    std::vector<int> patch_count(ns, 0);
                    for (uint32_t i = 0; i < total_samples; ++i) {
                        float z = h_grid[i].s[2];
                        if (z >= 1e29f) continue; // sentinel
                        float px = h_pos[i].s[0], py = h_pos[i].s[1], pz = h_pos[i].s[2];
                        if (px == 0 && py == 0 && pz == 0) { ++p_zero; continue; }
                        if (px != px || py != py || pz != pz) { ++p_bad; continue; } // NaN
                        px_sum += px; py_sum += py; pz_sum += pz;
                        if (px < px_min) px_min = px; if (px > px_max) px_max = px;
                        if (py < py_min) py_min = py; if (py > py_max) py_max = py;
                        if (pz < pz_min) pz_min = pz; if (pz > pz_max) pz_max = pz;
                        int bx = bucket(px), by = bucket(py), bz = bucket(pz);
                        if (bx >= 0) px_hist[bx]++;
                        if (by >= 0) py_hist[by]++;
                        if (bz >= 0) pz_hist[bz]++;
                        // Map sample i to its patch via sample_off (binary search).
                        uint32_t lo = 0, hi = ns;
                        while (hi - lo > 1u) {
                            uint32_t mid = (lo + hi) >> 1;
                            if (h_sample_off[mid] <= i) lo = mid; else hi = mid;
                        }
                        patch_pz_min[lo] = std::min(patch_pz_min[lo], pz);
                        patch_pz_max[lo] = std::max(patch_pz_max[lo], pz);
                        patch_count[lo]++;
                    }
                    std::printf("  [pos-diag] valid world pos: p_zero=%d p_bad=%d "
                                "xyz=[%.3f..%.3f, %.3f..%.3f, %.3f..%.3f] avg=(%.3f,%.3f,%.3f) valid=%d\n",
                                p_zero, p_bad, px_min, px_max, py_min, py_max, pz_min, pz_max,
                                px_sum/valid, py_sum/valid, pz_sum/valid, (int)valid);
                    // World-position histograms (bucket width 0.1, range -1.5..1.5).
                    // If all samples cluster in one bucket per axis, geometry is collapsed.
                    std::printf("  [pos-diag] px_hist:");
                    for (int b = 0; b < 30; ++b) if (px_hist[b]) std::printf(" [%.1f..%.1f]=%d", (b-15)*0.1f, (b-14)*0.1f, px_hist[b]);
                    std::printf("\n  [pos-diag] py_hist:");
                    for (int b = 0; b < 30; ++b) if (py_hist[b]) std::printf(" [%.1f..%.1f]=%d", (b-15)*0.1f, (b-14)*0.1f, py_hist[b]);
                    std::printf("\n  [pos-diag] pz_hist:");
                    for (int b = 0; b < 30; ++b) if (pz_hist[b]) std::printf(" [%.1f..%.1f]=%d", (b-15)*0.1f, (b-14)*0.1f, pz_hist[b]);
                    std::printf("\n");
                    // Per-patch z-span diagnostic: how many patches have z-range < 0.05?
                    // Those patches are flat in z -> they render as a collapsed sheet.
                    int flat_patches = 0, flat_samples = 0;
                    for (uint32_t p = 0; p < ns; ++p) {
                        float span = patch_pz_max[p] - patch_pz_min[p];
                        if (span < 0.05f) { flat_patches++; flat_samples += patch_count[p]; }
                    }
                    std::printf("  [pos-diag] per-patch z-span: flat(<0.05)=%u patches/%d samples, "
                                "total=%u patches\n", flat_patches, flat_samples, ns);
                    // Show z-span for the first 10 patches (debug detail).
                    std::printf("  [pos-diag] first 10 patch z-spans:\n");
                    for (uint32_t p = 0; p < std::min(ns, 10u); ++p) {
                        std::printf("    patch %3u: z=[%.4f..%.4f] span=%.4f count=%d\n",
                                    p, patch_pz_min[p], patch_pz_max[p],
                                    patch_pz_max[p]-patch_pz_min[p], patch_count[p]);
                    }
                    // Dump positions of samples spread across the VALID range (not just first 30),
                    // plus any sample with |z| > 0.3 (should be the torus tube top/bottom).
                    std::printf("  [pos-diag] spread positions (every ~300th valid sample):\n");
                    int shown = 0, stride = std::max(1, (int)valid / 30);
                    int seen = 0;
                    for (uint32_t i = 0; i < total_samples && shown < 30; ++i) {
                        float z = h_grid[i].s[2];
                        if (z >= 1e29f) continue; // sentinel/clipped
                        float px = h_pos[i].s[0], py = h_pos[i].s[1], pz = h_pos[i].s[2];
                        if (px == 0 && py == 0 && pz == 0) continue;
                        if (seen++ % stride != 0) continue;
                        std::printf("    [%3d] (%.4f, %.4f, %.4f)\n", i, px, py, pz);
                        ++shown;
                    }
                    // Dump any samples with significant |z| (torus tube should reach +/-0.35).
                    std::printf("  [pos-diag] samples with |z|>0.30 (torus tube top/bottom):\n");
                    shown = 0;
                    for (uint32_t i = 0; i < total_samples && shown < 20; ++i) {
                        float z = h_grid[i].s[2];
                        if (z >= 1e29f) continue;
                        float px = h_pos[i].s[0], py = h_pos[i].s[1], pz = h_pos[i].s[2];
                        if (px == 0 && py == 0 && pz == 0) continue;
                        if (std::fabs(pz) > 0.30f) {
                            std::printf("    [%3d] (%.4f, %.4f, %.4f)\n", i, px, py, pz);
                            ++shown;
                        }
                    }
                    if (shown == 0) std::printf("    (NONE — no sample reaches |z|>0.30, geometry is flat in z!)\n");
                }
            }
        }
        } // end KERNEL 1 block (psf_sample)
        // Actual triangle count written by tri_build (may be < total_cells*2
        // because tri_build skips cells with clipped vertices). Read back after
        // tri_build completes and used as tri_raster's global size so it never
        // reads unwritten buf_indices slots (VRAM garbage -> black triangles).
        cl_uint ntri = total_cells * 2;
        // Reset the triangle counter to 0 EVERY frame. buf_n_tri is only
        // zeroed when the buffer is reallocated (on growth), so on every
        // subsequent frame atomic_inc keeps incrementing the stale value from
        // the previous frame. That makes tri_raster run with a huge ntri,
        // reading unwritten buf_indices slots (VRAM garbage) -> out-of-range
        // indices -> garbage grid reads -> NaN positions -> all triangles
        // skipped -> hits=0. Zeroing here guarantees atomic_inc starts fresh.
        {
            const cl_uint zero = 0u;
            // NON-BLOCKING: tri_build runs on the same in-order queue AFTER this
            // write, so it is guaranteed to see the zero value without an explicit
            // sync. Using CL_TRUE here stalls the queue on the Intel Arc driver
            // even for a 4-byte write — the same ~27-31 ms/frame stall the
            // tri_build clFinish caused. CL_FALSE keeps the queue flowing.
            CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_n_tri, CL_FALSE, 0,
                                     sizeof(uint32_t), &zero, 0, nullptr, nullptr));
        }
        // --- KERNEL 2: tri_build ---
        {
            using Clock = std::chrono::steady_clock;
            auto t0 = Clock::now();
            cl_kernel k2 = (cl_kernel)k_tri_build;
            cl_uint a = 0;
            CLC(clSetKernelArg(k2, a++, sizeof(cl_mem), &buf_verts_grid));
            CLC(clSetKernelArg(k2, a++, sizeof(cl_mem), &buf_norms_grid));
            CLC(clSetKernelArg(k2, a++, sizeof(cl_uint), &ns));
            CLC(clSetKernelArg(k2, a++, sizeof(cl_uint), &base_uvn));
            CLC(clSetKernelArg(k2, a++, sizeof(cl_mem), &buf_indices));
            CLC(clSetKernelArg(k2, a++, sizeof(cl_mem), &buf_n_tri));
            CLC(clSetKernelArg(k2, a++, sizeof(cl_mem), &buf_patch_uvn));
            CLC(clSetKernelArg(k2, a++, sizeof(cl_mem), &buf_patch_cell_off));
            CLC(clSetKernelArg(k2, a++, sizeof(cl_mem), &buf_sample_off));
            cl_uint lod = lod_mode ? 1u : 0u;
            CLC(clSetKernelArg(k2, a++, sizeof(cl_uint), &lod));
            size_t g2 = (size_t(total_cells*2) + 255) & ~size_t(255);
            CLC(clEnqueueNDRangeKernel(q, k2, 1, nullptr, &g2, nullptr, 0, nullptr, nullptr));
            // NO clFinish here — tri_raster reads the actual triangle count from
            // device memory (buf_n_tri[0]) instead of a host round-trip. On the
            // same in-order queue, tri_raster is guaranteed to see tri_build's
            // writes to buf_n_tri, so no explicit sync is needed for correctness.
            // Removing this finish + the blocking readback below eliminates the
            // ~27-31 ms/frame stall that capped tri-infill FPS at ~31.
            last_tri_build_ms = std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
            stage_print("kernel2-tri_build-dispatch");
            // DIAGNOSTIC (DELIRIUM_TRI_DEBUG=1 only): read back the sample grid
            // indices written by tri_build. Verify they reference valid grid
            // slots (< total_samples) and that consecutive triangles share
            // vertices (structured connectivity). Garbage indices (> total_samples)
            // or random connectivity = tri_build bug.
            // GATED: this readback moves up to ~11 MB GPU->CPU every frame and
            // is the dominant cost (~32 ms/frame). Must stay off in production.
            if (tri_debug) {
                std::vector<cl_uint> h_idx(ntri * 3);
                CLC(clEnqueueReadBuffer(q, (cl_mem)buf_indices, CL_TRUE, 0,
                                        ntri * 3 * sizeof(cl_uint), h_idx.data(),
                                        0, nullptr, nullptr));
                int out_of_range = 0, max_idx = 0;
                for (uint32_t i = 0; i < ntri * 3; ++i) {
                    if ((int)h_idx[i] >= (int)total_samples) ++out_of_range;
                    if ((int)h_idx[i] > max_idx) max_idx = (int)h_idx[i];
                }
                std::printf("  [tri-diag] ntri=%u indices=%u out_of_range=%d max_idx=%d total_samples=%u\n",
                            ntri, ntri * 3, out_of_range, max_idx, total_samples);
                // Print first 12 indices to inspect connectivity pattern
                std::printf("  [tri-diag] first idx: ");
                for (int i = 0; i < 12 && i < (int)ntri * 3; ++i) std::printf("%u ", h_idx[i]);
                std::printf("\n");
            }
        }
        // --- KERNEL 3: tri_raster ---
        {
            using Clock = std::chrono::steady_clock;
            auto t0 = Clock::now();
            cl_kernel k3 = (cl_kernel)k_tri_raster;
            cl_uint a = 0;
            CLC(clSetKernelArg(k3, a++, sizeof(cl_mem), &buf_indices));
            CLC(clSetKernelArg(k3, a++, sizeof(cl_mem), &buf_n_tri));  // device-side triangle count (kernel reads buf_n_tri[0])
            CLC(clSetKernelArg(k3, a++, sizeof(cl_uint), &W));
            CLC(clSetKernelArg(k3, a++, sizeof(cl_uint), &H));
            CLC(clSetKernelArg(k3, a++, sizeof(cl_float), &INF));
            CLC(clSetKernelArg(k3, a++, sizeof(cl_float), &depth_bias));
            CLC(clSetKernelArg(k3, a++, sizeof(cl_mem), &buf_verts_grid));
            CLC(clSetKernelArg(k3, a++, sizeof(cl_mem), &buf_norms_grid));
            CLC(clSetKernelArg(k3, a++, sizeof(cl_mem), &buf_pos_grid));
            CLC(clSetKernelArg(k3, a++, sizeof(cl_float3), &light_dir));
            CLC(clSetKernelArg(k3, a++, sizeof(cl_float3), &eye));
            CLC(clSetKernelArg(k3, a++, sizeof(cl_mem), &buf_fb));
            CLC(clSetKernelArg(k3, a++, sizeof(cl_mem), &buf_depth32));
            size_t g3 = (size_t(ntri) + 255) & ~size_t(255);
            CLC(clEnqueueNDRangeKernel(q, k3, 1, nullptr, &g3, nullptr, 0, nullptr, nullptr));
            last_tri_raster_ms = std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
            stage_print("kernel3-tri_raster-dispatch");
            // DIAGNOSTIC: FULL-FRAME scan of buf_fb AFTER tri_raster (DEBUG ONLY).
            // Reports total hit pixels, total pure-black hit pixels, and the
            // bounding box of black-hit pixels. This locates WHERE pure-black
            // hits occur in the framebuffer directly (before resolve/infill).
            if (tri_debug) {
                CLC(clFinish(q));  // sync only when reading back for diagnostics
                std::vector<cl_ulong> h_fb((size_t)W * H);
                CLC(clEnqueueReadBuffer(q, (cl_mem)buf_fb, CL_TRUE, 0,
                                        (size_t)W * H * sizeof(cl_ulong), h_fb.data(),
                                        0, nullptr, nullptr));
                auto is_hit = [&](cl_ulong key) -> bool {
                    return key != 0xFFFFFFFFFFFFFFFFUL;
                };
                auto is_black_hit = [&](cl_ulong key) -> bool {
                    uint32_t rgba = (uint32_t)(key & 0xFFFFFFFFu);
                    uint8_t r = (rgba >> 24) & 0xFF, g = (rgba >> 16) & 0xFF, b = (rgba >> 8) & 0xFF;
                    return (r == 0 && g == 0 && b == 0);
                };
                int total_hits = 0, total_black = 0;
                // ALL-hit bbox (for CENTER-HOLE diagnostic): track extents of EVERY
                // hit pixel, not just black ones. Previously tracked only black-hit
                // bbox, which was empty (black_hits=0) -> CENTER-HOLE loop never ran.
                int all_bx0 = W, all_bx1 = -1, all_by0 = H, all_by1 = -1;
                int bx0 = W, bx1 = -1, by0 = H, by1 = -1;
                int min_depth = 0x7FFFFFFF, max_depth = 0;
                // Depth histogram inside the center hole: are center pixels being
                // written by near or far samples? (blob = center filled with near samples)
                int center_depth_hist[10] = {0}; // bucket depth into 10 bins [0,1)
                // Screen-space radial histogram of hits (distance from center).
                int rad_hist[20] = {0}; // 20 buckets from 0 to max(W,H)/2
                float max_rad = 0.5f * (W < H ? (float)W : (float)H);
                // Brightness histogram buckets for hit pixels: where does the
                // lighting collapse? If most hits land in the lowest bucket,
                // the lighting formula is producing ~0 despite bright base colors.
                int hist[11] = {0}; // 0, 1-25, 26-50, ..., 231-255, >255(clamp)
                float br_sum = 0, br_min = 1e30f, br_max = -1e30f;
                for (uint32_t y = 0; y < H; ++y) {
                    for (uint32_t x = 0; x < W; ++x) {
                        cl_ulong key = h_fb[(size_t)y * W + x];
                        if (!is_hit(key)) continue;
                        ++total_hits;
                        // ALL-hit bbox extents.
                        if ((int)x < all_bx0) all_bx0 = (int)x;
                        if ((int)x > all_bx1) all_bx1 = (int)x;
                        if ((int)y < all_by0) all_by0 = (int)y;
                        if ((int)y > all_by1) all_by1 = (int)y;
                        uint32_t depth = (uint32_t)(key >> 32);
                        if ((int)depth < min_depth) min_depth = (int)depth;
                        if ((int)depth > max_depth) max_depth = (int)depth;
                        uint32_t rgba = (uint32_t)(key & 0xFFFFFFFFu);
                        uint8_t r = (rgba >> 24) & 0xFF, g = (rgba >> 16) & 0xFF, b = (rgba >> 8) & 0xFF;
                        if (r == 0 && g == 0 && b == 0) {
                            ++total_black;
                            if ((int)x < bx0) bx0 = (int)x;
                            if ((int)x > bx1) bx1 = (int)x;
                            if ((int)y < by0) by0 = (int)y;
                            if ((int)y > by1) by1 = (int)y;
                        }
                        // Radial histogram bucket (distance from framebuffer center).
                        int dx = (int)x - (int)(W * 0.5f), dy = (int)y - (int)(H * 0.5f);
                        float rad = std::sqrt((float)(dx * dx + dy * dy));
                        int rb = (int)(rad / max_rad * 19.999f);
                        if (rb < 0) rb = 0; if (rb > 19) rb = 19;
                        rad_hist[rb]++;
                        // Brightness bucket
                        int br = (int)((r + g + b) / 3);
                        if (br <= 0) hist[0]++;
                        else if (br < 26) hist[1]++;
                        else if (br < 51) hist[2]++;
                        else if (br < 76) hist[3]++;
                        else if (br < 102) hist[4]++;
                        else if (br < 128) hist[5]++;
                        else if (br < 153) hist[6]++;
                        else if (br < 179) hist[7]++;
                        else if (br < 204) hist[8]++;
                        else if (br < 231) hist[9]++;
                        else hist[10]++;
                        br_sum += br;
                        if (br < br_min) br_min = (float)br;
                        if (br > br_max) br_max = (float)br;
                    }
                }
                // CENTER-HOLE diagnostic: count hits inside the torus hole region.
                // The torus hole projects to the CENTER of the hit bbox. Compute the
                // hole radius from the ALL-HIT bbox half-span AFTER the scan loop
                // completes. A correct RING has FEW hits in this disk; a filled BLOB
                // has MANY. Second pass over h_fb.
                int center_hits = 0, ring_hits = 0;
                if (total_hits > 0 && all_bx1 >= all_bx0 && all_by1 >= all_by0) {
                    float bbox_cx = 0.5f * (all_bx0 + all_bx1), bbox_cy = 0.5f * (all_by0 + all_by1);
                    float bbox_half_x = 0.5f * (all_bx1 - all_bx0), bbox_half_y = 0.5f * (all_by1 - all_by0);
                    float bbox_half = (bbox_half_x < bbox_half_y ? bbox_half_x : bbox_half_y);
                    int cx = (int)(bbox_cx + 0.5f), cy = (int)(bbox_cy + 0.5f);
                    int hole_r = (int)(bbox_half * 0.35f);
                    if (hole_r < 1) hole_r = 1;
                    float hole_r2 = (float)(hole_r * hole_r);
                    for (uint32_t y = 0; y < H; ++y) {
                        for (uint32_t x = 0; x < W; ++x) {
                            cl_ulong key = h_fb[(size_t)y * W + x];
                            if (!is_hit(key)) continue;
                            uint32_t depth = (uint32_t)(key >> 32);
                            int dx = (int)x - cx, dy = (int)y - cy;
                            float rad2 = (float)(dx * dx + dy * dy);
                            if (rad2 < hole_r2) {
                                ++center_hits;
                                // Bucket depth into 0..9 using SAFE integer arithmetic.
                                uint32_t db = (uint32_t)((uint64_t)depth * 10u / 0xFFFFFFFFu);
                                if (db > 9) db = 9;
                                center_depth_hist[db]++;
                            } else {
                                ++ring_hits;
                            }
                        }
                    }
                }
                std::printf("  [fb-diag] FULL-FRAME hits=%d/%d black_hits=%d (%.2f%%)\n",
                            total_hits, (int)W * (int)H, total_black,
                            100.0f * (float)total_black / (float)(W * H));
                if (total_black > 0)
                    std::printf("  [fb-diag] BLACK bbox=[x%d..%d y%d..%d] depth=[%d..%d]\n",
                                bx0, bx1, by0, by1, min_depth, max_depth);
                else
                    std::printf("  [fb-diag] NO pure-black hits in framebuffer -> raster clean.\n");
                // Brightness histogram of hit pixels.
                if (total_hits > 0) {
                    std::printf("  [fb-diag] BRIGHTNESS hist: ");
                    const char* labels[] = {"0", "1-25", "26-50", "51-75", "76-101",
                                            "102-127", "128-152", "153-178", "179-203",
                                            "204-230", "231+"};
                    for (int i = 0; i < 11; ++i)
                        std::printf("[%s=%d %5.1f%%] ", labels[i], hist[i],
                                    100.0f * (float)hist[i] / (float)total_hits);
                    std::printf("\n");
                    std::printf("  [fb-diag] BRIGHTNESS avg=%.1f min=%.0f max=%.0f "
                                "(ambient floor=38)\n",
                                br_sum / total_hits, br_min, br_max);
                }
                // CENTER-HOLE diagnostic: direct measurement of the blob symptom.
                // A correct torus RING has FEW hits inside the hole disk; a filled
                // BLOB has MANY hits there. hole_r = bbox_half_span*0.35 centered on
                // projected bbox center (targets the torus inner hole region).
                std::printf("  [fb-diag] CENTER-HOLE: center_hits=%d ring_hits=%d "
                            "(center fraction=%.1f%%; blob if >15%%)\n",
                            center_hits, ring_hits,
                            total_hits ? 100.0f * (float)center_hits / (float)total_hits : 0.0f);
                std::printf("  [fb-diag] center-depth hist:");
                for (int i = 0; i < 10; ++i) if (center_depth_hist[i])
                    std::printf(" [%.1f..%.1f)=%d", i*0.1f, (i+1)*0.1f, center_depth_hist[i]);
                std::printf("\n");
                // Screen-space radial histogram of hits.
                std::printf("  [fb-diag] radial hist:");
                for (int i = 0; i < 20; ++i) if (rad_hist[i])
                    std::printf(" r=%.1f..%.1f=%d", i*0.05f, (i+1)*0.05f, rad_hist[i]);
                std::printf("\n");
            }
        }
        last_tri_total_ms = last_tri_sample_ms + last_tri_build_ms + last_tri_raster_ms + last_tri_upload_ms;
        // Record timing diagnostics for the Profiler panel (last frame).
        last_tri_samples = total_samples;
        last_tri_tris = ntri;  // actual triangles written by tri_build (may be < total_cells*2)
        last_tri_uvn = base_uvn;
        last_tri_lod_mode = lod_mode;
        static int tilog = 0;
        if ((tilog++ % 60) == 0)
            std::printf("tri-infill: %ux%u patches=%u samples=%u tris=%u lod=%s uvn=%d "
                        "sample=%.3f build=%.3f raster=%.3f upload=%.3f total=%.3f ms\n",
                        W, H, ns, total_samples, total_cells*2,
                        lod_mode ? "adaptive" : "uniform", base_uvn,
                        last_tri_sample_ms, last_tri_build_ms, last_tri_raster_ms,
                        last_tri_upload_ms, last_tri_total_ms);
        last_spp = 1.f;
    }
    } else if (path_ == Path::SplatClassic && k_splat) {
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
    stage_print("kernel-resolve-dispatch");

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

    // DIAGNOSTIC: time the GPU compute wait separately from enqueue.
    // The per-kernel timings (sample/build/raster) measure enqueue return time
    // (async queueing), NOT GPU execution. The actual GPU compute happens
    // asynchronously and is waited on by this clFinish(q). If render bucket >>
    // enqueue total, the gap is the GPU compute time. Gate on DELIRIUM_FINISH_DIAG
    // so it adds zero overhead in production.
    stage_print("all-dispatches-complete");
    if (std::getenv("DELIRIUM_FINISH_DIAG")) {
        using Clock = std::chrono::steady_clock;
        auto ft0 = Clock::now();
        CLC(clFinish(q));
        double finish_ms = std::chrono::duration<double, std::milli>(Clock::now() - ft0).count();
        static int flog = 0;
        if ((flog++ % 30) == 0)
            std::printf("finish-diag: clFinish wait=%.2f ms (enqueue total=%.3f ms, render bucket=%.2f ms)\n",
                        finish_ms, last_tri_total_ms, finish_ms + last_tri_total_ms);
    } else {
        CLC(clFinish(q));
    }
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

    // DIAGNOSTIC: read back pixels from buf_img to verify resolve wrote the
    // background color (18,18,24) or object color. Pure black would mean
    // buf_img was never written (resolve failed) or the SDL blit is broken.
    if (width > 0 && height > 0) {
        const uint32_t npx32 = width * (uint32_t)height;
        const uint32_t center = npx32 / 2;
        uint8_t sample_px[3][4];
        // Read center pixel + first pixel + last pixel (RGBA = 4 bytes each).
        clEnqueueReadBuffer(q, (cl_mem)buf_img, CL_TRUE, center*4u, 4,
                            &sample_px[0][0], 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, (cl_mem)buf_img, CL_TRUE, 0, 4,
                            &sample_px[1][0], 0, nullptr, nullptr);
        clEnqueueReadBuffer(q, (cl_mem)buf_img, CL_TRUE, (npx32-1)*4u, 4,
                            &sample_px[2][0], 0, nullptr, nullptr);
        fprintf(stderr,
            "[present-diag] buf_img pixels: center=(%u,%u,%u,%u) px[0]=(%u,%u,%u,%u) px[last]=(%u,%u,%u,%u)\n",
            (unsigned)sample_px[0][0], (unsigned)sample_px[0][1], (unsigned)sample_px[0][2], (unsigned)sample_px[0][3],
            (unsigned)sample_px[1][0], (unsigned)sample_px[1][1], (unsigned)sample_px[1][2], (unsigned)sample_px[1][3],
            (unsigned)sample_px[2][0], (unsigned)sample_px[2][1], (unsigned)sample_px[2][2], (unsigned)sample_px[2][3]);
    }
}

void SplatRenderer::SetPresentRect(int x, int y, int w, int h) {
    present_x = x; present_y = y; present_w = w; present_h = h;
}

void SplatRenderer::Present() {
    if (!sdl_rnd || !sdl_tex) return;   // headless: nothing to blit
    void* px = nullptr; int pitch = 0;
    bool image_valid = false;
    using Clock = std::chrono::steady_clock;
    auto t_present = Clock::now();
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
        auto t_read = Clock::now();
        const cl_int read_err = clEnqueueReadBuffer((cl_command_queue)cl_queue,
            (cl_mem)buf_img, CL_TRUE, 0, size_t(width)*height*4,
            read_dst, 0, nullptr, nullptr);
        double read_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_read).count();
        if (read_err == CL_SUCCESS && !staging.empty()) {
            for (int y = 0; y < height; ++y)
                std::memcpy(dst + size_t(y)*pitch,
                            staging.data() + size_t(y)*width*4, size_t(width)*4);
        }
        image_valid = (read_err == CL_SUCCESS);
        SDL_UnlockTexture((SDL_Texture*)sdl_tex);
        double unlock_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_read).count() - read_ms;
        if (read_err != CL_SUCCESS)
            std::fprintf(stderr, "Present: OpenCL image readback failed (%d)\n", read_err);
        // DIAGNOSTIC: measure Present() readback + unlock cost. This is the
        // suspected bottleneck when GPU work << frame time. Log every 60 frames.
        {
            static int plog = 0;
            if ((plog++ % 60) == 0)
                std::printf("present-diag: read=%.2fms unlock=%.2fms total=%.2fms (GPU work=%.2fms)\n",
                            read_ms, unlock_ms,
                            std::chrono::duration<double, std::milli>(Clock::now() - t_present).count(),
                            last_tri_total_ms);
        }
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

bool SplatRenderer::ReflowBodyGPU(PSFBody& body, float alpha, float max_disp, int max_iters) {
    // GPU path disabled? Fall back to CPU.
    if (!k_reflow_accum || !k_reflow_disp || !k_reflow_update ||
        !k_reflow_refit || !buf_anchors || !buf_surfaces || !buf_stitch)
        return false;

    const uint32_t n_patches = (uint32_t)body.Patches().size();
    const uint32_t n_anchors = (uint32_t)body.AnchorBlob().size();
    if (n_patches == 0 || n_anchors == 0) return false;

    cl_command_queue q = (cl_command_queue)cl_queue;
    const cl_float3 bbmin = { body.BBoxMin().x, body.BBoxMin().y, body.BBoxMin().z };
    const cl_float3 bbspan = { body.BBoxSpan().x, body.BBoxSpan().y, body.BBoxSpan().z };
    const float s_scale = body.StitchScale();

    // Upload the body's CURRENT blobs into the shared device buffers first.
    // buf_anchors/buf_surfaces/buf_stitch are shared across bodies, so we must
    // write THIS body's data before running kernels against them.
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_anchors, CL_TRUE, 0,
                             size_t(n_anchors) * 64u, body.AnchorBlob().data(), 0, nullptr, nullptr));
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_surfaces, CL_TRUE, 0,
                             size_t(n_patches) * 64u, body.SurfaceBlob().data(), 0, nullptr, nullptr));
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_stitch, CL_TRUE, 0,
                             size_t(n_patches) * 8u, body.StitchBlob().data(), 0, nullptr, nullptr));

    // Lazily allocate temp buffers sized to the body.
    if (n_patches > cap_reflow_patches) {
        if (buf_reflow_target) clReleaseMemObject((cl_mem)buf_reflow_target);
        buf_reflow_target = nullptr;
        cl_int e = CL_SUCCESS;
        buf_reflow_target = clCreateBuffer((cl_context)cl_ctx, CL_MEM_READ_WRITE,
                                           size_t(n_patches) * 81u * sizeof(cl_float3), nullptr, &e);
        if (e != CL_SUCCESS) { std::fprintf(stderr, "ReflowBodyGPU: target buffer alloc failed (%d)\n", e); return false; }
        cap_reflow_patches = n_patches;
    }
    if (n_anchors > cap_reflow_anchors) {
        if (buf_reflow_disp) clReleaseMemObject((cl_mem)buf_reflow_disp);
        buf_reflow_disp = nullptr;
        cl_int e = CL_SUCCESS;
        buf_reflow_disp = clCreateBuffer((cl_context)cl_ctx, CL_MEM_READ_WRITE,
                                         size_t(n_anchors) * sizeof(cl_float3), nullptr, &e);
        if (e != CL_SUCCESS) { std::fprintf(stderr, "ReflowBodyGPU: disp buffer alloc failed (%d)\n", e); return false; }
        cap_reflow_anchors = n_anchors;
    }
    // error buffer (float per patch) and coef buffer (float[8] per patch) are
    // reused each iteration; allocate once sized to the body.
    if (n_patches > cap_reflow_patches || !buf_reflow_error) {
        if (buf_reflow_error) clReleaseMemObject((cl_mem)buf_reflow_error);
        buf_reflow_error = nullptr;
        cl_int e = CL_SUCCESS;
        buf_reflow_error = clCreateBuffer((cl_context)cl_ctx, CL_MEM_READ_WRITE,
                                          size_t(n_patches) * sizeof(float), nullptr, &e);
        if (e != CL_SUCCESS) { std::fprintf(stderr, "ReflowBodyGPU: error buffer alloc failed (%d)\n", e); return false; }
    }
    // per-anchor accumulation buffers (float sum_err + uint count); cleared each pass.
    if (n_anchors > cap_reflow_anchors || !buf_reflow_sum_err) {
        if (buf_reflow_sum_err) clReleaseMemObject((cl_mem)buf_reflow_sum_err);
        if (buf_reflow_count)   clReleaseMemObject((cl_mem)buf_reflow_count);
        buf_reflow_sum_err = buf_reflow_count = nullptr;
        cl_int e = CL_SUCCESS;
        buf_reflow_sum_err = clCreateBuffer((cl_context)cl_ctx, CL_MEM_READ_WRITE,
                                            size_t(n_anchors) * sizeof(uint32_t), nullptr, &e);
        buf_reflow_count   = clCreateBuffer((cl_context)cl_ctx, CL_MEM_READ_WRITE,
                                            size_t(n_anchors) * sizeof(uint32_t), nullptr, &e);
        if (e != CL_SUCCESS) { std::fprintf(stderr, "ReflowBodyGPU: accumulation buffer alloc failed (%d)\n", e); return false; }
    }
    if (n_patches > cap_reflow_patches || !buf_reflow_coef) {
        if (buf_reflow_coef) clReleaseMemObject((cl_mem)buf_reflow_coef);
        buf_reflow_coef = nullptr;
        cl_int e = CL_SUCCESS;
        buf_reflow_coef = clCreateBuffer((cl_context)cl_ctx, CL_MEM_READ_WRITE,
                                         size_t(n_patches) * 8u * sizeof(float), nullptr, &e);
        if (e != CL_SUCCESS) { std::fprintf(stderr, "ReflowBodyGPU: coef buffer alloc failed (%d)\n", e); return false; }
    }

    // Run max_iters passes of the shape-preserving pipeline. Each pass:
    //   snapshot_error_accumulate -> disp -> update -> refit_coefficients
    // The snapshot captures the surface BEFORE anchors move; refit reproduces
    // that snapshot over the NEW anchor frame, so iterations do not drift.
    // The fused kernel atomically accumulates error into per-anchor buffers,
    // eliminating the old O(n_patches*n_anchors) scan in compute_anchor_displacement.
    const size_t zero_u32 = 0u;
    for (int iter = 0; iter < max_iters; ++iter) {
        /* clear per-anchor accumulation buffers before each pass */
        CLC(clEnqueueFillBuffer(q, (cl_mem)buf_reflow_sum_err, &zero_u32, sizeof(float),
                                0, size_t(n_anchors) * sizeof(float), 0, nullptr, nullptr));
        CLC(clEnqueueFillBuffer(q, (cl_mem)buf_reflow_count, &zero_u32, sizeof(uint32_t),
                                0, size_t(n_anchors) * sizeof(uint32_t), 0, nullptr, nullptr));

        // snapshot_error_accumulate: anchors, surfaces, stitch, n_patches,
        //   bbmin, bbspan, s_scale, target_out, error_out, sum_err, count
        CLC(clSetKernelArg((cl_kernel)k_reflow_accum, 0, sizeof(cl_mem), &buf_anchors));
        CLC(clSetKernelArg((cl_kernel)k_reflow_accum, 1, sizeof(cl_mem), &buf_surfaces));
        CLC(clSetKernelArg((cl_kernel)k_reflow_accum, 2, sizeof(cl_mem), &buf_stitch));
        CLC(clSetKernelArg((cl_kernel)k_reflow_accum, 3, sizeof(uint32_t), &n_patches));
        CLC(clSetKernelArg((cl_kernel)k_reflow_accum, 4, sizeof(cl_float3), &bbmin));
        CLC(clSetKernelArg((cl_kernel)k_reflow_accum, 5, sizeof(cl_float3), &bbspan));
        CLC(clSetKernelArg((cl_kernel)k_reflow_accum, 6, sizeof(float), &s_scale));
        CLC(clSetKernelArg((cl_kernel)k_reflow_accum, 7, sizeof(cl_mem), &buf_reflow_target));
        CLC(clSetKernelArg((cl_kernel)k_reflow_accum, 8, sizeof(cl_mem), &buf_reflow_error));
        CLC(clSetKernelArg((cl_kernel)k_reflow_accum, 9, sizeof(cl_mem), &buf_reflow_sum_err));
        CLC(clSetKernelArg((cl_kernel)k_reflow_accum, 10, sizeof(cl_mem), &buf_reflow_count));
        size_t g0 = n_patches;
        CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_reflow_accum, 1, nullptr, &g0, nullptr, 0, nullptr, nullptr));

        // compute_anchor_displacement: anchors, sum_err, count, n_anchors,
        //   alpha, max_disp, disp_out
        CLC(clSetKernelArg((cl_kernel)k_reflow_disp, 0, sizeof(cl_mem), &buf_anchors));
        CLC(clSetKernelArg((cl_kernel)k_reflow_disp, 1, sizeof(cl_mem), &buf_reflow_sum_err));
        CLC(clSetKernelArg((cl_kernel)k_reflow_disp, 2, sizeof(cl_mem), &buf_reflow_count));
        CLC(clSetKernelArg((cl_kernel)k_reflow_disp, 3, sizeof(uint32_t), &n_anchors));
        CLC(clSetKernelArg((cl_kernel)k_reflow_disp, 4, sizeof(float), &alpha));
        CLC(clSetKernelArg((cl_kernel)k_reflow_disp, 5, sizeof(float), &max_disp));
        CLC(clSetKernelArg((cl_kernel)k_reflow_disp, 6, sizeof(cl_mem), &buf_reflow_disp));
        size_t g1 = n_anchors;
        CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_reflow_disp, 1, nullptr, &g1, nullptr, 0, nullptr, nullptr));

        // update_anchor_positions: anchors, disp, bbmin, bbspan, n_anchors
        CLC(clSetKernelArg((cl_kernel)k_reflow_update, 0, sizeof(cl_mem), &buf_anchors));
        CLC(clSetKernelArg((cl_kernel)k_reflow_update, 1, sizeof(cl_mem), &buf_reflow_disp));
        CLC(clSetKernelArg((cl_kernel)k_reflow_update, 2, sizeof(cl_float3), &bbmin));
        CLC(clSetKernelArg((cl_kernel)k_reflow_update, 3, sizeof(cl_float3), &bbspan));
        CLC(clSetKernelArg((cl_kernel)k_reflow_update, 4, sizeof(uint32_t), &n_anchors));
        CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_reflow_update, 1, nullptr, &g1, nullptr, 0, nullptr, nullptr));

        // refit_coefficients: anchors, surfaces, target, n_patches, bbmin, bbspan, coef_out
        CLC(clSetKernelArg((cl_kernel)k_reflow_refit, 0, sizeof(cl_mem), &buf_anchors));
        CLC(clSetKernelArg((cl_kernel)k_reflow_refit, 1, sizeof(cl_mem), &buf_surfaces));
        CLC(clSetKernelArg((cl_kernel)k_reflow_refit, 2, sizeof(cl_mem), &buf_reflow_target));
        CLC(clSetKernelArg((cl_kernel)k_reflow_refit, 3, sizeof(uint32_t), &n_patches));
        CLC(clSetKernelArg((cl_kernel)k_reflow_refit, 4, sizeof(cl_float3), &bbmin));
        CLC(clSetKernelArg((cl_kernel)k_reflow_refit, 5, sizeof(cl_float3), &bbspan));
        CLC(clSetKernelArg((cl_kernel)k_reflow_refit, 6, sizeof(cl_mem), &buf_reflow_coef));
        CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_reflow_refit, 1, nullptr, &g0, nullptr, 0, nullptr, nullptr));
    }

    // Sync and read back the updated anchors + coef to the body's CPU blobs.
    CLC(clFinish(q));

    // Read anchors back (whole blob changed).
    std::vector<AnchorRecord> anchors_out = body.AnchorBlob();
    CLC(clEnqueueReadBuffer(q, (cl_mem)buf_anchors, CL_TRUE, 0,
                            size_t(n_anchors) * 64u, anchors_out.data(), 0, nullptr, nullptr));

    // Read coef back and write into surface records.
    std::vector<SurfaceRecord> surfaces_out = body.Patches().size() > 0
        ? std::vector<SurfaceRecord>(body.SurfaceBlob()) : std::vector<SurfaceRecord>();
    std::vector<float> coef_in(n_patches * 8u);
    CLC(clEnqueueReadBuffer(q, (cl_mem)buf_reflow_coef, CL_TRUE, 0,
                            size_t(n_patches) * 8u * sizeof(float), coef_in.data(), 0, nullptr, nullptr));
    for (uint32_t p = 0; p < n_patches && p < surfaces_out.size(); ++p)
        for (int k = 0; k < 8; ++k)
            surfaces_out[p].coef[k] = coef_in[p * 8u + k];

    // Commit changes to the body and mark dirty so the next render re-uploads.
    body.SetAnchorBlob(std::move(anchors_out));
    body.SetSurfaceBlob(std::move(surfaces_out));
    body.MarkAnchorsDirty();
    body.RebuildViews();
    return true;
}

// =============================================================================
// GPU adaptive mesh refinement (live subdivision).
// Splits a batch of flagged patches entirely on the GPU via psf_refine.cl.
// The kernel evaluates the 5 hanging anchors on the parent equation, fits each
// child by LSQ over a 9x9 grid, and appends records via an atomic free-list.
// Returns the number of patches actually split (0 if the GPU path is disabled).
// The caller rebuilds the seam graph afterwards (seams.Build). ONLY legacy
// height-field patches (surface_def == SK_LEGACY) can be split on the GPU.
// =============================================================================
int SplatRenderer::SplitBodyGPU(PSFBody& body, const std::vector<uint32_t>& split_list) {
    // GPU path disabled? Return 0 so the caller falls back to CPU.
    if (!k_split) return 0;

    const uint32_t n_patches = (uint32_t)body.Patches().size();
    const uint32_t n_anchors = (uint32_t)body.AnchorBlob().size();
    if (n_patches == 0 || n_anchors == 0) return 0;
    if (split_list.empty()) return 0;

    // The kernel APPENDS new records, so buffers must have capacity for the
    // worst case: each split adds 5 anchors + 3 surfaces (parent slot reused).
    const uint32_t n_splits = (uint32_t)split_list.size();
    const uint32_t cap_anchors_needed = n_anchors + 5 * n_splits;
    const uint32_t cap_surfaces_needed = n_patches + 3 * n_splits;

    cl_command_queue q = (cl_command_queue)cl_queue;
    const cl_float3 bbmin = { body.BBoxMin().x, body.BBoxMin().y, body.BBoxMin().z };
    const cl_float3 bbspan = { body.BBoxSpan().x, body.BBoxSpan().y, body.BBoxSpan().z };
    const float s_scale = body.StitchScale();

    // Lazily allocate the AMR buffers sized to the required capacity.
    // They are separate from the render path's shared READ_ONLY buffers so
    // the split kernel can append records via the atomic free-list.
    if (cap_anchors_needed > cap_amr_anchors) {
        if (buf_amr_anchors) clReleaseMemObject((cl_mem)buf_amr_anchors);
        buf_amr_anchors = nullptr;
        cl_int e = CL_SUCCESS;
        buf_amr_anchors = clCreateBuffer((cl_context)cl_ctx, CL_MEM_READ_WRITE,
                                         size_t(cap_anchors_needed) * 64u, nullptr, &e);
        if (e != CL_SUCCESS) { std::fprintf(stderr, "SplitBodyGPU: amr anchor buffer alloc failed (%d)\n", e); return 0; }
        cap_amr_anchors = cap_anchors_needed;
    }
    if (cap_surfaces_needed > cap_amr_surfaces) {
        if (buf_amr_surfaces) clReleaseMemObject((cl_mem)buf_amr_surfaces);
        buf_amr_surfaces = nullptr;
        cl_int e = CL_SUCCESS;
        buf_amr_surfaces = clCreateBuffer((cl_context)cl_ctx, CL_MEM_READ_WRITE,
                                          size_t(cap_surfaces_needed) * 64u, nullptr, &e);
        if (e != CL_SUCCESS) { std::fprintf(stderr, "SplitBodyGPU: amr surface buffer alloc failed (%d)\n", e); return 0; }
        cap_amr_surfaces = cap_surfaces_needed;
    }
    // stitch buffer sized to the post-split surface count.
    if (cap_surfaces_needed * 8u > 0) {
        if (buf_amr_stitch) clReleaseMemObject((cl_mem)buf_amr_stitch);
        buf_amr_stitch = nullptr;
        cl_int e = CL_SUCCESS;
        buf_amr_stitch = clCreateBuffer((cl_context)cl_ctx, CL_MEM_READ_WRITE,
                                        size_t(cap_surfaces_needed) * 8u, nullptr, &e);
        if (e != CL_SUCCESS) { std::fprintf(stderr, "SplitBodyGPU: amr stitch buffer alloc failed (%d)\n", e); return 0; }
    }
    // alloc_heads: two uint cursors [anchor_cursor, surface_cursor].
    if (!buf_amr_alloc_heads) {
        cl_int e = CL_SUCCESS;
        buf_amr_alloc_heads = clCreateBuffer((cl_context)cl_ctx, CL_MEM_READ_WRITE,
                                             2 * sizeof(cl_uint), nullptr, &e);
        if (e != CL_SUCCESS) { std::fprintf(stderr, "SplitBodyGPU: alloc_heads buffer alloc failed (%d)\n", e); return 0; }
    }
    // split_list input buffer.
    if (n_splits > 0) {
        if (buf_amr_split_list) clReleaseMemObject((cl_mem)buf_amr_split_list);
        buf_amr_split_list = nullptr;
        cl_int e = CL_SUCCESS;
        buf_amr_split_list = clCreateBuffer((cl_context)cl_ctx, CL_MEM_READ_ONLY,
                                            size_t(n_splits) * sizeof(cl_uint), nullptr, &e);
        if (e != CL_SUCCESS) { std::fprintf(stderr, "SplitBodyGPU: split_list buffer alloc failed (%d)\n", e); return 0; }
    }

    // Upload the body's CURRENT blobs into the AMR device buffers first.
    // buf_amr_anchors/buf_amr_surfaces/buf_amr_stitch are local to this path,
    // so we write THIS body's data before running the kernel against them.
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_amr_anchors, CL_TRUE, 0,
                             size_t(n_anchors) * 64u, body.AnchorBlob().data(), 0, nullptr, nullptr));
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_amr_surfaces, CL_TRUE, 0,
                             size_t(n_patches) * 64u, body.SurfaceBlob().data(), 0, nullptr, nullptr));
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_amr_stitch, CL_TRUE, 0,
                             size_t(n_patches) * 8u, body.StitchBlob().data(), 0, nullptr, nullptr));
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_amr_split_list, CL_TRUE, 0,
                             size_t(n_splits) * sizeof(cl_uint), split_list.data(), 0, nullptr, nullptr));

    // Initialize alloc_heads cursors to the current counts (the kernel only
    // ever APPENDS starting from these positions).
    cl_uint heads[2] = { n_anchors, n_patches };
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_amr_alloc_heads, CL_TRUE, 0,
                             2 * sizeof(cl_uint), heads, 0, nullptr, nullptr));

    // Set kernel args and launch.
    // split_patches: 0 surfaces, 1 anchors, 2 split_list, 3 alloc_heads,
    //   4 n_splits, 5 bbmin, 6 bbspan, 7 stitch, 8 s_scale
    CLC(clSetKernelArg((cl_kernel)k_split, 0, sizeof(cl_mem), &buf_amr_surfaces));
    CLC(clSetKernelArg((cl_kernel)k_split, 1, sizeof(cl_mem), &buf_amr_anchors));
    CLC(clSetKernelArg((cl_kernel)k_split, 2, sizeof(cl_mem), &buf_amr_split_list));
    CLC(clSetKernelArg((cl_kernel)k_split, 3, sizeof(cl_mem), &buf_amr_alloc_heads));
    CLC(clSetKernelArg((cl_kernel)k_split, 4, sizeof(uint32_t), &n_splits));
    CLC(clSetKernelArg((cl_kernel)k_split, 5, sizeof(cl_float3), &bbmin));
    CLC(clSetKernelArg((cl_kernel)k_split, 6, sizeof(cl_float3), &bbspan));
    CLC(clSetKernelArg((cl_kernel)k_split, 7, sizeof(cl_mem), &buf_amr_stitch));
    CLC(clSetKernelArg((cl_kernel)k_split, 8, sizeof(float), &s_scale));
    size_t g0 = n_splits;
    CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_split, 1, nullptr, &g0, nullptr, 0, nullptr, nullptr));

    // Sync and read back the alloc_heads cursors to determine the new counts.
    CLC(clFinish(q));
    cl_uint new_heads[2] = { 0, 0 };
    CLC(clEnqueueReadBuffer(q, (cl_mem)buf_amr_alloc_heads, CL_TRUE, 0,
                            2 * sizeof(cl_uint), new_heads, 0, nullptr, nullptr));

    const uint32_t new_anchor_count = new_heads[0];
    const uint32_t new_surface_count = new_heads[1];
    // Number of patches split = new_surface_count - original n_patches
    // (each split appends 3 surfaces + reuses the parent slot = net +3, but
    // the cursor tracks the total surface count, so delta = splits * 3...
    // actually the kernel appends exactly 3 per split, so splits = (new - old)/3).
    // However the simplest correct accounting: the kernel processed every
    // valid entry in split_list; the number actually split is the count of
    // successful kernel work-items. We read new_count if the kernel writes it,
    // else infer from surface delta.
    int n_split = 0;
    if (new_surface_count > n_patches) {
        // Each successful split adds exactly 3 appended surfaces (child 1..3).
        n_split = (int)((new_surface_count - n_patches) / 3u);
    }
    if (n_split <= 0 || n_split > (int)n_splits) return 0;

    // Read back the updated anchors + surfaces into resized CPU blobs.
    std::vector<AnchorRecord> anchors_out(new_anchor_count);
    CLC(clEnqueueReadBuffer(q, (cl_mem)buf_amr_anchors, CL_TRUE, 0,
                            size_t(new_anchor_count) * 64u, anchors_out.data(), 0, nullptr, nullptr));

    std::vector<SurfaceRecord> surfaces_out(new_surface_count);
    CLC(clEnqueueReadBuffer(q, (cl_mem)buf_amr_surfaces, CL_TRUE, 0,
                            size_t(new_surface_count) * 64u, surfaces_out.data(), 0, nullptr, nullptr));

    // Read back the updated stitch blob (8 int8 per surface). The split kernel
    // zeroes the stitch slots for the 3 appended children, so the host MUST
    // commit the resized stitch buffer; otherwise stitch_blob stays short while
    // surface_blob grows, and SeamGraph::Build writes out-of-bounds past the old
    // buffer (corrupting AnchorPool::dedupe heap structure).
    std::vector<int8_t> stitch_out(size_t(new_surface_count) * 8u);
    CLC(clEnqueueReadBuffer(q, (cl_mem)buf_amr_stitch, CL_TRUE, 0,
                            size_t(new_surface_count) * 8u, stitch_out.data(), 0, nullptr, nullptr));

    // Commit changes to the body and mark dirty so the next render re-uploads.
    body.SetAnchorBlob(std::move(anchors_out));
    body.SetSurfaceBlob(std::move(surfaces_out));
    body.SetStitchBlob(std::move(stitch_out));
    body.MarkAnchorsDirty();
    body.MarkSurfacesDirty();
    body.RebuildViews();
    // The GPU kernel only splits patches; it does NOT update the seam graph.
    // Rebuild the full seam graph from scratch (handles duplicate hanging
    // anchors created by adjacent splits the same way the CPU SplitPatch does).
    body.Seams().Build(body);
    return n_split;
}

// =============================================================================
// M2: Scene-level frustum culling.
// Tests a body's WORLD-space AABB against the view frustum planes derived
// from the camera's MVP matrix. Returns true if the body is at least
// partially visible (should be rendered). Static — no renderer state needed.
// =============================================================================
bool SplatRenderer::BodyVisibleFrustum(const PSFBody& body, const Camera& cam,
                                       const float model[16]) {
    // Build the MVP matrix and extract frustum planes.
    float mvp[16];
    cam.BuildMVP(mvp, 1.0f);  // aspect doesn't matter for plane extraction here

    // Compute the body's WORLD-space AABB.
    Vec3 bmn = body.BBoxMin(), bsp = body.BBoxSpan();
    if (bsp.x < 1e-8f && bsp.y < 1e-8f && bsp.z < 1e-8f) {
        bmn = {-1.f, -1.f, -1.f}; bsp = {2.f, 2.f, 2.f};
    }

    // Transform the 8 corners of the body bbox by the model matrix (if given).
    Vec3 wmn{1e30f, 1e30f, 1e30f}, wmx{-1e30f, -1e30f, -1e30f};
    for (int c = 0; c < 8; ++c) {
        Vec3 corner{ bmn.x + ((c&1)?bsp.x:0.f),
                     bmn.y + ((c&2)?bsp.y:0.f),
                     bmn.z + ((c&4)?bsp.z:0.f) };
        Vec3 w = corner;
        if (model) {
            w.x = model[0]*corner.x + model[1]*corner.y + model[2]*corner.z + model[3];
            w.y = model[4]*corner.x + model[5]*corner.y + model[6]*corner.z + model[7];
            w.z = model[8]*corner.x + model[9]*corner.y + model[10]*corner.z + model[11];
        }
        wmn.x = std::min(wmn.x, w.x); wmn.y = std::min(wmn.y, w.y); wmn.z = std::min(wmn.z, w.z);
        wmx.x = std::max(wmx.x, w.x); wmx.y = std::max(wmx.y, w.y); wmx.z = std::max(wmx.z, w.z);
    }

    // Extract frustum planes from MVP (same logic as EquationAccel::FrustumFromMVP).
    const float r0[4] = {mvp[0], mvp[1], mvp[2], mvp[3]};
    const float r1[4] = {mvp[4], mvp[5], mvp[6], mvp[7]};
    const float r2[4] = {mvp[8], mvp[9], mvp[10], mvp[11]};
    const float r3[4] = {mvp[12], mvp[13], mvp[14], mvp[15]};
    float planes[6][4];
    auto set_plane = [&](int i, float a, float b, float c, float d) {
        float L = std::sqrt(a * a + b * b + c * c);
        if (L < 1e-12f) L = 1.f;
        planes[i][0] = a / L; planes[i][1] = b / L; planes[i][2] = c / L; planes[i][3] = d / L;
    };
    set_plane(0, r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]); // left
    set_plane(1, r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]); // right
    set_plane(2, r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]); // bottom
    set_plane(3, r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]); // top
    set_plane(4, r3[0] + r2[0], r3[1] + r2[1], r3[2] + r2[2], r3[3] + r2[3]); // near
    set_plane(5, r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]); // far

    // Test AABB against each plane: if the most-positive corner is still
    // outside (< 0), the body is fully outside the frustum.
    for (int i = 0; i < 6; ++i) {
        float px = planes[i][0] >= 0.f ? wmx.x : wmn.x;
        float py = planes[i][1] >= 0.f ? wmx.y : wmn.y;
        float pz = planes[i][2] >= 0.f ? wmx.z : wmn.z;
        if (planes[i][0] * px + planes[i][1] * py + planes[i][2] * pz + planes[i][3] < 0.f)
            return false;
    }
    return true;
}

// =============================================================================
// M2: Coarse occlusion culling.
// Tests a body's world-space AABB against a coarse depth tile buffer
// (16x16 tiles). Returns true if the body is NOT fully occluded by nearer
// geometry already rendered this frame. NOTE: coarse approximation — the
// GPU march/raymarch handles per-pixel occlusion.
// =============================================================================
bool SplatRenderer::BodyVisibleOcclusion(const PSFBody& body, const Camera& cam,
                                         const float model[16]) {
    // Guard: if occlusion buffer not allocated or GPU disabled, skip culling.
    if (!buf_occlusion_depth || !cl_ctx || !cl_queue) return true;

    cl_command_queue q = (cl_command_queue)cl_queue;

    // Compute the body's WORLD-space AABB (same as BodyVisibleFrustum).
    Vec3 bmn = body.BBoxMin(), bsp = body.BBoxSpan();
    if (bsp.x < 1e-8f && bsp.y < 1e-8f && bsp.z < 1e-8f) {
        bmn = {-1.f, -1.f, -1.f}; bsp = {2.f, 2.f, 2.f};
    }
    Vec3 wmn{1e30f, 1e30f, 1e30f}, wmx{-1e30f, -1e30f, -1e30f};
    for (int c = 0; c < 8; ++c) {
        Vec3 corner{ bmn.x + ((c&1)?bsp.x:0.f),
                     bmn.y + ((c&2)?bsp.y:0.f),
                     bmn.z + ((c&4)?bsp.z:0.f) };
        Vec3 w = corner;
        if (model) {
            w.x = model[0]*corner.x + model[1]*corner.y + model[2]*corner.z + model[3];
            w.y = model[4]*corner.x + model[5]*corner.y + model[6]*corner.z + model[7];
            w.z = model[8]*corner.x + model[9]*corner.y + model[10]*corner.z + model[11];
        }
        wmn.x = std::min(wmn.x, w.x); wmn.y = std::min(wmn.y, w.y); wmn.z = std::min(wmn.z, w.z);
        wmx.x = std::max(wmx.x, w.x); wmx.y = std::max(wmx.y, w.y); wmx.z = std::max(wmx.z, w.z);
    }

    // Project the AABB center to screen space using the camera's MVP.
    float mvp[16];
    cam.BuildMVP(mvp, 1.0f);
    Vec3 ctr{ 0.5f*(wmn.x+wmx.x), 0.5f*(wmn.y+wmx.y), 0.5f*(wmn.z+wmx.z) };
    float cx = mvp[0]*ctr.x + mvp[1]*ctr.y + mvp[2]*ctr.z + mvp[3];
    float cy = mvp[4]*ctr.x + mvp[5]*ctr.y + mvp[6]*ctr.z + mvp[7];
    float cw = mvp[12]*ctr.x + mvp[13]*ctr.y + mvp[14]*ctr.z + mvp[15];
    if (cw < 1e-5f) return true;  // behind camera — frustum cull should have caught this
    float sx = (cx/cw*0.5f + 0.5f) * float(width);
    float sy = (0.5f - cy/cw*0.5f) * float(height);
    // Convert to tile coordinates.
    int tile_x = int(sx * float(occlusion_tiles_x) / float(width));
    int tile_y = int(sy * float(occlusion_tiles_y) / float(height));
    tile_x = std::max(0, std::min(int(occlusion_tiles_x) - 1, tile_x));
    tile_y = std::max(0, std::min(int(occlusion_tiles_y) - 1, tile_y));

    // Read back the depth buffer (small: 16x16 floats) and check the tile
    // the body center projects to. If the stored depth at that tile is nearer
    // than the body's center depth, the body is likely occluded.
    std::vector<float> depths(occlusion_tiles_x * occlusion_tiles_y);
    cl_int e = clEnqueueReadBuffer(q, (cl_mem)buf_occlusion_depth, CL_TRUE, 0,
                                   size_t(occlusion_tiles_x * occlusion_tiles_y) * sizeof(float),
                                   depths.data(), 0, nullptr, nullptr);
    if (e != CL_SUCCESS) return true;  // readback failed — skip culling

    float stored_depth = depths[tile_y * occlusion_tiles_x + tile_x];
    float body_depth = std::sqrt(ctr.x*ctr.x + ctr.y*ctr.y + ctr.z*ctr.z);

    // Conservative test: if the stored depth is significantly nearer than the
    // body center, the body is likely occluded. Use a margin to avoid culling
    // bodies that are partially in front of the stored geometry.
    // NOTE: this is a coarse test. The real occlusion happens per-pixel in the
    // GPU kernel. Returning true (visible) is always safe; returning false
    // (occluded) may miss some visible bodies but never culls a fully visible one
    // incorrectly if the margin is chosen conservatively.
    const float occlusion_margin = 0.15f;  // allow 15% depth tolerance
    if (stored_depth < body_depth * (1.0f - occlusion_margin))
        return false;  // body is behind stored geometry — occluded
    return true;  // visible
}

// =============================================================================
// M2: GPU modal writeback.
// Applies modal amplitudes K to the body's coef[8] coefficients directly on
// the GPU (no CPU round-trip). Returns true if the GPU path ran; false if
// disabled. After a successful call the body's coef are dirty and must be
// re-uploaded before the next render.
// =============================================================================
bool SplatRenderer::ApplyModesGPU(PSFBody& body, const std::vector<float>& K) {
    // Guard: kernel disabled or no buffers.
    if (!k_apply_modes || !buf_surfaces || !buf_modal_shapes || !buf_modal_K)
        return false;

    cl_command_queue q = (cl_command_queue)cl_queue;
    const uint32_t n_surfaces = (uint32_t)body.SurfaceBlob().size();
    const uint32_t n_modes = (uint32_t)K.size();
    if (n_surfaces == 0 || n_modes == 0) return false;

    // Upload the amplitude vector K.
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_modal_K, CL_TRUE, 0,
                             size_t(n_modes) * sizeof(float), K.data(), 0, nullptr, nullptr));

    // Run apply_modes kernel: coef[s] += sum_m K[m] * mode_shapes[m][s][8].
    CLC(clSetKernelArg((cl_kernel)k_apply_modes, 0, sizeof(cl_mem), &buf_surfaces));
    CLC(clSetKernelArg((cl_kernel)k_apply_modes, 1, sizeof(cl_mem), &buf_modal_shapes));
    CLC(clSetKernelArg((cl_kernel)k_apply_modes, 2, sizeof(cl_mem), &buf_modal_K));
    CLC(clSetKernelArg((cl_kernel)k_apply_modes, 3, sizeof(uint32_t), &n_surfaces));
    CLC(clSetKernelArg((cl_kernel)k_apply_modes, 4, sizeof(uint32_t), &n_modes));
    size_t g0 = n_surfaces;
    CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_apply_modes, 1, nullptr, &g0, nullptr, 0, nullptr, nullptr));
    CLC(clFinish(q));

    // Read back the updated surfaces and commit to the body.
    std::vector<SurfaceRecord> surfaces_out = body.SurfaceBlob();
    CLC(clEnqueueReadBuffer(q, (cl_mem)buf_surfaces, CL_TRUE, 0,
                            size_t(n_surfaces) * 64u, surfaces_out.data(), 0, nullptr, nullptr));
    body.SetSurfaceBlob(std::move(surfaces_out));
    body.MarkSurfacesDirty();  // coef changed — re-upload before next render
    return true;
}

// =============================================================================
// M2: Upload modal shapes to GPU device buffer.
// mode_shapes layout: [n_modes][n_surfaces][8] floats.
// =============================================================================
bool SplatRenderer::SetModalShapesGPU(const std::vector<float>& mode_shapes,
                                      uint32_t n_modes, uint32_t n_surfaces) {
    cl_context ctx = (cl_context)cl_ctx;
    cl_command_queue q = (cl_command_queue)cl_queue;
    if (!ctx || !q) return false;

    size_t total = size_t(n_modes) * n_surfaces * 8u;
    if (total == 0) return false;

    // Allocate/reallocate buf_modal_shapes if needed.
    if (n_surfaces > cap_modal_surfaces) {
        if (buf_modal_shapes) clReleaseMemObject((cl_mem)buf_modal_shapes);
        buf_modal_shapes = nullptr;
        cl_int e = CL_SUCCESS;
        buf_modal_shapes = clCreateBuffer(ctx, CL_MEM_READ_ONLY, total * sizeof(float), nullptr, &e);
        if (e != CL_SUCCESS) { std::fprintf(stderr, "SetModalShapesGPU: shape buffer alloc failed (%d)\n", e); return false; }
        cap_modal_surfaces = n_surfaces;
    }
    // Allocate/reallocate buf_modal_K if needed.
    if (n_modes > cap_modal_modes) {
        if (buf_modal_K) clReleaseMemObject((cl_mem)buf_modal_K);
        buf_modal_K = nullptr;
        cl_int e = CL_SUCCESS;
        buf_modal_K = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size_t(n_modes) * sizeof(float), nullptr, &e);
        if (e != CL_SUCCESS) { std::fprintf(stderr, "SetModalShapesGPU: K buffer alloc failed (%d)\n", e); return false; }
        cap_modal_modes = n_modes;
    }

    // Upload the mode shapes.
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_modal_shapes, CL_TRUE, 0,
                             total * sizeof(float), mode_shapes.data(), 0, nullptr, nullptr));
    return true;
}

// =============================================================================
// GPU-FIRST modal writeback setup: uploads REST state ONCE to device buffers.
// Called when a modal body is first bound. REST state comes from ModalDynamics
// snapshots (rest_apos/rest_anrm/rest_coef captured at LoadModes), NOT the
// body blob (which physics mutates every frame). The kernels write the
// deformed anchors/surfaces DIRECTLY into the SHARED render buffers
// (buf_anchors/buf_surfaces/buf_snorm) that bake_surfels reads — no host
// round-trip, no per-frame upload (~85KB/frame eliminated).
// Returns true if buffers allocated.
// =============================================================================
bool SplatRenderer::SetupGPUModal(PSFBody& body, ModalDynamics& dyn) {
    if (!gpu_modal_on || !k_modal_writeback_anchors || !k_modal_writeback_surfaces
        || !k_modal_snorm_pass1 || !k_modal_snorm_pass2) {
        if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] SetupGPUModal FAIL: kernels not ready\n");
        return false;
    }
    cl_context ctx = (cl_context)cl_ctx;
    cl_command_queue q = (cl_command_queue)cl_queue;
    if (!ctx || !q) { if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] SetupGPUModal FAIL: no ctx/queue\n"); return false; }
    // DIAGNOSTIC readbacks below are gated behind DELIRIUM_GPU_MODAL_DIAG.
    // Without the gate, each blocking readback forces a CPU-GPU sync (~9-10ms
    // on Intel Arc) — the entire FPS bottleneck even though kernels cost <0.1ms.
    const bool modal_diag = gpu_modal_diag;

    const uint32_t n_anchors = dyn.AnchorCount();
    const uint32_t n_surfaces = dyn.SurfCount();
    const int n_modes = dyn.ModeCount();
    if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] SetupGPUModal enter: anchors=%u surfaces=%u modes=%d\n",
            n_anchors, n_surfaces, n_modes);
    if (n_anchors == 0 || n_surfaces == 0 || n_modes == 0) {
        if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] SetupGPUModal FAIL: zero counts\n");
        return false;
    }
    // Guard against topology mismatch after splits/merges.
    if (n_anchors != (uint32_t)body.AnchorBlob().size()
        || n_surfaces != (uint32_t)body.SurfaceBlob().size()) {
        if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] SetupGPUModal FAIL: topology mismatch (%u/%u vs body %zu/%zu)\n",
                n_anchors, n_surfaces, body.AnchorBlob().size(), body.SurfaceBlob().size());
        return false;
    }

    // Full shapes table layout: [n_modes][n_anchors + n_surfaces*8] floats.
    const size_t stride = n_anchors + (size_t)n_surfaces * 8u;
    const size_t shapes_bytes = stride * n_modes * sizeof(float);
    const float* full_shapes = dyn.FullModeShapes();
    if (!full_shapes) return false;

    auto rel = [](void* m) { if (m) clReleaseMemObject((cl_mem)m); };

    // Cache bbox for quantization (REST positions are Vec3 floats in ModalDynamics).
    Vec3 bmn = body.BBoxMin(), bsp = body.BBoxSpan();
    if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] SetupGPUModal bbox: min=(%.3f,%.3f,%.3f) span=(%.3f,%.3f,%.3f)\n",
            bmn.x, bmn.y, bmn.z, bsp.x, bsp.y, bsp.z);
    if (bsp.x < 1e-8f && bsp.y < 1e-8f && bsp.z < 1e-8f) {
        bmn = Vec3{-1.f,-1.f,-1.f}; bsp = Vec3{2.f,2.f,2.f};
        if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] SetupGPUModal bbox DEGENERATE -> fallback min=(-1,-1,-1) span=(2,2,2)\n");
    }
    modal_bbmin = cl_float3{{bmn.x, bmn.y, bmn.z}};
    modal_bbspan = cl_float3{{bsp.x, bsp.y, bsp.z}};
    modal_n_anchors = n_anchors;
    modal_n_surfaces = n_surfaces;
    modal_n_modes = (uint32_t)n_modes;
    modal_body = &body;

    // Build quantized REST anchor records from ModalDynamics rest_apos/rest_anrm.
    // ModalDynamics stores rest positions as Vec3 floats; the kernel reads
    // quantized ushort[3] at offset 32 of each AnchorRecord. Re-quantize using
    // the body's bbox (matches the original quantization since bbox is stable).
    const auto& rest_pos = dyn.RestAnchorPos();
    const auto& rest_nrm = dyn.RestAnchorNrm();
    const auto& rest_coef = dyn.RestCoef();
    std::vector<AnchorRecord> rest_anchor_records(n_anchors);
    for (uint32_t a = 0; a < n_anchors; ++a) {
        AnchorRecord rec{};
        const Vec3& p = rest_pos[a];
        rec.pos[0] = (uint16_t)(std::clamp((p.x - bmn.x) / bsp.x, 0.0f, 1.0f) * 65535.0f + 0.5f);
        rec.pos[1] = (uint16_t)(std::clamp((p.y - bmn.y) / bsp.y, 0.0f, 1.0f) * 65535.0f + 0.5f);
        rec.pos[2] = (uint16_t)(std::clamp((p.z - bmn.z) / bsp.z, 0.0f, 1.0f) * 65535.0f + 0.5f);
        const Vec3& n = rest_nrm[a];
        rec.normal[0] = (uint8_t)(std::clamp(n.x * 0.5f + 0.5f, 0.0f, 1.0f) * 255.0f + 0.5f);
        rec.normal[1] = (uint8_t)(std::clamp(n.y * 0.5f + 0.5f, 0.0f, 1.0f) * 255.0f + 0.5f);
        rec.normal[2] = (uint8_t)(std::clamp(n.z * 0.5f + 0.5f, 0.0f, 1.0f) * 255.0f + 0.5f);
        rest_anchor_records[a] = rec;
    }
    if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] SetupGPUModal quantized record[0]: pos=(%u,%u,%u) normal=(%u,%u,%u)\n",
            (unsigned)rest_anchor_records[0].pos[0], (unsigned)rest_anchor_records[0].pos[1], (unsigned)rest_anchor_records[0].pos[2],
            (unsigned)rest_anchor_records[0].normal[0], (unsigned)rest_anchor_records[0].normal[1], (unsigned)rest_anchor_records[0].normal[2]);
    // Upload REST anchors to buf_modal_rest_anchors (read-only, uploaded once).
    if (cap_modal_anchors < n_anchors || !buf_modal_rest_anchors) {
        rel(buf_modal_rest_anchors); buf_modal_rest_anchors = nullptr;
        cl_int e = CL_SUCCESS;
        // NOTE: READ_WRITE (not READ_ONLY) — Intel Arc drivers sometimes fail
        // to persist host writes to READ_ONLY buffers. Verified by readback diag.
        buf_modal_rest_anchors = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_anchors*64u, nullptr, &e);
        if (e != CL_SUCCESS) return false;
        cap_modal_anchors = n_anchors;
    }
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_modal_rest_anchors, CL_TRUE, 0,
                             n_anchors*64u, rest_anchor_records.data(), 0, nullptr, nullptr));

    // DIAGNOSTIC: read back immediately after write to verify the buffer persisted.
    // Gated behind DELIRIUM_GPU_MODAL_DIAG — blocking readback stalls the queue.
    if (modal_diag) {
        std::vector<AnchorRecord> verify_sample(4);
        clEnqueueReadBuffer(q, (cl_mem)buf_modal_rest_anchors, CL_TRUE, 0, 4*64u,
                            verify_sample.data(), 0, nullptr, nullptr);
        fprintf(stderr,
            "[gpu-modal-diag] REST write verify: anchor[0] pos=(%u,%u,%u) normal=(%u,%u,%u) buf=%p\n",
            (unsigned)verify_sample[0].pos[0], (unsigned)verify_sample[0].pos[1], (unsigned)verify_sample[0].pos[2],
            (unsigned)verify_sample[0].normal[0], (unsigned)verify_sample[0].normal[1], (unsigned)verify_sample[0].normal[2],
            buf_modal_rest_anchors);
    }

    // Upload REST normals (uchar[3]) to buf_modal_rest_normals (read-only).
    if (cap_modal_anchors < n_anchors || !buf_modal_rest_normals) {
        rel(buf_modal_rest_normals); buf_modal_rest_normals = nullptr;
        cl_int e = CL_SUCCESS;
        buf_modal_rest_normals = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_anchors*3u, nullptr, &e);
        if (e != CL_SUCCESS) return false;
    }
    {
        std::vector<cl_uchar> normals_host(n_anchors*3);
        for (uint32_t a = 0; a < n_anchors; ++a) {
            normals_host[a*3+0] = rest_anchor_records[a].normal[0];
            normals_host[a*3+1] = rest_anchor_records[a].normal[1];
            normals_host[a*3+2] = rest_anchor_records[a].normal[2];
        }
        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_modal_rest_normals, CL_TRUE, 0,
                                 normals_host.size()*sizeof(cl_uchar), normals_host.data(), 0, nullptr, nullptr));
    }

    // Upload REST coef [n_surf][8] to buf_modal_rest_coef (read-only).
    if (cap_modal_surfaces < n_surfaces || !buf_modal_rest_coef) {
        rel(buf_modal_rest_coef); buf_modal_rest_coef = nullptr;
        cl_int e = CL_SUCCESS;
        buf_modal_rest_coef = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n_surfaces*8u*sizeof(float), nullptr, &e);
        if (e != CL_SUCCESS) return false;
        cap_modal_surfaces = n_surfaces;
    }
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_modal_rest_coef, CL_TRUE, 0,
                             rest_coef.size()*sizeof(float), rest_coef.data(), 0, nullptr, nullptr));

    // Allocate the SHARED render buffers buf_anchors/buf_surfaces/buf_snorm
    // (READ_WRITE) that the writeback kernels write into each frame and that
    // bake_surfels reads. MUST be READ_WRITE (the grew block above allocates
    // READ_ONLY for non-modal bodies, so force reallocation here). Initialize
    // them to REST state so the first frame renders correctly even before the
    // kernels run.
    {
        rel(buf_anchors); buf_anchors = nullptr;
        cl_int e = CL_SUCCESS;
        buf_anchors = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_anchors*64u, nullptr, &e);
        if (e != CL_SUCCESS) return false;
        cap_anchors = n_anchors;
        // Copy REST anchors INTO buf_anchors (working buffer). NOTE: src=REST,
        // dst=buf_anchors — the original code had these swapped, which copied
        // empty buf_anchors OVER buf_modal_rest_anchors, zeroing the REST data
        // and producing a black screen (all anchors collapsed to bbmin).
        CLC(clEnqueueCopyBuffer(q, (cl_mem)buf_modal_rest_anchors, (cl_mem)buf_anchors, 0, 0, n_anchors*64u, 0, nullptr, nullptr));
    }
    {
        rel(buf_surfaces); buf_surfaces = nullptr;
        cl_int e = CL_SUCCESS;
        buf_surfaces = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_surfaces*64u, nullptr, &e);
        if (e != CL_SUCCESS) return false;
        cap_surfaces = n_surfaces;
        std::vector<SurfaceRecord> rest_surf_records(n_surfaces);
        for (uint32_t s = 0; s < n_surfaces; ++s) {
            SurfaceRecord rec{};
            const auto& S = body.SurfaceBlob()[s];
            rec.anchor_idx[0] = S.anchor_idx[0]; rec.anchor_idx[1] = S.anchor_idx[1];
            rec.anchor_idx[2] = S.anchor_idx[2]; rec.anchor_idx[3] = S.anchor_idx[3];
            rec.surface_def = S.surface_def;
            for (int i = 0; i < 8; ++i) rec.coef[i] = rest_coef[s*8 + i];
            std::memcpy(rec.color_pattern, S.color_pattern, 18);
            rest_surf_records[s] = rec;
        }
        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_surfaces, CL_TRUE, 0,
                                 n_surfaces*64u, rest_surf_records.data(), 0, nullptr, nullptr));
    }

    if (cap_snorm < n_anchors || !buf_snorm) {
        rel(buf_snorm); buf_snorm = nullptr;
        cl_int e = CL_SUCCESS;
        buf_snorm = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_anchors*sizeof(cl_float4), nullptr, &e);
        if (e != CL_SUCCESS) return false;
        cap_snorm = n_anchors;
    }

    // Upload full shapes table [n_modes][stride] floats.
    if (cap_modal_shapes_bytes < shapes_bytes) {
        rel(buf_modal_shapes);
        buf_modal_shapes = nullptr;
        cl_int e = CL_SUCCESS;
        buf_modal_shapes = clCreateBuffer(ctx, CL_MEM_READ_ONLY, shapes_bytes, nullptr, &e);
        if (e != CL_SUCCESS) return false;
        cap_modal_shapes_bytes = (uint32_t)shapes_bytes;
    }
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_modal_shapes, CL_TRUE, 0,
                             shapes_bytes, full_shapes, 0, nullptr, nullptr));

    // Allocate buf_modal_K [n_modes].
    if (cap_modal_modes < (uint32_t)n_modes) {
        rel(buf_modal_K);
        buf_modal_K = nullptr;
        cl_int e = CL_SUCCESS;
        buf_modal_K = clCreateBuffer(ctx, CL_MEM_READ_WRITE, (size_t)n_modes*sizeof(float), nullptr, &e);
        if (e != CL_SUCCESS) return false;
        cap_modal_modes = (uint32_t)n_modes;
    }

    // Allocate buf_raw_n scratch for snorm pass1 accumulator [n_anchors] float3.
    if (!buf_raw_n) {
        cl_int e = CL_SUCCESS;
        buf_raw_n = clCreateBuffer(ctx, CL_MEM_READ_WRITE, n_anchors*3*sizeof(float), nullptr, &e);
        if (e != CL_SUCCESS) return false;
    }
    // DIAGNOSTIC: read REST buffer at END of SetupGPUModal to confirm it's
    // still correct before returning. If this shows zeros, something AFTER
    // the verify readback but BEFORE return zeroes it. Gated behind DELIRIUM_GPU_MODAL_DIAG.
    if (modal_diag) {
        std::vector<AnchorRecord> end_sample(4);
        clEnqueueReadBuffer(q, (cl_mem)buf_modal_rest_anchors, CL_TRUE, 0, 4*64,
                            end_sample.data(), 0, nullptr, nullptr);
        fprintf(stderr,
            "[gpu-modal-diag] SetupGPUModal END REST readback: anchor[0] pos=(%u,%u,%u) normal=(%u,%u,%u) buf=%p\n",
            (unsigned)end_sample[0].pos[0], (unsigned)end_sample[0].pos[1], (unsigned)end_sample[0].pos[2],
            (unsigned)end_sample[0].normal[0], (unsigned)end_sample[0].normal[1], (unsigned)end_sample[0].normal[2],
            buf_modal_rest_anchors);
    }
    return true;
}

// =============================================================================
// GPU-FIRST modal writeback per frame: uploads only K, runs kernels that write
// deformed anchors/surfaces DIRECTLY into the shared render buffers
// (buf_anchors/buf_surfaces/buf_snorm). bake_surfels reads those buffers as
// usual. No host round-trip, no per-frame upload (~85KB/frame eliminated).
// Returns true if GPU path ran; false if disabled.
// =============================================================================
bool SplatRenderer::GPUModalWriteback(PSFBody& body, ModalDynamics& dyn, const float* K,
                                      uint32_t n_modes) {
    if (!gpu_modal_on || !k_modal_writeback_anchors || !k_modal_writeback_surfaces
        || !k_modal_snorm_pass1 || !k_modal_snorm_pass2) {
        if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] GPUModalWriteback FAIL: kernels disabled\n");
        return false;
    }
    if (modal_body != &body || modal_n_modes != n_modes) {
        if (gpu_modal_diag) fprintf(stderr, "[gpu-modal-diag] GPUModalWriteback FAIL: body/mode mismatch (modal=%p/%u, body=%p/%u)\n",
                (void*)modal_body, modal_n_modes, (void*)&body, n_modes);
        return false;
    }

    cl_command_queue q = (cl_command_queue)cl_queue;
    const uint32_t n_anchors = modal_n_anchors;
    const uint32_t n_surfaces = modal_n_surfaces;
    const uint32_t stride = modal_n_anchors + n_surfaces * 8u;
    // DIAGNOSTIC readbacks below are gated behind DELIRIUM_GPU_MODAL_DIAG.
    // Without the gate, each blocking readback forces a CPU-GPU sync (~9-10ms
    // each on Intel Arc) — THREE per frame = ~27-31ms stall, which was the
    // entire FPS bottleneck even though the GPU kernels themselves cost <0.1ms.
    const bool modal_diag = gpu_modal_diag;
    // STAGE TIMING inside GPUModalWriteback: DELIRIUM_MODAL_STAGES=1 prints
    // per-call timings to pinpoint which dispatch/write/fill forces the sync.
    const bool modal_stages = std::getenv("DELIRIUM_MODAL_STAGES") != nullptr;
    using ModalClock = std::chrono::steady_clock;
    auto modal_t = ModalClock::now();
    auto modal_stage = [&](const char* label) {
        if (!modal_stages) return;
        double dt = std::chrono::duration<double, std::milli>(ModalClock::now() - modal_t).count();
        modal_t = ModalClock::now();
        std::fprintf(stderr, "[modal-stage] %-24s %.2f ms\n", label, dt);
    };

    // DIAGNOSTIC: read REST buffer at START of writeback (before any kernels)
    // to see if it was zeroed between SetupGPUModal and here.
    if (modal_diag) {
        std::vector<AnchorRecord> pre_sample(4);
        clEnqueueReadBuffer(q, (cl_mem)buf_modal_rest_anchors, CL_TRUE, 0, 4*64,
                            pre_sample.data(), 0, nullptr, nullptr);
        fprintf(stderr,
            "[gpu-modal-diag] GPUModalWriteback PRE-kernel REST readback: anchor[0] pos=(%u,%u,%u) normal=(%u,%u,%u) buf=%p\n",
            (unsigned)pre_sample[0].pos[0], (unsigned)pre_sample[0].pos[1], (unsigned)pre_sample[0].pos[2],
            (unsigned)pre_sample[0].normal[0], (unsigned)pre_sample[0].normal[1], (unsigned)pre_sample[0].normal[2],
            buf_modal_rest_anchors);
    }

    // Upload only K (tiny: n_modes floats).
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_modal_K, CL_FALSE, 0,
                             (size_t)n_modes * sizeof(float), K, 0, nullptr, nullptr));
    modal_stage("write-K");

    // Run anchor writeback kernel (absolute from rest) writing into buf_anchors.
    cl_float3 bbmin = modal_bbmin, bbspan = modal_bbspan;
    CLC(clSetKernelArg((cl_kernel)k_modal_writeback_anchors, 0, sizeof(cl_mem), &buf_anchors));
    CLC(clSetKernelArg((cl_kernel)k_modal_writeback_anchors, 1, sizeof(cl_mem), &buf_modal_rest_anchors));  // rest anchors (read-only)
    CLC(clSetKernelArg((cl_kernel)k_modal_writeback_anchors, 2, sizeof(cl_mem), &buf_modal_rest_normals)); // rest normals (read-only)
    CLC(clSetKernelArg((cl_kernel)k_modal_writeback_anchors, 3, sizeof(cl_mem), &buf_modal_shapes));
    CLC(clSetKernelArg((cl_kernel)k_modal_writeback_anchors, 4, sizeof(cl_mem), &buf_modal_K));
    CLC(clSetKernelArg((cl_kernel)k_modal_writeback_anchors, 5, sizeof(cl_float3), &bbmin));
    CLC(clSetKernelArg((cl_kernel)k_modal_writeback_anchors, 6, sizeof(cl_float3), &bbspan));
    CLC(clSetKernelArg((cl_kernel)k_modal_writeback_anchors, 7, sizeof(uint32_t), &n_anchors));
    CLC(clSetKernelArg((cl_kernel)k_modal_writeback_anchors, 8, sizeof(uint32_t), &stride));
    CLC(clSetKernelArg((cl_kernel)k_modal_writeback_anchors, 9, sizeof(uint32_t), &n_modes));
    size_t g0 = n_anchors;
    CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_modal_writeback_anchors, 1, nullptr, &g0, nullptr, 0, nullptr, nullptr));
    modal_stage("anchors-kernel");

    // Run surface writeback kernel (absolute from rest) writing into buf_surfaces.
    CLC(clSetKernelArg((cl_kernel)k_modal_writeback_surfaces, 0, sizeof(cl_mem), &buf_surfaces));
    CLC(clSetKernelArg((cl_kernel)k_modal_writeback_surfaces, 1, sizeof(cl_mem), &buf_modal_rest_coef));  // rest coef (read-only)
    CLC(clSetKernelArg((cl_kernel)k_modal_writeback_surfaces, 2, sizeof(cl_mem), &buf_modal_shapes));
    CLC(clSetKernelArg((cl_kernel)k_modal_writeback_surfaces, 3, sizeof(cl_mem), &buf_modal_K));
    CLC(clSetKernelArg((cl_kernel)k_modal_writeback_surfaces, 4, sizeof(uint32_t), &n_surfaces));
    CLC(clSetKernelArg((cl_kernel)k_modal_writeback_surfaces, 5, sizeof(uint32_t), &n_anchors));
    CLC(clSetKernelArg((cl_kernel)k_modal_writeback_surfaces, 6, sizeof(uint32_t), &n_modes));
    size_t g1 = n_surfaces;
    CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_modal_writeback_surfaces, 1, nullptr, &g1, nullptr, 0, nullptr, nullptr));
    modal_stage("surfaces-kernel");

    // Run snorm pass1: area-weighted corner normal accumulation into buf_raw_n.
    CLC(clSetKernelArg((cl_kernel)k_modal_snorm_pass1, 0, sizeof(cl_mem), &buf_anchors));
    CLC(clSetKernelArg((cl_kernel)k_modal_snorm_pass1, 1, sizeof(cl_mem), &buf_surfaces));
    CLC(clSetKernelArg((cl_kernel)k_modal_snorm_pass1, 2, sizeof(cl_mem), &buf_raw_n));
    CLC(clSetKernelArg((cl_kernel)k_modal_snorm_pass1, 3, sizeof(cl_float3), &bbmin));
    CLC(clSetKernelArg((cl_kernel)k_modal_snorm_pass1, 4, sizeof(cl_float3), &bbspan));
    CLC(clSetKernelArg((cl_kernel)k_modal_snorm_pass1, 5, sizeof(uint32_t), &n_surfaces));
    CLC(clSetKernelArg((cl_kernel)k_modal_snorm_pass1, 6, sizeof(uint32_t), &n_anchors));
    // Zero buf_raw_n before pass1 (accumulator). NOTE: pattern_size MUST be a
    // power of 2 — use a single float (4 bytes) as the pattern, filling the
    // whole buffer with zeros.
    const cl_float zero = 0.0f;
    CLC(clEnqueueFillBuffer(q, (cl_mem)buf_raw_n, &zero, sizeof(zero), 0,
                            n_anchors*3*sizeof(float), 0, nullptr, nullptr));
    modal_stage("snorm-fill");
    CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_modal_snorm_pass1, 1, nullptr, &g1, nullptr, 0, nullptr, nullptr));
    modal_stage("snorm-pass1");

    // Run snorm pass2: normalize + Laplacian blend, write to buf_snorm.
    CLC(clSetKernelArg((cl_kernel)k_modal_snorm_pass2, 0, sizeof(cl_mem), &buf_raw_n));
    CLC(clSetKernelArg((cl_kernel)k_modal_snorm_pass2, 1, sizeof(cl_mem), &buf_surfaces));
    CLC(clSetKernelArg((cl_kernel)k_modal_snorm_pass2, 2, sizeof(cl_mem), &buf_snorm));
    CLC(clSetKernelArg((cl_kernel)k_modal_snorm_pass2, 3, sizeof(uint32_t), &n_surfaces));
    CLC(clSetKernelArg((cl_kernel)k_modal_snorm_pass2, 4, sizeof(uint32_t), &n_anchors));
    size_t g2 = n_anchors;
    CLC(clEnqueueNDRangeKernel(q, (cl_kernel)k_modal_snorm_pass2, 1, nullptr, &g2, nullptr, 0, nullptr, nullptr));
    modal_stage("snorm-pass2");

    // DIAGNOSTIC READBACK: sample first few anchors/snorm after writeback to
    // verify the kernels produced valid geometry (not NaN/degenerate).
    if (modal_diag && modal_n_modes == 12 && n_anchors >= 4) {
        std::vector<AnchorRecord> sample_anchors(4);
        clEnqueueReadBuffer(q, (cl_mem)buf_anchors, CL_TRUE, 0, 4*64, sample_anchors.data(), 0, nullptr, nullptr);
        std::vector<cl_float4> sample_snorm(4);
        clEnqueueReadBuffer(q, (cl_mem)buf_snorm, CL_TRUE, 0, 4*sizeof(cl_float4), sample_snorm.data(), 0, nullptr, nullptr);
        fprintf(stderr,
            "[gpu-modal-diag] writeback readback: anchor[0] pos=(%u,%u,%u) normal=(%u,%u,%u) snorm=(%.3f,%.3f,%.3f) | anchor[1] pos=(%u,%u,%u) | anchor[2] pos=(%u,%u,%u) | anchor[3] pos=(%u,%u,%u)\n",
            sample_anchors[0].pos[0], sample_anchors[0].pos[1], sample_anchors[0].pos[2],
            sample_anchors[0].normal[0], sample_anchors[0].normal[1], sample_anchors[0].normal[2],
            sample_snorm[0].s[0], sample_snorm[0].s[1], sample_snorm[0].s[2],
            sample_anchors[1].pos[0], sample_anchors[1].pos[1], sample_anchors[1].pos[2],
            sample_anchors[2].pos[0], sample_anchors[2].pos[1], sample_anchors[2].pos[2],
            sample_anchors[3].pos[0], sample_anchors[3].pos[1], sample_anchors[3].pos[2]);
        // Also sample REST anchors to compare.
        std::vector<AnchorRecord> rest_sample(4);
        clEnqueueReadBuffer(q, (cl_mem)buf_modal_rest_anchors, CL_TRUE, 0, 4*64, rest_sample.data(), 0, nullptr, nullptr);
        fprintf(stderr,
            "[gpu-modal-diag] REST readback: anchor[0] pos=(%u,%u,%u) normal=(%u,%u,%u)\n",
            rest_sample[0].pos[0], rest_sample[0].pos[1], rest_sample[0].pos[2],
            rest_sample[0].normal[0], rest_sample[0].normal[1], rest_sample[0].normal[2]);
    }

    // NO readback, NO mark dirty — bake_surfels reads buf_anchors/buf_surfaces/
    // buf_snorm directly from the GPU. RenderBody skips the host upload block
    // for modal bodies (detected via modal_body == &body).
    return true;
}

// =============================================================================
// Tear down GPU modal buffers (Shutdown path).
// =============================================================================
void SplatRenderer::ShutdownGPUModal() {
    auto rel = [](void* m) { if (m) clReleaseMemObject((cl_mem)m); };
    rel(buf_modal_rest_anchors); rel(buf_modal_rest_normals); rel(buf_modal_rest_coef);
    rel(buf_modal_shapes); rel(buf_modal_K);
    rel(buf_raw_n);
    buf_modal_rest_anchors = buf_modal_rest_normals = buf_modal_rest_coef = nullptr;
    buf_modal_shapes = buf_modal_K = nullptr;
    buf_raw_n = nullptr;
    cap_modal_anchors = cap_modal_surfaces = cap_modal_modes = cap_modal_shapes_bytes = 0;
    if (k_modal_writeback_anchors) { clReleaseKernel((cl_kernel)k_modal_writeback_anchors); k_modal_writeback_anchors = nullptr; }
    if (k_modal_writeback_surfaces) { clReleaseKernel((cl_kernel)k_modal_writeback_surfaces); k_modal_writeback_surfaces = nullptr; }
    if (k_modal_snorm_pass1) { clReleaseKernel((cl_kernel)k_modal_snorm_pass1); k_modal_snorm_pass1 = nullptr; }
    if (k_modal_snorm_pass2) { clReleaseKernel((cl_kernel)k_modal_snorm_pass2); k_modal_snorm_pass2 = nullptr; }
    modal_body = nullptr;
}

void SplatRenderer::Shutdown() {
    auto rel = [](void* m) { if (m) clReleaseMemObject((cl_mem)m); };
    rel(buf_fb); rel(buf_img); rel(buf_depth32);
    rel(buf_anchors); rel(buf_surfaces); rel(buf_stitch);
    rel(buf_snorm);
    // TriInfill frame buffers
    rel(buf_verts_grid); rel(buf_norms_grid); rel(buf_pos_grid); rel(buf_branch_trace);
    buf_verts_grid = buf_norms_grid = buf_pos_grid = buf_branch_trace = nullptr;
    cap_tri_samples = cap_tri_tris = cap_tri_patches = 0;
    rel(buf_soff); rel(buf_sss);
    // Anchor reflow temp buffers
    rel(buf_reflow_target); rel(buf_reflow_error);
    rel(buf_reflow_sum_err); rel(buf_reflow_count);
    rel(buf_reflow_disp); rel(buf_reflow_coef);
    buf_reflow_target = buf_reflow_error = buf_reflow_sum_err = buf_reflow_count = nullptr;
    buf_reflow_disp = buf_reflow_coef = nullptr;
    cap_reflow_patches = cap_reflow_anchors = 0;
    if (k_reflow_accum)  { clReleaseKernel((cl_kernel)k_reflow_accum);  k_reflow_accum = nullptr; }
    if (k_reflow_disp)   { clReleaseKernel((cl_kernel)k_reflow_disp);   k_reflow_disp = nullptr; }
    if (k_reflow_update) { clReleaseKernel((cl_kernel)k_reflow_update); k_reflow_update = nullptr; }
    if (k_reflow_refit)  { clReleaseKernel((cl_kernel)k_reflow_refit);  k_reflow_refit = nullptr; }
    // GPU AMR buffers + kernel
    rel(buf_amr_anchors); rel(buf_amr_surfaces); rel(buf_amr_stitch);
    rel(buf_amr_alloc_heads); rel(buf_amr_split_list);
    buf_amr_anchors = buf_amr_surfaces = buf_amr_stitch = nullptr;
    buf_amr_alloc_heads = buf_amr_split_list = nullptr;
    cap_amr_anchors = cap_amr_surfaces = 0;
    if (k_split) { clReleaseKernel((cl_kernel)k_split); k_split = nullptr; }
    // M2 occlusion culling buffer + modal writeback buffers/kernel
    rel(buf_occlusion_depth);
    buf_occlusion_depth = nullptr; cap_occlusion_tiles = 0;
    // GPU-FIRST modal writeback buffers/kernels (REST buffers, shapes, K, raw_n, kernels).
    ShutdownGPUModal();
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
