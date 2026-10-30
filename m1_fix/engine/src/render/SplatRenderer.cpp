#include "delirium/render/SplatRenderer.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/render/Camera.h"
#include "delirium/render/EquationAccel.h"
#include "delirium/render/LateralRayMarch.h"
#include "delirium/render/EquationAccel.h"
#include <CL/cl.h>
#include <SDL2/SDL.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cmath>
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


#define CLC(call) do { cl_int _e = (call); if (_e != CL_SUCCESS) { \
    std::fprintf(stderr, "CL error %d at %s:%d\n", _e, __FILE__, __LINE__); \
    return false; } } while (0)

bool SplatRenderer::Initialize(int w, int h, void* sdl_renderer,
                               const std::string& kernel_dir) {
    width = w; height = h; sdl_rnd = sdl_renderer;

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
    std::printf("OpenCL device: %s (64-bit atomics: %s)\n",
                device_name.c_str(), has_long_atomics ? "yes" : "no, two-pass");

    cl_int err = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    CLC(err); cl_ctx = ctx;
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    CLC(err); cl_queue = q;

    if (!BuildProgram(kernel_dir)) return false;

    buf_fb  = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size_t(w)*h*8, nullptr, &err); CLC(err);
    buf_img = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, size_t(w)*h*4, nullptr, &err); CLC(err);
    if (!has_long_atomics) {
        buf_depth32 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size_t(w)*h*4,
                                     nullptr, &err); CLC(err);
    }
    sdl_tex = SDL_CreateTexture((SDL_Renderer*)sdl_rnd,
                                SDL_PIXELFORMAT_RGBA32,
                                SDL_TEXTUREACCESS_STREAMING, w, h);
    return sdl_tex != nullptr;
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
        // LOD by the format: emit one sample per screen pixel along each
        // axis (factor 1.0 = exact coverage), plus a small margin, with NO
        // low fixed cap -- a patch that fills the screen when the camera is
        // close MUST be allowed to emit enough samples to cover every pixel
        // (the old cap of 192 left holes/stripes on close patches). The cap
        // scales with the larger projected edge and only bounds pathological
        // cases. Because the surface is a continuous mathematical patch, we
        // can evaluate it at whatever density coverage demands -- this is
        // the custom-LOD the format affords.
        auto clampg = [&](float e) {
            // Close-up: need >1 sample/px along projected edges or holes appear
            // and LOD-stitch smears (glitches near POV). Aim ~1.5 samples/px.
            int g = (int)(e * 1.5f) + 3;
            const int hi = (int)(std::fmax(eu, ev) * 1.55f) + 6;
            const int cap = hi < 12 ? 12 : (hi > 320 ? 320 : hi);
            return (uint32_t)(g < 10 ? 10 : (g > cap ? cap : g));
        };
        const uint32_t su = clampg(eu), sv = clampg(ev);
        h_sss[s] = (su << 16) | sv;
        h_soff[s] = (uint32_t)total;
        total += uint64_t(su) * sv;
    }
    h_soff[ns] = (uint32_t)total;
    return (uint32_t)total;
}

bool SplatRenderer::RenderBody(PSFBody& body, const Camera& cam) {
    cl_context ctx = (cl_context)cl_ctx;
    cl_command_queue q = (cl_command_queue)cl_queue;
    cl_int err = CL_SUCCESS;

    // True lateral / view-ray equation path (raw math). CPU builds hits; GPU only holds FB + optional AA.
    // DELIRIUM_RAYMARCH=0 falls back to classic psf_splat.
    // GPU equation ray-march (OpenCL). Opt out: DELIRIUM_RAYMARCH=0 → classic splat.
    bool use_ray = false;
    if (const char* e = std::getenv("DELIRIUM_RAYMARCH")) {
        if (e[0] == '1') use_ray = true;
    }
    if (use_ray && k_raymarch && buf_img) {
        const auto& A = body.AnchorBlob();
        const auto& S = body.SurfaceBlob();
        const uint32_t ns = (uint32_t)S.size();
        Vec3 bmn = body.BBoxMin(), bsp = body.BBoxSpan();
        if (bsp.x < 1e-8f && bsp.y < 1e-8f && bsp.z < 1e-8f) {
            bmn = Vec3{-1.f,-1.f,-1.f}; bsp = Vec3{2.f,2.f,2.f};
        }
        // ensure buffers
        const bool grew = cap_anchors < A.size() || cap_surfaces < S.size();
        if (grew || !buf_anchors) {
            auto rel = [](void* m) { if (m) clReleaseMemObject((cl_mem)m); };
            rel(buf_anchors); rel(buf_surfaces);
            buf_anchors  = clCreateBuffer(ctx, CL_MEM_READ_ONLY, A.size()*64, nullptr, &err); CLC(err);
            buf_surfaces = clCreateBuffer(ctx, CL_MEM_READ_ONLY, S.size()*64, nullptr, &err); CLC(err);
            cap_anchors = (uint32_t)A.size(); cap_surfaces = (uint32_t)S.size();
            CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_anchors, CL_FALSE, 0, A.size()*64, A.data(), 0, nullptr, nullptr));
            CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_surfaces, CL_FALSE, 0, S.size()*64, S.data(), 0, nullptr, nullptr));
        } else if (body.AnchorsDirty() || body.AnyDirty()) {
            CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_anchors, CL_FALSE, 0, A.size()*64, A.data(), 0, nullptr, nullptr));
            CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_surfaces, CL_FALSE, 0, S.size()*64, S.data(), 0, nullptr, nullptr));
            body.MarkAnchorsClean(); body.ClearDirty();
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
        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_frustum, CL_FALSE, 0, sizeof(planes), planes, 0, nullptr, nullptr));

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

    // (re)create on growth (live splits!), upload on dirty flags
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
    if (grew || body.AnchorsDirty()) {
        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_anchors, CL_FALSE, 0, A.size()*64, A.data(),  0, nullptr, nullptr));
        {
            const size_t stitch_bytes = St.empty() ? size_t(ns) * 8u : St.size();
            if (St.empty()) {
                std::vector<char> zeros(stitch_bytes, 0);
                CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_stitch, CL_FALSE, 0, stitch_bytes, zeros.data(), 0, nullptr, nullptr));
            } else {
                CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_stitch,  CL_FALSE, 0, St.size(),   St.data(), 0, nullptr, nullptr));
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
        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_snorm, CL_FALSE, 0,
                            snf.size()*sizeof(cl_float4), snf.data(), 0, nullptr, nullptr));
        body.MarkAnchorsClean();
    }
    if (grew || body.AnyDirty()) {
        CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_surfaces, CL_FALSE, 0, S.size()*64, S.data(), 0, nullptr, nullptr));
        body.ClearDirty();
    }

    float mvp[16];
    cam.BuildMVP(mvp, float(width) / float(height));
    const uint32_t total = BuildSampleGrids(body, mvp);
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
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_soff, CL_FALSE, 0, (ns+1)*4, h_soff.data(), 0, nullptr, nullptr));
    CLC(clEnqueueWriteBuffer(q, (cl_mem)buf_sss,  CL_FALSE, 0,  ns   *4, h_sss.data(),  0, nullptr, nullptr));

    const cl_ulong far_key = ~cl_ulong(0);
    CLC(clEnqueueFillBuffer(q, (cl_mem)buf_fb, &far_key, 8, 0,
                            size_t(width)*height*8, 0, nullptr, nullptr));
    if (!has_long_atomics) {
        const cl_uint far_d = ~cl_uint(0);
        CLC(clEnqueueFillBuffer(q, (cl_mem)buf_depth32, &far_d, 4, 0,
                                size_t(width)*height*4, 0, nullptr, nullptr));
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
    cl_float16 m16; std::memcpy(&m16, mvp, 64);
    const float lf = 1.0f / std::sqrt(0.45f*0.45f + 0.35f*0.35f + 0.82f*0.82f);
    const cl_float4 light = {{0.45f*lf, 0.35f*lf, -0.82f*lf, 0}};
    const float sscale = body.StitchScale();
    const cl_uint W = width, H = height;

    cl_kernel k = (cl_kernel)k_splat;
    cl_uint a = 0;
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
    CLC(clSetKernelArg(k, a++, sizeof(cl_uint), &W));
    CLC(clSetKernelArg(k, a++, sizeof(cl_uint), &H));
    CLC(clSetKernelArg(k, a++, sizeof(cl_mem), &buf_fb));

    size_t g = (size_t(total) + 255) & ~size_t(255);
    if (has_long_atomics) {
        CLC(clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr));
    } else {
        CLC(clSetKernelArg(k, a, sizeof(cl_mem), &buf_depth32));
        for (cl_uint pass = 0; pass < 2; ++pass) {
            CLC(clSetKernelArg(k, a + 1, sizeof(cl_uint), &pass));
            CLC(clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, nullptr, 0, nullptr, nullptr));
        }
    }

    cl_kernel kr = (cl_kernel)k_resolve;
    const cl_uint npx = W * H;
    CLC(clSetKernelArg(kr, 0, sizeof(cl_mem), &buf_fb));
    CLC(clSetKernelArg(kr, 1, sizeof(cl_uint), &npx));
    CLC(clSetKernelArg(kr, 2, sizeof(cl_mem), &buf_img));
    size_t gr = (size_t(npx) + 255) & ~size_t(255);
    CLC(clEnqueueNDRangeKernel(q, kr, 1, nullptr, &gr, nullptr, 0, nullptr, nullptr));

    // GPU-only hole fill: CPU does not touch sample data — only enqueue + Present
    int passes = 3;  // lower default; set DELIRIUM_GPU_INFILL=8 to thicken
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

        // Variable-LOD surface stitch: average coarse impacts with distance-aware radius
        {
            int lod_passes = 2;
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
    last_spp = float(total) / float(npx);
    return true;
}

void SplatRenderer::Present() {
    void* px = nullptr; int pitch = 0;
    if (SDL_LockTexture((SDL_Texture*)sdl_tex, nullptr, &px, &pitch) == 0) {
        clEnqueueReadBuffer((cl_command_queue)cl_queue, (cl_mem)buf_img,
                            CL_TRUE, 0, size_t(width)*height*4, px,
                            0, nullptr, nullptr);
        SDL_UnlockTexture((SDL_Texture*)sdl_tex);
    }
    SDL_RenderClear((SDL_Renderer*)sdl_rnd);
    SDL_RenderCopy((SDL_Renderer*)sdl_rnd, (SDL_Texture*)sdl_tex, nullptr, nullptr);
    SDL_RenderPresent((SDL_Renderer*)sdl_rnd);
}

void SplatRenderer::Resize(int, int) { /* TODO realloc fb/img/tex */ }

void SplatRenderer::Shutdown() {
    auto rel = [](void* m) { if (m) clReleaseMemObject((cl_mem)m); };
    rel(buf_fb); rel(buf_img); rel(buf_depth32);
    rel(buf_anchors); rel(buf_surfaces); rel(buf_stitch);
    rel(buf_snorm);
    rel(buf_soff); rel(buf_sss);
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
