// =============================================================================
// SpectralRenderer.cpp - OpenCL host for kernels/spf_splat.cl.
// =============================================================================
#include "delirium/render/SpectralRenderer.h"
#include "delirium/psf/SpectralPrimitive.h"
#include "delirium/psf/SpectralObject.h"
#include "delirium/render/Camera.h"
#include <CL/cl.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cmath>

namespace DeliRium {

#define CLC(call) do { cl_int _e = (call); if (_e != CL_SUCCESS) { \
    std::fprintf(stderr, "CL error %d at %s:%d\n", _e, __FILE__, __LINE__); \
    return false; } } while (0)

bool SpectralRenderer::Initialize(int w, int h, const std::string& kernel_dir) {
    width = w; height = h;
    image.assign((size_t)w * h * 4, 0);

    cl_platform_id plats[8]; cl_uint np = 0;
    if (clGetPlatformIDs(8, plats, &np) != CL_SUCCESS || !np) {
        std::fprintf(stderr, "no OpenCL platforms\n"); return false; }
    cl_device_id dev = nullptr;
    for (cl_uint p = 0; p < np && !dev; ++p) {
        cl_device_id d; cl_uint nd = 0;
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 1, &d, &nd)
            == CL_SUCCESS && nd) dev = d;
    }
    if (!dev)
        for (cl_uint p = 0; p < np && !dev; ++p) {
            cl_device_id d; cl_uint nd = 0;
            if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 1, &d, &nd)
                == CL_SUCCESS && nd) dev = d;
        }
    if (!dev) { std::fprintf(stderr, "no OpenCL device\n"); return false; }
    cl_dev = dev;

    char info[2048] = {};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(info), info, nullptr);
    device_name = info;
    clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sizeof(info), info, nullptr);
    has_long_atomics = std::strstr(info, "cl_khr_int64_extended_atomics")
                    && std::strstr(info, "cl_khr_int64_base_atomics");

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return false;
    cl_ctx = ctx;
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    if (err != CL_SUCCESS) return false;
    cl_queue = q;

    if (!BuildProgram(kernel_dir)) return false;

    cl_mem fb = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                               (size_t)w*h*sizeof(cl_ulong), nullptr, &err);
    if (err != CL_SUCCESS) return false;
    buf_fb = fb;
    cl_mem img = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                (size_t)w*h*4, nullptr, &err);
    if (err != CL_SUCCESS) return false;
    buf_img = img;
    return true;
}

bool SpectralRenderer::BuildProgram(const std::string& kernel_dir) {
    std::ifstream f(kernel_dir + "/spf_splat.cl");
    if (!f) { std::fprintf(stderr, "cannot open spf_splat.cl\n"); return false; }
    std::stringstream ss; ss << f.rdbuf();
    std::string src = ss.str();
    const char* csrc = src.c_str(); size_t len = src.size();
    cl_int err;
    cl_program p = clCreateProgramWithSource((cl_context)cl_ctx, 1, &csrc,
                                             &len, &err);
    if (err != CL_SUCCESS) return false;
    prog = p;
    cl_device_id dev = (cl_device_id)cl_dev;
    err = clBuildProgram(p, 1, &dev, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t ln = 0;
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &ln);
        std::string log(ln, 0);
        clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, ln, &log[0], nullptr);
        std::fprintf(stderr, "spf_splat.cl build failed:\n%s\n", log.c_str());
        return false;
    }
    k_clear   = clCreateKernel(p, "clear_fb", &err);   if (err) return false;
    k_splat   = clCreateKernel(p, "spf_splat", &err);  if (err) return false;
    k_resolve = clCreateKernel(p, "resolve_fb", &err); if (err) return false;
    return true;
}

void SpectralRenderer::BeginFrame() {
    cl_uint npix = (cl_uint)(width * height);
    clSetKernelArg((cl_kernel)k_clear, 0, sizeof(cl_mem), &buf_fb);
    clSetKernelArg((cl_kernel)k_clear, 1, sizeof(cl_uint), &npix);
    size_t g = ((npix + 255) / 256) * 256;
    clEnqueueNDRangeKernel((cl_command_queue)cl_queue, (cl_kernel)k_clear,
                           1, nullptr, &g, nullptr, 0, nullptr, nullptr);
}

void* SpectralRenderer::UploadCoeffs(const SpectralPrimitive& prim, bool color) {
    for (auto& c : coeff_cache)
        if (c.key == &prim) return color ? c.bc : c.bg;
    // not cached: upload both buffers
    cl_int err;
    const int ng = (prim.GeomOrder()+1)*(prim.GeomOrder()+1);
    const int nc = 3*(prim.ColorOrder()+1)*(prim.ColorOrder()+1);
    cl_mem bg = clCreateBuffer((cl_context)cl_ctx,
        CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, ng*sizeof(float),
        (void*)prim.GeomCoeffs(), &err);
    cl_mem bc = clCreateBuffer((cl_context)cl_ctx,
        CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, nc*sizeof(float),
        (void*)prim.ColorCoeffs(), &err);
    coeff_cache.push_back({&prim, bg, bc});
    return color ? bc : bg;
}

bool SpectralRenderer::DrawPrimitive(const SpectralPrimitive& prim,
                                     const float center_world[3], float scale,
                                     uint32_t lod_order, float color_lod, const Camera& cam) {
    cl_mem bg = (cl_mem)UploadCoeffs(prim, false);
    cl_mem bc = (cl_mem)UploadCoeffs(prim, true);

    float mvp[16]; cam.BuildMVP(mvp, float(width)/float(height));
    cl_int Lg = (cl_int)(lod_order ? lod_order : prim.GeomOrder());
    if (Lg > (cl_int)prim.GeomOrder()) Lg = prim.GeomOrder();
    cl_int Lc_stored = (cl_int)prim.ColorOrder();
    // color LOD as a CONTINUOUS order: each frequency band fades in with
    // proximity (the user's model: the image is a continuous combination of
    // bands revealed by distance), so there is no popping between levels.
    cl_float Lc = (color_lod > 0.0f) ? (cl_float)color_lod : (cl_float)Lc_stored;
    if (Lc > (cl_float)Lc_stored) Lc = (cl_float)Lc_stored;
    cl_float3 ccenter = {{center_world[0], center_world[1], center_world[2]}};
    cl_float  base_r = prim.BaseRadius() * scale;
    cl_uint n_samples = 900000, pad_unused = 0;
    cl_float16 cmvp; for (int i=0;i<16;++i) cmvp.s[i]=mvp[i];
    cl_float3 light = {{0.45f,0.35f,-0.82f}};
    float ll=std::sqrt(light.s[0]*light.s[0]+light.s[1]*light.s[1]+light.s[2]*light.s[2]);
    light.s[0]/=ll; light.s[1]/=ll; light.s[2]/=ll;
    cl_float3 eye = {{cam.position.x, cam.position.y, cam.position.z}};
    cl_uint W=(cl_uint)width, H=(cl_uint)height;

    cl_kernel k=(cl_kernel)k_splat; int a=0;
    CLC(clSetKernelArg(k,a++,sizeof(cl_mem),&bg));
    CLC(clSetKernelArg(k,a++,sizeof(cl_mem),&bc));
    CLC(clSetKernelArg(k,a++,sizeof(cl_int),&Lg));
    CLC(clSetKernelArg(k,a++,sizeof(cl_float),&Lc));
    CLC(clSetKernelArg(k,a++,sizeof(cl_int),&Lc_stored));
    CLC(clSetKernelArg(k,a++,sizeof(cl_float3),&ccenter));
    CLC(clSetKernelArg(k,a++,sizeof(cl_float),&base_r));
    CLC(clSetKernelArg(k,a++,sizeof(cl_uint),&n_samples));
    CLC(clSetKernelArg(k,a++,sizeof(cl_uint),&pad_unused));
    CLC(clSetKernelArg(k,a++,sizeof(cl_float16),&cmvp));
    CLC(clSetKernelArg(k,a++,sizeof(cl_float3),&light));
    CLC(clSetKernelArg(k,a++,sizeof(cl_float3),&eye));
    CLC(clSetKernelArg(k,a++,sizeof(cl_uint),&W));
    CLC(clSetKernelArg(k,a++,sizeof(cl_uint),&H));
    CLC(clSetKernelArg(k,a++,sizeof(cl_mem),&buf_fb));
    size_t g=(size_t)n_samples; g=((g+255)/256)*256;
    CLC(clEnqueueNDRangeKernel((cl_command_queue)cl_queue,k,1,nullptr,&g,
                               nullptr,0,nullptr,nullptr));
    return true;
}

bool SpectralRenderer::DrawModel(const SpectralModel& model, const Camera& cam) {
    for (size_t i=0;i<model.PartCount();++i) {
        const SpectralObject& part = model.Part(i);
        // world center = root + local position (rotation of composite TODO)
        float cw[3] = {
            model.Root().position.x + part.transform.position.x,
            model.Root().position.y + part.transform.position.y,
            model.Root().position.z + part.transform.position.z };
        float sc = part.transform.scale.x * model.Root().scale.x;
        // crude projected-size estimate for LOD: distance to camera
        float dx=cw[0]-cam.position.x, dy=cw[1]-cam.position.y, dz=cw[2]-cam.position.z;
        float dist=std::sqrt(dx*dx+dy*dy+dz*dz);
        float proj_px = dist>1e-3f ? (sc*400.0f/dist) : 400.0f;
        SpectralObject::Lod lod = part.LodPairForScreenSize(proj_px, part.prim->ColorOrder());
        if (!DrawPrimitive(*part.prim, cw, sc, lod.geom, lod.color, cam)) return false;
    }
    return true;
}

void SpectralRenderer::EndFrame() {
    cl_uint npix=(cl_uint)(width*height);
    clSetKernelArg((cl_kernel)k_resolve,0,sizeof(cl_mem),&buf_fb);
    clSetKernelArg((cl_kernel)k_resolve,1,sizeof(cl_mem),&buf_img);
    clSetKernelArg((cl_kernel)k_resolve,2,sizeof(cl_uint),&npix);
    size_t g=((npix+255)/256)*256;
    clEnqueueNDRangeKernel((cl_command_queue)cl_queue,(cl_kernel)k_resolve,
                           1,nullptr,&g,nullptr,0,nullptr,nullptr);
    clFinish((cl_command_queue)cl_queue);
    clEnqueueReadBuffer((cl_command_queue)cl_queue,(cl_mem)buf_img,CL_TRUE,0,
                        image.size(),image.data(),0,nullptr,nullptr);
}

void SpectralRenderer::Shutdown() {
    auto rel=[](void*m){ if(m) clReleaseMemObject((cl_mem)m); };
    for (auto& c : coeff_cache) { rel(c.bg); rel(c.bc); }
    coeff_cache.clear();
    rel(buf_fb); rel(buf_img);
    if (k_clear)   clReleaseKernel((cl_kernel)k_clear);
    if (k_splat)   clReleaseKernel((cl_kernel)k_splat);
    if (k_resolve) clReleaseKernel((cl_kernel)k_resolve);
    if (prog)      clReleaseProgram((cl_program)prog);
    if (cl_queue)  clReleaseCommandQueue((cl_command_queue)cl_queue);
    if (cl_ctx)    clReleaseContext((cl_context)cl_ctx);
}

} // namespace DeliRium
