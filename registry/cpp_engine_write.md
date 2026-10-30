```cpp
// FILEPATH: engine/src/render/SplatRenderer.cpp

#include <CL/cl.h>
#include <SDL2/SDL.h>
#include <string>
#include <vector>
#include "engine/include/delirium/render/SplatRenderer.h"
#include "engine/include/delirium/psf/PSFBody.h"

namespace DeliRium {

bool SplatRenderer::Initialize(int width, int height, void* sdl_renderer,
                               const std::string& kernel_dir) {
    this->width = width;
    this->height = height;
    this->sdl_rnd = sdl_renderer;

    // Initialize OpenCL context and device
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
    cl_ctx = clCreateContext(properties, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return false;

    // Create command queue
    cl_queue = clCreateCommandQueue(cl_ctx, device, 0, &err);
    if (err != CL_SUCCESS) return false;

    // Build OpenCL program from kernel file
    if (!BuildProgram(kernel_dir)) return false;

    // Allocate buffers and textures
    buf_fb = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, width * height * sizeof(uint32_t), NULL, &err);
    if (err != CL_SUCCESS) return false;

    sdl_tex = SDL_CreateTexture((SDL_Renderer*)sdl_rnd, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, width, height);

    return true;
}

void SplatRenderer::Shutdown() {
    clReleaseMemObject(buf_fb);
    clReleaseKernel(k_splat);
    clReleaseKernel(k_resolve);
    clReleaseProgram(prog);
    clReleaseCommandQueue(cl_queue);
    clReleaseContext(cl_ctx);
    SDL_DestroyTexture((SDL_Texture*)sdl_tex);
}

bool SplatRenderer::RenderBody(PSFBody& body, const Camera& cam) {
    // Build sample grids and fill buffers
    uint32_t total_samples = BuildSampleGrids(body, cam.mvp);

    // Set kernel arguments
    clSetKernelArg(k_splat, 0, sizeof(cl_mem), &buf_fb);
    clSetKernelArg(k_splat, 1, sizeof(uint32_t), &total_samples);

    // Enqueue kernel execution
    size_t global_work_size = total_samples;
    clEnqueueNDRangeKernel(cl_queue, k_splat, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);

    return true;
}

void SplatRenderer::Present() {
    SDL_UpdateTexture((SDL_Texture*)sdl_tex, NULL, buf_fb, width * sizeof(uint32_t));
    SDL_RenderCopy((SDL_Renderer*)sdl_rnd, (SDL_Texture*)sdl_tex, NULL, NULL);
    SDL_RenderPresent((SDL_Renderer*)sdl_rnd);
}

void SplatRenderer::Resize(int width, int height) {
    this->width = width;
    this->height = height;

    clReleaseMemObject(buf_fb);
    buf_fb = clCreateBuffer(cl_ctx, CL_MEM_READ_WRITE, width * height * sizeof(uint32_t), NULL, &err);

    SDL_DestroyTexture((SDL_Texture*)sdl_tex);
    sdl_tex = SDL_CreateTexture((SDL_Renderer*)sdl_rnd, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, width, height);
}

bool SplatRenderer::BuildProgram(const std::string& kernel_dir) {
    // Load and build OpenCL program from source
    // ...

    return true;
}

uint32_t SplatRenderer::BuildSampleGrids(PSFBody& body, const float mvp[16]) {
    // Build anisotropic grids and fill buffers
    // ...

    return total_samples;
}

}  // namespace DeliRium
```