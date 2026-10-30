// =============================================================================
// SplatRenderer.h - OpenCL host for kernels/psf_splat.cl.
// One frame: fill fb -> psf_splat (one work-item per surface sample,
// anisotropic per-patch grids) -> resolve_fb -> readback -> SDL texture.
// 64-bit packed atomic path when cl_khr_int64_extended_atomics exists;
// otherwise the kernel's two-pass 32-bit fallback (same image).
// CPU reference: tools/psf_validate.py (pixel-for-pixel comparable).
// =============================================================================
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace DeliRium {

class PSFBody;
struct Camera;

class SplatRenderer {
public:
    // sdl_renderer: an SDL_Renderer* passed as void* to keep this header
    // dependency-free. kernel_dir: where psf_splat.cl lives.
    bool Initialize(int width, int height, void* sdl_renderer,
                    const std::string& kernel_dir = "kernels");
    void Shutdown();

    // Milestone 3 scope: one body per frame (TODO scene batching).
    // clear=false skips the framebuffer clear so a second body (e.g. a
    // fracture fragment) can be composited into the same frame.
    // final=false defers the buf_fb->buf_img resolve + queue finish to the
    // LAST body of the frame (the caller passes final=true there), so a
    // multi-body frame pays the full-screen resolve and GPU sync ONCE
    // instead of once per body.
    bool RenderBody(PSFBody& body, const Camera& cam, bool clear = true, bool final = true,
                float depth_bias = 0.0f);
    void Present();                     // blit the resolved image
    // Save the resolved GPU framebuffer to a PPM (ground-truth capture).
    bool SaveFramePPM(const std::string& path) const;
    // Read the resolved GPU framebuffer into `out` (width*height*4 RGBA,
    // row-major, top row first -- the same layout Present() blits). Used by
    // the 10s recorder. Returns false on a readback failure.
    bool GrabFinalRGBA(std::vector<unsigned char>& out) const;
    void Resize(int width, int height);

    bool HasLongAtomics() const { return has_long_atomics; }
    const std::string& DeviceName() const { return device_name; }
    float LastSamplesPerPixel() const { return last_spp; }

private:
    bool  BuildProgram(const std::string& kernel_dir);
    // anisotropic grids: ssu/ssv from projected edge lengths (x1.4 + 2,
    // clamp [2,192]); returns total sample count, fills packed ss + offsets
    uint32_t BuildSampleGrids(PSFBody& body, const float mvp[16]);

    int  width = 0, height = 0;
    bool has_long_atomics = false;
    std::string device_name;
    float last_spp = 0;

    // opaque CL/SDL handles (void* so the header stays clean)
    void *cl_ctx = nullptr, *cl_queue = nullptr, *cl_dev = nullptr;
    void *prog = nullptr, *k_splat = nullptr, *k_resolve = nullptr;
    void *buf_fb = nullptr, *buf_img = nullptr, *buf_depth32 = nullptr;
    void *buf_tie_cnt = nullptr;   // SPLAT_TIE_DIAG depth-tie counter (1 uint)
    void *buf_anchors = nullptr, *buf_surfaces = nullptr, *buf_stitch = nullptr;
    void *buf_snorm = nullptr;   // smooth per-anchor shading normals
    void *buf_soff = nullptr, *buf_sss = nullptr;
    uint32_t cap_soff = 0, cap_anchors = 0, cap_surfaces = 0, cap_snorm = 0;
    // CROSS-BODY OWNERSHIP: buf_anchors/buf_surfaces/buf_snorm are SHARED
    // across every body rendered in a frame. The old upload gate
    // (grew || AnchorsDirty) let a smaller/clean body skip its upload and
    // render with the PREVIOUS body's data (the "everything is one color"
    // contamination). Tracking which body's data is actually in the buffers
    // forces a re-upload whenever the body changes, not just when it grows.
    const PSFBody* last_body = nullptr;
    void *sdl_rnd = nullptr, *sdl_tex = nullptr;

    // host staging for per-frame sample layout
    std::vector<uint32_t> h_soff, h_sss;
};

} // namespace DeliRium
