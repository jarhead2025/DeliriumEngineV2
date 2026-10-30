// =============================================================================
// SpectralRenderer.h - OpenCL host for kernels/spf_splat.cl.
//
// Renders spectral primitives (.spf): each work-item marches one sphere
// direction, evaluates the SH detail field, projects, and splats with the
// 64-bit packed-atomic depth test. Far simpler than SplatRenderer -- there
// are no anchors, no per-patch sample grids, no seams, no back-face pass.
// One coefficient buffer per primitive (uploaded once, reused every frame
// since the detail never changes; only the transform/camera move).
//
// Supports many primitives per frame (a scene): clear once, dispatch
// spf_splat per visible primitive into the shared depth buffer, resolve
// once. LOD is chosen per primitive from its projected size (fewer SH
// terms when far) -- the spectral analogue of mesh LOD, for free.
// =============================================================================
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace DeliRium {

class SpectralPrimitive;
struct SpectralObject;
class SpectralModel;
struct Camera;

class SpectralRenderer {
public:
    bool Initialize(int width, int height, const std::string& kernel_dir = "kernels");
    void Shutdown();

    // Begin a frame: clears the shared depth/color buffer.
    void BeginFrame();
    // Draw one primitive with a world transform (position/rotation/scale)
    // and a chosen SH order (lod_order; 0 = full detail). Uploads the
    // primitive's coefficients on first sight and caches them by pointer.
    bool DrawPrimitive(const SpectralPrimitive& prim,
                       const float center_world[3], float scale,
                       uint32_t lod_order, float color_lod, const Camera& cam);
    // Draw a whole composite model (each part with its transform + LOD).
    bool DrawModel(const SpectralModel& model, const Camera& cam);
    // Resolve the shared buffer into the RGBA image (host-readable).
    void EndFrame();

    // Read the resolved RGBA8 image back to host (width*height*4 bytes).
    const std::vector<uint8_t>& Image() const { return image; }

    const std::string& DeviceName() const { return device_name; }
    bool HasLongAtomics() const { return has_long_atomics; }

private:
    bool BuildProgram(const std::string& kernel_dir);
    // upload (or fetch cached) coefficient buffers for a primitive
    void* UploadCoeffs(const SpectralPrimitive& prim, bool color);

    int width = 0, height = 0;
    bool has_long_atomics = false;
    std::string device_name;

    void *cl_ctx=nullptr, *cl_queue=nullptr, *cl_dev=nullptr, *prog=nullptr;
    void *k_clear=nullptr, *k_splat=nullptr, *k_resolve=nullptr;
    void *buf_fb=nullptr, *buf_img=nullptr;

    // coefficient buffers cached by primitive pointer (detail is static)
    struct CoeffCache { const void* key; void* bg; void* bc; };
    std::vector<CoeffCache> coeff_cache;

    std::vector<uint8_t> image;
};

} // namespace DeliRium
