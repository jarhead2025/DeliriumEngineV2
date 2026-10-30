#pragma once
#include "../opengl/buffer.hpp"
#include "../types.h"
#include "../body_runtime.h"
#include <vector>

namespace delirium {

/* ── GPU compute renderer (surfel generation + point sprite render) ─
 *
 * Proven pipeline (mirrors original surfel_renderer.cpp + psf_splat.cl):
 *   Compute: each work item evaluates one (patch,u,v) sample → writes Surfel to SSBO
 *   Render:  GL_POINTS with hardware depth test, no image atomics
 *
 * ~115 FLOPs/sample, no CPU evaluation, no surfel cache.
 * ───────────────────────────────────────────────────────────────── */
struct SurfelVertex {
    float x, y, z;       // position
    float nx, ny, nz;    // normal
    float r, g, b, a;    // color
    float radius;        // point sprite size
};

static_assert(sizeof(SurfelVertex) == 44, "SurfelVertex must be 44 bytes");

class ComputeRenderer {
public:
    // Single-path overload: resolves resolve.vert/frag from same dir as compute shader
    bool initialize(const char* cs_path, uint32_t w, uint32_t h);
    bool initialize(const char* cs_path, const char* vert_path, const char* frag_path,
                    uint32_t w, uint32_t h);
    void shutdown();
    void resize(uint32_t w, uint32_t h);

    void upload(BodyRuntime* body);
    void render(Vec3 cam_pos, const float* view, const float* proj,
                uint32_t width, uint32_t height);

private:
    GLuint compute_prog_ = 0;
    GLuint render_prog_ = 0;

    // SSBOs for compute shader
    GLBuffer anchors_buf_;     // binding 0
    GLBuffer anchors_n_buf_;   // binding 1
    GLBuffer patches_buf_;     // binding 2
    GLBuffer coef_buf_;        // binding 3
    GLBuffer mat_buf_;         // binding 4
    GLBuffer meta_buf_;        // binding 5

    // Surfel output buffer (compute writes, render reads via VAO)
    GLBuffer surfel_buf_;      // binding 6 (compute) / GL_ARRAY_BUFFER (render)
    GLBuffer counter_buf_;     // binding 7 (atomic counter)
    GLuint   surfel_vao_ = 0;
    uint32_t max_surfels_ = 0;
    uint32_t surfel_count_ = 0;

    // CPU staging
    std::vector<float> anchor_data_;
    std::vector<float> anchor_n_data_;
    std::vector<int>   patch_data_;
    std::vector<float> coef_data_;
    std::vector<float> mat_data_;
    uint32_t meta_[6] = {};

    uint32_t na_ = 0, np_ = 0;
    uint32_t win_w_ = 0, win_h_ = 0;
    bool topo_dirty_ = true;
    float bbox_diag_ = 1.0f;

    uint32_t compute_samples(uint32_t np, const float* mvp, const float* anchor_data, uint32_t na) const;
};

} // namespace delirium