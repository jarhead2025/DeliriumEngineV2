#pragma once
#include "../opengl/shader.hpp"
#include "../opengl/buffer.hpp"
#include "../types.h"
#include "../body_runtime.h"
#include <string>
#include <vector>

namespace delirium {

class RayMarchRenderer {
public:
    bool initialize();
    void shutdown();
    void update_geometry(BodyRuntime* body);
    void draw(ShaderProgram& shader, Vec3 cam_pos, const float* view_inv, const float* proj_inv,
              uint32_t width, uint32_t height);
    uint32_t anchor_count() const { return anchor_count_; }
    uint32_t patch_count() const { return patch_count_; }
private:
    GLBuffer vbo_;
    uint32_t anchor_count_ = 0;
    uint32_t patch_count_ = 0;
    std::vector<float> anchor_data_;
    std::vector<int> patch_data_;
    std::vector<int> patch_basis_data_;
    std::vector<float> patch_coef_data_;  // 8 floats per patch
    std::vector<float> color_data_;
    uint32_t texture_ = 0;
    bool has_texture_ = false;
    bool topology_dirty_ = false;
};

} // namespace delirium