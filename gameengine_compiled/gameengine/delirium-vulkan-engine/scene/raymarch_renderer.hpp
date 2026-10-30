#pragma once
#include "math/mat4.hpp"
#include "opengl/shader.hpp"
#include "opengl/buffer.hpp"
#include "camera/camera.hpp"
#include "delirium_m8/body_runtime.hpp"
#include <string>
#include <vector>

namespace delirium {

// Ray marching renderer — full implicit surface rendering with:
// - 8 basis surface types per patch (Bilinear, Plano, Cupula, Silla, ValleU/V, Eggbox, Twist)
// - UV interpolation from anchor coordinates for texture sampling
// - GL texture support for reconstructing the original model's appearance
class RayMarchRenderer {
public:
    bool initialize();
    void shutdown();

    // Upload anchor positions, UVs, patch topology, basis types from M8 bodies
    void update_geometry(const std::vector<m8::BodyRuntime*>& bodies);

    // Load a texture for the model (optional — falls back to patch colors)
    bool load_texture(const std::string& path);
    void set_texture_unit(uint32_t unit) { texture_unit_ = unit; }

    // Render the full-screen ray marching quad
    void draw(ShaderProgram& shader, const Camera& camera, uint32_t width, uint32_t height);

    uint32_t anchor_count() const { return anchor_count_; }
    uint32_t patch_count() const { return patch_count_; }

private:
    GLBuffer vbo_;  // Full-screen quad vertices
    uint32_t anchor_count_ = 0;
    uint32_t patch_count_ = 0;

    // Cached uniform data (uploaded every frame as physics state changes)
    std::vector<float> anchor_data_;     // 3 floats per anchor (xyz)
    std::vector<float> anchor_uv_data_;  // 2 floats per anchor (uv)
    std::vector<int> patch_data_;        // 4 ints per patch (anchor indices)
    std::vector<int> patch_basis_data_;  // 1 int per patch (basis type)
    std::vector<float> patch_amp_data_;  // 1 float per patch (shape amplitude)
    std::vector<float> color_data_;      // 3 floats per patch (RGB fallback)

    // Texture
    uint32_t texture_ = 0;
    bool has_texture_ = false;
    uint32_t texture_unit_ = 0;
};

} // namespace delirium