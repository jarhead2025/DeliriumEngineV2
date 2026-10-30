#pragma once
#include "math/mat4.hpp"
#include "opengl/shader.hpp"
#include "opengl/buffer.hpp"
#include "delirium_m8/body_runtime.hpp"
#include "delirium_m8/surfel_cache.hpp"
#include <vector>

namespace delirium {

// GPU surfel layout: pos(3) norm(3) color(4) radius(1) = 11 floats per surfel
constexpr int SURFEL_STRIDE = 11;

class SurfelRenderer {
public:
    bool initialize();
    void shutdown();

    // Upload surfels from M8's SurfelCache to GPU
    void update_surfels(const std::vector<m8::BodyRuntime*>& bodies,
                        const Vec3& camera_pos);
    void draw(ShaderProgram& shader, uint32_t width, uint32_t height);

    uint32_t surfel_count() const { return surfel_count_; }

private:
    GLBuffer vbo_;
    uint32_t surfel_count_ = 0;
    uint32_t max_surfels_ = 200000;
};

} // namespace delirium