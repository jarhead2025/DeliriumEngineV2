#pragma once
#include "../core/types.h"
#include "shader_program.h"
#include <vector>

namespace delirium::render {

/* ── Surfel point sprite renderer ──────────────────────────────────
 * Consumes surfels from BodyRuntime cache → VBO → GL_POINTS
 * Layout per surfel: pos(3) norm(3) color(4) radius(1) = 11 floats
 */
class SurfelRenderer {
public:
    static constexpr int STRIDE = 11;
    static constexpr int MAX_SURFELS = 100000;

    bool initialize();
    void shutdown();

    void update_surfels(const std::vector<Surfel>& surfels);
    void draw(ShaderProgram& shader) const;

    int count() const { return count_; }

private:
    VBO vbo_;
    int count_ = 0;
};

} // namespace delirium::render