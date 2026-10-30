#include "surfel_renderer.h"
#include <GL/glew.h>
#include <algorithm>

namespace delirium::render {

bool SurfelRenderer::initialize() {
    vbo_.create(GL_ARRAY_BUFFER, MAX_SURFELS * STRIDE * sizeof(float));
    glEnable(GL_PROGRAM_POINT_SIZE);  // let vertex shader set gl_PointSize
    return true;
}

void SurfelRenderer::shutdown() { vbo_.shutdown(); }

void SurfelRenderer::update_surfels(const std::vector<Surfel>& surfels) {
    if (surfels.empty()) { count_ = 0; return; }

    std::vector<float> data;
    data.reserve(std::min(surfels.size(), (size_t)MAX_SURFELS) * STRIDE);

    for (const auto& s : surfels) {
        if ((int)data.size() >= MAX_SURFELS * STRIDE) break;
        data.push_back(s.position.x);
        data.push_back(s.position.y);
        data.push_back(s.position.z);
        data.push_back(s.normal.x);
        data.push_back(s.normal.y);
        data.push_back(s.normal.z);
        data.push_back(s.r / 255.0f);
        data.push_back(s.g / 255.0f);
        data.push_back(s.b / 255.0f);
        data.push_back(s.a / 255.0f);
        data.push_back(s.radius);
    }
    count_ = (int)(data.size() / STRIDE);
    vbo_.update(data.data(), data.size() * sizeof(float));
}

void SurfelRenderer::draw(ShaderProgram& shader) const {
    if (count_ == 0) return;
    shader.use();
    glPointSize(6.0f);
    vbo_.bind(GL_ARRAY_BUFFER);

    auto bind_attr = [&](const char* name, int comps, float off) {
        GLint l = glGetAttribLocation(shader.id(), name);
        if (l < 0) return;
        glEnableVertexAttribArray(l);
        glVertexAttribPointer(l, comps, GL_FLOAT, GL_FALSE,
            STRIDE * sizeof(float), (void*)static_cast<std::size_t>(off * sizeof(float)));
    };
    auto unbind_attr = [&](const char* name) {
        GLint l = glGetAttribLocation(shader.id(), name);
        if (l >= 0) glDisableVertexAttribArray(l);
    };

    bind_attr("a_pos", 3, 0);
    bind_attr("a_normal", 3, 3);
    bind_attr("a_color", 4, 6);
    bind_attr("a_radius", 1, 10);

    glDrawArrays(GL_POINTS, 0, count_);

    unbind_attr("a_pos");
    unbind_attr("a_normal");
    unbind_attr("a_color");
    unbind_attr("a_radius");

    vbo_.unbind(GL_ARRAY_BUFFER);
}

} // namespace delirium::render