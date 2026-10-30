#include "surfel_renderer.hpp"
#define GL_GLEXT_PROTOTYPES
#include <cstring>
#include <iostream>
#include <GL/gl.h>

namespace delirium {

bool SurfelRenderer::initialize() {
    vbo_.create(GL_ARRAY_BUFFER, max_surfels_ * SURFEL_STRIDE * sizeof(float));
    return true;
}

void SurfelRenderer::shutdown() {
    vbo_.shutdown();
}

void SurfelRenderer::update_surfels(const std::vector<m8::BodyRuntime*>& bodies,
                                     const Vec3& camera_pos) {
    std::vector<float> surfel_data;
    surfel_data.reserve(max_surfels_ * SURFEL_STRIDE);
    surfel_count_ = 0;

    for (auto* body : bodies) {
        if (!body) continue;

        // Use M8's built-in surfel cache with LOD
        m8::SurfelCache cache;
        Vec3 bc{body->bounding_center().x, body->bounding_center().y, body->bounding_center().z};
        float dist = length(camera_pos - bc);

        std::string error;
        body->rebuild_cache_with_lod(cache, dist, &error);
        if (!error.empty()) {
            std::cerr << "[Surfel] Cache rebuild warning: " << error << "\n";
        }

        const auto& surfels = cache.surfels();
        for (const auto& s : surfels) {
            if (static_cast<uint32_t>(surfel_data.size()) >= max_surfels_ * SURFEL_STRIDE) break;

            // Position
            surfel_data.push_back(s.position.x);
            surfel_data.push_back(s.position.y);
            surfel_data.push_back(s.position.z);

            // Normal (computed from basis evaluation)
            surfel_data.push_back(s.normal.x);
            surfel_data.push_back(s.normal.y);
            surfel_data.push_back(s.normal.z);

            // Color (from material, with alpha for blending)
            surfel_data.push_back(static_cast<float>(s.r) / 255.0f);
            surfel_data.push_back(static_cast<float>(s.g) / 255.0f);
            surfel_data.push_back(static_cast<float>(s.b) / 255.0f);
            surfel_data.push_back(static_cast<float>(s.a) / 255.0f);

            // Radius
            surfel_data.push_back(s.radius);

            surfel_count_++;
        }
    }

    if (surfel_count_ == 0) return;

    vbo_.update(surfel_data.data(), surfel_data.size() * sizeof(float));
}

void SurfelRenderer::draw(ShaderProgram& shader, uint32_t width, uint32_t height) {
    if (surfel_count_ == 0) return;

    shader.use();

    // Set point size based on viewport — larger for closer camera
    glPointSize(6.0f);

    vbo_.bind(GL_ARRAY_BUFFER);

    // Layout: pos(3) norm(3) color(4) radius(1) = 11 floats
    GLsizei stride = SURFEL_STRIDE * sizeof(float);

    // Position: offset 0, 3 floats
    GLint loc_pos = glGetAttribLocation(shader.id(), "a_pos");
    if (loc_pos >= 0) {
        glEnableVertexAttribArray(loc_pos);
        glVertexAttribPointer(loc_pos, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    }

    // Normal: offset 3, 3 floats
    GLint loc_norm = glGetAttribLocation(shader.id(), "a_normal");
    if (loc_norm >= 0) {
        glEnableVertexAttribArray(loc_norm);
        glVertexAttribPointer(loc_norm, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
    }

    // Color: offset 6, 4 floats
    GLint loc_color = glGetAttribLocation(shader.id(), "a_color");
    if (loc_color >= 0) {
        glEnableVertexAttribArray(loc_color);
        glVertexAttribPointer(loc_color, 4, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));
    }

    // Radius: offset 10, 1 float
    GLint loc_radius = glGetAttribLocation(shader.id(), "a_radius");
    if (loc_radius >= 0) {
        glEnableVertexAttribArray(loc_radius);
        glVertexAttribPointer(loc_radius, 1, GL_FLOAT, GL_FALSE, stride, (void*)(10 * sizeof(float)));
    }

    glDrawArrays(GL_POINTS, 0, (GLsizei)surfel_count_);

    // Clean up attribs
    if (loc_pos >= 0) glDisableVertexAttribArray(loc_pos);
    if (loc_norm >= 0) glDisableVertexAttribArray(loc_norm);
    if (loc_color >= 0) glDisableVertexAttribArray(loc_color);
    if (loc_radius >= 0) glDisableVertexAttribArray(loc_radius);

    vbo_.unbind(GL_ARRAY_BUFFER);
}

} // namespace delirium