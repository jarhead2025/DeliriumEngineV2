#include "raymarch_renderer.hpp"
#define GL_GLEXT_PROTOTYPES
#include <cstring>
#include <iostream>
#include <fstream>
#include <GL/gl.h>

namespace delirium {

static uint32_t generate_checkerboard_texture() {
    const int size = 256;
    std::vector<uint8_t> data(size * size * 4);
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int tx = x / 32, ty = y / 32;
            bool white = ((tx + ty) % 2) == 0;
            int idx = (y * size + x) * 4;
            if (white) { data[idx]=200; data[idx+1]=190; data[idx+2]=180; data[idx+3]=255; }
            else { data[idx]=100; data[idx+1]=95; data[idx+2]=90; data[idx+3]=255; }
        }
    }
    uint32_t tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
    glGenerateMipmap(GL_TEXTURE_2D);
    return tex;
}

bool RayMarchRenderer::initialize() {
    static const float quad_vertices[6 * 2] = {
        -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
    };
    vbo_.create(GL_ARRAY_BUFFER, sizeof(quad_vertices));
    vbo_.update(quad_vertices, sizeof(quad_vertices));
    texture_ = generate_checkerboard_texture();
    has_texture_ = true;
    return true;
}

void RayMarchRenderer::shutdown() {
    vbo_.shutdown();
    if (texture_) { glDeleteTextures(1, &texture_); texture_ = 0; }
}

void RayMarchRenderer::update_geometry(BodyRuntime* body) {
    if (!body) return;

    const auto& state = body->read_state();
    const auto& topology = body->topology();

    // Resize buffers on topology change
    if (anchor_data_.size() != state.anchors.size() * 3) {
        anchor_data_.resize(state.anchors.size() * 3);
        anchor_count_ = (uint32_t)state.anchors.size();
    }
    if (patch_data_.size() != topology.patch_count() * 4) {
        patch_data_.resize(topology.patch_count() * 4);
        patch_basis_data_.resize(topology.patch_count());
        patch_coef_data_.resize(topology.patch_count() * 8);
        color_data_.resize(topology.patch_count() * 3);
        patch_count_ = (uint32_t)topology.patch_count();
        // Static topology data (upload once)
        for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
            const auto& patch = topology.patch(pid);
            patch_data_[pid * 4 + 0] = (int)patch.anchors[0];
            patch_data_[pid * 4 + 1] = (int)patch.anchors[1];
            patch_data_[pid * 4 + 2] = (int)patch.anchors[2];
            patch_data_[pid * 4 + 3] = (int)patch.anchors[3];
            patch_basis_data_[pid] = (int)(uint8_t)patch.basis;
            const auto& mat = body->materials().get(patch.material);
            color_data_[pid * 3 + 0] = mat.r / 255.0f;
            color_data_[pid * 3 + 1] = mat.g / 255.0f;
            color_data_[pid * 3 + 2] = mat.b / 255.0f;
        }
        topology_dirty_ = true;
    }

    // Dynamic anchor positions (upload every frame)
    for (std::size_t i = 0; i < state.anchors.size(); ++i) {
        anchor_data_[i * 3 + 0] = state.anchors[i].position.x;
        anchor_data_[i * 3 + 1] = state.anchors[i].position.y;
        anchor_data_[i * 3 + 2] = state.anchors[i].position.z;
    }

    // Dynamic coefficients from STATE (not topology — modal deformation lives here)
    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        for (int j = 0; j < 8; ++j) {
            patch_coef_data_[pid * 8 + j] = state.patches[pid].coefficients[j];
        }
    }
}

void RayMarchRenderer::draw(ShaderProgram& shader, Vec3 cam_pos, const float* view_inv,
                            const float* proj_inv, uint32_t width, uint32_t height) {
    if (anchor_count_ == 0 || patch_count_ == 0) return;
    shader.use();
    shader.set_vec3("u_camera_pos", cam_pos.x, cam_pos.y, cam_pos.z);
    shader.set_mat4("u_view_inv", view_inv);
    shader.set_mat4("u_proj_inv", proj_inv);
    shader.set_vec2("u_resolution", (float)width, (float)height);
    shader.set_int("u_anchor_count", (int)anchor_count_);
    shader.set_int("u_patch_count", (int)patch_count_);

    // Dynamic: anchor positions every frame
    GLint loc_anchors = glGetUniformLocation(shader.id(), "u_anchors");
    if (loc_anchors >= 0) glUniform3fv(loc_anchors, (GLsizei)anchor_count_, anchor_data_.data());

    // Static: topology only when dirty
    if (topology_dirty_) {
        GLint loc_patches = glGetUniformLocation(shader.id(), "u_patches");
        if (loc_patches >= 0) glUniform4iv(loc_patches, (GLsizei)patch_count_, patch_data_.data());
        GLint loc_basis = glGetUniformLocation(shader.id(), "u_patch_basis");
        if (loc_basis >= 0) glUniform1iv(loc_basis, (GLsizei)patch_count_, patch_basis_data_.data());
        GLint loc_colors = glGetUniformLocation(shader.id(), "u_patch_colors");
        if (loc_colors >= 0) glUniform3fv(loc_colors, (GLsizei)patch_count_, color_data_.data());
        topology_dirty_ = false;
    }

    // Dynamic: coefficients every frame (modal deformation)
    GLint loc_amp = glGetUniformLocation(shader.id(), "u_patch_coef");
    if (loc_amp >= 0) glUniform4fv(loc_amp, (GLsizei)(patch_coef_data_.size() / 4), patch_coef_data_.data());

    shader.set_int("u_has_texture", has_texture_ ? 1 : 0);
    if (has_texture_) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_);
        shader.set_int("u_texture", 0);
    }

    vbo_.bind(GL_ARRAY_BUFFER);
    GLint loc_pos = glGetAttribLocation(shader.id(), "a_pos");
    if (loc_pos >= 0) {
        glEnableVertexAttribArray(loc_pos);
        glVertexAttribPointer(loc_pos, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    }
    glDrawArrays(GL_TRIANGLES, 0, 6);
    if (loc_pos >= 0) glDisableVertexAttribArray(loc_pos);
    vbo_.unbind(GL_ARRAY_BUFFER);
}

} // namespace delirium