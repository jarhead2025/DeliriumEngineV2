#include "raymarch_renderer.hpp"
#define GL_GLEXT_PROTOTYPES
#include <cstring>
#include <iostream>
#include <fstream>
#include <GL/gl.h>

namespace delirium {

// ─── Simple DXT/RAW texture loader (no stb_image dependency) ─────
// Loads a raw RGBA texture file or falls back to a generated checkerboard.
// For production, replace with stb_image or SOIL.
static uint32_t load_raw_texture(const std::string& path) {
    // Try to load as raw RGBA (width=256, height=256, 4 bytes per pixel)
    // This is a minimal loader — replace with stb_image for production
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return 0;

    // Read file size
    file.seekg(0, std::ios::end);
    std::size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Try common texture sizes (256x256, 512x512, 1024x1024)
    int width = 0, height = 0;
    for (int s : {1024, 512, 256, 128, 64}) {
        if (file_size == s * s * 4) { width = s; height = s; break; }
        if (file_size == s * s * 3) { width = s; height = s; break; }
    }

    if (width == 0) {
        std::cerr << "[Texture] Unsupported format: " << path << " (" << file_size << " bytes)\n";
        return 0;
    }

    bool has_alpha = (file_size == width * height * 4);
    std::vector<uint8_t> data(file_size);
    file.read((char*)data.data(), file_size);

    uint32_t tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    if (has_alpha) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
    } else {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data.data());
    }
    glGenerateMipmap(GL_TEXTURE_2D);

    std::cout << "[Texture] Loaded: " << path << " (" << width << "x" << height << ")\n";
    return tex;
}

// ─── Generate a checkerboard texture as fallback ─────────────────
static uint32_t generate_checkerboard_texture() {
    const int size = 256;
    std::vector<uint8_t> data(size * size * 4);

    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            int tx = x / 32;
            int ty = y / 32;
            bool white = ((tx + ty) % 2) == 0;

            int idx = (y * size + x) * 4;
            if (white) {
                data[idx] = 200; data[idx+1] = 190; data[idx+2] = 180; data[idx+3] = 255;
            } else {
                data[idx] = 100; data[idx+1] = 95; data[idx+2] = 90; data[idx+3] = 255;
            }
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

// ─── RayMarchRenderer ────────────────────────────────────────────
bool RayMarchRenderer::initialize() {
    // Full-screen quad: 2 triangles covering clip space [-1, 1]
    static const float quad_vertices[6 * 2] = {
        -1.0f, -1.0f,
         1.0f, -1.0f,
         1.0f,  1.0f,
        -1.0f, -1.0f,
         1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    vbo_.create(GL_ARRAY_BUFFER, sizeof(quad_vertices));
    vbo_.update(quad_vertices, sizeof(quad_vertices));

    // Generate a default checkerboard texture
    texture_ = generate_checkerboard_texture();
    has_texture_ = true;

    return true;
}

void RayMarchRenderer::shutdown() {
    vbo_.shutdown();
    if (texture_) {
        glDeleteTextures(1, &texture_);
        texture_ = 0;
    }
}

bool RayMarchRenderer::load_texture(const std::string& path) {
    if (texture_) {
        glDeleteTextures(1, &texture_);
        texture_ = 0;
    }

    texture_ = load_raw_texture(path);
    has_texture_ = (texture_ != 0);

    if (!has_texture_) {
        std::cerr << "[Texture] Failed to load: " << path << ", using checkerboard fallback\n";
        texture_ = generate_checkerboard_texture();
        has_texture_ = true;
    }

    return has_texture_;
}

void RayMarchRenderer::update_geometry(const std::vector<m8::BodyRuntime*>& bodies) {
    anchor_data_.clear();
    anchor_uv_data_.clear();
    patch_data_.clear();
    patch_basis_data_.clear();
    patch_amp_data_.clear();
    color_data_.clear();
    anchor_count_ = 0;
    patch_count_ = 0;

    for (auto* body : bodies) {
        if (!body) continue;

        const auto& state = body->read_state();
        const auto& topo = body->topology();
        const auto& mats = body->materials();

        // Upload anchor positions + UVs
        for (const auto& anchor : state.anchors) {
            anchor_data_.push_back(anchor.position.x);
            anchor_data_.push_back(anchor.position.y);
            anchor_data_.push_back(anchor.position.z);

            anchor_uv_data_.push_back(anchor.uv_u);
            anchor_uv_data_.push_back(anchor.uv_v);

            anchor_count_++;
        }

        // Upload patch connectivity, basis types, and colors
        for (m8::PatchId pid = 0; pid < topo.patch_count(); ++pid) {
            const auto& patch = topo.patch(pid);

            // Patch anchor indices (4 per patch)
            patch_data_.push_back(patch.anchors[0]);
            patch_data_.push_back(patch.anchors[1]);
            patch_data_.push_back(patch.anchors[2]);
            patch_data_.push_back(patch.anchors[3]);

            // Basis type (0-7)
            int basis_int = static_cast<int>(static_cast<std::underlying_type_t<m8::BasisType>>(patch.basis));
            patch_basis_data_.push_back(basis_int);

            // Shape amplitude: compute from coefficients
            // Use the max residual magnitude as the amplitude
            float max_residual = 0.0f;
            for (int i = 0; i < 4; i++) {
                float r = patch.coefficients.residuals[i].x * patch.coefficients.residuals[i].x +
                          patch.coefficients.residuals[i].y * patch.coefficients.residuals[i].y +
                          patch.coefficients.residuals[i].z * patch.coefficients.residuals[i].z;
                if (r > max_residual) max_residual = r;
            }
            max_residual = std::sqrt(max_residual);

            // Also use scalar coefficients if present
            float scalar_amp = 0.0f;
            for (int i = 0; i < 8; i++) {
                if (std::abs(patch.coefficients.scalars[i]) > scalar_amp)
                    scalar_amp = std::abs(patch.coefficients.scalars[i]);
            }

            // Amplitude is the max of residual magnitude and scalar coefficient
            float amplitude = std::max(max_residual, scalar_amp);
            patch_amp_data_.push_back(amplitude);

            // Patch color from material
            if (mats.has(patch.material)) {
                const auto& mat = mats.get(patch.material);
                color_data_.push_back(static_cast<float>(mat.r) / 255.0f);
                color_data_.push_back(static_cast<float>(mat.g) / 255.0f);
                color_data_.push_back(static_cast<float>(mat.b) / 255.0f);
            } else {
                color_data_.push_back(0.8f);
                color_data_.push_back(0.8f);
                color_data_.push_back(0.8f);
            }
            patch_count_++;
        }
    }
}

void RayMarchRenderer::draw(ShaderProgram& shader, const Camera& camera,
                            uint32_t width, uint32_t height) {
    if (anchor_count_ == 0 || patch_count_ == 0) return;

    shader.use();

    // Camera uniforms
    shader.set_vec3("u_camera_pos", camera.position().x, camera.position().y, camera.position().z);
    shader.set_mat4("u_view_inv", camera.view_inv().m.data());
    shader.set_mat4("u_proj_inv", camera.proj_inv().m.data());
    shader.set_vec2("u_resolution", (float)width, (float)height);

    // Geometry uniforms (implicit surface definition)
    shader.set_int("u_anchor_count", (int)anchor_count_);
    shader.set_int("u_patch_count", (int)patch_count_);

    // Upload anchor positions as uniform array
    GLint loc_anchors = glGetUniformLocation(shader.id(), "u_anchors");
    if (loc_anchors >= 0) {
        glUniform3fv(loc_anchors, (GLsizei)anchor_count_, anchor_data_.data());
    }

    // Upload anchor UVs as uniform array
    GLint loc_uvs = glGetUniformLocation(shader.id(), "u_anchor_uvs");
    if (loc_uvs >= 0) {
        glUniform2fv(loc_uvs, (GLsizei)anchor_count_, anchor_uv_data_.data());
    }

    // Upload patch connectivity as uniform array
    GLint loc_patches = glGetUniformLocation(shader.id(), "u_patches");
    if (loc_patches >= 0) {
        glUniform4iv(loc_patches, (GLsizei)patch_count_, patch_data_.data());
    }

    // Upload patch basis types
    GLint loc_basis = glGetUniformLocation(shader.id(), "u_patch_basis");
    if (loc_basis >= 0) {
        glUniform1iv(loc_basis, (GLsizei)patch_count_, patch_basis_data_.data());
    }

    // Upload patch amplitudes
    GLint loc_amp = glGetUniformLocation(shader.id(), "u_patch_amplitude");
    if (loc_amp >= 0) {
        glUniform1fv(loc_amp, (GLsizei)patch_count_, patch_amp_data_.data());
    }

    // Upload patch colors as uniform array
    GLint loc_colors = glGetUniformLocation(shader.id(), "u_patch_colors");
    if (loc_colors >= 0) {
        glUniform3fv(loc_colors, (GLsizei)patch_count_, color_data_.data());
    }

    // Texture
    shader.set_int("u_has_texture", has_texture_ ? 1 : 0);
    if (has_texture_) {
        glActiveTexture(GL_TEXTURE0 + texture_unit_);
        glBindTexture(GL_TEXTURE_2D, texture_);
        shader.set_int("u_texture", (int)texture_unit_);
    }

    // Draw full-screen quad
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