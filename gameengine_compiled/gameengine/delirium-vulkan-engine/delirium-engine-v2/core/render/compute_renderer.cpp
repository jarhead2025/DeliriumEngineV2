#include "compute_renderer.hpp"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <GL/gl.h>

namespace delirium {

static GLuint compile_shader(GLenum type, const char* path) {
    std::ifstream f(path);
    if (!f) { std::cerr << "[Compute] Cannot open " << path << "\n"; return 0; }
    std::string src((std::istreambuf_iterator<char>(f)), {});
    GLuint s = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(s, 1, &c, nullptr);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512]; glGetShaderInfoLog(s, 512, nullptr, log);
        std::cerr << "[Compute] Shader error: " << log << "\n";
        glDeleteShader(s); return 0;
    }
    return s;
}

// Single-path overload: resolves resolve.vert/frag from same dir as compute shader
bool ComputeRenderer::initialize(const char* cs_path, uint32_t w, uint32_t h) {
    std::string cs(cs_path);
    auto pos = cs.rfind('/');
    std::string dir = (pos != std::string::npos) ? cs.substr(0, pos) : ".";
    return initialize(cs_path, (dir + "/resolve.vert.glsl").c_str(),
                      (dir + "/resolve.frag.glsl").c_str(), w, h);
}

bool ComputeRenderer::initialize(const char* cs_path, const char* vert_path, const char* frag_path,
                                  uint32_t w, uint32_t h) {
    // Compute shader
    GLuint cs = compile_shader(GL_COMPUTE_SHADER, cs_path);
    if (!cs) return false;
    compute_prog_ = glCreateProgram();
    glAttachShader(compute_prog_, cs);
    glLinkProgram(compute_prog_);
    glDeleteShader(cs);
    {
        GLint ok; glGetProgramiv(compute_prog_, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[512]; glGetProgramInfoLog(compute_prog_, 512, nullptr, log);
            std::cerr << "[Compute] Link: " << log << "\n"; return false;
        }
    }

    // Render shaders (point sprite vertex + fragment)
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vert_path);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, frag_path);
    if (!vs || !fs) return false;
    render_prog_ = glCreateProgram();
    glAttachShader(render_prog_, vs);
    glAttachShader(render_prog_, fs);
    glLinkProgram(render_prog_);
    glDeleteShader(vs); glDeleteShader(fs);
    {
        GLint ok; glGetProgramiv(render_prog_, GL_LINK_STATUS, &ok);
        if (!ok) {
            char log[512]; glGetProgramInfoLog(render_prog_, 512, nullptr, log);
            std::cerr << "[Render] Link: " << log << "\n"; return false;
        }
    }

    // SSBOs (allocated lazily in upload() when topology is known)
    anchors_buf_.create(GL_SHADER_STORAGE_BUFFER, 0);
    anchors_n_buf_.create(GL_SHADER_STORAGE_BUFFER, 0);
    patches_buf_.create(GL_SHADER_STORAGE_BUFFER, 0);
    coef_buf_.create(GL_SHADER_STORAGE_BUFFER, 0);
    mat_buf_.create(GL_SHADER_STORAGE_BUFFER, 0);
    meta_buf_.create(GL_SHADER_STORAGE_BUFFER, 6*sizeof(uint32_t));

    // Surfel output buffer (compute writes, render reads)
    surfel_buf_.create(GL_SHADER_STORAGE_BUFFER, 0);
    max_surfels_ = 512 * 512;  // up to 262K surfels
    surfel_buf_.update(nullptr, max_surfels_ * sizeof(SurfelVertex));

    // Counter buffer — atomic counter for actual surfel count
    counter_buf_.create(GL_SHADER_STORAGE_BUFFER, sizeof(uint32_t));
    uint32_t zero = 0;
    counter_buf_.update(&zero, sizeof(uint32_t));

    // VAO/VBO for point sprite rendering — VAO reads directly from surfel_buf_
    glGenVertexArrays(1, &surfel_vao_);
    glBindVertexArray(surfel_vao_);
    // Bind the SSBO as ARRAY_BUFFER so VAO reads compute output
    glBindBuffer(GL_ARRAY_BUFFER, surfel_buf_.id());
    // Layout: pos(3) + normal(3) + color(4) + radius(1) = 11 floats
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(SurfelVertex), (void*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(SurfelVertex), (void*)(3*4));
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(SurfelVertex), (void*)(6*4));
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(SurfelVertex), (void*)(10*4));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);
    glBindVertexArray(0);

    win_w_ = w; win_h_ = h;
    std::cout << "[Compute] Initialized (" << w << "x" << h << "), " << max_surfels_ << " max surfels\n";
    return true;
}

void ComputeRenderer::shutdown() {
    anchors_buf_.shutdown(); anchors_n_buf_.shutdown(); patches_buf_.shutdown();
    coef_buf_.shutdown(); mat_buf_.shutdown(); meta_buf_.shutdown();
    surfel_buf_.shutdown();
    if (surfel_vao_) glDeleteVertexArrays(1, &surfel_vao_);
    if (compute_prog_) glDeleteProgram(compute_prog_);
    if (render_prog_) glDeleteProgram(render_prog_);
}

void ComputeRenderer::resize(uint32_t w, uint32_t h) {
    win_w_ = w; win_h_ = h;
}

/* ── Adaptive sample count (mirrors SplatRenderer.cpp anisotropic grid) ── */
uint32_t ComputeRenderer::compute_samples(uint32_t np, const float* mvp,
                                           const float* anchor_data, uint32_t na) const {
    // Heuristic: project center to screen, estimate extent
    float cx = 0, cy = 0, cz = 0;
    for (uint32_t i = 0; i < na; ++i) {
        cx += anchor_data[i*3+0];
        cy += anchor_data[i*3+1];
        cz += anchor_data[i*3+2];
    }
    cx /= (float)na; cy /= (float)na; cz /= (float)na;

    // Project center (column-major MVP)
    float pw = mvp[3]*cx + mvp[7]*cy + mvp[11]*cz + mvp[15];
    if (pw <= 0.0f) {
        static bool warned = true;
        if (warned) { warned = false; std::cerr << "[Compute] pw=" << pw << " falling back to default ss\n"; }
        return 32;  // fallback: 32 samples/edge
    }

    // Project a point at bounding radius
    float radius = 1.0f;
    float qx = mvp[0]*(cx+radius) + mvp[4]*cy + mvp[8]*cz + mvp[12];
    float qw = mvp[3]*(cx+radius) + mvp[7]*cy + mvp[11]*cz + mvp[15];
    float px = mvp[0]*cx + mvp[4]*cy + mvp[8]*cz + mvp[12];

    float ndc_extent = std::abs((qx/qw) - (px/pw));
    float screen_px = ndc_extent * 0.5f * (float)win_w_;

    // Target: 5 samples per pixel of projected extent (denser coverage)
    uint32_t ss = (uint32_t)std::sqrt(screen_px * 5.0f);
    ss = std::max(16u, std::min(128u, ss));
    static bool first_diag = true;
    if (first_diag) {
        first_diag = false;
        std::cout << "[Compute] screen_px=" << screen_px << " ss=" << ss << " np=" << np << " total=" << (uint64_t)np*ss*ss << "\n";
    }
    return ss;
}

void ComputeRenderer::upload(BodyRuntime* body) {
    const auto& state = body->read_state();
    const auto& topo  = body->topology();
    const auto& mats  = body->materials();

    uint32_t na = (uint32_t)state.anchors.size();
    uint32_t np = (uint32_t)topo.patch_count();
    bool dirty = topo_dirty_ || na != na_ || np != np_;

    if (dirty) {
        na_ = na; np_ = np;
        anchor_data_.resize(na * 3);
        anchor_n_data_.resize(na * 3);
        patch_data_.resize(np * 4);
        coef_data_.resize(np * 8);
        mat_data_.resize(np * 4);

        for (uint32_t i = 0; i < np; ++i) {
            const auto& pt = topo.patch(i);
            for (int j = 0; j < 4; ++j)
                patch_data_[i*4+j] = (int)pt.anchors[j];
            auto m = mats.get(topo.patch(i).material);
            mat_data_[i*4+0] = m.r/255.0f;
            mat_data_[i*4+1] = m.g/255.0f;
            mat_data_[i*4+2] = m.b/255.0f;
            mat_data_[i*4+3] = 1.0f;
        }
        patches_buf_.update(patch_data_.data(), np*4*sizeof(int));
        mat_buf_.update(mat_data_.data(), np*4*sizeof(float));
        topo_dirty_ = false;
    }

    // Per-frame: anchors + normals + coefs
    for (uint32_t i = 0; i < na; ++i) {
        anchor_data_[i*3+0] = state.anchors[i].position.x;
        anchor_data_[i*3+1] = state.anchors[i].position.y;
        anchor_data_[i*3+2] = state.anchors[i].position.z;
        anchor_n_data_[i*3+0] = state.anchors[i].normal.x;
        anchor_n_data_[i*3+1] = state.anchors[i].normal.y;
        anchor_n_data_[i*3+2] = state.anchors[i].normal.z;
    }
    anchors_buf_.update(anchor_data_.data(), na*3*sizeof(float));
    anchors_n_buf_.update(anchor_n_data_.data(), na*3*sizeof(float));

    for (uint32_t i = 0; i < np; ++i)
        for (int j = 0; j < 8; ++j)
            coef_data_[i*8+j] = state.patches[i].coefficients[j];
    coef_buf_.update(coef_data_.data(), np*8*sizeof(float));

    // One-time diagnostic: verify CPU-side data is non-zero
    static bool upload_diag = true;
    if (upload_diag) {
        upload_diag = false;
        std::cout << "[Upload] First anchor=(" << anchor_data_[0] << "," << anchor_data_[1] << "," << anchor_data_[2] << ")\\n";
        std::cout << "[Upload] First coef=(" << coef_data_[0] << "," << coef_data_[1] << "," << coef_data_[2] << ")\\n";
    }
}

void ComputeRenderer::render(Vec3 cam_pos, const float* view, const float* proj,
                              uint32_t width, uint32_t height) {
    if (!compute_prog_ || np_ == 0) return;

    // Build MVP = proj * view (column-major)
    float mvp[16] = {};
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
            for (int k = 0; k < 4; ++k)
                mvp[c*4+r] += proj[c*4+k] * view[k*4+r];

    // Normalize light direction
    float lx = 0.5f, ly = 1.0f, lz = 0.3f;
    float llen = std::sqrt(lx*lx + ly*ly + lz*lz);
    lx /= llen; ly /= llen; lz /= llen;

    // ── Compute pass: generate surfels ──────────────────────────────
    glUseProgram(compute_prog_);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, anchors_buf_.id());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, anchors_n_buf_.id());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, patches_buf_.id());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, coef_buf_.id());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, mat_buf_.id());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, meta_buf_.id());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, surfel_buf_.id());
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, counter_buf_.id());

    auto uloc = [this](const char* n) { return glGetUniformLocation(compute_prog_, n); };
    glUniformMatrix4fv(uloc("u_mvp"), 1, GL_FALSE, mvp);
    glUniform3f(uloc("u_cam_pos"), cam_pos.x, cam_pos.y, cam_pos.z);
    glUniform3f(uloc("u_light_dir"), lx, ly, lz);
    glUniform1ui(uloc("u_width"), width);
    glUniform1ui(uloc("u_height"), height);
    glUniform1f(uloc("u_bbox_diag"), bbox_diag_);

    uint32_t ss = compute_samples(np_, mvp, anchor_data_.data(), na_);
    uint32_t total = np_ * ss * ss;
    if (total > max_surfels_) total = max_surfels_;
    uint32_t groups = (total + 255) / 256;

    meta_[0] = na_; meta_[1] = np_; meta_[2] = ss;
    meta_[3] = total; meta_[4] = 0;  // surfel_count reset
    meta_buf_.update(meta_, 6*sizeof(uint32_t));

    glDispatchCompute(groups, 1, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);

    // Read back actual surfel count from atomic counter
    {
        uint32_t actual = 0;
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, counter_buf_.id());
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(uint32_t), &actual);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        surfel_count_ = actual;
    }

    // Diagnostic: read back first surfel to verify compute output
    static bool first_frame = true;
    if (first_frame) {
        first_frame = false;
        glFinish();

        std::cout << "[Compute] Dispatched " << groups << " groups, " << total << " work items, " << surfel_count_ << " surfels written, " << ss << " samples/edge\\n";
        std::cout << "[Compute] na=" << na_ << " np=" << np_ << " ss=" << ss << "\\n";

        float check[11] = {};
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, surfel_buf_.id());
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 44, check);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        std::cout << "[Compute] First surfel pos=(" << check[0] << "," << check[1] << "," << check[2] << ")\\n";
        std::cout << "[Compute] First surfel radius=" << check[10] << "\\n";

        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            std::cerr << "[Compute] GL error after dispatch: 0x" << std::hex << err << std::dec << "\\n";
        }
    }

    // ── Render pass: point sprites with hardware depth test ─────────
    glUseProgram(render_prog_);

    // MVP uniform for render
    auto rloc = [this](const char* n) { return glGetUniformLocation(render_prog_, n); };
    glUniformMatrix4fv(rloc("u_mvp"), 1, GL_FALSE, mvp);
    glUniform3f(rloc("u_light_dir"), lx, ly, lz);
    glUniform3f(rloc("u_cam_pos"), cam_pos.x, cam_pos.y, cam_pos.z);
    glUniform1f(rloc("u_point_size"), 3.0f);

    // Bind surfel buffer as vertex array source (VAO already bound to surfel_buf_)
    glBindVertexArray(surfel_vao_);

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);  // REQUIRED: allows vertex shader to set gl_PointSize
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glDrawArrays(GL_POINTS, 0, (GLsizei)surfel_count_);

    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

    glDisable(GL_BLEND);
    glBindVertexArray(0);
}

} // namespace delirium