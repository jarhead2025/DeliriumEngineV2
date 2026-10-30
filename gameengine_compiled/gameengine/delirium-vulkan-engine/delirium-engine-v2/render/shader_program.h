#pragma once
#include <GL/glew.h>
#include <string>
#include <vector>

namespace delirium::render {

/* ── Minimal shader program ─────────────────────────────────────── */
class ShaderProgram {
public:
    bool compile(const char* vert_path, const char* frag_path);
    void use() const { glUseProgram(id_); }
    void shutdown();
    uint32_t id() const { return id_; }

    /* Uniform setters */
    void set_int(const char* name, int v) const {
        GLint l = glGetUniformLocation(id_, name); if (l >= 0) glUniform1i(l, v);
    }
    void set_float(const char* name, float v) const {
        GLint l = glGetUniformLocation(id_, name); if (l >= 0) glUniform1f(l, v);
    }
    void set_vec3(const char* name, float x, float y, float z) const {
        GLint l = glGetUniformLocation(id_, name); if (l >= 0) glUniform3f(l, x, y, z);
    }
    void set_mat4(const char* name, const float* m) const {
        GLint l = glGetUniformLocation(id_, name); if (l >= 0) glUniformMatrix4fv(l, 1, GL_FALSE, m);
    }
    void set_vec2(const char* name, float x, float y) const {
        GLint l = glGetUniformLocation(id_, name); if (l >= 0) glUniform2f(l, x, y);
    }

private:
    uint32_t id_ = 0;
    static std::string read_file(const char* path);
};

/* ── VBO wrapper ────────────────────────────────────────────────── */
class VBO {
public:
    void create(GLenum target, size_t size) {
        glGenBuffers(1, &id_);
        glBindBuffer(target, id_);
        glBufferData(target, size, nullptr, target == GL_ARRAY_BUFFER ? GL_STATIC_DRAW : GL_DYNAMIC_DRAW);
    }
    void update(const void* data, size_t size) {
        glBindBuffer(GL_ARRAY_BUFFER, id_);
        glBufferSubData(GL_ARRAY_BUFFER, 0, size, data);
    }
    void bind(GLenum target) const { glBindBuffer(target, id_); }
    void unbind(GLenum target) const { glBindBuffer(target, 0); }
    void shutdown() { if (id_) { glDeleteBuffers(1, &id_); id_ = 0; } }
    uint32_t id() const { return id_; }
private:
    uint32_t id_ = 0;
};

} // namespace delirium::render