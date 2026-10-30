#pragma once
#define GL_GLEXT_PROTOTYPES
#include <string>
#include <vector>
#include <GL/gl.h>

namespace delirium {

class ShaderProgram {
public:
    bool compile(const char* vert_path, const char* frag_path);
    void use();
    void shutdown();

    GLuint id() const { return program_; }

    // Uniform helpers
    void set_mat4(const char* name, const float* m);
    void set_vec3(const char* name, float x, float y, float z);
    void set_vec2(const char* name, float x, float y);
    void set_float(const char* name, float v);
    void set_int(const char* name, int v);

private:
    GLuint compile_shader(GLenum type, const char* source);
    std::string read_file(const char* path);

    GLuint program_ = 0;
};

} // namespace delirium