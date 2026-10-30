#include "shader.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

namespace delirium {

std::string ShaderProgram::read_file(const char* path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file) {
        std::cerr << "[Shader] Failed to open: " << path << "\n";
        return {};
    }
    size_t size = file.tellg();
    file.seekg(0);
    std::string buffer(size, '\0');
    file.read(buffer.data(), size);
    return buffer;
}

GLuint ShaderProgram::compile_shader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        std::cerr << "[Shader] Compile error:\n" << log << "\n";
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

bool ShaderProgram::compile(const char* vert_path, const char* frag_path) {
    std::string vert_src = read_file(vert_path);
    std::string frag_src = read_file(frag_path);
    if (vert_src.empty() || frag_src.empty()) return false;

    GLuint vert = compile_shader(GL_VERTEX_SHADER, vert_src.c_str());
    GLuint frag = compile_shader(GL_FRAGMENT_SHADER, frag_src.c_str());
    if (!vert || !frag) return false;

    program_ = glCreateProgram();
    glAttachShader(program_, vert);
    glAttachShader(program_, frag);
    glLinkProgram(program_);

    GLint ok;
    glGetProgramiv(program_, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(program_, 512, nullptr, log);
        std::cerr << "[Shader] Link error:\n" << log << "\n";
        glDeleteProgram(program_);
        program_ = 0;
        return false;
    }

    glDeleteShader(vert);
    glDeleteShader(frag);
    std::cerr << "[Shader] Program compiled OK (id=" << program_ << ")\n";
    return true;
}

void ShaderProgram::use() {
    glUseProgram(program_);
}

void ShaderProgram::shutdown() {
    if (program_) glDeleteProgram(program_);
    program_ = 0;
}

void ShaderProgram::set_mat4(const char* name, const float* m) {
    // Mat4 stores column-major (OpenGL native) — no transpose needed
    glUniformMatrix4fv(glGetUniformLocation(program_, name), 1, GL_FALSE, m);
}

void ShaderProgram::set_vec3(const char* name, float x, float y, float z) {
    glUniform3f(glGetUniformLocation(program_, name), x, y, z);
}

void ShaderProgram::set_vec2(const char* name, float x, float y) {
    glUniform2f(glGetUniformLocation(program_, name), x, y);
}

void ShaderProgram::set_float(const char* name, float v) {
    glUniform1f(glGetUniformLocation(program_, name), v);
}

void ShaderProgram::set_int(const char* name, int v) {
    glUniform1i(glGetUniformLocation(program_, name), v);
}

} // namespace delirium