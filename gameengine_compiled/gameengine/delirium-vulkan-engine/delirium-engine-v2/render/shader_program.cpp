#include "shader_program.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace delirium::render {

std::string ShaderProgram::read_file(const char* path) {
    std::ifstream f(path);
    if (!f) return "";
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

bool ShaderProgram::compile(const char* vp, const char* fp) {
    auto vs = read_file(vp);
    auto fs = read_file(fp);
    if (vs.empty() || fs.empty()) { std::cerr << "[Shader] Cannot read shader files\n"; return false; }

    auto mk = [](const char* src, GLenum t) -> uint32_t {
        uint32_t s = glCreateShader(t);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[512]; glGetShaderInfoLog(s, 512, nullptr, log);
            std::cerr << "[Shader] Compile error: " << log << "\n";
            glDeleteShader(s); return 0;
        }
        return s;
    };

    uint32_t v = mk(vs.c_str(), GL_VERTEX_SHADER);
    uint32_t f = mk(fs.c_str(), GL_FRAGMENT_SHADER);
    if (!v || !f) return false;

    id_ = glCreateProgram();
    glAttachShader(id_, v); glAttachShader(id_, f);
    glLinkProgram(id_);
    GLint ok; glGetProgramiv(id_, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512]; glGetProgramInfoLog(id_, 512, nullptr, log);
        std::cerr << "[Shader] Link error: " << log << "\n";
        return false;
    }
    return true;
}

void ShaderProgram::shutdown() {
    if (id_) { glDeleteProgram(id_); id_ = 0; }
}

} // namespace delirium::render