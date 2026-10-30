#pragma once
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <vector>

namespace delirium {

class GLBuffer {
public:
    void create(GLenum target, const void* data, GLsizeiptr size);
    void create(GLenum target, GLsizeiptr size);
    void update(const void* data, GLsizeiptr size);
    void bind(GLenum target) const;
    void unbind(GLenum target) const;
    void shutdown();

    GLuint id() const { return buffer_; }

private:
    GLuint buffer_ = 0;
};

} // namespace delirium