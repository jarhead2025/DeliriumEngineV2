#include "buffer.hpp"

namespace delirium {

void GLBuffer::create(GLenum target, const void* data, GLsizeiptr size) {
    if (buffer_) glDeleteBuffers(1, &buffer_);
    glGenBuffers(1, &buffer_);
    glBindBuffer(target, buffer_);
    glBufferData(target, size, data, GL_DYNAMIC_DRAW);
    glBindBuffer(target, 0);
}

void GLBuffer::create(GLenum target, GLsizeiptr size) {
    if (buffer_) glDeleteBuffers(1, &buffer_);
    glGenBuffers(1, &buffer_);
    glBindBuffer(target, buffer_);
    glBufferData(target, size, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(target, 0);
}

void GLBuffer::update(const void* data, GLsizeiptr size) {
    glBindBuffer(GL_ARRAY_BUFFER, buffer_);
    glBufferSubData(GL_ARRAY_BUFFER, 0, size, data);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLBuffer::bind(GLenum target) const {
    glBindBuffer(target, buffer_);
}

void GLBuffer::unbind(GLenum target) const {
    glBindBuffer(target, 0);
}

void GLBuffer::shutdown() {
    if (buffer_) glDeleteBuffers(1, &buffer_);
    buffer_ = 0;
}

} // namespace delirium