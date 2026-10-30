#version 430 core

// Full-screen quad vertex shader — no geometry, just a screen-filling triangle pair.
// The real work happens in the fragment shader via ray marching.

layout(location = 0) in vec2 a_pos;  // xy in [-1, 1] clip space, zw ignored

void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
}