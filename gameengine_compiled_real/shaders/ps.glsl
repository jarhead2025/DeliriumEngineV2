// FILEPATH: shaders/ps.glsl

#version 450 core

layout(std140, binding = 0) buffer SSBO {
    float data[];
};

out vec4 FragColor;

void main() {
    // Example shader using the SSBO data
    int index = gl_FragCoord.x + (gl_FragCoord.y * 800); // Assuming a resolution of 800x600
    FragColor = vec4(data[index], 0.0, 0.0, 1.0);
}
