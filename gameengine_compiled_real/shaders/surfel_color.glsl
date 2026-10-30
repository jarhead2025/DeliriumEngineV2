// FILEPATH: shaders/surfel_color.glsl

#version 330 core

out vec4 FragColor;

uniform float time;
uniform sampler1D colorMap;

void main() {
    float t = fract(time * 0.5);
    vec3 color = texture(colorMap, t).rgb;
    FragColor = vec4(color, 1.0);
}
