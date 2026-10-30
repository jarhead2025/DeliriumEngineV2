#version 430 core

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec4 a_color;
layout(location = 3) in float a_radius;

uniform mat4 u_mvp;
uniform vec3 u_cam_pos;

out vec4 v_color;
out vec3 v_normal;
out vec3 v_world_pos;
out float v_radius;

void main() {
    gl_Position = u_mvp * vec4(a_pos, 1.0);
    gl_PointSize = max(1.0, a_radius * 200.0 / gl_Position.w);

    v_color = a_color;
    v_normal = a_normal;
    v_world_pos = a_pos;
    v_radius = a_radius;
}