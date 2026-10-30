#version 430 compatibility

/* ── surfel_point.vert.glsl ────────────────────────────────────────
 *
 * Point sprite vertex shader: transforms surfel positions, sets size.
 * Attributes:
 *   0: position (3 floats)
 *   1: normal (3 floats)
 *   2: color (4 floats)
 *   3: radius (1 float)
 * ────────────────────────────────────────────────────────────────── */

uniform mat4 u_mvp;
uniform float u_point_size;

in vec3 a_position;
in vec3 a_normal;
in vec4 a_color;
in float a_radius;

out vec3 v_normal;
out vec4 v_color;

void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    gl_PointSize = a_radius;
    v_normal = a_normal;
    v_color = a_color;
}