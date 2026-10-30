#version 430 core

in vec4  v_color;
in vec3  v_normal;
in vec3  v_world_pos;
in float v_radius;

uniform vec3 u_light_dir;
uniform vec3 u_cam_pos;

out vec4 frag_color;

void main() {
    vec2 center = gl_PointCoord - 0.5;
    if (dot(center, center) > 0.25) discard;

    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_dir);
    vec3 V = normalize(u_cam_pos - v_world_pos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 32.0);

    vec3 light_color = vec3(1.0, 0.95, 0.9);
    vec3 ambient = v_color.rgb * 0.3;
    vec3 diffuse = v_color.rgb * diff * 0.7;
    vec3 specular = vec3(1.0) * spec * 0.3;

    vec3 final_color = ambient + diffuse + specular;
    float alpha = 1.0 - smoothstep(0.3, 0.5, length(center));

    frag_color = vec4(final_color, v_color.a * alpha);
}