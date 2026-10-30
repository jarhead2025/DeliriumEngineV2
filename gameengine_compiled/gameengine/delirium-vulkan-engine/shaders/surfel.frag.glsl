#version 430 core

in vec4 v_color;
in vec3 v_normal;
in vec3 v_world_pos;
in float v_radius;

uniform vec3 u_cam_pos;

out vec4 frag_color;

void main() {
    // Discard pixels outside the point sprite circle
    vec2 center = gl_PointCoord - 0.5;
    if (dot(center, center) > 0.25) discard;

    // Simple lighting
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_cam_pos - v_world_pos);
    vec3 V = normalize(u_cam_pos - v_world_pos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 32.0);

    // Ambient + directional light
    vec3 light_color = vec3(1.0, 0.95, 0.9);
    vec3 ambient = v_color.rgb * 0.3;
    vec3 diffuse = v_color.rgb * diff * 0.7;
    vec3 specular = vec3(1.0) * spec * 0.3;

    vec3 final_color = ambient + diffuse + specular;

    // Fade edges of point sprite
    float alpha = 1.0 - smoothstep(0.3, 0.5, length(center));

    frag_color = vec4(final_color, v_color.a * alpha);
}