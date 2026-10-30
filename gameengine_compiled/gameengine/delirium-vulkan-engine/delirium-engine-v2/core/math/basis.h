#pragma once
#include "vec3.h"
#include "mat4.h"
#include "coefficient.h"

namespace delirium {

/// Basis function types — the "types" in Σ Kᵢ · typeᵢ.
enum class BasisType : uint8_t {
    SPHERE = 0, CYLINDER = 1, SADDLE = 2, PLANE = 3,
    CONE = 4, TORUS = 5, BOX = 6, GAUSSIAN = 7,
    COSINE = 8, LINE = 9, DISK = 10, CAPSULE = 11,
    COUNT
};

/// Core math: basis function evaluation.
struct BasisFunction {
    static float evaluate(BasisType type, float x, float y, float z, float sigma = 1.0f);
    static float evaluate_sum(const CoefficientRegister& reg, float px, float py, float pz);
    static void evaluate_with_grad(const Coefficient& c, float px, float py, float pz,
                                    float& val, vec3& grad);
    static bool query_surface(const CoefficientRegister& reg, float px, float py, float pz,
                               float& depth, vec3& normal, vec3& color, float& material);
    static vec3 evaluate_color(const Coefficient& c, float u, float v, int lod = 4);
};

/// Ray definition for raymarching.
struct Ray {
    vec3 origin;
    vec3 direction;
    float t_min;
    float t_max;
    Ray() : origin(0), direction(0,0,1), t_min(0), t_max(1000) {}
    Ray(const vec3& o, const vec3& d, float tn = 0, float tf = 1000)
        : origin(o), direction(d), t_min(tn), t_max(tf) {}
    vec3 point_at(float t) const { return origin + direction * t; }
};

/// Camera — defines the fixed FOV point for ray generation.
struct Camera {
    vec3 position;
    vec3 target;
    vec3 up;
    float fov;
    float aspect;
    float near;
    float far;
    int width;
    int height;

    Camera() : position(0,0,5), target(0), up(0,1,0),
               fov(1.047f), aspect(16.0f/9.0f),
               near(0.1f), far(100.0f),
               width(1920), height(1080) {}

    mat4 view_matrix() const { return mat4::lookAt(position, target, up); }
    mat4 proj_matrix() const { return mat4::perspective(fov, aspect, near, far); }
    mat4 view_inv() const { return view_matrix().inverse(); }
    mat4 proj_inv() const { return proj_matrix().inverse(); }

    Ray ray_for_pixel(int x, int y) const;
};

} // namespace delirium