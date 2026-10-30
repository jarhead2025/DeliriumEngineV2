#include "basis.h"
#include <cmath>
#include <algorithm>

namespace delirium {

// ---------------------------------------------------------------------------
// BasisFunction::evaluate — the core math of the engine
// Each basis type is a smooth function that contributes to surface shape.
// The surface is Σ Kᵢ · typeᵢ — a weighted sum of these basis evaluations.
// ---------------------------------------------------------------------------

float BasisFunction::evaluate(BasisType type, float x, float y, float z, float sigma) {
    float r2 = x*x + y*y + z*z;

    switch(type) {
        case BasisType::SPHERE:
            return expf(-r2 / (2.0f * sigma * sigma));

        case BasisType::CYLINDER: {
            float rho2 = x*x + y*y;
            return expf(-rho2 / (2.0f * sigma * sigma));
        }

        case BasisType::SADDLE:
            return expf(-(x*x - y*y) / (2.0f * sigma * sigma));

        case BasisType::PLANE:
            return expf(-(x*x + y*y) / (2.0f * sigma * sigma));

        case BasisType::CONE: {
            float h = fabsf(z);
            return expf(-(r2 - h*h) / (2.0f * sigma * sigma));
        }

        case BasisType::TORUS: {
            float rho = sqrtf(x*x + y*y);
            float R = sigma;
            return expf(-(rho-R)*(rho-R) / (0.5f*sigma*sigma) - z*z/(2.0f*sigma*sigma));
        }

        case BasisType::BOX: {
            float s = sigma;
            return (fabsf(x)<s && fabsf(y)<s && fabsf(z)<s) ? 1.0f : 0.0f;
        }

        case BasisType::GAUSSIAN:
            return expf(-r2 / (2.0f * sigma * sigma));

        case BasisType::COSINE:
            return cosf(M_PI * x / sigma) * cosf(M_PI * y / sigma) * cosf(M_PI * z / sigma);

        case BasisType::LINE:
            return expf(-(x*x + y*y) / (2.0f * sigma * sigma * 0.01f));

        case BasisType::DISK: {
            float rho = sqrtf(x*x + y*y);
            return expf(-(rho-sigma)*(rho-sigma)/(0.1f*sigma*sigma))
                 * expf(-z*z / (0.01f * sigma * sigma));
        }

        case BasisType::CAPSULE: {
            float half = sigma;
            float dz = fabsf(z) - half;
            float rho = sqrtf(x*x + y*y);
            if(dz > 0)
                return expf(-(rho*sigma*sigma + dz*dz) / (sigma*sigma));
            return expf(-rho*sigma*sigma / (sigma*sigma));
        }

        default:
            return 0.0f;
    }
}

// ---------------------------------------------------------------------------
// evaluate_sum — Σ Kᵢ · typeᵢ over the entire coefficient register
// This is the surface field query: "what value exists at (px,py,pz)?"
// Called by the ray marcher every step. Fixed compute per coefficient.
// ---------------------------------------------------------------------------

float BasisFunction::evaluate_sum(const CoefficientRegister& reg, float px, float py, float pz) {
    float sum = 0.0f;
    for(int i = 0; i < reg.count; i++) {
        const auto& c = reg.coeffs[i];
        // Transform to local space
        float lx = (px - c.cx) / c.sx;
        float ly = (py - c.cy) / c.sy;
        float lz = (pz - c.cz) / c.sz;

        // Sum K_j * basis(lx, ly, lz) for active coefficients
        int n = c.active_count();
        for(int j = 0; j < n; j++) {
            float basis_val = evaluate(
                static_cast<BasisType>(j % static_cast<int>(BasisType::COUNT)),
                lx, ly, lz, 1.0f);
            sum += c.k[j] * basis_val;
        }
    }
    return sum;
}

// ---------------------------------------------------------------------------
// evaluate_with_grad — basis value + gradient (for normals)
// Uses central differences for accuracy.
// ---------------------------------------------------------------------------

void BasisFunction::evaluate_with_grad(const Coefficient& c, float px, float py, float pz,
                                        float& val, vec3& grad) {
    val = c.evaluate(px, py, pz);
    float eps = 0.001f;
    grad = vec3(
        (c.evaluate(px+eps, py, pz) - c.evaluate(px-eps, py, pz)) / (2.0f*eps),
        (c.evaluate(px, py+eps, pz) - c.evaluate(px, py-eps, pz)) / (2.0f*eps),
        (c.evaluate(px, py, pz+eps) - c.evaluate(px, py, pz-eps)) / (2.0f*eps)
    );
}

// ---------------------------------------------------------------------------
// query_surface — "is there surface here?" + properties
// Threshold at 0.5: if Σ Kᵢ·typeᵢ > 0.5, we're inside/on surface.
// Computes normal via gradient, color from nearest coefficient.
// ---------------------------------------------------------------------------

bool BasisFunction::query_surface(const CoefficientRegister& reg, float px, float py, float pz,
                                   float& depth, vec3& normal, vec3& color, float& material) {
    float val = evaluate_sum(reg, px, py, pz);

    // Threshold: value > 0.5 means surface exists here
    if(val < 0.5f) return false;

    // Compute normal via gradient (central differences)
    float eps = 0.001f;
    float fx = evaluate_sum(reg, px+eps, py, pz);
    float fy = evaluate_sum(reg, px, py+eps, pz);
    float fz = evaluate_sum(reg, px, py, pz+eps);
    float fx0 = evaluate_sum(reg, px-eps, py, pz);
    float fy0 = evaluate_sum(reg, px, py-eps, pz);
    float fz0 = evaluate_sum(reg, px, py, pz-eps);

    normal = vec3(
        (fx - fx0) / (2.0f*eps),
        (fy - fy0) / (2.0f*eps),
        (fz - fz0) / (2.0f*eps)
    );
    float len = normal.length();
    if(len > 1e-8f) normal = normal / len;
    else normal = vec3(0, 0, 1);

    depth = val;

    // Get color/material from nearest coefficient
    color = vec3(1, 1, 1);
    material = 0.5f;
    float best_dist = 1e30f;
    for(int i = 0; i < reg.count; i++) {
        const auto& c = reg.coeffs[i];
        float d = (px-c.cx)*(px-c.cx) + (py-c.cy)*(py-c.cy) + (pz-c.cz)*(pz-c.cz);
        float extent = c.sx * c.sx;
        if(d < extent && d < best_dist) {
            best_dist = d;
            color = vec3(c.cr, c.cg, c.cb);
            material = c.roughness;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// evaluate_color — spectral color field (2D cosine basis)
// Color is a spectral field on top of geometry, separate from shape.
// LOD scales with frequency — fewer terms for distant objects.
// ---------------------------------------------------------------------------

vec3 BasisFunction::evaluate_color(const Coefficient& c, float u, float v, int lod) {
    // 2D cosine basis: Σ K_cr·cos(nπu)·cos(mπv) for each channel
    vec3 color(0, 0, 0);
    int terms = std::min(lod, MAX_ORDERS);

    for(int n = 0; n < terms; n++) {
        for(int m = 0; m < terms; m++) {
            float cu = cosf(n * M_PI * u);
            float cv = cosf(m * M_PI * v);
            float basis = cu * cv;

            int idx = n * terms + m;
            if(idx < c.active_count()) {
                // Store color coefficients interleaved after geometry coefficients
                int r_idx = idx;
                int g_idx = idx + c.active_count();
                int b_idx = idx + 2*c.active_count();

                if(r_idx < MAX_ORDERS*MAX_ORDERS*MAX_ORDERS)
                    color.x += c.k[r_idx] * basis;
                if(g_idx < MAX_ORDERS*MAX_ORDERS*MAX_ORDERS)
                    color.y += c.k[g_idx] * basis;
                if(b_idx < MAX_ORDERS*MAX_ORDERS*MAX_ORDERS)
                    color.z += c.k[b_idx] * basis;
            }
        }
    }

    // Fallback to stored color if spectral field is empty
    if(color.length_sq() < 0.01f)
        return vec3(c.cr, c.cg, c.cb);

    return color;
}

// ---------------------------------------------------------------------------
// Camera::ray_for_pixel — generates one ray per pixel
// Fixed W×H rays per frame. No scene complexity penalty.
// ---------------------------------------------------------------------------

Ray Camera::ray_for_pixel(int x, int y) const {
    // Normalize pixel to NDC [-1, 1]
    float nx = (2.0f * x / width - 1.0f) * aspect * tanf(fov * 0.5f);
    float ny = 1.0f - 2.0f * y / height;

    vec3 forward = (target - position).normalized();
    vec3 right = up.cross(forward).normalized();
    vec3 actual_up = forward.cross(right);

    vec3 dir = forward + right * nx + actual_up * ny;
    dir = dir.normalized();

    return Ray(position, dir, near, far);
}

// ---------------------------------------------------------------------------
// Coefficient::evaluate — single coefficient basis evaluation
// ---------------------------------------------------------------------------

float Coefficient::evaluate(float px, float py, float pz) const {
    float lx = (px - cx) / sx;
    float ly = (py - cy) / sy;
    float lz = (pz - cz) / sz;

    float sum = 0.0f;
    int n = active_count();
    for(int i = 0; i < n; i++) {
        float basis_val = BasisFunction::evaluate(
            static_cast<BasisType>(i % static_cast<int>(BasisType::COUNT)),
            lx, ly, lz, 1.0f);
        sum += k[i] * basis_val;
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Coefficient::transform — update center by matrix multiplication
// Used for bone/anchor animation: transform coefficient center.
// ---------------------------------------------------------------------------

void Coefficient::transform(const float mat[4][4]) {
    vec3 p(cx, cy, cz);
    vec3 result(
        mat[0][0]*p.x + mat[0][1]*p.y + mat[0][2]*p.z + mat[0][3],
        mat[1][0]*p.x + mat[1][1]*p.y + mat[1][2]*p.z + mat[1][3],
        mat[2][0]*p.x + mat[2][1]*p.y + mat[2][2]*p.z + mat[2][3]
    );
    cx = result.x; cy = result.y; cz = result.z;
}

} // namespace delirium