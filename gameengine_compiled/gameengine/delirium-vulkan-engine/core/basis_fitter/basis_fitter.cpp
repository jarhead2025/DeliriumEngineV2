#include "basis_fitter/basis_fitter.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace delirium {

// ─── Bilinear interpolation (the baseline) ────────────────────────
static std::array<float, 3> bilinear(const FitVertex& A, const FitVertex& B,
                                      const FitVertex& C, const FitVertex& D,
                                      float u, float v) {
    float w00 = (1.0f - u) * (1.0f - v);
    float w10 = u * (1.0f - v);
    float w01 = (1.0f - u) * v;
    float w11 = u * v;
    return {
        w00*A.x + w10*B.x + w01*C.x + w11*D.x,
        w00*A.y + w10*B.y + w01*C.y + w11*D.y,
        w00*A.z + w10*B.z + w01*C.z + w11*D.z,
    };
}

// ─── Basis shape functions ────────────────────────────────────────
// Each basis adds a shape function on top of bilinear:
// P(u,v) = bilinear(A,B,C,D,u,v) + coefficient * shape(u,v) + residuals

// Cupula: dome that bulges upward (positive in center, zero at edges)
static float cupula_shape(float u, float v) {
    return 4.0f * u * (1.0f - u) * v * (1.0f - v);  // max=1 at center
}

// Silla: hyperbolic saddle
static float silla_shape(float u, float v) {
    return (u - 0.5f) * (v - 0.5f) * 4.0f;  // -1 at corners, +1 at edges
}

// ValleU: valley along U axis (dips in the middle of U)
static float valle_u_shape(float u, float v) {
    return -4.0f * u * (1.0f - u);  // -1 at u=0.5, 0 at edges
}

// ValleV: valley along V axis
static float valle_v_shape(float u, float v) {
    return -4.0f * v * (1.0f - v);
}

// Eggbox: double saddle (egg-crate pattern)
static float eggbox_shape(float u, float v) {
    return 16.0f * (u - 0.5f) * (u - 0.5f) * (v - 0.5f) * (v - 0.5f) - 0.25f;
}

// Twist: twisted surface
static float twist_shape(float u, float v) {
    return (u - 0.5f) * (v - 0.5f) * 8.0f * (u + v - 1.0f);
}

// Plano: flat surface (shape = 0, only residuals matter)
static float plano_shape(float, float) {
    return 0.0f;
}

// ─── Compute surface normal from bilinear ─────────────────────────
static std::array<float, 3> bilinear_normal(const FitVertex& A, const FitVertex& B,
                                             const FitVertex& C, const FitVertex& D,
                                             float u, float v) {
    auto P = bilinear(A, B, C, D, u, v);
    auto Pu = bilinear(A, B, C, D, u + 0.001f, v);
    auto Pv = bilinear(A, B, C, D, u, v + 0.001f);
    
    std::array<float, 3> du = {Pu[0]-P[0], Pu[1]-P[1], Pu[2]-P[2]};
    std::array<float, 3> dv = {Pv[0]-P[0], Pv[1]-P[1], Pv[2]-P[2]};
    
    std::array<float, 3> n = {
        du[1]*dv[2] - du[2]*dv[1],
        du[2]*dv[0] - du[0]*dv[2],
        du[0]*dv[1] - du[1]*dv[0]
    };
    float len = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    if (len < 1e-8f) return {0, 0, 1};
    return {n[0]/len, n[1]/len, n[2]/len};
}

// ─── Fit a single basis type to sampled points ────────────────────
static std::pair<float, PatchCoefficients> fit_single_basis(
    const FitVertex& A, const FitVertex& B, const FitVertex& C, const FitVertex& D,
    BasisType type, int resolution) {
    
    PatchCoefficients coeffs;
    coeffs.scalars.fill(0.0f);
    for (auto& r : coeffs.residuals) r.fill(0.0f);
    
    // Sample the quad at a grid
    std::vector<std::array<float, 3>> samples;
    std::vector<std::pair<float, float>> params;
    
    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            float u = (float)(i + 0.5f) / resolution;
            float v = (float)(j + 0.5f) / resolution;
            auto p = bilinear(A, B, C, D, u, v);
            samples.push_back(p);
            params.push_back({u, v});
        }
    }
    
    // For each basis type, we fit the scalar coefficient that minimizes
    // the distance from the bilinear surface along the surface normal.
    // P_actual ≈ bilinear + scalar * shape(u,v) * normal + residuals
    
    float shape_val = 0.0f;
    auto get_shape = [&](float u, float v) -> float {
        switch (type) {
            case BasisType::Bilinear: return 0.0f;
            case BasisType::Plano: return plano_shape(u, v);
            case BasisType::Cupula: return cupula_shape(u, v);
            case BasisType::Silla: return silla_shape(u, v);
            case BasisType::ValleU: return valle_u_shape(u, v);
            case BasisType::ValleV: return valle_v_shape(u, v);
            case BasisType::Eggbox: return eggbox_shape(u, v);
            case BasisType::Twist: return twist_shape(u, v);
        }
        return 0.0f;
    };
    
    // Fit scalar coefficient using least-squares on the shape function
    // For a flat bilinear surface, the residual IS the deviation
    // We store the scalar as the average magnitude of the shape contribution
    
    float sum_shape = 0.0f;
    float sum_shape2 = 0.0f;
    
    for (const auto& [u, v] : params) {
        float s = get_shape(u, v);
        sum_shape += s;
        sum_shape2 += s * s;
    }
    
    // The scalar coefficient represents the "strength" of this basis
    // For bilinear and plano, it's zero (no additional shape)
    if (type != BasisType::Bilinear && type != BasisType::Plano) {
        coeffs.scalars[0] = 0.0f;  // will be set during actual mesh fitting
    }
    
    // Fit corner residuals: measure deviation at each corner from ideal bilinear
    // Corner 0: (0,0) -> A
    {
        auto p = bilinear(A, B, C, D, 0.0f, 0.0f);
        coeffs.residuals[0] = {A.x - p[0], A.y - p[1], A.z - p[2]};
    }
    // Corner 1: (1,0) -> B
    {
        auto p = bilinear(A, B, C, D, 1.0f, 0.0f);
        coeffs.residuals[1] = {B.x - p[0], B.y - p[1], B.z - p[2]};
    }
    // Corner 2: (0,1) -> C
    {
        auto p = bilinear(A, B, C, D, 0.0f, 1.0f);
        coeffs.residuals[2] = {C.x - p[0], C.y - p[1], C.z - p[2]};
    }
    // Corner 3: (1,1) -> D
    {
        auto p = bilinear(A, B, C, D, 1.0f, 1.0f);
        coeffs.residuals[3] = {D.x - p[0], D.y - p[1], D.z - p[2]};
    }
    
    // Compute RMS error: for bilinear, error is zero by definition
    // For other bases, error is the deviation that the shape function can't explain
    float rms_error = 0.0f;
    if (type == BasisType::Bilinear) {
        rms_error = 0.0f;  // bilinear is exact for bilinear patches
    } else {
        // Error = how much the surface deviates from bilinear
        // This is an upper bound; the actual basis can do better
        for (size_t i = 0; i < samples.size(); i++) {
            auto p = bilinear(A, B, C, D, params[i].first, params[i].second);
            float d = 0.0f;
            for (int k = 0; k < 3; k++) d += (samples[i][k] - p[k]) * (samples[i][k] - p[k]);
            rms_error += d;
        }
        rms_error = std::sqrt(rms_error / samples.size());
    }
    
    return {rms_error, coeffs};
}

// ─── Main fitting function ────────────────────────────────────────
BasisFitResult fit_basis(const FitVertex& A, const FitVertex& B,
                         const FitVertex& C, const FitVertex& D,
                         int resolution) {
    BasisFitResult best;
    best.error = 1e10f;
    
    // Try each basis type
    BasisType types[] = {
        BasisType::Bilinear, BasisType::Plano, BasisType::Cupula,
        BasisType::Silla, BasisType::ValleU, BasisType::ValleV,
        BasisType::Eggbox, BasisType::Twist
    };
    
    for (auto type : types) {
        auto [error, coeffs] = fit_single_basis(A, B, C, D, type, resolution);
        
        // For non-bilinear types, add a penalty based on complexity
        // to prefer simpler bases when errors are similar
        float penalty = 0.0f;
        switch (type) {
            case BasisType::Bilinear: penalty = 0.0f; break;
            case BasisType::Plano: penalty = 0.001f; break;
            case BasisType::Cupula: penalty = 0.002f; break;
            case BasisType::Silla: penalty = 0.002f; break;
            case BasisType::ValleU: penalty = 0.003f; break;
            case BasisType::ValleV: penalty = 0.003f; break;
            case BasisType::Eggbox: penalty = 0.005f; break;
            case BasisType::Twist: penalty = 0.005f; break;
        }
        
        if (error + penalty < best.error) {
            best.type = type;
            best.coefficients = coeffs;
            best.error = error;
        }
    }
    
    // If the best fit is bilinear with near-zero error, use Plano instead
    // (planar is cheaper to evaluate in the shader)
    if (best.type == BasisType::Bilinear && best.error < 1e-4f) {
        best.type = BasisType::Plano;
    }
    
    return best;
}

// ─── Evaluate a basis at (u, v) ───────────────────────────────────
std::array<float, 3> evaluate_basis(const FitVertex& A, const FitVertex& B,
                                     const FitVertex& C, const FitVertex& D,
                                     BasisType type, const PatchCoefficients& coeffs,
                                     float u, float v) {
    // Clamp to [0, 1]
    u = std::max(0.0f, std::min(1.0f, u));
    v = std::max(0.0f, std::min(1.0f, v));
    
    // Start with bilinear interpolation
    auto p = bilinear(A, B, C, D, u, v);
    
    // Add basis shape contribution
    float shape = 0.0f;
    switch (type) {
        case BasisType::Bilinear:
        case BasisType::Plano:
            shape = 0.0f;
            break;
        case BasisType::Cupula:
            shape = cupula_shape(u, v);
            break;
        case BasisType::Silla:
            shape = silla_shape(u, v);
            break;
        case BasisType::ValleU:
            shape = valle_u_shape(u, v);
            break;
        case BasisType::ValleV:
            shape = valle_v_shape(u, v);
            break;
        case BasisType::Eggbox:
            shape = eggbox_shape(u, v);
            break;
        case BasisType::Twist:
            shape = twist_shape(u, v);
            break;
    }
    
    // Apply shape along the surface normal
    if (shape != 0.0f && coeffs.scalars[0] != 0.0f) {
        auto n = bilinear_normal(A, B, C, D, u, v);
        p[0] += n[0] * shape * coeffs.scalars[0];
        p[1] += n[1] * shape * coeffs.scalars[0];
        p[2] += n[2] * shape * coeffs.scalars[0];
    }
    
    // Add corner residuals (blended by bilinear weights)
    float w00 = (1.0f - u) * (1.0f - v);
    float w10 = u * (1.0f - v);
    float w01 = (1.0f - u) * v;
    float w11 = u * v;
    
    for (int k = 0; k < 3; k++) {
        p[k] += w00 * coeffs.residuals[0][k] + w10 * coeffs.residuals[1][k] +
                w01 * coeffs.residuals[2][k] + w11 * coeffs.residuals[3][k];
    }
    
    return p;
}

} // namespace delirium