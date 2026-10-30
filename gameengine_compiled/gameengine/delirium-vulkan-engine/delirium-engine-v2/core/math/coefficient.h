#pragma once
#include <cstring>
#include <cstdint>

namespace delirium {

/// Maximum basis orders per axis.
/// Orders 0-2 contain 99.9% of shape energy — proven on roughball.
static constexpr int MAX_ORDERS = 32;

/// A coefficient record — the fundamental data unit of the engine.
/// 64-byte aligned, fixed-size memory register.
/// Stores K_i weights for one basis function instance.
/// The rendering summation never changes — only these coefficients update.
struct alignas(16) Coefficient {
    // Basis center position (world space)
    float cx, cy, cz;

    // Scale along each axis (defines basis extent)
    float sx, sy, sz;

    // Basis order (degree) — higher = more detail
    // 99.9% of shape energy lives in orders 0-2
    int order;

    // K_i coefficients: the weights for Σ Kᵢ · typeᵢ
    // For a 3D basis of order N: (N+1)³ coefficients
    float k[MAX_ORDERS * MAX_ORDERS * MAX_ORDERS];

    // Color/RGB coefficients (spectral color field)
    float cr, cg, cb, ca;

    // Material properties for rendering
    float roughness, metallic, reflectivity;

    // Bone/anchor binding index — links to skeletal transform
    int bone_id;

    // Basis type index (0=SPHERE, 1=CYLINDER, etc.)
    int basis_type;

    // Padding for alignment
    float pad[4];

    Coefficient() {
        memset(this, 0, sizeof(*this));
        order = 2;
        sx = sy = sz = 1.0f;
        cr = cg = cb = ca = 1.0f;
        roughness = 0.5f;
        metallic = 0.0f;
        reflectivity = 0.3f;
        bone_id = -1;
        basis_type = 0;
        // Default: sphere basis with unit weight
        k[0] = 1.0f;
    }

    /// Evaluate basis function at point (world space)
    float evaluate(float px, float py, float pz) const;

    /// Number of active coefficients for this order
    int active_count() const { return (order+1)*(order+1)*(order+1); }

    /// Transform by matrix (updates center position)
    void transform(const float mat[4][4]);
};

/// Memory register: array of coefficients for one body/object.
/// This is what the GPU ray marcher reads every frame.
/// Swap coefficients → swap geometry. No mesh upload needed.
struct CoefficientRegister {
    static constexpr int MAX_COEFFICIENTS = 4096;

    Coefficient coeffs[MAX_COEFFICIENTS];
    int count = 0;

    /// Dirty tracking for physics → render pipeline
    /// When physics modifies coefficients, it marks them dirty.
    /// Renderer only re-uploads dirty coefficients.
    bool dirty[MAX_COEFFICIENTS];
    bool any_dirty = false;

    CoefficientRegister() { memset(dirty, 0, sizeof(dirty)); }

    void mark_dirty(int idx) {
        if(idx >= 0 && idx < count) {
            dirty[idx] = true;
            any_dirty = true;
        }
    }
    void mark_all_dirty() {
        for(int i=0; i<count; i++) dirty[i] = true;
        any_dirty = true;
    }
    void clear_dirty() {
        memset(dirty, 0, count * sizeof(bool));
        any_dirty = false;
    }

    void add_coefficient(const Coefficient& c) {
        if(count < MAX_COEFFICIENTS) {
            coeffs[count++] = c;
        }
    }

    void reset() {
        count = 0;
        clear_dirty();
    }
};

} // namespace delirium