#include "modal_system.h"
#include <cmath>

namespace delirium {

void ModalSystem::add_mode(ModalMode mode) {
    modes_.push_back(std::move(mode));
}

/* ── Semi-implicit Euler (symplectic, no energy leak) ──────────────
 *
 * PROVEN MIRROR (ModalDynamics.cpp:131-139):
 *   a = f - 2*zeta*omega*v - omega²*q
 *   v += a * dt    (velocity FIRST — symplectic)
 *   q += v * dt    (position, using NEW velocity)
 *
 * Damping is in the acceleration term (2*zeta*omega*v), NOT as a
 * post-hoc velocity multiplier.  The original engine's fixed step
 * (1/120) keeps dt*omega < 2 for stability.
 * ───────────────────────────────────────────────────────────────── */
void ModalSystem::integrate(std::vector<float>& amplitudes,
                             std::vector<float>& velocities,
                             float dt) const {
    for (std::size_t i = 0; i < modes_.size(); ++i) {
        const auto& mode = modes_[i];
        float omega = mode.frequency * kTwoPi;
        // Fixed damping ratio — mode.amplitude is initial displacement, not damping
        float zeta = 0.05f;  // light damping (original default)

        // Acceleration: spring + damper (no external force in v2 — impulses via velocity)
        float a = -2.0f * zeta * omega * velocities[i]
                  - omega * omega * amplitudes[i];

        // Velocity FIRST (symplectic)
        velocities[i] += a * dt;
        // Position (uses NEW velocity)
        amplitudes[i] += velocities[i] * dt;
    }
}

/* ── Per-coefficient mode shapes ───────────────────────────────────
 *
 * Each mode type affects specific coefficients of the 8-term basis:
 *   b[0,1] = u(1-u) terms (horizontal bulge)
 *   b[2,3] = v(1-v) terms (vertical bulge)
 *   b[4,5] = u(1-u)(2u-1) terms (horizontal twist)
 *   b[6,7] = v(1-v)(2v-1) terms (vertical twist)
 *
 * Breathing: uniform radial displacement (all 8 terms)
 * BendingX:  horizontal curvature (b[0,1,4,5])
 * BendingY:  vertical curvature (b[2,3,6,7])
 * BendingZ:  diagonal curvature (all, alternating sign)
 * TorsionX:  horizontal twist (b[4,5])
 * TorsionY:  vertical twist (b[6,7])
 *
 * This mirrors the original engine's ModalDynamics mode shapes
 * where each mode has a per-coefficient influence vector.
 * ────────────────────────────────────────────────────────────────── */
static void mode_shape(ModalMode::Type type, float shape[8]) {
    // Zero out all coefficients
    for (int i = 0; i < 8; ++i) shape[i] = 0.0f;

    switch (type) {
        case ModalMode::Type::Breathing:
            // Uniform radial — all 8 terms equal weight
            for (int i = 0; i < 8; ++i) shape[i] = 1.0f;
            break;
        case ModalMode::Type::BendingX:
            // Horizontal curvature — u terms
            shape[0] = 1.0f;  shape[1] = 1.0f;
            shape[4] = 0.5f;  shape[5] = -0.5f;  // twist component
            break;
        case ModalMode::Type::BendingY:
            // Vertical curvature — v terms
            shape[2] = 1.0f;  shape[3] = 1.0f;
            shape[6] = 0.5f;  shape[7] = -0.5f;  // twist component
            break;
        case ModalMode::Type::BendingZ:
            // Diagonal — all terms, alternating sign
            shape[0] = 1.0f;  shape[1] = -1.0f;
            shape[2] = -1.0f; shape[3] = 1.0f;
            shape[4] = 0.5f;  shape[5] = -0.5f;
            shape[6] = -0.5f; shape[7] = 0.5f;
            break;
        case ModalMode::Type::TorsionX:
            // Horizontal twist — higher-order u terms
            shape[4] = 1.0f;  shape[5] = -1.0f;
            break;
        case ModalMode::Type::TorsionY:
            // Vertical twist — higher-order v terms
            shape[6] = 1.0f;  shape[7] = -1.0f;
            break;
    }
}

/* ── Write modal displacement back onto patch coefficients ───────── */
void ModalSystem::write_back(const std::vector<float>& amplitudes,
                              BodyState& state,
                              const BodyTopology& topology) const {
    for (PatchId pid = 0; pid < topology.patch_count(); ++pid) {
        const auto& rest = topology.patch(pid).coefficients;
        auto& ps = state.patches[pid];

        // Recompute from rest + modal delta (no accumulation drift)
        for (int c = 0; c < 8; ++c) {
            float delta = 0.0f;
            for (std::size_t mi = 0; mi < modes_.size() && mi < amplitudes.size(); ++mi) {
                if (std::abs(amplitudes[mi]) < 1e-8f) continue;

                float shape[8];
                mode_shape(modes_[mi].type, shape);
                delta += amplitudes[mi] * shape[c];
            }
            ps.coefficients[c] = rest.scalars[c] + delta;
        }
        ps.cumulative_plastic_strain = 0.0f;
    }
}

} // namespace delirium