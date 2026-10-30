#pragma once
#include "delirium/core/RID.h"
#include <unordered_map>
#include <vector>

namespace DeliRium {

// Equation-domain step: advances simple modal / coeff state (no mesh).
class PhysicsServer {
public:
    static PhysicsServer& get_singleton();

    void init();
    void finish();

    RID body_create();
    void body_set_modal(RID id, const float* modes, size_t n);
    void body_free(RID id);

    void step(double dt);

    // Read back coeffs after step (for RenderServer upload)
    const std::vector<float>* body_coeffs(RID id) const;

private:
    PhysicsServer() = default;
    uint64_t next_id_ = 1;
    struct BodyState {
        std::vector<float> coeffs;
        std::vector<float> modes;
        float phase = 0.f;
    };
    std::unordered_map<uint64_t, BodyState> bodies_;
};

} // namespace DeliRium
