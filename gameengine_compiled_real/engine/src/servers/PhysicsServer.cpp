// FILEPATH: engine/src/servers/PhysicsServer.cpp
#include "delirium/servers/PhysicsServer.h"
#include <cmath>

namespace DeliRium {

PhysicsServer& PhysicsServer::get_singleton() {
    static PhysicsServer s;
    return s;
}

void PhysicsServer::init() {}
void PhysicsServer::finish() { bodies_.clear(); }

RID PhysicsServer::body_create() {
    RID r;
    r.id = next_id_++;
    bodies_[r.id] = BodyState{};
    return r;
}

void PhysicsServer::body_set_modal(RID id, const float* modes, size_t n) {
    if (!id.is_valid()) return;
    auto it = bodies_.find(id.id);
    if (it == bodies_.end()) return;
    it->second.modes.assign(modes, modes + n);
    if (it->second.coeffs.size() < n) it->second.coeffs.assign(n, 0.f);
}

void PhysicsServer::body_free(RID id) { bodies_.erase(id.id); }

void PhysicsServer::step(double dt) {
    for (auto& kv : bodies_) {
        BodyState& b = kv.second;
        b.phase += static_cast<float>(dt);
        if (b.coeffs.empty() && !b.modes.empty()) {
            b.coeffs.assign(b.modes.size(), 0.f);
        }
        for (size_t i = 0; i < b.coeffs.size(); ++i) {
            const float m = i < b.modes.size() ? b.modes[i] : 0.1f;
            // Simple modal oscillation — equation domain, not mesh
            b.coeffs[i] = m * std::sin(b.phase * (1.f + 0.15f * static_cast<float>(i)));
        }
    }
}

const std::vector<float>* PhysicsServer::body_coeffs(RID id) const {
    auto it = bodies_.find(id.id);
    if (it == bodies_.end()) return nullptr;
    return &it->second.coeffs;
}

} // namespace DeliRium
