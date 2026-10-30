// FILEPATH: engine/src/scene/SplatView.cpp
#include "delirium/scene/SplatView.h"
#include "delirium/servers/PhysicsServer.h"
#include "delirium/servers/RenderServer.h"

namespace DeliRium {

void SplatView::_enter_tree() {
    auto& phys = PhysicsServer::get_singleton();
    auto& rend = RenderServer::get_singleton();
    phys_body_ = phys.body_create();
    render_body_ = rend.body_create();
    if (demo_) {
        const float modes[8] = {0.4f, 0.3f, 0.2f, 0.15f, 0.1f, 0.08f, 0.05f, 0.03f};
        phys.body_set_modal(phys_body_, modes, 8);
    }
}

void SplatView::_exit_tree() {
    PhysicsServer::get_singleton().body_free(phys_body_);
    RenderServer::get_singleton().body_free(render_body_);
    phys_body_ = {};
    render_body_ = {};
}

void SplatView::_physics_process(double dt) {
    (void)dt;
    // PhysicsServer.step is called from main; node only ensures RID alive
}

void SplatView::_process(double dt) {
    (void)dt;
    auto* coeffs = PhysicsServer::get_singleton().body_coeffs(phys_body_);
    if (coeffs && !coeffs->empty()) {
        RenderServer::get_singleton().body_set_coeffs(
            render_body_, coeffs->data(), coeffs->size());
    }
}

} // namespace DeliRium
