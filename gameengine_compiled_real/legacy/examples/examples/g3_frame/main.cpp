// FILEPATH: examples/g3_frame/main.cpp
// G3 vertical slice: SceneTree + PhysicsServer + OpenCL RenderServer hit/fill demo.
#include "delirium/core/SceneTree.h"
#include "delirium/scene/SplatView.h"
#include "delirium/servers/PhysicsServer.h"
#include "delirium/servers/RenderServer.h"

#include <cstdio>
#include <memory>

int main() {
    using namespace DeliRium;

    auto& rend = RenderServer::get_singleton();
    auto& phys = PhysicsServer::get_singleton();
    phys.init();
    if (!rend.init()) {
        std::fprintf(stderr, "G3 FAIL: OpenCL RenderServer::init (Intel platform)\n");
        return 1;
    }
    if (!rend.set_viewport_size(128, 128)) {
        std::fprintf(stderr, "G3 FAIL: viewport\n");
        return 1;
    }

    SceneTree tree;
    auto view = std::make_unique<SplatView>();
    view->set_name("splat_view");
    view->set_demo_body(true);
    Node* root = tree.get_root();
    root->add_child(std::move(view));

    const double dt = 1.0 / 60.0;
    const int frames = 30;
    for (int f = 0; f < frames; ++f) {
        phys.step(dt);
        tree.physics_process(dt);
        tree.process(dt);
        if (!rend.draw()) {
            std::fprintf(stderr, "G3 FAIL: draw frame %d\n", f);
            return 1;
        }
    }

    const auto& fb = rend.get_framebuffer();
    unsigned nonzero = 0;
    for (uint32_t p : fb) {
        if ((p & 0x00FFFFFFu) != 0x00101010u) ++nonzero;
    }
    std::printf(
        "G3 frame OK: OpenCL ready, %d frames, fb=%zu pixels, lit≈%u\n",
        frames, fb.size(), nonzero);
    std::printf("Core path: modal coeffs → sparse hits → pattern average fill\n");

    rend.finish();
    phys.finish();
    return 0;
}
