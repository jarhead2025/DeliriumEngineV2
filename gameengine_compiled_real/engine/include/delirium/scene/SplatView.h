#pragma once
#include "delirium/core/Node.h"
#include "delirium/core/RID.h"

namespace DeliRium {

// Node that registers a body with Physics + Render servers each frame.
class SplatView : public Node {
public:
    void _enter_tree() override;
    void _exit_tree() override;
    void _process(double dt) override;
    void _physics_process(double dt) override;

    void set_demo_body(bool on) { demo_ = on; }

private:
    RID phys_body_{};
    RID render_body_{};
    bool demo_ = true;
};

} // namespace DeliRium
