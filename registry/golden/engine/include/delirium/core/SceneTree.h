#pragma once
#include "delirium/core/Node.h"
#include <memory>

namespace DeliRium {

// Owns the root node and drives process / physics ticks.
class SceneTree {
public:
    SceneTree();
    ~SceneTree();

    Node* get_root() { return root_.get(); }
    void set_root(std::unique_ptr<Node> root);

    void process(double dt);
    void physics_process(double dt);

    bool quit_requested() const { return quit_; }
    void quit() { quit_ = true; }

private:
    std::unique_ptr<Node> root_;
    bool quit_ = false;
};

} // namespace DeliRium
