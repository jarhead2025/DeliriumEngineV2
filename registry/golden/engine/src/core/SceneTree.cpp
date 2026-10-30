// FILEPATH: engine/src/core/SceneTree.cpp
#include "delirium/core/SceneTree.h"

namespace DeliRium {

SceneTree::SceneTree() {
    root_ = std::make_unique<Node>();
    root_->set_name("root");
    root_->set_tree(this);
}

SceneTree::~SceneTree() {
    if (root_) {
        root_->set_tree(nullptr);
    }
}

void SceneTree::set_root(std::unique_ptr<Node> root) {
    if (root_) root_->set_tree(nullptr);
    root_ = std::move(root);
    if (root_) root_->set_tree(this);
}

void SceneTree::process(double dt) {
    if (root_) root_->propagate_process(dt);
}

void SceneTree::physics_process(double dt) {
    if (root_) root_->propagate_physics(dt);
}

} // namespace DeliRium
