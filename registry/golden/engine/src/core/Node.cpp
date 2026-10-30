// FILEPATH: engine/src/core/Node.cpp
#include "delirium/core/Node.h"
#include "delirium/core/SceneTree.h"

namespace DeliRium {

Node::~Node() {
    for (auto& c : children_) {
        if (c) {
            c->set_tree(nullptr);
            c->parent_ = nullptr;
        }
    }
}

void Node::add_child(std::unique_ptr<Node> child) {
    if (!child) return;
    child->parent_ = this;
    child->set_tree(tree_);
    child->_enter_tree();
    children_.push_back(std::move(child));
}

void Node::set_tree(SceneTree* t) {
    tree_ = t;
    for (auto& c : children_) {
        if (c) c->set_tree(t);
    }
}

void Node::propagate_process(double dt) {
    if (process_) _process(dt);
    for (auto& c : children_) {
        if (c) c->propagate_process(dt);
    }
}

void Node::propagate_physics(double dt) {
    if (process_) _physics_process(dt);
    for (auto& c : children_) {
        if (c) c->propagate_physics(dt);
    }
}

} // namespace DeliRium
