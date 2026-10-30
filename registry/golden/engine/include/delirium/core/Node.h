#pragma once
#include "delirium/core/Object.h"
#include <memory>
#include <vector>

namespace DeliRium {

class SceneTree;

// Godot-like Node: ownership tree + process hooks.
class Node : public Object {
public:
    ~Node() override;

    void add_child(std::unique_ptr<Node> child);
    Node* get_parent() const { return parent_; }
    const std::vector<std::unique_ptr<Node>>& get_children() const { return children_; }

    void set_process(bool v) { process_ = v; }
    bool is_processing() const { return process_; }

    virtual void _enter_tree() {}
    virtual void _exit_tree() {}
    virtual void _process(double dt) { (void)dt; }
    virtual void _physics_process(double dt) { (void)dt; }

    void propagate_process(double dt);
    void propagate_physics(double dt);

    SceneTree* get_tree() const { return tree_; }
    void set_tree(SceneTree* t);

private:
    Node* parent_ = nullptr;
    SceneTree* tree_ = nullptr;
    std::vector<std::unique_ptr<Node>> children_;
    bool process_ = true;
};

} // namespace DeliRium
