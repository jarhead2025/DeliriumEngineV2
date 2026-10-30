#pragma once
#include <string>

namespace DeliRium {

// Lightweight base for tree objects (not full Godot Object).
class Object {
public:
    virtual ~Object() = default;
    void set_name(std::string n) { name_ = std::move(n); }
    const std::string& get_name() const { return name_; }

protected:
    std::string name_;
};

} // namespace DeliRium
