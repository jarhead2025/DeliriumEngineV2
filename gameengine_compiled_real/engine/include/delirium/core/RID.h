#pragma once
#include <cstdint>

namespace DeliRium {

// Opaque handle (Godot-style RID). 0 = invalid.
struct RID {
    uint64_t id = 0;
    bool is_valid() const { return id != 0; }
    bool operator==(const RID& o) const { return id == o.id; }
    bool operator!=(const RID& o) const { return id != o.id; }
};

} // namespace DeliRium
