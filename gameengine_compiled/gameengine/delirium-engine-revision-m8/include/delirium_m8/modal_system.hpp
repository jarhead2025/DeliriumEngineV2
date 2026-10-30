#pragma once

#include "delirium_m8/types.hpp"
#include <vector>

namespace delirium::m8 {

class ModalSystem {
public:
    void add_mode(ModalMode mode);
    const std::vector<ModalMode>& modes() const { return modes_; }
    std::size_t count() const { return modes_.size(); }
    float evaluate_displacement(std::size_t mode_index, float time) const;
private:
    std::vector<ModalMode> modes_;
};

} // namespace delirium::m8