#include "delirium_m8/shared_material_pool.hpp"

namespace delirium::m8 {

MaterialId SharedMaterialPool::register_material(MaterialProperties props) {
    MaterialId id = next_id_++;
    materials_[id] = std::move(props);
    refs_[id] = 1;
    return id;
}

const MaterialProperties& SharedMaterialPool::get(MaterialId id) const {
    static MaterialProperties empty;
    auto it = materials_.find(id);
    return it != materials_.end() ? it->second : empty;
}

std::uint32_t SharedMaterialPool::reference_count(MaterialId id) const {
    auto it = refs_.find(id);
    return it != refs_.end() ? it->second : 0;
}

void SharedMaterialPool::add_reference(MaterialId id) {
    auto it = refs_.find(id);
    if (it != refs_.end()) ++it->second;
}

void SharedMaterialPool::remove_reference(MaterialId id) {
    auto it = refs_.find(id);
    if (it != refs_.end() && it->second > 0) {
        --it->second;
        if (it->second == 0) {
            materials_.erase(id);
            refs_.erase(it);
        }
    }
}

} // namespace delirium::m8