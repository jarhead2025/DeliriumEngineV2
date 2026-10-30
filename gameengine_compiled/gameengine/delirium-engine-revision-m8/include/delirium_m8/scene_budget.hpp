#pragma once
#include "delirium_m8/types.hpp"
namespace delirium::m8 {
inline bool SceneBudget::is_within_budget(std::uint32_t bodies, std::uint32_t surfels, std::uint32_t constraints) const {
    return bodies <= max_bodies && surfels <= max_surfels && constraints <= max_constraints;
}
} // namespace delirium::m8