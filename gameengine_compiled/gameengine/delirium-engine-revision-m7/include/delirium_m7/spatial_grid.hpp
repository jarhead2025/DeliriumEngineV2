#pragma once
#include "delirium_m7/types.hpp"
#include "delirium_m7/body_topology.hpp"
#include "delirium_m7/body_state.hpp"
#include <string>
namespace delirium::m7 {
class SpatialGrid {
public:
    SpatialGrid(std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z,
                const Vec3& min_corner, const Vec3& max_corner);
    void build(const BodyState& state, const BodyTopology& topology);
    std::tuple<std::uint32_t, std::uint32_t, std::uint32_t> position_to_cell(const Vec3& p) const;
    const std::vector<SpatialCell>& cells() const { return cells_; }
    std::size_t occupied_count() const { return occupied_; }
private:
    std::uint32_t dim_x_, dim_y_, dim_z_;
    Vec3 min_corner_, max_corner_, inv_dim_;
    std::vector<SpatialCell> cells_;
    std::size_t occupied_ = 0;
};
} // namespace delirium::m7