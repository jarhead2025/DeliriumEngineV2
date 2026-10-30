#include "delirium_m8/spatial_grid.hpp"
#include <cmath>

namespace delirium::m8 {

SpatialGrid::SpatialGrid(std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z,
                         const Vec3& min_corner, const Vec3& max_corner)
    : dim_x_(dim_x), dim_y_(dim_y), dim_z_(dim_z), min_corner_(min_corner), max_corner_(max_corner) {
    cells_.resize(static_cast<std::size_t>(dim_x) * dim_y * dim_z);
    inv_dim_ = {
        dim_x > 1 ? 1.0f / (max_corner.x - min_corner.x) : 0.0f,
        dim_y > 1 ? 1.0f / (max_corner.y - min_corner.y) : 0.0f,
        dim_z > 1 ? 1.0f / (max_corner.z - min_corner.z) : 0.0f,
    };
}

std::tuple<std::uint32_t, std::uint32_t, std::uint32_t> SpatialGrid::position_to_cell(const Vec3& p) const {
    float fx = (p.x - min_corner_.x) * inv_dim_.x * dim_x_;
    float fy = (p.y - min_corner_.y) * inv_dim_.y * dim_y_;
    float fz = (p.z - min_corner_.z) * inv_dim_.z * dim_z_;

    std::uint32_t x = static_cast<std::uint32_t>(std::floor(fx));
    std::uint32_t y = static_cast<std::uint32_t>(std::floor(fy));
    std::uint32_t z = static_cast<std::uint32_t>(std::floor(fz));

    if (x >= dim_x_) x = dim_x_ - 1;
    if (y >= dim_y_) y = dim_y_ - 1;
    if (z >= dim_z_) z = dim_z_ - 1;

    return {x, y, z};
}

void SpatialGrid::build(const BodyState& state, const BodyTopology& topology) {
    cells_.assign(cells_.size(), SpatialCell{});
    occupied_ = 0;

    std::size_t idx = 0;
    for (const auto& anchor : state.anchors) {
        auto [cx, cy, cz] = position_to_cell(anchor.position);
        std::size_t ci = static_cast<std::size_t>(cx) +
                         static_cast<std::size_t>(cy) * dim_x_ +
                         static_cast<std::size_t>(cz) * dim_x_ * dim_y_;
        if (cells_[ci].count == 0) occupied_++;
        cells_[ci].first = static_cast<std::uint32_t>(idx);
        cells_[ci].count++;
        idx++;
    }
}

} // namespace delirium::m8