#include "delirium_m4/spatial_grid.hpp"

#include <algorithm>
#include <stdexcept>

namespace delirium::m4 {

SpatialGrid::SpatialGrid() = default;

SpatialGrid::SpatialGrid(std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z)
    : dim_x_(dim_x), dim_y_(dim_y), dim_z_(dim_z) {}

void SpatialGrid::clear() {
    cells_.clear();
    surfel_indices_.clear();
}

std::size_t SpatialGrid::build(const std::vector<Surfel>& surfels) {
    if (surfels.empty()) return 0;

    min_bound_ = surfels[0].position;
    max_bound_ = surfels[0].position;
    for (const auto& s : surfels) {
        if (s.position.x < min_bound_.x) min_bound_.x = s.position.x;
        if (s.position.y < min_bound_.y) min_bound_.y = s.position.y;
        if (s.position.z < min_bound_.z) min_bound_.z = s.position.z;
        if (s.position.x > max_bound_.x) max_bound_.x = s.position.x;
        if (s.position.y > max_bound_.y) max_bound_.y = s.position.y;
        if (s.position.z > max_bound_.z) max_bound_.z = s.position.z;
    }

    Vec3 padding{0.001f, 0.001f, 0.001f};
    min_bound_ = min_bound_ - padding;
    max_bound_ = max_bound_ + padding;

    cell_size_.x = (max_bound_.x - min_bound_.x) / dim_x_;
    cell_size_.y = (max_bound_.y - min_bound_.y) / dim_y_;
    cell_size_.z = (max_bound_.z - min_bound_.z) / dim_z_;

    std::size_t total = static_cast<std::size_t>(dim_x_) * dim_y_ * dim_z_;
    cells_.assign(total, SpatialCell{});

    surfel_indices_.clear();
    surfel_indices_.reserve(surfels.size());

    std::vector<std::vector<std::uint32_t>> cell_surfels(total);

    for (std::uint32_t i = 0; i < surfels.size(); ++i) {
        auto [cx, cy, cz] = position_to_cell(surfels[i].position);
        std::size_t cell_idx = (static_cast<std::size_t>(cz) * dim_x_ * dim_y_) +
                               (static_cast<std::size_t>(cy) * dim_x_) +
                               static_cast<std::size_t>(cx);

        if (cell_idx < total) {
            cell_surfels[cell_idx].push_back(i);
            cells_[cell_idx].material_mask |= (1u << (surfels[i].material & 31));
        }
    }

    std::uint32_t offset = 0;
    std::size_t occupied = 0;
    for (std::size_t i = 0; i < total; ++i) {
        if (!cell_surfels[i].empty()) {
            cells_[i].first = offset;
            cells_[i].count = static_cast<std::uint32_t>(cell_surfels[i].size());
            surfel_indices_.insert(surfel_indices_.end(),
                                   cell_surfels[i].begin(),
                                   cell_surfels[i].end());
            offset += cells_[i].count;
            occupied++;
        }
    }

    return occupied;
}

const SpatialCell& SpatialGrid::cell(std::uint32_t x, std::uint32_t y, std::uint32_t z) const {
    if (x >= dim_x_ || y >= dim_y_ || z >= dim_z_) {
        static const SpatialCell empty;
        return empty;
    }
    std::size_t idx = (static_cast<std::size_t>(z) * dim_x_ * dim_y_) +
                      (static_cast<std::size_t>(y) * dim_x_) +
                      static_cast<std::size_t>(x);
    return cells_[idx];
}

const SpatialCell& SpatialGrid::cell(std::size_t index) const {
    if (index >= cells_.size()) {
        static const SpatialCell empty;
        return empty;
    }
    return cells_[index];
}

std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>
SpatialGrid::position_to_cell(const Vec3& pos) const {
    std::uint32_t x = static_cast<std::uint32_t>(
        (pos.x - min_bound_.x) / cell_size_.x);
    std::uint32_t y = static_cast<std::uint32_t>(
        (pos.y - min_bound_.y) / cell_size_.y);
    std::uint32_t z = static_cast<std::uint32_t>(
        (pos.z - min_bound_.z) / cell_size_.z);

    if (x >= dim_x_) x = dim_x_ - 1;
    if (y >= dim_y_) y = dim_y_ - 1;
    if (z >= dim_z_) z = dim_z_ - 1;

    return {x, y, z};
}

const std::vector<std::uint32_t>& SpatialGrid::surfels_in_cell(std::size_t) const {
    return surfel_indices_;
}

} // namespace delirium::m4