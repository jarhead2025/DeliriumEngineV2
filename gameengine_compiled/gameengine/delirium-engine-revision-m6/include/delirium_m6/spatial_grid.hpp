#pragma once
#include "delirium_m6/types.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>
namespace delirium::m6 {
class SpatialGrid {
public:
    SpatialGrid();
    explicit SpatialGrid(std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z);
    std::size_t total_cells() const { return static_cast<std::size_t>(dim_x_) * dim_y_ * dim_z_; }
    void clear();
    std::size_t build(const std::vector<Surfel>& surfels);
    const SpatialCell& cell(std::uint32_t x, std::uint32_t y, std::uint32_t z) const;
    const SpatialCell& cell(std::size_t index) const;
    std::tuple<std::uint32_t, std::uint32_t, std::uint32_t> position_to_cell(const Vec3& pos) const;
private:
    std::uint32_t dim_x_ = 32, dim_y_ = 32, dim_z_ = 32;
    std::vector<SpatialCell> cells_;
    std::vector<std::uint32_t> surfel_indices_;
    Vec3 min_bound_{}, max_bound_{}, cell_size_{};
};
} // namespace delirium::m6