#pragma once

#include "delirium_m5/types.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace delirium::m5 {

class SpatialGrid {
public:
    SpatialGrid();
    explicit SpatialGrid(std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z);
    std::uint32_t dim_x() const { return dim_x_; }
    std::uint32_t dim_y() const { return dim_y_; }
    std::uint32_t dim_z() const { return dim_z_; }
    std::size_t total_cells() const { return static_cast<std::size_t>(dim_x_) * dim_y_ * dim_z_; }
    void clear();
    std::size_t build(const std::vector<Surfel>& surfels);
    const SpatialCell& cell(std::uint32_t x, std::uint32_t y, std::uint32_t z) const;
    const SpatialCell& cell(std::size_t index) const;
    std::tuple<std::uint32_t, std::uint32_t, std::uint32_t> position_to_cell(const Vec3& pos) const;
    Vec3 min_bound() const { return min_bound_; }
    Vec3 max_bound() const { return max_bound_; }
private:
    std::uint32_t dim_x_ = 32, dim_y_ = 32, dim_z_ = 32;
    std::vector<SpatialCell> cells_;
    std::vector<std::uint32_t> surfel_indices_;
    Vec3 min_bound_{}, max_bound_{}, cell_size_{};
};

} // namespace delirium::m5