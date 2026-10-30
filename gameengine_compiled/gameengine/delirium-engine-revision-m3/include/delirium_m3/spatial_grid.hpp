#pragma once

#include "delirium_m3/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace delirium::m3 {

// ---------------------------------------------------------------------------
// SpatialGrid — multi-surfel spatial cells
// ---------------------------------------------------------------------------
//
// Replaces the legacy single-surfel-per-cell grid. Each cell holds a compact
// list of surfel indices. Material identity is preserved per-surfel, and
// material masks allow quick rejection of incompatible blending.
// ---------------------------------------------------------------------------

class SpatialGrid {
public:
    // Construct a 3D grid with the given dimensions
    SpatialGrid();
    explicit SpatialGrid(std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z);

    // Grid dimensions
    std::uint32_t dim_x() const { return dim_x_; }
    std::uint32_t dim_y() const { return dim_y_; }
    std::uint32_t dim_z() const { return dim_z_; }
    std::size_t total_cells() const { return static_cast<std::size_t>(dim_x_) * dim_y_ * dim_z_; }

    // Clear all cells
    void clear();

    // Build the grid from a surfel array
    // Returns the number of occupied cells
    std::size_t build(const std::vector<Surfel>& surfels);

    // Get cell at grid coordinates
    const SpatialCell& cell(std::uint32_t x, std::uint32_t y, std::uint32_t z) const;

    // Get cell at linear index
    const SpatialCell& cell(std::size_t index) const;

    // Convert world position to grid coordinates
    std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>
    position_to_cell(const Vec3& pos) const;

    // Get all surfel indices in a cell
    const std::vector<std::uint32_t>& surfels_in_cell(std::size_t cell_index) const;

    // Bounding box of the grid
    Vec3 min_bound() const { return min_bound_; }
    Vec3 max_bound() const { return max_bound_; }

private:
    std::uint32_t dim_x_ = 32;
    std::uint32_t dim_y_ = 32;
    std::uint32_t dim_z_ = 32;

    std::vector<SpatialCell> cells_;
    std::vector<std::uint32_t> surfel_indices_;  // flat array of all surfel indices
    Vec3 min_bound_{};
    Vec3 max_bound_{};
    Vec3 cell_size_{};
};

} // namespace delirium::m3