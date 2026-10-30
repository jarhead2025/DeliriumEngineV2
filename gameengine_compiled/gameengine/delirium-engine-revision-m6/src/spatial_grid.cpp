#include "delirium_m6/spatial_grid.hpp"
#include <algorithm>
namespace delirium::m6 {
SpatialGrid::SpatialGrid() = default;
SpatialGrid::SpatialGrid(std::uint32_t dx, std::uint32_t dy, std::uint32_t dz)
    : dim_x_(dx), dim_y_(dy), dim_z_(dz) {}
void SpatialGrid::clear() { cells_.clear(); surfel_indices_.clear(); }
std::size_t SpatialGrid::build(const std::vector<Surfel>& surfels) {
    if (surfels.empty()) return 0;
    min_bound_ = max_bound_ = surfels[0].position;
    for (const auto& s : surfels) {
        if (s.position.x < min_bound_.x) min_bound_.x = s.position.x;
        if (s.position.y < min_bound_.y) min_bound_.y = s.position.y;
        if (s.position.z < min_bound_.z) min_bound_.z = s.position.z;
        if (s.position.x > max_bound_.x) max_bound_.x = s.position.x;
        if (s.position.y > max_bound_.y) max_bound_.y = s.position.y;
        if (s.position.z > max_bound_.z) max_bound_.z = s.position.z;
    }
    Vec3 pad{0.001f,0.001f,0.001f}; min_bound_ -= pad; max_bound_ += pad;
    cell_size_.x = (max_bound_.x-min_bound_.x)/dim_x_;
    cell_size_.y = (max_bound_.y-min_bound_.y)/dim_y_;
    cell_size_.z = (max_bound_.z-min_bound_.z)/dim_z_;
    std::size_t total = static_cast<std::size_t>(dim_x_)*dim_y_*dim_z_;
    cells_.assign(total, SpatialCell{});
    std::vector<std::vector<std::uint32_t>> cs(total);
    for (std::uint32_t i = 0; i < surfels.size(); ++i) {
        auto [cx,cy,cz] = position_to_cell(surfels[i].position);
        std::size_t idx = (static_cast<std::size_t>(cz)*dim_x_*dim_y_) + (static_cast<std::size_t>(cy)*dim_x_) + cx;
        if (idx < total) { cs[idx].push_back(i); cells_[idx].material_mask |= (1u << (surfels[i].material & 31)); }
    }
    std::uint32_t off = 0; std::size_t occ = 0;
    for (std::size_t i = 0; i < total; ++i) {
        if (!cs[i].empty()) {
            cells_[i].first = off; cells_[i].count = static_cast<std::uint32_t>(cs[i].size());
            surfel_indices_.insert(surfel_indices_.end(), cs[i].begin(), cs[i].end());
            off += cells_[i].count; occ++;
        }
    }
    return occ;
}
const SpatialCell& SpatialGrid::cell(std::uint32_t x, std::uint32_t y, std::uint32_t z) const {
    if (x>=dim_x_||y>=dim_y_||z>=dim_z_) { static const SpatialCell e; return e; }
    return cells_[(static_cast<std::size_t>(z)*dim_x_*dim_y_)+(static_cast<std::size_t>(y)*dim_x_)+x];
}
const SpatialCell& SpatialGrid::cell(std::size_t index) const {
    if (index >= cells_.size()) { static const SpatialCell e; return e; }
    return cells_[index];
}
std::tuple<std::uint32_t,std::uint32_t,std::uint32_t> SpatialGrid::position_to_cell(const Vec3& p) const {
    std::uint32_t x = static_cast<std::uint32_t>((p.x-min_bound_.x)/cell_size_.x);
    std::uint32_t y = static_cast<std::uint32_t>((p.y-min_bound_.y)/cell_size_.y);
    std::uint32_t z = static_cast<std::uint32_t>((p.z-min_bound_.z)/cell_size_.z);
    if (x>=dim_x_) x=dim_x_-1;
    if (y>=dim_y_) y=dim_y_-1;
    if (z>=dim_z_) z=dim_z_-1;
    return {x,y,z};
}
} // namespace delirium::m6