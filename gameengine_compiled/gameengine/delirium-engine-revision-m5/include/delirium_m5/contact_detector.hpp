#pragma once

#include "delirium_m5/types.hpp"
#include <vector>

namespace delirium::m5 {

// ContactDetector — finds overlapping surfels between two bodies using spatial grids
class ContactDetector {
public:
    explicit ContactDetector(float min_penetration = 0.0f);

    // Detect contacts between two surfel arrays
    std::vector<ContactPair> detect(const std::vector<Surfel>& surfels_a,
                                    const std::vector<Surfel>& surfels_b,
                                    BodyId body_a, BodyId body_b) const;

    // Detect contacts using spatial grids for broad-phase culling
    std::vector<ContactPair> detect_with_grids(const std::vector<Surfel>& surfels_a,
                                                const std::vector<Surfel>& surfels_b,
                                                BodyId body_a, BodyId body_b) const;

    float min_penetration() const { return min_penetration_; }
    void set_min_penetration(float p) { min_penetration_ = p; }

private:
    float min_penetration_;
};

} // namespace delirium::m5