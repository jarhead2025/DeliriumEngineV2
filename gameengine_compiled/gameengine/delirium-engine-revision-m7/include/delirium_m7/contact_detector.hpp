#pragma once
#include "delirium_m7/types.hpp"
#include <vector>
namespace delirium::m7 {
class ContactDetector {
public:
    explicit ContactDetector(float min_penetration = 0.0f);
    std::vector<ContactPair> detect(const std::vector<Surfel>& a, const std::vector<Surfel>& b,
                                    BodyId body_a, BodyId body_b) const;
    float min_penetration() const { return min_penetration_; }
private:
    float min_penetration_;
};
} // namespace delirium::m7