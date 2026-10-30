#include "delirium_m5/contact_detector.hpp"
#include <algorithm>

namespace delirium::m5 {

ContactDetector::ContactDetector(float min_penetration) : min_penetration_(min_penetration) {}

std::vector<ContactPair> ContactDetector::detect(const std::vector<Surfel>& surfels_a,
                                                   const std::vector<Surfel>& surfels_b,
                                                   BodyId body_a, BodyId body_b) const {
    std::vector<ContactPair> contacts;
    for (std::uint32_t i = 0; i < surfels_a.size(); ++i) {
        for (std::uint32_t j = 0; j < surfels_b.size(); ++j) {
            Vec3 diff = surfels_a[i].position - surfels_b[j].position;
            float dist = length(diff);
            float overlap = surfels_a[i].radius + surfels_b[j].radius - dist;
            if (overlap > min_penetration_) {
                ContactPair cp;
                cp.body_a = body_a; cp.body_b = body_b;
                cp.surfel_a = i; cp.surfel_b = j;
                cp.penetration = overlap;
                cp.contact_point = (surfels_a[i].position + surfels_b[j].position) * 0.5f;
                cp.normal = dist > 1e-8f ? diff * (1.0f / dist) : surfels_a[i].normal;
                contacts.push_back(cp);
            }
        }
    }
    return contacts;
}

std::vector<ContactPair> ContactDetector::detect_with_grids(const std::vector<Surfel>& surfels_a,
                                                              const std::vector<Surfel>& surfels_b,
                                                              BodyId body_a, BodyId body_b) const {
    // For M5, fall back to brute-force (grid-based broadphase is a future optimization)
    return detect(surfels_a, surfels_b, body_a, body_b);
}

} // namespace delirium::m5