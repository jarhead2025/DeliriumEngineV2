#include "delirium_m7/contact_detector.hpp"

namespace delirium::m7 {

ContactDetector::ContactDetector(float min_penetration) : min_penetration_(min_penetration) {}

std::vector<ContactPair> ContactDetector::detect(const std::vector<Surfel>& a,
                                                  const std::vector<Surfel>& b,
                                                  BodyId body_a, BodyId body_b) const {
    std::vector<ContactPair> contacts;

    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < b.size(); ++j) {
            Vec3 diff = a[i].position - b[j].position;
            float dist = length(diff);
            float overlap = (a[i].radius + b[j].radius) - dist;

            if (overlap > min_penetration_) {
                ContactPair cp;
                cp.body_a = body_a;
                cp.body_b = body_b;
                cp.surfel_a = static_cast<std::uint32_t>(i);
                cp.surfel_b = static_cast<std::uint32_t>(j);
                cp.contact_point = (a[i].position + b[j].position) * 0.5f;
                cp.normal = dist > 1e-8f ? diff * (1.0f / dist) : a[i].normal;
                cp.penetration = overlap;
                contacts.push_back(cp);
            }
        }
    }
    return contacts;
}

} // namespace delirium::m7