#include "delirium_m6/contact_detector.hpp"
namespace delirium::m6 {
ContactDetector::ContactDetector(float mp) : min_penetration_(mp) {}
std::vector<ContactPair> ContactDetector::detect(const std::vector<Surfel>& a, const std::vector<Surfel>& b,
                                                   BodyId ba, BodyId bb) const {
    std::vector<ContactPair> contacts;
    for (std::uint32_t i = 0; i < a.size(); ++i) {
        for (std::uint32_t j = 0; j < b.size(); ++j) {
            Vec3 diff = a[i].position - b[j].position;
            float dist = length(diff);
            float overlap = a[i].radius + b[j].radius - dist;
            if (overlap > min_penetration_) {
                ContactPair cp; cp.body_a = ba; cp.body_b = bb;
                cp.surfel_a = i; cp.surfel_b = j; cp.penetration = overlap;
                cp.contact_point = (a[i].position + b[j].position) * 0.5f;
                cp.normal = dist > 1e-8f ? diff * (1.0f/dist) : a[i].normal;
                contacts.push_back(cp);
            }
        }
    }
    return contacts;
}
} // namespace delirium::m6