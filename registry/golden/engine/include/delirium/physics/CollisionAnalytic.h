// =============================================================================
// CollisionAnalytic.h - Contacts directly on the equations: no proxy
// mesh. Body-vs-plane finds the deepest surface points per patch via
// Newton on (u,v) minimizing the plane distance; contact normals are
// exact, and each contact carries (patch,u,v) so impacts can feed the
// modal deformation at the precise surface location.
// =============================================================================
#pragma once
#include <vector>
#include "../psf/PSFBody.h"
#include "../core/GameObject.h"

namespace DeliRium {

struct Contact {
    Vec3 point, normal;      // world space
    float depth;             // penetration (>0 = inside)
    uint32_t patch; float u, v;
};

class CollisionAnalytic {
public:
    // plane: n . x = d (n unit, world space)
    static void BodyVsPlane(const PSFBody& body, const Transform& xf,
                            const Vec3& n, float d,
                            std::vector<Contact>& out, int max_contacts = 8);
    static bool Raycast(const PSFBody& body, const Transform& xf,
                        const Vec3& o, const Vec3& dir, Contact& out);
};

} // namespace DeliRium
