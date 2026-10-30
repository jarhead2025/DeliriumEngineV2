#pragma once

#include "body_runtime.h"
#include "surfel_cache.h"
#include <vector>

namespace delirium {

/* ── Scene: collection of BodyRuntimes + shared surfel cache ─────── */
class Scene {
public:
    /* Add a body to the scene. */
    BodyId add_body(BodyRuntime* body);

    /* Remove a body from the scene. */
    void remove_body(BodyId id);

    /* Get a body by ID. */
    BodyRuntime* get_body(BodyId id);
    const BodyRuntime* get_body(BodyId id) const;

    /* Step all bodies by dt. */
    bool step(float dt, std::string* error = nullptr);

    /* Rebuild the shared surfel cache from all bodies. */
    bool rebuild_cache(SurfelCache& cache, float cam_dist, std::string* error = nullptr);

    /* Iterate over all bodies. */
    const std::vector<BodyRuntime*>& bodies() const { return bodies_; }
    std::vector<BodyRuntime*>& bodies_mut() { return bodies_; }

    /* Scene-level bounding box. */
    Vec3 bounding_center() const;
    float bounding_radius() const;

private:
    std::vector<BodyRuntime*> bodies_;
    BodyId next_id_ = 0;
};

} // namespace delirium