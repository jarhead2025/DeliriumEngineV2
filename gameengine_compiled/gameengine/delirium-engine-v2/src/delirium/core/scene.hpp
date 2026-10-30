#pragma once
#include "delirium/geometry/patch/patch.hpp"
#include "delirium/render/surfel/surfel_cache.hpp"
#include "delirium/math/mat4.hpp"
#include <vector>

namespace delirium {

struct Camera {
    vec3 position;
    vec3 target;
    vec3 up;
    float fov;
    float aspect;
    mat4 view;
    mat4 projection;

    void update_matrices() {
        view = mat4::look_at(position, target, up);
        projection = mat4::perspective(fov, aspect, 0.1f, 100.0f);
    }
};

class Scene {
public:
    std::vector<AnchorRecord> anchors;
    std::vector<Patch> patches;
    std::vector<BoneRecord> bones;
    std::vector<MaterialRecord> materials;

    SurfelCache surfel_cache;
    Camera camera;

    Scene();

    /// Build a demo scene (8×8 grid of patches)
    void build_demo();

    /// Regenerate surfel cache from current patch state
    void update_surfel_cache();

    SurfelCache& get_surfel_cache() { return surfel_cache; }
    Camera& get_camera() { return camera; }

    /// Global generation counter (increments when any patch changes)
    uint32_t generation() const { return generation_id_; }
    void increment_generation() { generation_id_++; }

private:
    uint32_t generation_id_ = 0;
};

} // namespace delirium