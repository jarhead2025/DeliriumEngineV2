#pragma once
#include "delirium_m8/types.hpp"
#include "delirium_m8/body_runtime.hpp"
#include "delirium_m8/contact_detector.hpp"
#include "delirium_m8/collision_response.hpp"
#include "delirium_m8/fracture_system.hpp"
#include "delirium_m8/scene_budget.hpp"
#include "delirium_m8/frustum_culler.hpp"
#include "delirium_m8/shared_material_pool.hpp"
#include "delirium_m8/interaction_layer.hpp"
#include <memory>
#include <vector>
#include <string>

namespace delirium::m8 {

class MultiBodyManager {
public:
    MultiBodyManager();
    explicit MultiBodyManager(SceneBudget budget);

    // Budget
    SceneBudget budget() const { return budget_; }
    void set_budget(SceneBudget b) { budget_ = b; }
    bool is_within_budget() const;
    std::uint32_t total_surfels() const;
    std::uint32_t total_constraints() const;

    // Bodies
    BodyId add_body(BodyRuntime runtime);
    void remove_body(BodyId id);
    BodyRuntime* get_body(BodyId id);
    const BodyRuntime* get_body(BodyId id) const;
    std::size_t body_count() const { return bodies_.size(); }
    std::size_t active_body_count() const;

    // Scene management
    std::vector<BodyId> get_visible_bodies(const Frustum& frustum) const;
    std::vector<Vec3> get_body_centers() const;

    // Materials
    SharedMaterialPool& material_pool() { return material_pool_; }
    const SharedMaterialPool& material_pool() const { return material_pool_; }

    // Interaction
    InteractionLayer& interaction() { return interaction_; }
    const InteractionLayer& interaction() const { return interaction_; }
    PickResult pick(const Vec3& ray_origin, const Vec3& ray_dir, float max_dist = 100.0f);

    // External forces
    void set_external_forces(ExternalForce f);
    void add_point_force(PointForce pf);

    // Fracture
    void set_fracture_params(FractureParams p);

    // Step — only steps visible bodies when frustum is set
    bool step(float fixed_dt, const Frustum* frustum = nullptr, std::string* error = nullptr);

    // Collision
    std::vector<ContactPair> detect_all_contacts();
    void resolve_all_collisions();
    std::vector<FractureEvent> check_fractures() const;

private:
    struct BodyEntry {
        BodyId id;
        BodyRuntime runtime;
        bool active;
        BodyEntry(BodyId i, BodyRuntime r) : id(i), runtime(std::move(r)), active(true) {}
    };

    std::vector<BodyEntry> bodies_;
    BodyId next_id_ = 0;

    SceneBudget budget_;
    FrustumCuller culler_;
    SharedMaterialPool material_pool_;
    InteractionLayer interaction_;

    ExternalForce external_force_;
    std::vector<PointForce> point_forces_;
    FractureSystem fracture_system_;
    ContactDetector contact_detector_;
    CollisionResponse collision_response_;
};

} // namespace delirium::m8