#include "delirium_m8/multi_body_manager.hpp"
#include <algorithm>

namespace delirium::m8 {

MultiBodyManager::MultiBodyManager()
    : contact_detector_(0.0f), collision_response_() {}

MultiBodyManager::MultiBodyManager(SceneBudget budget)
    : budget_(std::move(budget)), contact_detector_(0.0f), collision_response_() {}

bool MultiBodyManager::is_within_budget() const {
    std::uint32_t active = static_cast<std::uint32_t>(active_body_count());
    std::uint32_t surfels = total_surfels();
    std::uint32_t constraints = total_constraints();
    return budget_.is_within_budget(active, surfels, constraints);
}

std::uint32_t MultiBodyManager::total_surfels() const {
    std::uint32_t total = 0;
    for (const auto& b : bodies_) {
        if (!b.active) continue;
        // Each patch generates ~4 surfels (one per corner anchor)
        total += static_cast<std::uint32_t>(b.runtime.read_state().patches.size()) * 4;
    }
    return total;
}

std::uint32_t MultiBodyManager::total_constraints() const {
    std::uint32_t total = 0;
    for (const auto& b : bodies_) {
        if (!b.active) continue;
        total += static_cast<std::uint32_t>(b.runtime.constraint_solver().constraint_count());
    }
    return total;
}

BodyId MultiBodyManager::add_body(BodyRuntime runtime) {
    std::uint32_t active = static_cast<std::uint32_t>(active_body_count()) + 1;
    if (active > budget_.max_bodies) {
        // Budget exceeded — still add but caller should check
    }
    bodies_.emplace_back(next_id_, std::move(runtime));
    return next_id_++;
}

void MultiBodyManager::remove_body(BodyId id) {
    for (auto& b : bodies_) {
        if (b.id == id) {
            b.active = false;
            break;
        }
    }
}

BodyRuntime* MultiBodyManager::get_body(BodyId id) {
    for (auto& b : bodies_) {
        if (b.id == id && b.active) return &b.runtime;
    }
    return nullptr;
}

const BodyRuntime* MultiBodyManager::get_body(BodyId id) const {
    for (const auto& b : bodies_) {
        if (b.id == id && b.active) return &b.runtime;
    }
    return nullptr;
}

std::size_t MultiBodyManager::active_body_count() const {
    std::size_t count = 0;
    for (const auto& b : bodies_) {
        if (b.active) ++count;
    }
    return count;
}

std::vector<BodyId> MultiBodyManager::get_visible_bodies(const Frustum& frustum) const {
    std::vector<BodyId> active_ids;
    std::vector<Vec3> centers;
    active_ids.reserve(bodies_.size());
    centers.reserve(bodies_.size());

    for (const auto& b : bodies_) {
        if (!b.active) continue;
        active_ids.push_back(b.id);
        centers.push_back(b.runtime.bounding_center());
    }
    return culler_.cull(active_ids, centers, frustum);
}

std::vector<Vec3> MultiBodyManager::get_body_centers() const {
    std::vector<Vec3> centers;
    centers.reserve(bodies_.size());
    for (const auto& b : bodies_) {
        if (b.active) centers.push_back(b.runtime.bounding_center());
    }
    return centers;
}

PickResult MultiBodyManager::pick(const Vec3& ray_origin, const Vec3& ray_dir, float max_dist) {
    std::vector<BodyRuntime*> ptrs;
    ptrs.reserve(bodies_.size());
    for (auto& b : bodies_) {
        if (b.active) ptrs.push_back(&b.runtime);
    }
    return interaction_.pick(ptrs, ray_origin, ray_dir, max_dist);
}

void MultiBodyManager::set_external_forces(ExternalForce f) {
    external_force_ = std::move(f);
}

void MultiBodyManager::add_point_force(PointForce pf) {
    point_forces_.push_back(std::move(pf));
}

void MultiBodyManager::set_fracture_params(FractureParams p) {
    fracture_system_.set_params(std::move(p));
}

bool MultiBodyManager::step(float fixed_dt, const Frustum* frustum, std::string* error) {
    std::vector<BodyId> to_step;

    if (frustum) {
        to_step = get_visible_bodies(*frustum);
    } else {
        for (const auto& b : bodies_) {
            if (b.active) to_step.push_back(b.id);
        }
    }

    for (BodyId id : to_step) {
        auto* body = get_body(id);
        if (!body) continue;

        body->external_forces_mut().set_force(external_force_);
        for (const auto& pf : point_forces_) {
            if (pf.body == id) {
                body->external_forces_mut().add_point_force(pf);
            }
        }

        if (!body->step(fixed_dt, error)) return false;
    }

    resolve_all_collisions();

    auto events = check_fractures();
    for (const auto& e : events) {
        if (e.triggered) {
            auto* body = get_body(e.source_body);
            if (body) {
                // Fracture detected — mark for removal
            }
        }
    }

    return true;
}

std::vector<ContactPair> MultiBodyManager::detect_all_contacts() {
    std::vector<ContactPair> all_contacts;

    for (std::size_t i = 0; i < bodies_.size(); ++i) {
        if (!bodies_[i].active) continue;
        for (std::size_t j = i + 1; j < bodies_.size(); ++j) {
            if (!bodies_[j].active) continue;

            SurfelCache cache_a, cache_b;
            std::string error;
            bodies_[i].runtime.rebuild_cache(cache_a, &error);
            bodies_[j].runtime.rebuild_cache(cache_b, &error);

            auto contacts = contact_detector_.detect(
                cache_a.surfels(), cache_b.surfels(),
                bodies_[i].id, bodies_[j].id);
            all_contacts.insert(all_contacts.end(), contacts.begin(), contacts.end());
        }
    }
    return all_contacts;
}

void MultiBodyManager::resolve_all_collisions() {
    auto contacts = detect_all_contacts();
    for (const auto& cp : contacts) {
        auto* body_a = get_body(cp.body_a);
        auto* body_b = get_body(cp.body_b);
        if (!body_a || !body_b) continue;

        std::vector<ContactPair> single{cp};
        collision_response_.resolve(
            body_a->write_state(), body_b->write_state(),
            single, 0.016f);
    }
}

std::vector<FractureEvent> MultiBodyManager::check_fractures() const {
    std::vector<FractureEvent> events;
    for (const auto& b : bodies_) {
        if (!b.active) continue;
        auto event = fracture_system_.evaluate(
            b.runtime.read_state(), b.runtime.topology(), b.id);
        events.push_back(event);
    }
    return events;
}

} // namespace delirium::m8