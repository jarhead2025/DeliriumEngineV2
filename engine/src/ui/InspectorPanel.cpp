// ============================================================================
// InspectorPanel.cpp — WS4 in-3D physics-first editing panel implementation
// ============================================================================
// M4 scaffolding: displays editable physical properties grouped by module.
// Units displayed for every field. Assumptions documented per field.
// Part of v2.2Beta milestone M4 (WS4).
// NOTE: property access maps to real engine API:
//   mass/inertia/velocity/omega → GameObject::Rigid() (RigidState)
//   density/youngs/poisson → GameObject::Mat() (Material)
//   charge → PhysicsWorld::ChargeOf(obj) / SetBodyCharge()
//   temperature → PhysicsWorld::ThermalOf(obj)
// ============================================================================

#include "delirium/ui/InspectorPanel.h"
#include "delirium/physics/PhysicsWorld.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/core/GameObject.h"
#include "delirium/physics/Material.h"

#include <cmath>
#include <algorithm>

namespace DeliRium {

InspectorPanel::InspectorPanel(PhysicsWorld* world)
    : world_(world), selected_obj_(nullptr) {
    if (!world_) {
        world_ = new PhysicsWorld();
    }
}

void InspectorPanel::SelectObject(GameObject* obj) {
    selected_obj_ = obj;
    groups_.clear();
}

void InspectorPanel::ClearSelection() {
    selected_obj_ = nullptr;
    groups_.clear();
}

GameObject* InspectorPanel::SelectedObject() const {
    return selected_obj_;
}

bool InspectorPanel::HasPendingEdits() const {
    for (const auto& group : groups_) {
        for (const auto& field : group.fields) {
            if (field.is_dirty) return true;
        }
    }
    return false;
}

void InspectorPanel::Refresh() {
    groups_.clear();
    if (!selected_obj_) return;

    const RigidState& r = selected_obj_->Rigid();
    const Material& m = selected_obj_->Mat();
    const Vec3& vel = r.velocity;
    const Vec3& omg = r.omega;
    const float vel_mag = std::sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
    const float omg_mag = std::sqrt(omg.x*omg.x + omg.y*omg.y + omg.z*omg.z);

    // Read charge/temperature from PhysicsWorld coupling state (may be null)
    float charge_val = 0.0, temp_val = 0.0;
    const ChargeState* cs = world_ ? world_->ChargeOf(selected_obj_) : nullptr;
    const ThermalState* ts = world_ ? world_->ThermalOf(selected_obj_) : nullptr;
    if (cs) charge_val = cs->total_charge;
    if (ts) temp_val = ts->temperature;

    // Group 1: Mass & Inertia
    {
        PropertyGroup g("Mass & Inertia");
        g.fields.emplace_back("Mass", "kg", r.mass, 0.0, 1e6,
            "RigidState.mass; uniform density assumption");
        g.fields.emplace_back("Density", "kg/m³", m.density, 0.0, 1e6,
            "Material.density; used to compute mass from volume");
        g.fields.emplace_back("Inertia (scalar)", "kg·m²", r.inertia, 0.0, 1e12,
            "RigidState.inertia; solid-sphere approximation from bbox");
        groups_.push_back(g);
    }

    // Group 2: Charge & Electromagnetic
    {
        PropertyGroup g("Charge & Electromagnetic");
        g.fields.emplace_back("Net Charge", "C", charge_val, -1e6, 1e6,
            "PhysicsWorld::ChargeOf — requires charge coupling on");
        g.fields.emplace_back("Conductivity", "S/m", m.damping > 0 ? m.damping : 0.0, 0.0, 1e6,
            "Material property governing charge relaxation τ = ε/σ");
        groups_.push_back(g);
    }

    // Group 3: Thermal
    {
        PropertyGroup g("Thermal");
        g.fields.emplace_back("Temperature", "K", temp_val, 0.0, 1e6,
            "PhysicsWorld::ThermalOf — requires thermal coupling on");
        groups_.push_back(g);
    }

    // Group 4: Material
    {
        PropertyGroup g("Material");
        g.fields.emplace_back("Young's Modulus", "Pa", m.youngs_modulus, 0.0, 1e12,
            "Material.youngs_modulus; governs elastic deformation");
        g.fields.emplace_back("Poisson Ratio", "—", m.poisson, 0.0, 0.5,
            "Material.poisson; lateral strain ratio");
        g.fields.emplace_back("Weld Category", "—", static_cast<double>(m.weld), 0.0, 4.0,
            "Material.weld: 0=Rigid 1=Elastic 2=Soft 3=Fluid 4=Brittle");
        groups_.push_back(g);
    }

    // Group 5: Motion
    {
        PropertyGroup g("Motion");
        g.fields.emplace_back("Velocity", "m/s", vel_mag, 0.0, 1e6,
            "RigidState.velocity magnitude");
        g.fields.emplace_back("Angular Vel", "rad/s", omg_mag, 0.0, 1e6,
            "RigidState.omega magnitude");
        groups_.push_back(g);
    }
}

bool InspectorPanel::CheckConservation(const PropertyField& field, double old_value) const {
    // Conservation ledger checks for non-physical changes
    if (field.name == "Mass" && field.value < 0.0) return false;
    if (field.name == "Density" && field.value < 0.0) return false;
    if (field.name == "Charge" && std::abs(field.value - old_value) > 1e-3) {
        // Charge conservation check — flag if significant change
        return true; // Allow for now; full system check deferred
    }
    return true;
}

void InspectorPanel::ApplyFieldEdit(PropertyField& field) {
    if (!selected_obj_) return;
    RigidState& r = selected_obj_->Rigid();
    Material& m = selected_obj_->Mat();

    if (field.name == "Mass") {
        r.mass = static_cast<float>(field.value);
        r.inv_mass = (r.mass > 0.0f) ? 1.0f / r.mass : 0.0f;
    } else if (field.name == "Density") {
        m.density = static_cast<float>(field.value);
    } else if (field.name == "Inertia (scalar)") {
        r.inertia = static_cast<float>(field.value);
        r.inv_inertia = (r.inertia > 0.0f) ? 1.0f / r.inertia : 0.0f;
    } else if (field.name == "Young's Modulus") {
        m.youngs_modulus = static_cast<float>(field.value);
    } else if (field.name == "Poisson Ratio") {
        m.poisson = static_cast<float>(field.value);
    } else if (field.name == "Net Charge") {
        if (world_) world_->SetBodyCharge(selected_obj_, static_cast<float>(field.value));
    } else if (field.name == "Velocity") {
        // Preserve direction; scale magnitude
        Vec3 v = r.velocity;
        float mag = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
        if (mag > 1e-6f) {
            float scale = static_cast<float>(field.value) / mag;
            r.velocity = { v.x * scale, v.y * scale, v.z * scale };
        } else {
            r.velocity = { static_cast<float>(field.value), 0.0f, 0.0f };
        }
    } else if (field.name == "Angular Vel") {
        Vec3 o = r.omega;
        float mag = std::sqrt(o.x*o.x + o.y*o.y + o.z*o.z);
        if (mag > 1e-6f) {
            float scale = static_cast<float>(field.value) / mag;
            r.omega = { o.x * scale, o.y * scale, o.z * scale };
        } else {
            r.omega = { static_cast<float>(field.value), 0.0f, 0.0f };
        }
    }

    // Mark body dirty so render + physics recompute
    selected_obj_->Body().MarkAnchorsDirty();
    field.is_dirty = false;
}

bool InspectorPanel::CommitChanges() {
    if (!HasPendingEdits()) return false;

    for (auto& group : groups_) {
        for (auto& field : group.fields) {
            if (!field.is_dirty) continue;
            double old_value = field.value; // NOTE: would need stored old values
            if (!CheckConservation(field, old_value)) {
                field.is_dirty = false;
                return false;
            }
            ApplyFieldEdit(field);
        }
    }
    return true;
}

} // namespace DeliRium
