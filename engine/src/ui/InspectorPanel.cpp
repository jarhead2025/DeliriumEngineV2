// ============================================================================
// InspectorPanel.cpp — WS4 in-3D physics-first editing panel implementation
// ============================================================================
// M4 scaffolding: displays editable physical properties grouped by module.
// Units displayed for every field. Assumptions documented per field.
// Part of v2.2Beta milestone M4 (WS4).
// ============================================================================

#include "delirium/ui/InspectorPanel.h"
#include "delirium/physics/PhysicsWorld.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/core/GameObject.h"
#include "delirium/core/Scene.h"
#include "delirium/render/SplatRenderer.h"
#include "delirium/core/ConservationLedger.h"

#include <SDL2/SDL.h>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace delirium {

InspectorPanel::InspectorPanel(PhysicsWorld* world)
    : world_(world), selected_handle_(0), needs_refresh_(true) {
    if (!world_) {
        // Create a default physics world if none provided
        world_ = new PhysicsWorld();
    }
}

void InspectorPanel::SelectObject(uint32_t handle) {
    selected_handle_ = handle;
    needs_refresh_ = true;
}

void InspectorPanel::ClearSelection() {
    selected_handle_ = 0;
    groups_.clear();
    needs_refresh_ = false;
}

GameObject* InspectorPanel::SelectedObject() const {
    if (!world_ || selected_handle_ == 0) return nullptr;
    return world_->GetScene()->GetObject(selected_handle_);
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
    if (!needs_refresh_ && !HasPendingEdits()) return;
    BuildPropertyGroups();
    needs_refresh_ = false;
}

void InspectorPanel::BuildPropertyGroups() {
    groups_.clear();
    GameObject* obj = SelectedObject();
    if (!obj) return;

    PSFBody* body = obj->Body();
    if (!body) return;

    // Group 1: Mass & Inertia
    {
        PropertyGroup g("Mass & Inertia");
        g.fields.emplace_back("Mass", "kg", body->mass(), 0.0, 1e6,
            "Uniform density assumption; mass = density × volume");
        g.fields.emplace_back("Density", "kg/m³", body->density(), 0.0, 1e6,
            "Material property; used to compute mass from volume");
        g.fields.emplace_back("Volume", "m³", body->volume(), 0.0, 1e6,
            "Computed from PSF surface mesh");
        g.fields.emplace_back("Inertia Ixx", "kg·m²", body->inertia(0, 0), 0.0, 1e12,
            "Diagonal inertia tensor component");
        g.fields.emplace_back("Inertia Iyy", "kg·m²", body->inertia(1, 1), 0.0, 1e12,
            "Diagonal inertia tensor component");
        g.fields.emplace_back("Inertia Izz", "kg·m²", body->inertia(2, 2), 0.0, 1e12,
            "Diagonal inertia tensor component");
        groups_.push_back(g);
    }

    // Group 2: Charge & Electromagnetic
    {
        PropertyGroup g("Charge & Electromagnetic");
        g.fields.emplace_back("Net Charge", "C", body->charge(), -1e6, 1e6,
            "Total charge; assumed point charge for far-field");
        g.fields.emplace_back("Charge Density", "C/m³", body->charge_density(), -1e6, 1e6,
            "Uniform charge density assumption");
        g.fields.emplace_back("Permittivity", "F/m", body->permittivity(), 1e-12, 1e-6,
            "Material permittivity; vacuum = 8.854e-12 F/m");
        groups_.push_back(g);
    }

    // Group 3: Thermal
    {
        PropertyGroup g("Thermal");
        g.fields.emplace_back("Surface Temp", "K", body->surface_temperature(), 0.0, 1e6,
            "Average surface temperature; distributed field in PathTempField");
        g.fields.emplace_back("Thermal Conductivity", "W/(m·K)", body->thermal_conductivity(), 0.0, 1e3,
            "Material property; governs heat diffusion");
        g.fields.emplace_back("Specific Heat", "J/(kg·K)", body->specific_heat(), 0.0, 1e4,
            "Material property; energy per unit mass per degree");
        groups_.push_back(g);
    }

    // Group 4: Material & Stress
    {
        PropertyGroup g("Material & Stress");
        g.fields.emplace_back("Young's Modulus", "Pa", body->youngs_modulus(), 0.0, 1e12,
            "Material stiffness; governs elastic deformation");
        g.fields.emplace_back("Poisson Ratio", "—", body->poisson_ratio(), 0.0, 0.5,
            "Lateral strain ratio; dimensionless");
        g.fields.emplace_back("Yield Strength", "Pa", body->yield_strength(), 0.0, 1e12,
            "Stress at which plastic deformation begins");
        g.fields.emplace_back("Max Stress", "Pa", body->max_stress(), 0.0, 1e12,
            "Current maximum von Mises stress in body");
        groups_.push_back(g);
    }

    // Group 5: Gravity & Motion
    {
        PropertyGroup g("Gravity & Motion");
        g.fields.emplace_back("Velocity", "m/s", body->velocity().Length(), 0.0, 1e6,
            "Linear velocity magnitude");
        g.fields.emplace_back("Angular Vel", "rad/s", body->omega().Length(), 0.0, 1e6,
            "Angular velocity magnitude");
        g.fields.emplace_back("Gravity Scale", "—", body->gravity_scale(), 0.0, 10.0,
            "Multiplier on gravitational acceleration; 1 = full gravity");
        groups_.push_back(g);
    }
}

bool InspectorPanel::CheckConservation(const PropertyField& field, double old_value) const {
    // Conservation ledger checks for non-physical changes
    // Example checks:
    // - Mass cannot go negative
    // - Charge conservation (if body is part of a system)
    // - Energy conservation (kinetic + potential + thermal)
    // - Momentum conservation

    if (field.name == "Mass" && field.value < 0.0) {
        return false; // Mass cannot be negative
    }
    if (field.name == "Density" && field.value < 0.0) {
        return false; // Density cannot be negative
    }
    if (field.name == "Charge" && std::abs(field.value - old_value) > 1e-3) {
        // Charge conservation check — flag if significant change
        // In a full system, this would check total charge conservation
        return true; // Allow for now; full system check deferred
    }
    return true; // Default: allow
}

void InspectorPanel::ApplyFieldEdit(PropertyField& field) {
    GameObject* obj = SelectedObject();
    if (!obj) return;
    PSFBody* body = obj->Body();
    if (!body) return;

    // Apply edit to body state based on field name
    if (field.name == "Mass") {
        body->set_mass(field.value);
    } else if (field.name == "Density") {
        body->set_density(field.value);
    } else if (field.name == "Charge") {
        body->set_charge(field.value);
    } else if (field.name == "Charge Density") {
        body->set_charge_density(field.value);
    } else if (field.name == "Surface Temp") {
        body->set_surface_temperature(field.value);
    } else if (field.name == "Thermal Conductivity") {
        body->set_thermal_conductivity(field.value);
    } else if (field.name == "Young's Modulus") {
        body->set_youngs_modulus(field.value);
    } else if (field.name == "Gravity Scale") {
        body->set_gravity_scale(field.value);
    }

    // Mark body dirty so render + physics recompute
    body->SetDirty();
    field.is_dirty = false;
}

bool InspectorPanel::CommitChanges() {
    if (!HasPendingEdits()) return false;

    // Check conservation ledger before applying each edit
    for (auto& group : groups_) {
        for (auto& field : group.fields) {
            if (!field.is_dirty) continue;
            double old_value = field.value; // Would need to store old values separately
            if (!CheckConservation(field, old_value)) {
                // Rejected by ledger — revert edit
                field.is_dirty = false;
                return false;
            }
            ApplyFieldEdit(field);
        }
    }
    return true;
}

void InspectorPanel::Render() {
    // SDL rendering calls would go here
    // For now, this is a placeholder — actual UI rendering deferred to M4 implementation
    Refresh();

    if (groups_.empty()) return;

    // Render group headers + fields (placeholder SDL calls)
    for (const auto& group : groups_) {
        // Render group title
        // Render each field with name, value, unit, assumption tooltip
    }
}

} // namespace delirium
