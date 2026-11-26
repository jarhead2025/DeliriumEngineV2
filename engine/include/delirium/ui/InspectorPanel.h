// ============================================================================
// InspectorPanel.h — WS4 in-3D physics-first editing panel header
// ============================================================================
// M4 scaffolding: displays editable physical properties grouped by module.
// Units displayed for every field. Assumptions documented per field.
// Part of v2.2Beta milestone M4 (WS4).
// ============================================================================

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <functional>

namespace delirium {

// Forward declarations
class PhysicsWorld;
class PSFBody;
class GameObject;
class FieldVisualizer;

// ============================================================================
// PropertyField — a single editable physical property
// ============================================================================
struct PropertyField {
    std::string name;        // Display name (e.g. "Mass", "Charge")
    std::string unit;        // Physical unit (e.g. "kg", "C")
    double value;            // Current value
    double min;              // Lower bound for editing
    double max;              // Upper bound for upper bound
    bool is_dirty;           // Has been modified since last commit
    std::string assumption;  // Documented assumption for this field

    PropertyField() : value(0.0), min(0.0), max(1e6), is_dirty(false) {}
    PropertyField(const std::string& n, const std::string& u, double v,
                  double lo, double hi, const std::string& asm_ = "")
        : name(n), unit(u), value(v), min(lo), max(hi), is_dirty(false), assumption(asm_) {}
};

// ============================================================================
// PropertyGroup — grouped properties for one physics module
// ============================================================================
struct PropertyGroup {
    std::string title;                       // Module name (e.g. "Mass & Inertia")
    std::vector<PropertyField> fields;       // Editable fields in this group

    PropertyGroup() = default;
    explicit PropertyGroup(const std::string& t) : title(t) {}
};

// ============================================================================
// InspectorPanel — WS4 in-3D editing panel
// ============================================================================
// Displays editable physical properties for the selected GameObject.
// Groups fields by physics module. Each field shows units + assumptions.
// On commit, updates PhysicsWorld state and triggers re-bake if needed.
// Conservation ledger checks for non-physical changes before applying.
//
// Usage:
//   InspectorPanel panel(world);
//   panel.SelectObject(obj_handle);
//   panel.Render();        // Draw panel UI
//   panel.CommitChanges(); // Apply edits, check conservation ledger
// ============================================================================
class InspectorPanel {
public:
    explicit InspectorPanel(PhysicsWorld* world);
    ~InspectorPanel() = default;

    // Select an object to inspect (by GameObject handle)
    void SelectObject(uint32_t handle);

    // Clear selection
    void ClearSelection();

    // Refresh property values from current PhysicsWorld state
    void Refresh();

    // Render the panel UI (SDL rendering calls)
    void Render();

    // Commit pending edits — checks conservation ledger before applying
    // Returns true if changes were applied, false if rejected by ledger
    bool CommitChanges();

    // Get the currently selected object handle (0 = none)
    uint32_t SelectedHandle() const { return selected_handle_; }

    // Get the selected object (nullptr if none)
    GameObject* SelectedObject() const;

    // Check if any field has pending edits
    bool HasPendingEdits() const;

private:
    // Populate property groups from selected object's physics state
    void BuildPropertyGroups();

    // Check conservation ledger before committing a field change
    // Returns true if the change is physically valid
    bool CheckConservation(const PropertyField& field, double old_value) const;

    // Apply a single field edit to PhysicsWorld state
    void ApplyFieldEdit(PropertyField& field);

    PhysicsWorld* world_;              // Physics world (read/write state)
    uint32_t selected_handle_;         // Selected GameObject handle (0 = none)
    std::vector<PropertyGroup> groups_; // Property groups for selected object
    bool needs_refresh_;               // Rebuild groups on next Refresh()
};

} // namespace delirium
