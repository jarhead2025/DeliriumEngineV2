#include "delirium_m1/body_runtime.hpp"

#include <cmath>
#include <iostream>
#include <string>

using namespace delirium::m1;

namespace {
int failures = 0;

void expect(bool condition, const char* label) {
    if (condition) std::cout << "PASS: " << label << "\n";
    else { std::cerr << "FAIL: " << label << "\n"; ++failures; }
}

BodyTopology make_topology() {
    BodyTopology body;
    for (int i = 0; i < 6; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 0, {0, 12});
    body.add_patch({2, 3, 4, 5}, 1, {12, 12});
    std::string error;
    expect(body.finalize(&error), "topology finalizes with valid anchors/materials");
    return body;
}

AnchorInfluence one_bone(BoneId bone) {
    AnchorInfluence influence;
    influence.bones[0] = bone;
    influence.weights[0] = 65535;
    return influence;
}

void test_topology_and_dirty_propagation() {
    BodyTopology topology = make_topology();
    expect(topology.patches_for_anchor(0).size() == 1, "anchor 0 has one incident patch");
    expect(topology.patches_for_anchor(2).size() == 2, "shared anchor has both incident patches");

    BodyRuntime runtime(std::move(topology));
    std::vector<Vec3> rest = {{0,0,0}, {1,0,0}, {0,1,0}, {1,1,0}, {2,1,0}, {2,0,0}};
    std::string error;
    expect(runtime.initialize(rest, &error), "runtime initializes from rest anchors");

    ArticulationRig rig;
    const BoneId moving = rig.add_bone(kInvalidBone, {});
    const BoneId fixed = rig.add_bone(kInvalidBone, {});
    for (AnchorId a = 0; a < rest.size(); ++a) {
        expect(rig.bind_anchor(a, one_bone(a == 2 ? moving : fixed), &error), "explicit anchor binding accepted");
    }
    rig.set_target_pose(moving, {{0.5f, 0.0f, 0.0f}});
    expect(runtime.bind_rig(std::move(rig), &error), "normalized articulation rig binds to body");

    for (int i = 0; i < 120; ++i) expect(runtime.step_articulation(1.0f / 120.0f, &error), "fixed articulation step publishes state");
    const BodyState& state = runtime.read_state();
    expect(state.generation == 120, "generation increases only through completed publishes");
    expect(state.dirty_anchors.marked(2), "moved anchor is dirty");
    expect(!state.dirty_anchors.marked(0), "unbound-to-moving anchor remains clean");
    expect(state.dirty_patches.marked(0) && state.dirty_patches.marked(1), "shared anchor dirties both incident patches");
    expect(state.anchors[2].position.x > rest[2].x + 0.2f, "spring articulation moves influenced anchor");
    expect(nearly_equal(state.anchors[0].position, rest[0]), "fixed influence preserves unrelated anchor");
}

void test_invalid_influence_is_rejected() {
    ArticulationRig rig;
    rig.add_bone(kInvalidBone, {});
    AnchorInfluence bad;
    bad.bones[0] = 0;
    bad.weights[0] = 65000;
    std::string error;
    expect(!rig.bind_anchor(0, bad, &error), "non-normalized influence rejected");
}

void test_material_boundaries_are_topological() {
    BodyTopology body;
    for (int i = 0; i < 4; ++i) body.add_anchor();
    body.add_patch({0, 1, 2, 3}, 7, {0, 8});
    std::string error;
    expect(body.finalize(&error), "material-bearing patch finalizes");
    expect(body.patch(0).material == 7, "patch retains explicit material ID for future composition");
}

} // namespace

int main() {
    test_topology_and_dirty_propagation();
    test_invalid_influence_is_rejected();
    test_material_boundaries_are_topological();
    std::cout << (failures ? "M1 TESTS FAILED\n" : "M1 TESTS PASSED\n");
    return failures ? 1 : 0;
}
