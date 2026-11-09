// =============================================================================
// move_probe - proves an object MOVES without re-quantizing its records.
//
// THE INVARIANT (this is the whole point of the Transform layer):
//   moving an object must NOT change the frozen 64-byte records.
//
// If placement lived inside the anchors, every drag would re-quantize the
// body: the anchors' bytes would change, the body's identity would be
// lost, and a saved scene would bake the placement in permanently. The
// Transform exists so the records stay byte-identical and the placement
// lives OUTSIDE them, applied as a GPU model matrix.
//
// The probe asserts that contract:
//   1. move the object -> the transform changes
//   2. move the object -> the anchor bytes are BIT-IDENTICAL
//   3. the world position of an anchor DOES change (the move is real)
//   4. the model matrix carries the translation
//   5. the matrix is invertible and its inverse returns the local point
//
// Also checks the matrix CONVENTION, because a transposed matrix silently
// drops the translation (the bug this layer was written to fix).
// =============================================================================
#include "delirium/core/Transform.h"
#include "delirium/core/GameObject.h"
#include "delirium/core/Scene.h"
#include "delirium/psf/PSFBody.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace DeliRium;

static int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...) do { \
    ++g_checks; \
    if (!(cond)) { ++g_fail; std::printf("FAIL %s:%d  ", __FILE__, __LINE__); \
                   std::printf(__VA_ARGS__); std::printf("\n"); } \
} while (0)

int main() {
    // --- build a real body ------------------------------------------------
    auto body = PSFBody::FromFile("assets/torus.psf");
    if (!body) {
        std::printf("move_probe: could not load assets/torus.psf\n");
        return 1;
    }
    Scene scene("probe");
    GameObject* obj = scene.Create("ring", std::move(body));
    CHECK(obj != nullptr, "scene created the object");
    if (!obj) return 1;
    CHECK(obj->HasBody(), "object has a body");

    // --- the record bytes BEFORE the move ---------------------------------
    const std::vector<AnchorRecord> before = obj->Body().AnchorBlob();
    CHECK(!before.empty(), "body has anchors");
    const size_t patch_bytes_before = obj->Body().SurfaceBlob().size();
    const Vec3 local_before = obj->Body().BBoxMin();

    // --- MOVE IT ----------------------------------------------------------
    Transform& tr = obj->GetTransform();
    tr.position = { 3.25f, -1.5f, 0.75f };

    // --- the records must be BIT-IDENTICAL -------------------------------
    const std::vector<AnchorRecord> after = obj->Body().AnchorBlob();
    CHECK(after.size() == before.size(), "anchor blob size unchanged");
    bool identical = (after.size() == before.size());
    if (identical) {
        for (size_t i = 0; i < before.size(); ++i)
            if (std::memcmp(&before[i], &after[i], sizeof(AnchorRecord)) != 0) {
                identical = false; break;
            }
    }
    CHECK(identical, "moving the object did NOT touch the frozen records");
    CHECK(obj->Body().SurfaceBlob().size() == patch_bytes_before,
          "moving the object did not change the surface records");

    // ...and the LOCAL geometry must be unchanged too (the bbox is local).
    const Vec3 local_after = obj->Body().BBoxMin();
    CHECK(std::fabs(local_after.x - local_before.x) < 1e-6f &&
          std::fabs(local_after.y - local_before.y) < 1e-6f &&
          std::fabs(local_after.z - local_before.z) < 1e-6f,
          "the local bbox did not move");

    // --- but the WORLD position must have moved ---------------------------
    // Transform the body's local bbox min by the model matrix.
    const Mat4 m = tr.ToMatrix();
    const Vec3 world = Mat4_TransformPoint(m, local_before);
    CHECK(std::fabs(world.x - (local_before.x + 3.25f)) < 1e-4f,
          "world x moved by the translation (%.4f)", world.x);
    CHECK(std::fabs(world.y - (local_before.y - 1.5f)) < 1e-4f,
          "world y moved by the translation (%.4f)", world.y);
    CHECK(std::fabs(world.z - (local_before.z + 0.75f)) < 1e-4f,
          "world z moved by the translation (%.4f)", world.z);

    // --- the matrix convention -------------------------------------------
    // A pure translation must produce the identity in the 3x3 block and the
    // offset in the LAST COLUMN (m[3], m[7], m[11]) -- that is where both
    // Mat4_TransformPoint and the OpenCL kernel read it from.
    Transform t2;
    t2.position = { 2.f, 3.f, 4.f };
    const Mat4 mt = t2.ToMatrix();
    CHECK(std::fabs(mt.m[3] - 2.f) < 1e-6f, "translation is in m[3] (%.4f)", mt.m[3]);
    CHECK(std::fabs(mt.m[7] - 3.f) < 1e-6f, "translation is in m[7] (%.4f)", mt.m[7]);
    CHECK(std::fabs(mt.m[11] - 4.f) < 1e-6f, "translation is in m[11] (%.4f)", mt.m[11]);
    // The 3x3 block must still be the identity for a pure translation.
    //
    // WHICH indices are the 3x3 block: in a row-major 4x4 with the
    // translation in the LAST COLUMN, the rotation/scale block is rows 0-2
    // x columns 0-2 -> indices {0,1,2, 4,5,6, 8,9,10}. Indices 3, 7 and 11
    // are the TRANSLATION column, not off-diagonals of the 3x3 block.
    const int diag[] = { 0, 5, 10 };
    const int off[]  = { 1, 2, 4, 6, 8, 9 };
    for (int i : diag)
        CHECK(std::fabs(mt.m[i] - 1.f) < 1e-6f, "rotation block diagonal m[%d]==1", i);
    for (int i : off)
        CHECK(std::fabs(mt.m[i]) < 1e-6f, "rotation block off-diagonal m[%d]==0", i);
    // The last row must stay the homogeneous row (0,0,0,1).
    CHECK(std::fabs(mt.m[12]) < 1e-6f && std::fabs(mt.m[13]) < 1e-6f &&
          std::fabs(mt.m[14]) < 1e-6f && std::fabs(mt.m[15] - 1.f) < 1e-6f,
          "the last row is the homogeneous row");

    // --- inverse round trip ----------------------------------------------
    // InversePoint assumes M = T * R * S. If ToMatrix disagreed with that
    // convention the round trip would not close.
    const Vec3 probe_pt{ 0.21f, -0.37f, 0.44f };
    const Vec3 fwd = Mat4_TransformPoint(mt, probe_pt);
    const Vec3 back = t2.InversePoint(fwd);
    CHECK(std::fabs(back.x - probe_pt.x) < 1e-4f &&
          std::fabs(back.y - probe_pt.y) < 1e-4f &&
          std::fabs(back.z - probe_pt.z) < 1e-4f,
          "inverse round trip closes (%.4f,%.4f,%.4f)", back.x, back.y, back.z);

    // The same round trip must close for the MOVE transform (the one the
    // inspector actually drives), including its translation.
    const Vec3 fwd2 = Mat4_TransformPoint(m, probe_pt);
    const Vec3 back2 = tr.InversePoint(fwd2);
    CHECK(std::fabs(back2.x - probe_pt.x) < 1e-4f &&
          std::fabs(back2.y - probe_pt.y) < 1e-4f &&
          std::fabs(back2.z - probe_pt.z) < 1e-4f,
          "the moved transform's round trip closes (%.4f,%.4f,%.4f)",
          back2.x, back2.y, back2.z);

    // --- scale and rotation also work ------------------------------------
    Transform t3;
    t3.scale = { 2.f, 3.f, 4.f };
    const Vec3 scaled = Mat4_TransformPoint(t3.ToMatrix(), probe_pt);
    CHECK(std::fabs(scaled.x - probe_pt.x * 2.f) < 1e-4f, "scale x");
    CHECK(std::fabs(scaled.y - probe_pt.y * 3.f) < 1e-4f, "scale y");
    CHECK(std::fabs(scaled.z - probe_pt.z * 4.f) < 1e-4f, "scale z");

    // A 90-degree rotation about Z must send +X to +Y.
    // The rotation is a QUATERNION (x,y,z,w); a 90-degree turn about Z is
    // (0, 0, sin(45), cos(45)).
    Transform t4;
    const float h = 3.14159265358979f / 4.0f;   // half of 90 degrees
    t4.rotation = Quat{ 0.f, 0.f, std::sin(h), std::cos(h) };
    const Vec3 rot = Mat4_TransformPoint(t4.ToMatrix(), Vec3{ 1.f, 0.f, 0.f });
    CHECK(std::fabs(rot.x) < 1e-3f && std::fabs(rot.y - 1.f) < 1e-3f,
          "90 deg about Z maps +X to +Y (%.4f,%.4f)", rot.x, rot.y);

    std::printf("move_probe: %d checks, %d failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
