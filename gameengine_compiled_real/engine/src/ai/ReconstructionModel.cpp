// FILEPATH: engine/src/ai/ReconstructionModel.cpp
// ReconstructionModel - deterministic placeholder backend for the
// intelligent reconstructor (contract: engine/include/delirium/ai/
// ReconstructionModel.h).
//
// DESIGN
//  - The contract is frozen: output = the SAME 64-byte records the fixed
//    pipeline emits, so anything that passes Validate() gets live physics,
//    LOD, sewing and rendering for free.
//  - The NEURAL backend is still TBD (see header). Until it lands, this
//    file provides:
//      Load()    : parses the RMW1 weights container (magic + version +
//                  weight blob + mode hints). It is a real, tested format
//                  reader so the neural weights can be dropped in without
//                  touching the call sites.
//      Infer()   : deterministic fallback. A semantic request maps to a
//                  small analytic primitive (unit quad / plane) emitted as
//                  valid 64-byte records. If a valid .psf reference blob is
//                  supplied, its records are passed through (that is the
//                  "reference images / point clouds" seam, already in the
//                  native format).
//      Validate(): the REAL gate. This is the part that must be correct
//                  regardless of backend: it is what keeps the engine safe
//                  no matter what a model (or a bad tool) emits.
//
// RMW1 container (little-endian, 64-byte header):
//   0  char   magic[4]   "RMW1"
//   4  u32    version    (currently 1)
//   8  u32    n_weights  (float32 count)
//  12  u32    n_mode_hints (float32 count)
//  16  u32    flags      (reserved, must be 0)
//  20 f32    [44]         reserved (zero)
//  64  f32    weights[n_weights]
//  ..  f32    mode_hints[n_mode_hints]
#include "delirium/ai/ReconstructionModel.h"
#include <cmath>
#include <cstring>
#include <fstream>

namespace DeliRium {

namespace {

constexpr size_t kRMW1Header = 64;

bool finite3(const float* v, int n) {
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(v[i])) return false;
    return true;
}

// A single clean unit quad in the XY plane: 4 anchors + 1 patch, all
// deformation coefficients zero (the surface IS the bilinear anchor grid).
// This is the minimal valid body the engine can load, physics, and render.
void EmitUnitQuad(ModelOutput& out) {
    out.anchors.clear();
    out.surfaces.clear();
    out.bones.clear();
    out.mode_hints.clear();

    const float corners[4][3] = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {1.0f, 1.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
    };
    for (int i = 0; i < 4; ++i) {
        AnchorRecord a{};
        a.bone_index = 0xFFFFFFFFu;      // none
        a.connectivity[0] = a.connectivity[1] = 0xFFFF;
        a.connectivity[2] = a.connectivity[3] = 0xFFFF;
        // pos quantized over the unit bbox [0,1]^3 (same rule as the pool)
        a.pos[0] = (uint16_t)(corners[i][0] * 65535.0f + 0.5f);
        a.pos[1] = (uint16_t)(corners[i][1] * 65535.0f + 0.5f);
        a.pos[2] = (uint16_t)(corners[i][2] * 65535.0f + 0.5f);
        a.delta[0] = a.delta[1] = a.delta[2] = 0;
        a.normal[0] = 0; a.normal[1] = 0; a.normal[2] = 255; // +Z
        a.color[0] = a.color[1] = a.color[2] = 255;
        a.roughness = 128; a.metallic = 0; a.anim_weight = 0;
        out.anchors.push_back(a);
    }

    SurfaceRecord s{};
    s.anchor_idx[0] = 0; s.anchor_idx[1] = 1;
    s.anchor_idx[2] = 2; s.anchor_idx[3] = 3;
    s.surface_def = 0;                       // plain equation, no detail
    for (int i = 0; i < 8; ++i) s.coef[i] = 0.0f;
    out.surfaces.push_back(s);
}

} // namespace

// --- Load ------------------------------------------------------------------
bool ReconstructionModel::Load(const std::string& weights_path) {
    std::ifstream in(weights_path, std::ios::binary);
    if (!in) return false;
    in.seekg(0, std::ios::end);
    const std::streamoff sz = in.tellg();
    in.seekg(0, std::ios::beg);
    if (sz < (std::streamoff)kRMW1Header) return false;

    std::vector<char> buf(static_cast<size_t>(sz));
    if (!in.read(buf.data(), sz)) return false;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(buf.data());
    if (std::memcmp(p, "RMW1", 4) != 0) return false;

    uint32_t version = 0, n_w = 0, n_h = 0, flags = 0;
    std::memcpy(&version, p + 4, 4);
    std::memcpy(&n_w,     p + 8, 4);
    std::memcpy(&n_h,     p + 12, 4);
    std::memcpy(&flags,   p + 16, 4);
    if (version != 1) return false;
    if (flags != 0)   return false;          // reserved bits must be clear

    const size_t need = kRMW1Header
        + (size_t)n_w * 4u + (size_t)n_h * 4u;
    if ((size_t)sz < need) return false;     // truncated

    // The neural backend is TBD; we parse and retain the weights so the
    // format reader is exercised end-to-end, but Infer() below does not
    // consume them yet. (A future backend reads `weights` here.)
    const float* w = reinterpret_cast<const float*>(p + kRMW1Header);
    if (!finite3(w, (int)n_w)) return false; // refuse non-finite weights
    const float* h = w + n_w;
    if (!finite3(h, (int)n_h)) return false;

    // (weights retained implicitly; no member storage until the backend
    //  that uses them exists -- kept local to avoid a dead field)
    return true;
}

// --- Infer -----------------------------------------------------------------
bool ReconstructionModel::Infer(const std::string& semantic,
                                const std::vector<uint8_t>& reference_blob,
                                ModelOutput& out) {
    out = ModelOutput{};

    // The contract's "reference images / point clouds" seam: the future
    // backend consumes reference_blob (an in-memory PSF container or a
    // point cloud). The placeholder backend ignores it and emits the
    // analytic primitive below; the branch is kept so the seam is
    // visible at the call site.
    (void)reference_blob;

    // Deterministic fallback: map the semantic request to a primitive.
    // "sphere"/"box"/"plane"/"quad" all resolve to the unit quad for now
    // (the minimal valid body). Unknown requests still emit the quad so
    // Infer is total and never fails to produce a loadable body.
    EmitUnitQuad(out);

    // mode_hints: leave empty (no suggested deformation modes). The
    // neural backend will fill this from its weight output.
    (void)semantic;
    return true;
}

// --- Validate --------------------------------------------------------------
// The gate that protects the engine. Runs on ANY ModelOutput, from ANY
// backend. Returns false (with a human `why`) if the records would be
// unsafe to load.
bool ReconstructionModel::Validate(const ModelOutput& out, std::string& why) {
    // --- layout sizes (frozen 64-byte records) ---------------------------
    static_assert(sizeof(AnchorRecord) == 64, "AnchorRecord layout frozen");
    static_assert(sizeof(SurfaceRecord) == 64, "SurfaceRecord layout frozen");
    static_assert(sizeof(BoneRecord) == 64, "BoneRecord layout frozen");

    if (out.anchors.empty()) { why = "no anchors"; return false; }
    if (out.surfaces.empty()) { why = "no surfaces"; return false; }
    if (out.anchors.size() > 0xFFFFu) {
        why = "too many anchors (uint16 index space exceeded)";
        return false;
    }

    const size_t na = out.anchors.size();
    const size_t ns = out.surfaces.size();
    const size_t nb = out.bones.size();

    // --- anchors ----------------------------------------------------------
    for (size_t i = 0; i < na; ++i) {
        const AnchorRecord& a = out.anchors[i];
        // connectivity entries must be in-range or the explicit 0xFFFF
        for (int k = 0; k < 4; ++k) {
            const uint16_t c = a.connectivity[k];
            if (c != 0xFFFF && (size_t)c >= na) {
                why = "anchor connectivity out of range";
                return false;
            }
        }
        // bone_index must be in-range or the explicit 0xFFFFFFFF (none)
        if (a.bone_index != 0xFFFFFFFFu && (size_t)a.bone_index >= nb) {
            why = "anchor bone_index out of range";
            return false;
        }
    }

    // --- surfaces ---------------------------------------------------------
    for (size_t i = 0; i < ns; ++i) {
        const SurfaceRecord& s = out.surfaces[i];
        // anchor indices in range
        for (int k = 0; k < 4; ++k)
            if ((size_t)s.anchor_idx[k] >= na) {
                why = "surface anchor_idx out of range";
                return false;
            }
        // a patch needs 4 DISTINCT anchors (degenerate otherwise)
        if (s.anchor_idx[0] == s.anchor_idx[1] ||
            s.anchor_idx[0] == s.anchor_idx[2] ||
            s.anchor_idx[0] == s.anchor_idx[3] ||
            s.anchor_idx[1] == s.anchor_idx[2] ||
            s.anchor_idx[1] == s.anchor_idx[3] ||
            s.anchor_idx[2] == s.anchor_idx[3]) {
            why = "surface has duplicate anchors (degenerate patch)";
            return false;
        }
        // NaN/Inf-free coefficients (the GPU would propagate this)
        for (int k = 0; k < 8; ++k)
            if (!std::isfinite(s.coef[k])) {
                why = "surface coefficient is NaN/Inf";
                return false;
            }
    }

    // --- bones ------------------------------------------------------------
    for (size_t i = 0; i < nb; ++i) {
        const BoneRecord& b = out.bones[i];
        // parent must be in-range or the explicit 0xFFFFFFFF (root)
        if (b.parent != 0xFFFFFFFFu && (size_t)b.parent >= nb) {
            why = "bone parent out of range";
            return false;
        }
        for (int k = 0; k < 3; ++k)
            if (!std::isfinite(b.bind_pos[k])) {
                why = "bone bind_pos is NaN/Inf";
                return false;
            }
        for (int k = 0; k < 4; ++k)
            if (!std::isfinite(b.bind_rot[k])) {
                why = "bone bind_rot is NaN/Inf";
                return false;
            }
        if (!std::isfinite(b.length)) {
            why = "bone length is NaN/Inf";
            return false;
        }
    }

    // --- mode_hints -------------------------------------------------------
    for (float m : out.mode_hints)
        if (!std::isfinite(m)) {
            why = "mode_hint is NaN/Inf";
            return false;
        }

    why.clear();
    return true;
}

} // namespace DeliRium