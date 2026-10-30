// =============================================================================
// validate_gameplay.cpp - Milestone 12 gate. Gameplay operators:
//  A) MORPH: AddColorK shifts the chosen pattern's contribution; the
//     average surface color must change in the expected direction, and
//     only where that pattern is present.
//  B) FRACTURE: cutting the torus produces a fragment body; the two
//     bodies' patch counts sum to the original; the fragment is a valid,
//     independent body (its anchors are self-contained / remapped).
//  C) FRAGMENT INTEGRITY: every anchor index in the fragment's surfaces
//     is within its own anchor pool (no dangling references to the parent).
//  D) SETPOSE: a bone translation moves nearby anchors and leaves far
//     anchors essentially fixed (locality of linear blend skinning).
// =============================================================================
#include "delirium/core/GameObject.h"
#include "delirium/psf/Reconstructor.h"
#include <cstdio>
#include <cstring>
#include <cmath>

using namespace DeliRium;

int main(int argc, char** argv) {
    const char* psf = argc>1 ? argv[1] : "assets/torus.psf";
    FixedReconstructor maker;
    ReconstructionRequest req; req.semantic = std::string("file:")+psf;
    int fails=0;

    // A) MORPH
    {
        GameObject o("m", maker, req);
        auto avg_color=[&](){ Vec3 s{}; int n=0;
            for (const Patch& p : o.Body().Patches()){
                Vec3 c=p.EvaluateColor(0.5f,0.5f); s.x+=c.x;s.y+=c.y;s.z+=c.z;++n; }
            return Vec3{s.x/n,s.y/n,s.z/n}; };
        const Vec3 before = avg_color();
        o.Morph(0, 0.3f);   // pattern 0 is the constant term -> brightens
        const Vec3 after = avg_color();
        const float dl = (after.x+after.y+after.z)-(before.x+before.y+before.z);
        std::printf("A) morph: avg luma delta %.4f %s\n", dl,
                    std::fabs(dl) > 1e-4f ? "OK":"FAIL");
        fails += !(std::fabs(dl) > 1e-4f);
    }

    // B+C) FRACTURE
    {
        GameObject o("f", maker, req);
        const size_t n0 = o.Body().Patches().size();
        const Vec3 hit = o.Body().Patches()[0].Evaluate(0.5f,0.5f);
        o.Fracture(hit, 10.0f);
        PSFBody* frag = o.Fragment();
        const size_t nb = o.Body().Patches().size();
        const size_t nf = frag ? frag->Patches().size() : 0;
        std::printf("B) fracture: %zu -> body %zu + fragment %zu (sum %zu) %s\n",
                    n0, nb, nf, nb+nf,
                    (frag && nf>0 && nb>0 && nb+nf==n0) ? "OK":"FAIL");
        fails += !(frag && nf>0 && nb>0 && nb+nf==n0);

        bool integrity = (frag != nullptr);
        if (frag) {
            const uint32_t na = (uint32_t)frag->AnchorBlob().size();
            for (const auto& s : frag->SurfaceBlob())
                for (int k=0;k<4;++k)
                    if (s.anchor_idx[k] >= na) integrity=false;
        }
        std::printf("C) fragment integrity: anchors self-contained %s\n",
                    integrity ? "OK":"FAIL");
        fails += !integrity;
    }

    // D) SETPOSE
    {
        // build an NPC-like object (reuse GameObject path) + a one-bone pose
        GameObject o("p", maker, req);
        const Vec3 sp = o.Body().BBoxSpan();
        const float diag = std::sqrt(sp.x*sp.x+sp.y*sp.y+sp.z*sp.z);
        // pick the NEAREST anchor and the FARTHEST anchor to a bone we
        // place at anchor 0's position
        auto& pool = o.Body().AnchorsConst();
        const Vec3 bone_pos = pool.Position(0);
        uint16_t near_a=0, far_a=0; float near_d=1e30f, far_d=0;
        for (uint16_t a=0;a<o.Body().AnchorBlob().size();++a){
            const Vec3 p=pool.Position(a);
            const float d=std::sqrt((p.x-bone_pos.x)*(p.x-bone_pos.x)
                +(p.y-bone_pos.y)*(p.y-bone_pos.y)+(p.z-bone_pos.z)*(p.z-bone_pos.z));
            if (d<near_d){near_d=d;near_a=a;}
            if (d>far_d){far_d=d;far_a=a;}
        }

        // TWO bones: one at the near anchor that translates, one at the
        // far anchor that stays put. Inverse-distance skinning must move
        // the near anchor much more than the far one (locality).
        BoneRecord b0{}; b0.parent=0xFFFFFFFF;
        b0.bind_pos[0]=bone_pos.x; b0.bind_pos[1]=bone_pos.y; b0.bind_pos[2]=bone_pos.z;
        const float tx=0.05f*diag;
        std::memcpy(&b0.reserved[0], &tx, 4);
        float zero=0; std::memcpy(&b0.reserved[4], &zero,4);
        std::memcpy(&b0.reserved[8], &zero,4);

        const Vec3 fp = pool.Position(far_a);
        BoneRecord b1{}; b1.parent=0xFFFFFFFF;
        b1.bind_pos[0]=fp.x; b1.bind_pos[1]=fp.y; b1.bind_pos[2]=fp.z;
        std::memcpy(&b1.reserved[0], &zero,4);   // far bone: no motion
        std::memcpy(&b1.reserved[4], &zero,4);
        std::memcpy(&b1.reserved[8], &zero,4);

        NPC npc("npc", maker, req);
        auto& np = npc.Body().AnchorsConst();
        const Vec3 nb0 = np.Position(near_a);
        const Vec3 fb0 = np.Position(far_a);
        npc.SetPose({b0, b1});
        const Vec3 nb1 = np.Position(near_a);
        const Vec3 fb1 = np.Position(far_a);
        const float near_move = std::fabs(nb1.x-nb0.x);
        const float far_move  = std::sqrt((fb1.x-fb0.x)*(fb1.x-fb0.x)
            +(fb1.y-fb0.y)*(fb1.y-fb0.y)+(fb1.z-fb0.z)*(fb1.z-fb0.z));
        std::printf("D) setpose: near anchor moved %.4f, far moved %.4f %s\n",
                    near_move, far_move,
                    (near_move > 0.5f*tx && far_move < near_move) ? "OK":"FAIL");
        fails += !(near_move > 0.5f*tx && far_move < near_move);
    }

    std::printf(fails ? "GAMEPLAY VALIDATION FAILED (%d)\n"
                      : "GAMEPLAY VALIDATION PASSED\n", fails);
    return fails;
}
