// =============================================================================
// camera_probe.cpp - is the ORB's silhouette asymmetry in the CAMERA or in
// the RENDERER?
//
// The orb is the ellipsoid substance: z = sz*sqrt(1 - x^2/sx^2 - y^2/sy^2)
// -- the UPPER hemisphere.  It is rotationally symmetric about Z, so its
// silhouette under a yaw sweep MUST be invariant.  Measured on screen, it
// is not: the surface pixels swing 76% across the sweep.
//
// This probe does the projection ON THE CPU, using the SAME Camera::BuildMVP
// the renderer uses, over the body's REAL anchor positions:
//
//   * the CPU silhouette (the anchor cloud's convex-ish extent) per yaw
//   * the CPU silhouette of the ANALYTIC dome (the ground truth)
//
// If the CPU silhouettes are INVARIANT and the on-screen one is not, the
// camera is fine and the defect is in the RENDERER's rasterization.
//
// It also reports the CAMERA BASIS (fwd/right/up) per yaw, because the
// march kernel builds its rays from right/up while the splat kernel uses
// the MVP matrix: a disagreement between those two conventions would show
// up exactly here -- as a yaw-dependent, axis-aligned clipping.
// =============================================================================
#include "delirium/render/Camera.h"
#include "delirium/ai/ElementSystem.h"
#include "delirium/psf/PSFBody.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace DeliRium;

int main() {
    const float extent = 1.0f;
    auto tree = std::make_unique<Element>();
    tree->kind = "ellipsoid_body";
    tree->substance = std::make_unique<EllipsoidSubstance>(0.5f, 0.5f, 0.5f);

    ModelOutput mo; std::string why;
    if (!ElementSystem::Bake(*tree, 48, extent, mo, why)) {
        std::printf("bake failed: %s\n", why.c_str());
        return 1;
    }

    // The object's world transform (the workspace places it at x = -0.7).
    const Vec3 obj_pos{-0.7f, 0.f, 0.f};

    // The analytic dome's sample cloud (the ground truth silhouette).
    std::vector<Vec3> truth;
    for (int i = 0; i < 48; ++i) {
        for (int j = 0; j < 48; ++j) {
            const float u = i / 47.f, v = j / 47.f;
            const float x = (u - 0.5f) * extent, y = (v - 0.5f) * extent;
            const float q = (x*x)/(0.5f*0.5f) + (y*y)/(0.5f*0.5f);
            if (q >= 1.f) continue;
            const float z = 0.5f * std::sqrt(1.f - q);
            truth.push_back(Vec3{obj_pos.x + x, obj_pos.y + y, obj_pos.z + z});
        }
    }

    // The baked anchor cloud, dequantized to world space.
    std::vector<Vec3> baked;
    {
        const auto& ab = mo.anchors;
        Vec3 mn{1e30f,1e30f,1e30f}, mx{-1e30f,-1e30f,-1e30f};
        // Recover the bbox the way BakeBody does: recompute from the anchors'
        // quantized positions is circular, so use the analytic extent of the
        // substance instead (the bake centres the canvas on it).
        const Vec3 bmn{-0.5f, -0.5f, 0.f};
        const Vec3 bsp{1.0f, 1.0f, 0.5f};
        (void)mn; (void)mx;
        for (const AnchorRecord& a : ab) {
            // The out-of-support default sits at the bbox centre; skip it.
            if (a.pos[0] == 32767 && a.pos[1] == 32767 && a.pos[2] == 32767) continue;
            baked.push_back(Vec3{ bmn.x + a.pos[0]/65535.f*bsp.x + obj_pos.x,
                                  bmn.y + a.pos[1]/65535.f*bsp.y + obj_pos.y,
                                  bmn.z + a.pos[2]/65535.f*bsp.z + obj_pos.z });
        }
    }
    std::printf("truth samples=%zu  baked anchors=%zu\n", truth.size(), baked.size());

    const int W = 762, H = 511;
    const float aspect = float(W) / float(H);

    // Project a point cloud through the camera; return the pixel-count and
    // the bounding box in pixels.
    auto project = [&](const std::vector<Vec3>& pts, const Camera& cam,
                       int& lo_x, int& hi_x, int& lo_y, int& hi_y) {
        float mvp[16];
        cam.BuildMVP(mvp, aspect);
        lo_x = lo_y = 1<<30; hi_x = hi_y = -(1<<30);
        int n = 0;
        for (const Vec3& p : pts) {
            const float x = p.x, y = p.y, z = p.z;
            const float cx = mvp[0]*x + mvp[1]*y + mvp[2]*z + mvp[3];
            const float cy = mvp[4]*x + mvp[5]*y + mvp[6]*z + mvp[7];
            const float cw = mvp[12]*x + mvp[13]*y + mvp[14]*z + mvp[15];
            if (cw <= 1e-6f) continue;
            const float ndcx = cx/cw, ndcy = cy/cw;
            const int px = int((ndcx*0.5f+0.5f)*W);
            const int py = int((0.5f-ndcy*0.5f)*H);
            if (px < lo_x) lo_x = px;
            if (px > hi_x) hi_x = px;
            if (py < lo_y) lo_y = py;
            if (py > hi_y) hi_y = py;
            ++n;
        }
        return n;
    };

    std::printf("\n== CPU projection of the SAME clouds, camera orbiting ==\n");
    std::printf("  yaw   truth_bbox(WxH)   baked_bbox(WxH)   truth_n  baked_n\n");
    const float dist = 2.25f, pitch = 0.35f;
    for (int k = 0; k < 8; ++k) {
        const float yaw = k * 0.8f;
        const float cp = std::cos(pitch), sp = std::sin(pitch);
        const Vec3 dir{ std::sin(yaw)*cp, sp, std::cos(yaw)*cp };
        Camera cam;
        cam.position = Vec3{ dir.x*dist, dir.y*dist, dir.z*dist };
        cam.target   = Vec3{ 0,0,0 };
        cam.up       = Vec3{ 0,1,0 };
        cam.fov_deg  = 45.f;

        int tlx,thx,tly,thy, blx,bhx,bly,bhy;
        int tn = project(truth, cam, tlx,thx,tly,thy);
        int bn = project(baked, cam, blx,bhx,bly,bhy);
        std::printf("  %.1f   %4dx%-4d          %4dx%-4d        %4d    %4d\n",
                    yaw, thx-tlx+1, thy-tly+1, bhx-blx+1, bhy-bly+1, tn, bn);
    }

    // Now the MARCH KERNEL's ray basis vs the MVP's. The march kernel builds
    // its rays as eye + fwd*d + right*(ndcx*tanfov) + up*(ndcy*tanfov), while
    // the MVP projects with P = [t/aspect, 0, 0, 0; 0, t, 0, 0; ...].  For
    // these to agree, the march's right/up must equal the MVP's, AND the
    // horizontal scale must carry the SAME aspect factor.
    std::printf("\n== march ray basis vs the MVP, per yaw ==\n");
    std::printf("  yaw    fwd                     right                  up            "
                "tan_h    aspect");
    for (int k = 0; k < 4; ++k) {
        const float yaw = k * 0.8f;
        const float cp = std::cos(pitch), sp = std::sin(pitch);
        const Vec3 dir{ std::sin(yaw)*cp, sp, std::cos(yaw)*cp };
        Camera cam;
        cam.position = Vec3{ dir.x*dist, dir.y*dist, dir.z*dist };
        cam.target   = Vec3{ 0,0,0 };
        cam.up       = Vec3{ 0,1,0 };
        cam.fov_deg  = 45.f;

        Vec3 f{ cam.target.x-cam.position.x, cam.target.y-cam.position.y,
                cam.target.z-cam.position.z };
        const float fl = std::sqrt(f.x*f.x+f.y*f.y+f.z*f.z);
        f = Vec3{ f.x/fl, f.y/fl, f.z/fl };
        Vec3 r{ f.y*cam.up.z-f.z*cam.up.y, f.z*cam.up.x-f.x*cam.up.z,
                f.x*cam.up.y-f.y*cam.up.x };
        const float rl = std::sqrt(r.x*r.x+r.y*r.y+r.z*r.z);
        r = Vec3{ r.x/rl, r.y/rl, r.z/rl };
        Vec3 u{ r.y*f.z-r.z*f.y, r.z*f.x-r.x*f.z, r.x*f.y-r.y*f.x };
        const float th = std::tan(cam.fov_deg * 3.14159265f / 360.f);
        std::printf("  %.1f   (%5.2f,%5.2f,%5.2f)  (%5.2f,%5.2f,%5.2f)  (%5.2f,%5.2f,%5.2f)  "
                    "%.4f  %.4f\n",
                    yaw, f.x,f.y,f.z, r.x,r.y,r.z, u.x,u.y,u.z, th, aspect);
    }
    return 0;
}
