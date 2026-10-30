// =============================================================================
// SurfelRasterizer.cpp
// =============================================================================
#include "delirium/render/SurfelRasterizer.h"
#include <cmath>
#include <algorithm>

namespace DeliRium {

static inline Vec3 sub(Vec3 a,Vec3 b){return{a.x-b.x,a.y-b.y,a.z-b.z};}
static inline float dot(Vec3 a,Vec3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}

void SurfelRasterizer::GeneratePatch(const TreeModel& tree, size_t patch,
                                     float projected_px,
                                     std::vector<Surfel>& out) {
    // sampling density from projected size: more samples when the patch is
    // bigger on screen (closer). g samples per axis. This is the LOD: far =>
    // small projected_px => few surfels.
    int g = (int)std::ceil(std::sqrt(std::max(4.0f, projected_px)));
    g = std::min(std::max(g, 3), 48);
    // estimate patch world extent by sampling corners (for surfel radius)
    Vec3 c00=tree.EvalPatch(patch,0,0,16,16,16).pos;
    Vec3 c11=tree.EvalPatch(patch,1,1,16,16,16).pos;
    float extent = std::sqrt(dot(sub(c11,c00),sub(c11,c00)));
    float spacing = (g>1)? extent/(g-1) : extent;
    float radius = spacing * 0.75f;        // disk covers the gap to neighbors
    for (int iu=0; iu<g; ++iu) {
        for (int iv=0; iv<g; ++iv) {
            float u=iu/(g-1.0f), v=iv/(g-1.0f);
            auto pt = tree.EvalPatch(patch, u, v, 16, 16, 16);
            out.push_back(Surfel{pt.pos, pt.normal, pt.color, radius});
        }
    }
}

void SurfelRasterizer::Splat(const std::vector<Surfel>& surfels,
                             const float eye[3], const float fwd[3],
                             const float right[3], const float up[3],
                             float focal, int W, int H,
                             std::vector<float>& rgb, std::vector<float>& zbuf,
                             bool hole_fill) {
    rgb.assign((size_t)W*H*3, 0.06f);
    zbuf.assign((size_t)W*H, 1e9f);
    Vec3 E{eye[0],eye[1],eye[2]}, F{fwd[0],fwd[1],fwd[2]};
    Vec3 Rg{right[0],right[1],right[2]}, U{up[0],up[1],up[2]};
    Vec3 light{0.3f,-0.6f,0.5f};
    float ll=std::sqrt(dot(light,light)); light={light.x/ll,light.y/ll,light.z/ll};
    for (const auto& s : surfels) {
        Vec3 rel=sub(s.pos,E); float z=dot(rel,F); if (z<1e-3f) continue;
        float px=dot(rel,Rg)/z*focal + W*0.5f;
        float py=H*0.5f - dot(rel,U)/z*focal;
        int r=std::max(1,(int)(s.radius/z*focal)); r=std::min(r,12);
        int cx=(int)px, cy=(int)py;
        float lam=std::max(0.0f,dot(s.normal,light))*0.7f+0.3f;
        for (int dy=-r; dy<=r; ++dy) for (int dx=-r; dx<=r; ++dx) {
            if (dx*dx+dy*dy>r*r) continue;
            int X=cx+dx, Y=cy+dy;
            if (X<0||X>=W||Y<0||Y>=H) continue;
            size_t idx=(size_t)Y*W+X;
            if (z<zbuf[idx]) { zbuf[idx]=z;
                rgb[idx*3+0]=std::min(1.0f,s.color.x*lam);
                rgb[idx*3+1]=std::min(1.0f,s.color.y*lam);
                rgb[idx*3+2]=std::min(1.0f,s.color.z*lam);
            }
        }
    }
    if (!hole_fill) return;
    // SAFETY NET (user's idea): fill interior holes by averaging filled
    // neighbors. A hole = empty pixel with >=3 filled 4-neighbors.
    std::vector<float> out=rgb;
    for (int y=1;y<H-1;++y) for (int x=1;x<W-1;++x) {
        size_t idx=(size_t)y*W+x;
        if (zbuf[idx]<1e9f) continue;
        int filled=0; float acc[3]={0,0,0};
        const int off[4][2]={{-1,0},{1,0},{0,-1},{0,1}};
        for (auto&o:off){ size_t n=(size_t)(y+o[1])*W+(x+o[0]);
            if (zbuf[n]<1e9f){ filled++; acc[0]+=out[n*3]; acc[1]+=out[n*3+1]; acc[2]+=out[n*3+2]; } }
        if (filled>=3){ rgb[idx*3]=acc[0]/filled; rgb[idx*3+1]=acc[1]/filled; rgb[idx*3+2]=acc[2]/filled; }
    }
}

} // namespace DeliRium
