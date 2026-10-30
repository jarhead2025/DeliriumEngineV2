// =============================================================================
// TreeFormat.cpp - .dlt load/save and per-patch tree evaluation.
// =============================================================================
#include "delirium/psf/TreeFormat.h"
#include "delirium/psf/SurfaceTypeLibrary.h"
#include "delirium/psf/PatchColorField.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>

namespace DeliRium {

// detail type basis (2D Fourier) -- must match the encoder
static float detail_type_eval(uint32_t idx, float u, float v) {
    // idx enumerates (fu,fv) pairs with cos and sin variants, matching
    // encode_patches.detail_types order.
    const float PI = 3.14159265358979323846f;
    int per_pair = 2;            // cos, sin
    int pairs_per_fu = 0;
    // reconstruct (fu,fv,variant) from linear idx, max_freq known by caller;
    // we store the basis order so detail uses the same enumeration. For the
    // evaluator we accept idx as: pair = idx/2, variant = idx%2, and
    // (fu,fv) enumerated fu=0..MF, fv=0..MF skipping (0,0).
    const int MF = 6;
    int pair = idx / 2, variant = idx % 2;
    int count = 0, fu = 0, fv = 0;
    for (int a = 0; a <= MF && count <= pair; ++a)
        for (int b = 0; b <= MF; ++b) {
            if (a == 0 && b == 0) continue;
            if (count == pair) { fu = a; fv = b; }
            ++count;
        }
    (void)pairs_per_fu; (void)per_pair;
    if (variant == 0) return std::cos(fu*PI*u) * std::cos(fv*PI*v);
    else              return std::sin(fu*PI*u) * std::cos(fv*PI*v);
}

TreeModel::Point TreeModel::EvalPatch(size_t pi, float u, float v,
                                      float lod_shape, float lod_detail,
                                      float lod_color) const {
    const TreePatch& p = patches_[pi];
    Point pt;
    // --- shape layer: sum of surface types (vector K) ---
    Vec3 pos{0,0,0};
    int ns = (int)std::min((float)p.shape_idx.size(),
                           std::floor(lod_shape) + 1.0f);
    if (ns > (int)p.shape_idx.size()) ns = (int)p.shape_idx.size();
    for (int i = 0; i < ns; ++i) {
        float b = SurfaceTypeLibrary::Eval(p.shape_idx[i], u, v);
        pos.x += p.shape_k[i].x * b;
        pos.y += p.shape_k[i].y * b;
        pos.z += p.shape_k[i].z * b;
    }
    // --- detail layer: sum of detail types ---
    int nd = (int)std::min((float)p.detail_idx.size(),
                           std::floor(lod_detail) + 1.0f);
    if (nd > (int)p.detail_idx.size()) nd = (int)p.detail_idx.size();
    for (int j = 0; j < nd; ++j) {
        float b = detail_type_eval(p.detail_idx[j], u, v);
        pos.x += p.detail_k[j].x * b;
        pos.y += p.detail_k[j].y * b;
        pos.z += p.detail_k[j].z * b;
    }
    pt.pos = pos;
    // --- normal via finite difference of the shape+detail field ---
    const float e = 1e-3f;
    auto eval_pos = [&](float uu, float vv){
        Vec3 q{0,0,0};
        for (int i = 0; i < ns; ++i) {
            float b = SurfaceTypeLibrary::Eval(p.shape_idx[i], uu, vv);
            q.x += p.shape_k[i].x*b; q.y += p.shape_k[i].y*b; q.z += p.shape_k[i].z*b;
        }
        for (int j = 0; j < nd; ++j) {
            float b = detail_type_eval(p.detail_idx[j], uu, vv);
            q.x += p.detail_k[j].x*b; q.y += p.detail_k[j].y*b; q.z += p.detail_k[j].z*b;
        }
        return q;
    };
    Vec3 du = eval_pos(u+e, v), dv = eval_pos(u, v+e);
    Vec3 a{du.x-pos.x, du.y-pos.y, du.z-pos.z};
    Vec3 b{dv.x-pos.x, dv.y-pos.y, dv.z-pos.z};
    Vec3 nrm{a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
    float nl = std::sqrt(nrm.x*nrm.x+nrm.y*nrm.y+nrm.z*nrm.z);
    if (nl > 1e-9f) { nrm.x/=nl; nrm.y/=nl; nrm.z/=nl; }
    pt.normal = nrm;
    // --- color layer ---
    if (p.color_order > 0 && !p.color_coeff.empty()) {
        PatchColorField cf;
        const int n = (p.color_order+1)*(p.color_order+1);
        cf.Set(p.color_order, p.color_coeff.data(),
               p.color_coeff.data()+n, p.color_coeff.data()+2*n);
        pt.color = cf.Eval(u, v, lod_color);
    } else {
        pt.color = Vec3{0.7f,0.7f,0.7f};
    }
    return pt;
}

bool TreeModel::Save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    { const char mg[4]={'D','L','T',0}; f.write(mg,4); }
    uint32_t pc = (uint32_t)patches_.size();
    f.write((char*)&pc, 4);
    for (const auto& p : patches_) {
        uint32_t ns = (uint32_t)p.shape_idx.size();
        uint32_t nd = (uint32_t)p.detail_idx.size();
        int32_t  co = p.color_order;
        f.write((char*)&ns, 4);
        f.write((char*)&nd, 4);
        f.write((char*)&co, 4);
        for (uint32_t i=0;i<ns;++i){ f.write((char*)&p.shape_idx[i],1);
            f.write((char*)&p.shape_k[i], sizeof(Vec3)); }
        for (uint32_t j=0;j<nd;++j){ f.write((char*)&p.detail_idx[j],1);
            f.write((char*)&p.detail_k[j], sizeof(Vec3)); }
        uint32_t cc = (uint32_t)p.color_coeff.size();
        f.write((char*)&cc, 4);
        if (cc) f.write((char*)p.color_coeff.data(), cc*sizeof(float));
        f.write((char*)p.neighbor, 16);
    }
    return true;
}

bool TreeModel::Load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4]; f.read(magic,4);
    if (std::memcmp(magic,"DLT\0",4)!=0) return false;
    uint32_t pc; f.read((char*)&pc,4);
    patches_.clear(); patches_.reserve(pc);
    for (uint32_t pi=0; pi<pc; ++pi) {
        TreePatch p;
        uint32_t ns,nd; int32_t co;
        f.read((char*)&ns,4); f.read((char*)&nd,4); f.read((char*)&co,4);
        p.color_order = co;
        p.shape_idx.resize(ns); p.shape_k.resize(ns);
        for (uint32_t i=0;i<ns;++i){ f.read((char*)&p.shape_idx[i],1);
            f.read((char*)&p.shape_k[i], sizeof(Vec3)); }
        p.detail_idx.resize(nd); p.detail_k.resize(nd);
        for (uint32_t j=0;j<nd;++j){ f.read((char*)&p.detail_idx[j],1);
            f.read((char*)&p.detail_k[j], sizeof(Vec3)); }
        uint32_t cc; f.read((char*)&cc,4);
        p.color_coeff.resize(cc);
        if (cc) f.read((char*)p.color_coeff.data(), cc*sizeof(float));
        f.read((char*)p.neighbor, 16);
        patches_.push_back(std::move(p));
    }
    return true;
}

size_t TreeModel::ByteSize() const {
    size_t s = 8;
    for (const auto& p : patches_)
        s += 12 + p.shape_idx.size()*(1+sizeof(Vec3))
               + p.detail_idx.size()*(1+sizeof(Vec3))
               + 4 + p.color_coeff.size()*sizeof(float) + 16;
    return s;
}

} // namespace DeliRium
