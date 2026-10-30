#include "delirium/physics/ModalDynamics.h"
#include <cstdio>
#include <cstring>
#include <cmath>

namespace DeliRium {

bool ModalDynamics::LoadModes(const std::string& path, const PSFBody& body) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint8_t hdr[24];
    if (std::fread(hdr, 1, 24, f) != 24 || std::memcmp(hdr, "PSM\0", 4)) {
        std::fclose(f); return false;
    }
    uint32_t version, n_modes, ns, na;
    std::memcpy(&version, hdr + 4,  4);
    std::memcpy(&n_modes, hdr + 8,  4);
    std::memcpy(&ns,      hdr + 12, 4);
    std::memcpy(&na,      hdr + 16, 4);
    float z; std::memcpy(&z, hdr + 20, 4);
    if (version != 2 || ns != body.Patches().size()
        || na != body.AnchorBlob().size()) { std::fclose(f); return false; }
    zeta = z; n_surf = ns; n_anchor = na;

    freq.resize(n_modes);
    shapes.resize(size_t(n_modes) * (na + size_t(ns)*8));
    bool ok = std::fread(freq.data(),   4, n_modes, f) == n_modes
           && std::fread(shapes.data(), 4, shapes.size(), f) == shapes.size();
    std::fclose(f);
    if (!ok) return false;

    q.assign(n_modes, 0.0f);
    q_dot.assign(n_modes, 0.0f);
    f_ext.assign(n_modes, 0.0f);
    rest_coef.resize(size_t(ns) * 8);
    for (uint32_t s = 0; s < ns; ++s)
        for (int i = 0; i < 8; ++i)
            rest_coef[s*8 + i] = body.Patches()[s].GetCoefficient(i);
    rest_apos.resize(na); rest_anrm.resize(na);
    auto& pool = body.AnchorsConst();
    for (uint32_t a = 0; a < na; ++a) {
        rest_apos[a] = pool.Position((uint16_t)a);
        rest_anrm[a] = pool.Normal((uint16_t)a);
    }
    aidx.resize(size_t(ns)*4);
    for (uint32_t s = 0; s < ns; ++s)
        for (int k = 0; k < 4; ++k)
            aidx[s*4+k] = body.SurfaceBlob()[s].anchor_idx[k];
    const Vec3 sp = body.BBoxSpan();
    const float diag = std::sqrt(sp.x*sp.x + sp.y*sp.y + sp.z*sp.z);
    amp_max = 0.15f * diag;            // modes can deflect up to ~15% of size
    return true;
}

void ModalDynamics::Reset() {
    std::fill(q.begin(), q.end(), 0.0f);
    std::fill(q_dot.begin(), q_dot.end(), 0.0f);
    std::fill(f_ext.begin(), f_ext.end(), 0.0f);
}

void ModalDynamics::Step(float dt) {
    if (freq.empty() || dt <= 0.0f) return;
    // Substep so the highest-frequency mode stays inside the semi-implicit
    // Euler stability region (dt_sub * w_max <= 1.0, a comfortable margin
    // under the |.| <= 2 limit). Without this, high modes (w ~ 393 rad/s)
    // at a 120 Hz step (dt*w ~ 3.3) gain energy every step and the geometry
    // is driven into a large fast oscillation (normals flip -> black patch).
    float w_max = 0.0f;
    for (float w : freq) w_max = std::max(w_max, w);
    const int nsub = std::max(1, (int)std::ceil(dt * w_max));
    const float h = dt / (float)nsub;
    for (int s = 0; s < nsub; ++s) {
        for (size_t m = 0; m < freq.size(); ++m) {
            const float w = freq[m];
            const float acc = f_ext[m] - 2.0f*zeta*w*q_dot[m] - w*w*q[m];
            q_dot[m] += acc * h;            // semi-implicit Euler:
            q[m]     += q_dot[m] * h;       // velocity first, then position
            if (amp_max > 0.0f) {           // saturate, bleed off excess speed
                if (q[m] >  amp_max) { q[m] =  amp_max; if (q_dot[m] > 0) q_dot[m] = 0; }
                if (q[m] < -amp_max) { q[m] = -amp_max; if (q_dot[m] < 0) q_dot[m] = 0; }
            }
        }
    }
    for (size_t m = 0; m < freq.size(); ++m) f_ext[m] = 0.0f;
}

void ModalDynamics::ApplyImpulseAt(uint32_t s, float u, float v, float j) {
    if (s >= n_surf) return;
    const float bu = u*(1-u), bv = v*(1-v);
    const float au = 2*u-1,   av = 2*v-1;
    const float B[8] = { bu*(1-v), bu*v, bv*(1-u), bv*u,
                         bu*au*(1-v), bu*au*v, bv*av*(1-u), bv*av*u };
    const float Wc[4] = { (1-u)*(1-v), (1-u)*v, u*v, u*(1-v) };  // A0..A3
    const size_t stride = n_anchor + size_t(n_surf)*8;
    for (size_t m = 0; m < freq.size(); ++m) {
        const float* sh = &shapes[m*stride];
        float phi = 0;
        for (int k = 0; k < 4; ++k) phi += Wc[k] * sh[aidx[s*4+k]];
        const float* sc = sh + n_anchor + size_t(s)*8;
        for (int i = 0; i < 8; ++i) phi += B[i] * sc[i];
        q_dot[m] += phi * j;             // impulse -> velocity jump
    }
}

void ModalDynamics::WriteBack(PSFBody& body) const {
    if (body.Patches().size() != n_surf
        || body.AnchorBlob().size() != n_anchor) return;  // stale after splits
    const size_t stride = n_anchor + size_t(n_surf)*8;
    auto& patches = const_cast<std::vector<Patch>&>(body.Patches());
    for (uint32_t s = 0; s < n_surf; ++s) {
        float c[8];
        for (int i = 0; i < 8; ++i) c[i] = rest_coef[s*8 + i];
        for (size_t m = 0; m < freq.size(); ++m) {
            const float* sc = &shapes[m*stride] + n_anchor + size_t(s)*8;
            const float qm = q[m];
            for (int i = 0; i < 8; ++i) c[i] += qm * sc[i];
        }
        for (int i = 0; i < 8; ++i) patches[s].SetCoefficient(i, c[i]);
    }
    auto& pool = body.Anchors();
    for (uint32_t a = 0; a < n_anchor; ++a) {
        float d = 0;
        for (size_t m = 0; m < freq.size(); ++m)
            d += q[m] * shapes[m*stride + a];
        const Vec3& p = rest_apos[a]; const Vec3& n = rest_anrm[a];
        pool.SetPosition((uint16_t)a, { p.x + d*n.x, p.y + d*n.y,
                                        p.z + d*n.z });
    }
    body.MarkAnchorsDirty();
}

void ModalDynamics::SetAmplitude(int m, float a, float ad) {
    q[m] = a; q_dot[m] = ad;
}

float ModalDynamics::TotalEnergy() const {
    float e = 0;
    for (size_t m = 0; m < freq.size(); ++m)
        e += 0.5f*(q_dot[m]*q_dot[m] + freq[m]*freq[m]*q[m]*q[m]);
    return e;
}

} // namespace DeliRium
