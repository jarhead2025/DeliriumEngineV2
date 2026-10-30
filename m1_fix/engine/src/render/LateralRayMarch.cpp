// Lateral / primary view-ray march against PSF equation surfaces.
// Full math path: bilinear anchors + deform_h(coef[8]) + analytic N.
// Lighting: key + fill + optional emissive; shadow via secondary occlusion probe.
// AA is NOT done here — caller runs fb_aa on GPU after upload.
#include "delirium/render/LateralRayMarch.h"
#include "delirium/psf/PSFBody.h"
#include "delirium/render/Camera.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace DeliRium {
namespace {

inline float deform_h(const float c[8], float u, float v) {
    const float bu = u * (1.f - u), bv = v * (1.f - v);
    const float au = 2.f * u - 1.f, av = 2.f * v - 1.f;
    const float w = 1.f - v, x = 1.f - u;
    return fma(c[0], bu * w,
           fma(c[1], bu * v,
           fma(c[2], bv * x,
           fma(c[3], bv * u,
           fma(c[4], bu * au * w,
           fma(c[5], bu * au * v,
           fma(c[6], bv * av * x, c[7] * bv * av * u)))))));
}

inline Vec3 vadd(Vec3 a, Vec3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
inline Vec3 vsub(Vec3 a, Vec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline Vec3 vscale(Vec3 a, float s) { return {a.x*s,a.y*s,a.z*s}; }
inline float vdot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline Vec3 vcross(Vec3 a, Vec3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
inline Vec3 vnorm(Vec3 a) {
    float L = std::sqrt(vdot(a,a));
    if (L < 1e-12f) return {0,1,0};
    return {a.x/L, a.y/L, a.z/L};
}

struct PatchGPU {
    Vec3 P[4];
    Vec3 Nd[4];
    float coef[8];
    Vec3 emissive; // glow color
    Vec3 albedo;
};

inline Vec3 dequant_bytes(const AnchorRecord& a, Vec3 bbmin, Vec3 bbspan) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&a);
    uint16_t px, py, pz;
    std::memcpy(&px, b+32, 2);
    std::memcpy(&py, b+34, 2);
    std::memcpy(&pz, b+36, 2);
    return {bbmin.x + (px/65535.f)*bbspan.x,
            bbmin.y + (py/65535.f)*bbspan.y,
            bbmin.z + (pz/65535.f)*bbspan.z};
}

inline Vec3 nd_from_anchor(const AnchorRecord& a) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(&a);
    float nx = b[44]/255.f*2.f-1.f;
    float ny = b[45]/255.f*2.f-1.f;
    float nz = b[46]/255.f*2.f-1.f;
    return vnorm({nx,ny,nz});
}

inline void eval_patch(const PatchGPU& p, float u, float v, Vec3& out_p, Vec3& out_n) {
    float um=1.f-u, vm=1.f-v;
    Vec3 bil = vadd(vadd(vscale(p.P[0], um*vm), vscale(p.P[1], u*vm)),
                    vadd(vscale(p.P[2], u*v), vscale(p.P[3], um*v)));
    Vec3 Nd = vadd(vadd(vscale(p.Nd[0], um*vm), vscale(p.Nd[1], u*vm)),
                   vadd(vscale(p.Nd[2], u*v), vscale(p.Nd[3], um*v)));
    Nd = vnorm(Nd);
    float h = deform_h(p.coef, u, v);
    out_p = vadd(bil, vscale(Nd, h));
    Vec3 dPdu = vadd(vscale(vsub(p.P[1], p.P[0]), vm), vscale(vsub(p.P[2], p.P[3]), v));
    Vec3 dPdv = vadd(vscale(vsub(p.P[3], p.P[0]), um), vscale(vsub(p.P[2], p.P[1]), u));
    const float eps = 1e-3f;
    float up = std::min(1.f,u+eps), um2 = std::max(0.f,u-eps);
    float vp = std::min(1.f,v+eps), vm2 = std::max(0.f,v-eps);
    float dhu = (deform_h(p.coef,up,v)-deform_h(p.coef,um2,v))/(up-um2);
    float dhv = (deform_h(p.coef,u,vp)-deform_h(p.coef,u,vm2))/(vp-vm2);
    dPdu = vadd(dPdu, vscale(Nd, dhu));
    dPdv = vadd(dPdv, vscale(Nd, dhv));
    out_n = vnorm(vcross(dPdu, dPdv));
}

// Closest point on ray to a surface sample; returns t along ray if near enough.
inline bool probe_hit(const Vec3& ro, const Vec3& rd, const Vec3& S, float max_dist,
                      float& t_out, float& dist_out) {
    Vec3 rel = vsub(S, ro);
    float t = vdot(rel, rd);
    if (t < 1e-4f) return false;
    Vec3 closest = vadd(ro, vscale(rd, t));
    float d = std::sqrt(vdot(vsub(S, closest), vsub(S, closest)));
    if (d > max_dist) return false;
    t_out = t;
    dist_out = d;
    return true;
}

} // namespace

bool LateralRayMarch(const PSFBody& body, const Camera& cam, int W, int H,
                     const RayMarchLights& lights,
                     std::vector<uint32_t>& out_rgba) {
    const auto& A = body.AnchorBlob();
    const auto& S = body.SurfaceBlob();
    if (A.empty() || S.empty() || W <= 0 || H <= 0) return false;

    Vec3 bbmin = body.BBoxMin(), bbspan = body.BBoxSpan();
    if (bbspan.x < 1e-8f && bbspan.y < 1e-8f && bbspan.z < 1e-8f) {
        bbmin = {-1.f,-1.f,-1.f};
        bbspan = {2.f,2.f,2.f};
    }

    std::vector<PatchGPU> patches;
    patches.reserve(S.size());
    for (size_t si = 0; si < S.size(); ++si) {
        const auto& s = S[si];
        PatchGPU p{};
        bool ok = true;
        for (int k = 0; k < 4; ++k) {
            if (s.anchor_idx[k] >= A.size()) { ok = false; break; }
            p.P[k] = dequant_bytes(A[s.anchor_idx[k]], bbmin, bbspan);
            p.Nd[k] = nd_from_anchor(A[s.anchor_idx[k]]);
        }
        if (!ok) continue;
        for (int k = 0; k < 8; ++k) p.coef[k] = s.coef[k];
        // Albedo from surface index + coef energy; emissive if glow enabled
        float e = 0.f;
        for (int k = 0; k < 8; ++k) e += std::fabs(p.coef[k]);
        float hue = std::fmod(float(si) * 0.07f, 1.f);
        // cheap HSV-ish
        auto ch = [&](float h) {
            float x = std::fabs(std::fmod(h, 1.f) * 6.f - 3.f) - 1.f;
            return std::max(0.f, std::min(1.f, x));
        };
        p.albedo = {0.35f + 0.5f * ch(hue), 0.35f + 0.5f * ch(hue+0.33f), 0.35f + 0.5f * ch(hue+0.66f)};
        if (lights.emissive_strength > 0.f) {
            float pulse = 0.5f + 0.5f * std::sin(lights.time_sec * 1.7f + float(si) * 0.11f);
            p.emissive = vscale(p.albedo, lights.emissive_strength * (0.4f + 0.6f * pulse) * (0.2f + e));
        } else {
            p.emissive = {0,0,0};
        }
        patches.push_back(p);
    }
    if (patches.empty()) return false;

    // Camera basis
    Vec3 eye = cam.position;
    Vec3 f = vnorm(vsub(cam.target, cam.position));
    Vec3 r = vnorm(vcross(f, cam.up));
    Vec3 u = vcross(r, f);
    float aspect = float(W) / float(H);
    float tan_half = std::tan(cam.fov_deg * 3.14159265f / 360.f);

    // UV resolution per patch for hit probes (lateral density; denser = better)
    const int uvn = lights.uv_samples > 0 ? lights.uv_samples : 6;
    const float hit_radius = lights.hit_radius > 0.f ? lights.hit_radius : 0.045f;

    out_rgba.assign(size_t(W) * size_t(H), 0xFF0A0A10u);
    std::vector<float> zbuf(size_t(W) * size_t(H), 1e30f);

    // Optional resolution scale for interactive rates
    const int step = lights.pixel_stride > 0 ? lights.pixel_stride : 1;

    for (int y = 0; y < H; y += step) {
        for (int x = 0; x < W; x += step) {
            // Primary view ray through pixel center
            float ndc_x = (2.f * (float(x) + 0.5f) / float(W) - 1.f) * aspect * tan_half;
            float ndc_y = (1.f - 2.f * (float(y) + 0.5f) / float(H)) * tan_half;
            Vec3 rd = vnorm(vadd(vadd(f, vscale(r, ndc_x)), vscale(u, ndc_y)));

            float best_t = 1e30f;
            float best_dist = 1e30f;
            Vec3 best_p{}, best_n{}, best_alb{}, best_em{};

            // Equation-domain probes: dense UV on each patch, pick nearest to ray
            for (const auto& pat : patches) {
                for (int iu = 0; iu < uvn; ++iu) {
                    for (int iv = 0; iv < uvn; ++iv) {
                        float uu = (iu + 0.5f) / float(uvn);
                        float vv = (iv + 0.5f) / float(uvn);
                        Vec3 Sp, Sn;
                        eval_patch(pat, uu, vv, Sp, Sn);
                        float t, d;
                        if (!probe_hit(eye, rd, Sp, hit_radius, t, d)) continue;
                        // Prefer closer t; break ties with smaller lateral distance
                        if (t < best_t - 1e-4f || (std::fabs(t - best_t) < 1e-4f && d < best_dist)) {
                            best_t = t;
                            best_dist = d;
                            best_p = Sp;
                            best_n = Sn;
                            best_alb = pat.albedo;
                            best_em = pat.emissive;
                            // Face toward camera
                            if (vdot(best_n, rd) > 0.f) best_n = vscale(best_n, -1.f);
                        }
                    }
                }
            }

            if (best_t > 1e29f) continue;

            // --- Lighting (generic lights + emissive) ---
            Vec3 V = vscale(rd, -1.f); // toward eye
            Vec3 col = best_em;       // start with glow
            float ambient = lights.ambient;
            col = vadd(col, vscale(best_alb, ambient));

            for (int li = 0; li < lights.n_lights && li < 4; ++li) {
                const auto& Lsrc = lights.lights[li];
                Vec3 Lvec;
                float atten = Lsrc.intensity;
                if (Lsrc.type == LightType::Directional) {
                    Lvec = vnorm(vscale(Lsrc.dir_or_pos, -1.f)); // dir points toward surface convention: light dir FROM light
                    // store as direction the light travels; L for shading = -travel
                    Lvec = vnorm(vscale(Lsrc.dir_or_pos, -1.f));
                } else if (Lsrc.type == LightType::Headlight) {
                    Lvec = V;
                } else { // Point
                    Lvec = vsub(Lsrc.dir_or_pos, best_p);
                    float dist2 = vdot(Lvec, Lvec);
                    Lvec = vnorm(Lvec);
                    atten *= 1.f / (1.f + dist2 * Lsrc.attenuation);
                }
                float ndotl = std::max(0.f, vdot(best_n, Lvec));
                // Shadow: probe toward light for closer equation samples
                float shadow = 1.f;
                if (lights.shadows && ndotl > 0.f) {
                    Vec3 so = vadd(best_p, vscale(best_n, 0.02f));
                    for (const auto& pat : patches) {
                        for (int iu = 0; iu < 3; ++iu) {
                            for (int iv = 0; iv < 3; ++iv) {
                                float uu = (iu + 0.5f) / 3.f, vv = (iv + 0.5f) / 3.f;
                                Vec3 Sp, Sn;
                                eval_patch(pat, uu, vv, Sp, Sn);
                                Vec3 toS = vsub(Sp, so);
                                float along = vdot(toS, Lvec);
                                if (along < 0.03f || along > 4.f) continue;
                                Vec3 perp = vsub(toS, vscale(Lvec, along));
                                if (vdot(perp, perp) < hit_radius * hit_radius * 0.5f) {
                                    shadow = 0.25f;
                                    goto shadow_done;
                                }
                            }
                        }
                    }
                shadow_done:;
                }
                Vec3 H = vnorm(vadd(Lvec, V));
                float spec = std::pow(std::max(0.f, vdot(best_n, H)), lights.shininess);
                col = vadd(col, vscale(vadd(vscale(best_alb, ndotl),
                                            vscale(Lsrc.color, spec * 0.35f)),
                                       atten * shadow));
            }

            auto q = [](float x) {
                if (x < 0.f) x = 0.f;
                if (x > 1.f) x = 1.f;
                return uint32_t(x * 255.f + 0.5f);
            };
            uint32_t rgba = (255u << 24) | (q(col.z) << 16) | (q(col.y) << 8) | q(col.x);
            for (int dy = 0; dy < step && y + dy < H; ++dy)
                for (int dx = 0; dx < step && x + dx < W; ++dx) {
                    size_t idx = size_t(y + dy) * size_t(W) + size_t(x + dx);
                    if (best_t < zbuf[idx]) {
                        zbuf[idx] = best_t;
                        out_rgba[idx] = rgba;
                    }
                }
        }
    }

    static int slog = 0;
    if ((slog++ % 30) == 0)
        std::printf("lateral-ray: %dx%d patches=%zu uvn=%d lights=%d emissive=%.2f shadows=%d\n",
                    W, H, patches.size(), uvn, lights.n_lights, lights.emissive_strength,
                    lights.shadows ? 1 : 0);
    return true;
}

} // namespace DeliRium
