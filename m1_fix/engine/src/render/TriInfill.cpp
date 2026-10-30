#include "delirium/render/TriInfill.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace DeliRium {
namespace {

inline uint32_t pack_rgb(float r, float g, float b) {
    auto q = [](float x) -> uint32_t {
        if (x < 0.f) x = 0.f;
        if (x > 255.f) x = 255.f;
        return uint32_t(x + 0.5f);
    };
    return (255u << 24) | (q(b) << 16) | (q(g) << 8) | q(r);
}

// Barycentric raster of one triangle (edge function), depth test, color lerp.
void raster_tri(const SampleDot& A, const SampleDot& B, const SampleDot& C,
                int W, int H, std::vector<uint32_t>& rgba, std::vector<float>& zbuf) {
    const float minx = std::floor(std::min({A.sx, B.sx, C.sx}));
    const float maxx = std::ceil(std::max({A.sx, B.sx, C.sx}));
    const float miny = std::floor(std::min({A.sy, B.sy, C.sy}));
    const float maxy = std::ceil(std::max({A.sy, B.sy, C.sy}));
    const int x0 = std::max(0, (int)minx);
    const int x1 = std::min(W - 1, (int)maxx);
    const int y0 = std::max(0, (int)miny);
    const int y1 = std::min(H - 1, (int)maxy);
    if (x0 > x1 || y0 > y1) return;

    const float area = (B.sx - A.sx) * (C.sy - A.sy) - (B.sy - A.sy) * (C.sx - A.sx);
    if (std::fabs(area) < 1e-8f) return;
    const float invA = 1.f / area;

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            const float px = float(x) + 0.5f, py = float(y) + 0.5f;
            // barycentric coords
            const float w0 = ((B.sx - px) * (C.sy - py) - (B.sy - py) * (C.sx - px)) * invA;
            const float w1 = ((C.sx - px) * (A.sy - py) - (C.sy - py) * (A.sx - px)) * invA;
            const float w2 = 1.f - w0 - w1;
            if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;
            const float z = w0 * A.z + w1 * B.z + w2 * C.z;
            const size_t idx = size_t(y) * size_t(W) + size_t(x);
            if (z >= zbuf[idx]) continue;
            zbuf[idx] = z;
            rgba[idx] = pack_rgb(w0 * A.r + w1 * B.r + w2 * C.r,
                                 w0 * A.g + w1 * B.g + w2 * C.g,
                                 w0 * A.b + w1 * B.b + w2 * C.b);
        }
    }
}

struct Tri {
    int a, b, c;
};

// Bowyer–Watson Delaunay for small-to-medium point sets (screen-space).
std::vector<Tri> delaunay(const std::vector<SampleDot>& pts) {
    const int n = (int)pts.size();
    std::vector<Tri> tris;
    if (n < 3) return tris;

    float minx = pts[0].sx, maxx = pts[0].sx, miny = pts[0].sy, maxy = pts[0].sy;
    for (const auto& p : pts) {
        minx = std::min(minx, p.sx);
        maxx = std::max(maxx, p.sx);
        miny = std::min(miny, p.sy);
        maxy = std::max(maxy, p.sy);
    }
    const float dx = maxx - minx, dy = maxy - miny;
    const float d = std::max(dx, dy) * 10.f + 10.f;
    // super-triangle vertices appended as n, n+1, n+2
    std::vector<SampleDot> P = pts;
    P.push_back({minx - d, miny - d, 0, 0, 0, 0});
    P.push_back({minx + dx * 0.5f, maxy + d, 0, 0, 0, 0});
    P.push_back({maxx + d, miny - d, 0, 0, 0, 0});
    const int s0 = n, s1 = n + 1, s2 = n + 2;
    tris.push_back({s0, s1, s2});

    auto in_circum = [&](int i, int j, int k, int p) {
        const float ax = P[i].sx - P[p].sx, ay = P[i].sy - P[p].sy;
        const float bx = P[j].sx - P[p].sx, by = P[j].sy - P[p].sy;
        const float cx = P[k].sx - P[p].sx, cy = P[k].sy - P[p].sy;
        const float det = (ax * ax + ay * ay) * (bx * cy - by * cx)
                        - (bx * bx + by * by) * (ax * cy - ay * cx)
                        + (cx * cx + cy * cy) * (ax * by - ay * bx);
        const float orient = (P[j].sx - P[i].sx) * (P[k].sy - P[i].sy)
                           - (P[j].sy - P[i].sy) * (P[k].sx - P[i].sx);
        return orient > 0 ? det > 0 : det < 0;
    };

    for (int p = 0; p < n; ++p) {
        struct Edge {
            int u, v;
        };
        std::vector<Edge> bad_edges;
        std::vector<Tri> next;
        next.reserve(tris.size());
        for (const auto& t : tris) {
            if (in_circum(t.a, t.b, t.c, p)) {
                bad_edges.push_back({t.a, t.b});
                bad_edges.push_back({t.b, t.c});
                bad_edges.push_back({t.c, t.a});
            } else {
                next.push_back(t);
            }
        }
        // boundary = edges that appear once
        std::vector<char> skip(bad_edges.size(), 0);
        for (size_t i = 0; i < bad_edges.size(); ++i) {
            if (skip[i]) continue;
            for (size_t j = i + 1; j < bad_edges.size(); ++j) {
                if (skip[j]) continue;
                if ((bad_edges[i].u == bad_edges[j].v && bad_edges[i].v == bad_edges[j].u) ||
                    (bad_edges[i].u == bad_edges[j].u && bad_edges[i].v == bad_edges[j].v)) {
                    skip[i] = skip[j] = 1;
                }
            }
        }
        for (size_t i = 0; i < bad_edges.size(); ++i) {
            if (skip[i]) continue;
            next.push_back({bad_edges[i].u, bad_edges[i].v, p});
        }
        tris.swap(next);
    }

    std::vector<Tri> out;
    for (const auto& t : tris) {
        if (t.a >= n || t.b >= n || t.c >= n) continue;
        out.push_back(t);
    }
    return out;
}

} // namespace

void TriInfillStructured(const std::vector<SampleDot>& dots, int su, int sv, int W, int H,
                         std::vector<uint32_t>& rgba, std::vector<float>& zbuf) {
    if (su < 2 || sv < 2 || (int)dots.size() < su * sv) return;
    if ((int)rgba.size() != W * H) {
        rgba.assign(size_t(W) * H, 0xFF121218u);
        zbuf.assign(size_t(W) * H, 1e30f);
    }
    for (int j = 0; j < sv - 1; ++j) {
        for (int i = 0; i < su - 1; ++i) {
            const int i00 = j * su + i;
            const int i10 = j * su + (i + 1);
            const int i01 = (j + 1) * su + i;
            const int i11 = (j + 1) * su + (i + 1);
            raster_tri(dots[i00], dots[i10], dots[i01], W, H, rgba, zbuf);
            raster_tri(dots[i10], dots[i11], dots[i01], W, H, rgba, zbuf);
        }
    }
}

void TriInfillDelaunay(const std::vector<SampleDot>& dots, int W, int H,
                       std::vector<uint32_t>& rgba, std::vector<float>& zbuf) {
    if (dots.size() < 3) return;
    if ((int)rgba.size() != W * H) {
        rgba.assign(size_t(W) * H, 0xFF121218u);
        zbuf.assign(size_t(W) * H, 1e30f);
    }
    // Cap for interactive rates
    std::vector<SampleDot> use = dots;
    if (use.size() > 2500) {
        // simple stride subsample
        std::vector<SampleDot> thin;
        const int step = (int)std::ceil(std::sqrt(double(use.size()) / 2000.0));
        for (size_t i = 0; i < use.size(); i += size_t(std::max(1, step))) thin.push_back(use[i]);
        use.swap(thin);
    }
    const auto tris = delaunay(use);
    for (const auto& t : tris)
        raster_tri(use[t.a], use[t.b], use[t.c], W, H, rgba, zbuf);
}

} // namespace DeliRium
