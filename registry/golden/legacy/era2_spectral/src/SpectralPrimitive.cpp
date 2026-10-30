// =============================================================================
// SpectralPrimitive.cpp - evaluate a baked spherical-harmonic detail field.
// Real SH convention matches the Python baker (scipy sph_harm_y, real form).
// =============================================================================
#include "delirium/psf/SpectralPrimitive.h"
#include <fstream>
#include <cmath>
#include <cstring>

namespace DeliRium {

// Fully-normalized associated Legendre seed N_m^m (no overflow at high
// order; built incrementally with the sin(theta)^m factor folded in).
static double norm_plm_seed(int m, double x) {
    const double sint = std::sqrt(std::fmax(0.0, 1.0 - x * x));
    double val = std::sqrt(1.0 / (4.0 * M_PI));     // N_0^0
    for (int k = 1; k <= m; ++k)
        val *= std::sqrt((2.0 * k + 1.0) / (2.0 * k)) * sint;
    return val;
}

void SpectralPrimitive::EvalSH(int L, float theta, float phi,
                               std::vector<float>& out) {
    // Produce coefficients in l-major layout (l=0..L, m=-l..l), matching the
    // baker's column order, using the stable normalized column recurrence
    // identical to the GPU kernel (verified to reproduce the baker exactly).
    out.assign((L + 1) * (L + 1), 0.0f);
    const double x = std::cos(theta);
    for (int m = 0; m <= L; ++m) {
        const double cmp = std::cos((double)m * phi);
        const double smp = std::sin((double)m * phi);
        double prev = 0.0, cur = norm_plm_seed(m, x);
        for (int l = m; l <= L; ++l) {
            double nlm;
            if (l == m) nlm = cur;
            else {
                const double a = std::sqrt(((2.0*l-1.0)*(2.0*l+1.0)) /
                                           ((double)(l - m) * (l + m)));
                const double b = std::sqrt(((2.0*l+1.0)*(l+m-1.0)*(l-m-1.0)) /
                                           ((double)(l - m) * (l + m) * (2.0*l-3.0)));
                nlm = a * x * cur - b * prev;
                prev = cur; cur = nlm;
            }
            const int row = l * l + l;       // index of (l, 0)
            if (m == 0) out[row] = (float)nlm;
            else {
                out[row + m] = (float)(1.4142135623730951 * nlm * cmp);
                out[row - m] = (float)(1.4142135623730951 * nlm * smp);
            }
        }
    }
}

float SpectralPrimitive::Field(const std::vector<float>& coef, int L,
                               float theta, float phi) const {
    std::vector<float> Y;
    EvalSH(L, theta, phi, Y);
    double s = 0;
    const size_t n = std::min(Y.size(), coef.size());
    for (size_t i = 0; i < n; ++i) s += double(Y[i]) * coef[i];
    return (float)s;
}

Vec3 SpectralPrimitive::Surface(float theta, float phi) const {
    const float R = base_r_ + Field(gcoef_, Lg_, theta, phi);
    const float st = std::sin(theta), ct = std::cos(theta);
    const Vec3 dir{ st * std::cos(phi), st * std::sin(phi), ct };
    return Vec3{ center_.x + dir.x * R, center_.y + dir.y * R,
                 center_.z + dir.z * R };
}

Vec3 SpectralPrimitive::Color(float theta, float phi) const {
    const int nc = (Lc_ + 1) * (Lc_ + 1);
    Vec3 c{0,0,0};
    if ((int)ccoef_.size() >= 3 * nc) {
        std::vector<float> ch0(ccoef_.begin(), ccoef_.begin()+nc);
        std::vector<float> ch1(ccoef_.begin()+nc, ccoef_.begin()+2*nc);
        std::vector<float> ch2(ccoef_.begin()+2*nc, ccoef_.begin()+3*nc);
        c.x = Field(ch0, Lc_, theta, phi);
        c.y = Field(ch1, Lc_, theta, phi);
        c.z = Field(ch2, Lc_, theta, phi);
    }
    auto cl=[](float v){ return v<0?0:(v>1?1:v); };
    return Vec3{cl(c.x), cl(c.y), cl(c.z)};
}

Vec3 SpectralPrimitive::Normal(float theta, float phi) const {
    const float e = 1e-3f;
    const Vec3 p   = Surface(theta, phi);
    const Vec3 pt  = Surface(std::min((float)M_PI, theta+e), phi);
    const Vec3 pp  = Surface(theta, phi+e);
    const Vec3 dt{pt.x-p.x, pt.y-p.y, pt.z-p.z};
    const Vec3 dp{pp.x-p.x, pp.y-p.y, pp.z-p.z};
    Vec3 n{ dt.y*dp.z-dt.z*dp.y, dt.z*dp.x-dt.x*dp.z, dt.x*dp.y-dt.y*dp.x };
    const float l = std::sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
    if (l>1e-9f){n.x/=l;n.y/=l;n.z/=l;}
    // orient outward
    const Vec3 rad{p.x-center_.x, p.y-center_.y, p.z-center_.z};
    if (n.x*rad.x+n.y*rad.y+n.z*rad.z < 0){n.x=-n.x;n.y=-n.y;n.z=-n.z;}
    return n;
}

bool SpectralPrimitive::LoadFromMemory(const uint8_t* d, size_t n) {
    if (n < 24 || std::memcmp(d, "SPF\0", 4) != 0) return false;
    size_t o = 4;
    std::memcpy(&center_, d+o, 12); o += 12;
    std::memcpy(&base_r_, d+o, 4);  o += 4;
    std::memcpy(&Lg_, d+o, 4);      o += 4;
    std::memcpy(&Lc_, d+o, 4);      o += 4;
    const int ng = (Lg_+1)*(Lg_+1);
    const int ncl = 3*(Lc_+1)*(Lc_+1);
    if (n < o + size_t(ng+ncl)*4) return false;
    gcoef_.resize(ng); std::memcpy(gcoef_.data(), d+o, ng*4); o += ng*4;
    ccoef_.resize(ncl); std::memcpy(ccoef_.data(), d+o, ncl*4); o += ncl*4;
    bytes_ = o;
    return true;
}

bool SpectralPrimitive::Load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
    return LoadFromMemory(buf.data(), buf.size());
}

} // namespace DeliRium
