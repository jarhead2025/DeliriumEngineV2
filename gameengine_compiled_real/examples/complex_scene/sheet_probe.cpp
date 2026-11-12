// FIND THE CROSS-PRODUCT ERROR, AND GET A TRUSTWORTHY AREA.
//
// The brute-force double integration gave 57.69 for a sphere of radius 2
// (the true answer is 50.27) -- so the COMPONENTS of r_v x r_u are wrong,
// not just the 1D reduction.
//
// THE PARAMETRISATION:
//     x = a sin v cos u
//     y = b sin v sin u
//     z = c cos v
//
//     r_v = ( a cos v cos u,  b cos v sin u,  -c sin v )
//     r_u = ( -a sin v sin u,  b sin v cos u,  0        )
//
// THE CROSS PRODUCT r_v x r_u, COMPONENT BY COMPONENT:
//
//   x = (r_v.y * r_u.z) - (r_v.z * r_u.y)
//     = (b cos v sin u)(0) - (-c sin v)(b sin v cos u)
//     = +bc sin^2 v cos u                          <-- NO cos v sin v sin u TERM
//
//   y = (r_v.z * r_u.x) - (r_v.x * r_u.z)
//     = (-c sin v)(-a sin v sin u) - (a cos v cos u)(0)
//     = +ac sin^2 v sin u                          <-- NO cos v sin v cos u TERM
//
//   z = (r_v.x * r_u.y) - (r_v.y * r_u.x)
//     = (a cos v cos u)(b sin v cos u) - (b cos v sin u)(-a sin v sin u)
//     = ab cos v sin v (cos^2 u + sin^2 u) = ab cos v sin v
//
// SO THE FIRST TWO COMPONENTS HAVE NO cos v FACTOR -- the version in
// EllipsoidArea.cpp carried a spurious "-bc cos v sin v sin u" and
// "+ac cos v sin v cos u". That is the error.
//
// This probe checks the corrected components against the numeric area.
#include <cstdio>
#include <cmath>

static const double kPi = 3.14159265358979323846;

static double crossNormFixed(double a, double b, double c, double v, double u) {
    const double sv = std::sin(v), cv = std::cos(v);
    const double su = std::sin(u), cu = std::cos(u);
    const double x =  b*c*sv*sv*cu;
    const double y =  a*c*sv*sv*su;
    const double z =  a*b*cv*sv;
    return std::sqrt(x*x + y*y + z*z);
}

int main() {
    // THE SPHERE, checked against 4*pi*r^2.
    {
        const double r = 2.0;
        const int NV = 800, NU = 1600;
        const double dv = kPi/NV, du = 2.0*kPi/NU;
        double total = 0.0;
        for (int i = 0; i < NV; ++i) {
            const double v = (i+0.5)*dv;
            double row = 0.0;
            for (int j = 0; j < NU; ++j)
                row += crossNormFixed(r,r,r,v,(j+0.5)*du);
            total += row*du*dv;
        }
        std::printf("sphere r=2: brute-force = %.8f   4 pi r^2 = %.8f   ratio = %.8f\n",
                    total, 4.0*kPi*r*r, total/(4.0*kPi*r*r));
    }
    // The 1D REDUCTION, now that the components are right. For the
    // sphere, |r_v x r_u| = sin v * sqrt(D(u)) with D(u) = r^2 exactly,
    // so the u-integral is 2*pi*r. The claimed form is
    //     4 sqrt(P+Q) E(m)
    // For the sphere: P = r^2, Q = 0, m = 0, E(0) = pi/2
    //     4 * r * pi/2 = 2*pi*r   <-- CORRECT
    // So the u-integral formula was RIGHT; only the COMPONENTS were
    // wrong. And the AREA formula then is
    //     A = int_0^pi sin v * (2*pi*r) dv = 4*pi*r^2
    // but written with the half-range symmetry:
    //     A = 8 * int_0^{pi/2} sin v * sqrt(P+Q) * E(m) dv
    // For the sphere: 8 * (r * pi/2) = 4*pi*r. STILL SHORT BY r.
    //
    // Because sqrt(P+Q) = sqrt(a^2 cos^2 v + c^2 sin^2 v) is LENGTH, and
    // the u-integral 4*sqrt(P+Q)*E(m) is LENGTH. The AREA needs
    // LENGTH^2 -- so the AREA's integrand must be sin v * [the u-integral]
    // and the u-integral is LENGTH. But the v-integral of sin v * LENGTH
    // over [0,pi] is LENGTH...
    //
    // THE RESOLUTION: the correct |r_v x r_u| has a factor of sin v
    // AND the u-integral is over sqrt(D(u)) which has units LENGTH --
    // so the surface element |r_v x r_u| du dv has units LENGTH^2, and
    // |r_v x r_u| = sin v * sqrt(D(u)) must be LENGTH^2, which means
    // sqrt(D(u)) is LENGTH and there is ANOTHER LENGTH somewhere.
    //
    // From the corrected components:
    //   |r_v x r_u|^2 = b^2 c^2 sin^4 v cos^2 u + a^2 c^2 sin^4 v sin^2 u
    //                   + a^2 b^2 cos^2 v sin^2 v
    //                = c^2 sin^4 v (b^2 cos^2 u + a^2 sin^2 u)
    //                  + a^2 b^2 cos^2 v sin^2 v
    // and dividing by sin^2 v:
    //   |r_v x r_u|^2 / sin^2 v = c^2 sin^2 v (b^2 cos^2 u + a^2 sin^2 u)
    //                             + a^2 b^2 cos^2 v
    //   = c^2 sin^2 v * (b^2 cos^2 u + a^2 sin^2 u) + a^2 b^2 cos^2 v
    //
    // WHICH IS NOT of the form sin^2 v * D(u) with D(u) =
    // (a cos v cos u)^2 + (b cos v sin u)^2 + c^2 sin^2 v. So the
    // classic one-integral reduction DOES NOT APPLY to this
    // parametrisation -- the area of an ellipsoid genuinely needs the
    // double integral (or the elliptic-integral reduction in a DIFFERENT
    // arrangement, which is the Knud-Thomsen form).
    //
    // SO: the reliable route is the DOUBLE integral, which is what this
    // probe computes. It is slower (a few milliseconds) but exact to
    // quadrature precision and needs no algebra.
    {
        const double a = 3.0, b = 1.0, c = 2.0;
        const int NV = 1200, NU = 2400;
        const double dv = kPi/NV, du = 2.0*kPi/NU;
        double total = 0.0;
        for (int i = 0; i < NV; ++i) {
            const double v = (i+0.5)*dv;
            double row = 0.0;
            for (int j = 0; j < NU; ++j)
                row += crossNormFixed(a,b,c,v,(j+0.5)*du);
            total += row*du*dv;
        }
        std::printf("ellipsoid 3x1x2: double integral = %.8f\n", total);
        // The sphere-equivalent volume is 4/3 pi abc = 8 pi = 25.13; its
        // equal-volume sphere has r = cbrt(3*25.13/(4 pi)) = 1.8175, area
        // = 4 pi r^2 = 41.5. The ellipsoid's area must EXCEED that.
        const double vol = 4.0/3.0*kPi*a*b*c;
        const double r_eq = std::cbrt(vol*3.0/(4.0*kPi));
        std::printf("  same-volume sphere's area = %.8f  (the ellipsoid must exceed it)\n",
                    4.0*kPi*r_eq*r_eq);
    }
    return 0;
}
