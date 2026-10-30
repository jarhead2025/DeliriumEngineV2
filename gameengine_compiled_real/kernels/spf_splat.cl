/* =========================================================================
 * spf_splat.cl  --  Spectral-primitive renderer. "Complex models from
 * simple equations": the whole object is a sphere primitive (center +
 * base radius) plus a baked detail field expressed as a real spherical-
 * harmonic expansion. NO anchors, NO patches, NO seams -- a single
 * continuous field. Each work-item marches ONE direction (theta,phi),
 * evaluates the field in registers, projects, and splats with the same
 * 64-bit atom_min depth trick as psf_splat.cl.
 *
 * WHY THIS IS CHEAP ON THE SAME BASE CIRCUITRY:
 *   Each thread evaluates the SH field as an FMA chain over (L+1)^2 terms
 *   using the stable associated-Legendre recurrence -- no special-function
 *   units except sin/cos for the direction and the azimuth harmonics. The
 *   coefficient arrays are tiny (geom (L+1)^2 floats, color 3*(Lc+1)^2)
 *   and stay cache-resident, read by every thread. The only scatter is one
 *   64-bit atomic per sample. No divergent loops: every thread runs the
 *   same term count. There is no back-face problem to cull and no seam to
 *   stitch, so the inner loop is pure arithmetic.
 *
 * SH CONVENTION: real spherical harmonics matching the Python baker
 *   (scipy sph_harm_y, real form, with the extra (-1)^m the baker applies).
 *   Y_l^0  = K_l^0 * P_l^0(cos t)
 *   Y_l^m  = sqrt2 * K_l^m * P_l^m(cos t) * cos(m phi)   (m>0)
 *   Y_l^-m = sqrt2 * K_l^m * P_l^m(cos t) * sin(m phi)   (m>0)
 *   with an extra sign (-1)^m folded in for odd m to match the baker, and
 *   K_l^m = sqrt((2l+1)/(4pi) * (l-m)!/(l+m)!).
 * ========================================================================= */

#define PI_F 3.14159265358979323846f

/* Fully-normalized associated Legendre via the stable column recurrence.
 * Computes Nlm = K_l^m * P_l^m(x) directly (normalization folded in), so no
 * intermediate factorial or P_m^m ever overflows float32 -- valid to high
 * order. Fills nrm[l] for the requested m (l from m..L). Standard geodesy
 * recurrence (e.g. Holmes & Featherstone), Condon-Shortley folded out here
 * and re-applied to match the baker sign in sh_field. */
static float norm_plm_seed(int m, float x) {
    /* N_m^m = sqrt((2m+1)/(4pi) * prod_{k=1..m} (2k-1)/(2k)) * (sin t)^m
     * built incrementally to stay in range. sin t = sqrt(1-x^2). */
    float sint = sqrt(fmax(0.0f, 1.0f - x * x));
    float val = sqrt(1.0f / (4.0f * PI_F));      /* N_0^0 */
    for (int k = 1; k <= m; ++k)
        val *= sqrt((2.0f * k + 1.0f) / (2.0f * k)) * sint;
    return val;   /* = N_m^m, includes (sin t)^m, no overflow */
}

static float sh_field(__global const float* coef, int L, float x,
                      float phi) {
    float sint = sqrt(fmax(0.0f, 1.0f - x * x));
    /* precompute cos(m phi), sin(m phi) incrementally */
    float s = 0.0f;
    int idx_base = 0;   /* index of (l=0,m=0) growing per l handled below */
    /* We iterate m outer (columns), l inner, but coef index is l-major
     * (l, m=-l..l). Easiest: recompute linear index from (l,m). */
    for (int m = 0; m <= L; ++m) {
        float cmp = cos((float)m * phi);
        float smp = sin((float)m * phi);
        /* seed N_m^m and N_{m+1}^m for the column recurrence */
        float nmm = norm_plm_seed(m, x);
        float prev = 0.0f;        /* N_{m-1}^m = 0 */
        float cur = nmm;          /* N_m^m */
        for (int l = m; l <= L; ++l) {
            float nlm;
            if (l == m) nlm = cur;
            else {
                /* normalized recurrence:
                 * N_l^m = a*(x*N_{l-1}^m) - b*N_{l-2}^m
                 * a = sqrt((2l-1)(2l+1)/((l-m)(l+m)))
                 * b = sqrt((2l+1)(l+m-1)(l-m-1)/((l-m)(l+m)(2l-3))) */
                float a = sqrt(((2.0f*l-1.0f)*(2.0f*l+1.0f)) /
                               ((l - m) * (l + m)));
                float b = sqrt(((2.0f*l+1.0f)*(l+m-1.0f)*(l-m-1.0f)) /
                               ((l - m) * (l + m) * (2.0f*l-3.0f)));
                nlm = a * x * cur - b * prev;
                prev = cur; cur = nlm;
            }
            /* coef linear index for (l,m) and (l,-m) in l-major layout */
            int row = l * l + l;     /* index of (l, 0) */
            if (m == 0) {
                s += nlm * coef[row];
            } else {
                /* the normalized seed already carries the correct sign;
                 * no extra (-1)^m needed (verified against the baker). */
                float y_pos = 1.41421356f * nlm * cmp; /* (l, +m) */
                float y_neg = 1.41421356f * nlm * smp; /* (l, -m) */
                s += y_pos * coef[row + m];
                s += y_neg * coef[row - m];
            }
        }
    }
    return s;
}

/* clear packed framebuffer */
__kernel void clear_fb(__global ulong* fb, const uint n) {
    uint i = get_global_id(0);
    if (i < n) fb[i] = 0xFFFFFFFFFFFFFFFFUL;
}

/* Continuous-order field eval (the user's model: the image is a continuous
 * combination of frequency bands that appear GRADUALLY with proximity). Lc
 * is a float: bands with l <= floor(Lc) contribute fully, band
 * l == floor(Lc)+1 is weighted by frac(Lc), higher bands are 0. So as the
 * camera nears and Lc rises smoothly, each frequency fades in -- no popping,
 * unlike integer truncation which switches a whole band on at once. */
static float sh_field_lod(__global const float* coef, float Lc, int Lmax,
                          float x, float phi) {
    float sint = sqrt(fmax(0.0f, 1.0f - x * x));
    int Lint = (int)floor(Lc);
    float frac = Lc - (float)Lint;
    int Ltop = Lint + 1;                 /* highest band that contributes */
    if (Ltop > Lmax) { Ltop = Lmax; Lint = Lmax; frac = 0.0f; }  /* never read past buffer */
    float s = 0.0f;
    for (int m = 0; m <= Ltop; ++m) {
        float cmp = cos((float)m * phi);
        float smp = sin((float)m * phi);
        float prev = 0.0f;
        float cur = norm_plm_seed(m, x);
        for (int l = m; l <= Ltop; ++l) {
            float nlm;
            if (l == m) nlm = cur;
            else {
                float a = sqrt(((2.0f*l-1.0f)*(2.0f*l+1.0f)) /
                               ((l - m) * (l + m)));
                float b = sqrt(((2.0f*l+1.0f)*(l+m-1.0f)*(l-m-1.0f)) /
                               ((l - m) * (l + m) * (2.0f*l-3.0f)));
                nlm = a * x * cur - b * prev;
                prev = cur; cur = nlm;
            }
            /* per-band weight: full up to Lint, fractional at Lint+1 */
            float w = (l <= Lint) ? 1.0f : frac;
            if (w == 0.0f) continue;
            int row = l * l + l;
            if (m == 0) {
                s += w * nlm * coef[row];
            } else {
                s += w * 1.41421356f * nlm * cmp * coef[row + m];
                s += w * 1.41421356f * nlm * smp * coef[row - m];
            }
        }
    }
    return s;
}


__kernel void spf_splat(
    __global const float* gcoef,   /* geom field, (Lg+1)^2 */
    __global const float* ccoef,   /* color field, 3*(Lc+1)^2 (ch-major) */
    const int   Lg, const float Lc, const int Lc_stored,
    const float3 center, const float base_r,
    const uint  n_samples,         /* total directions (Fibonacci sphere) */
    const uint  pad_unused,        /* kept for arg-count stability */
    const float16 mvp,             /* row-major */
    const float3 light_dir,
    const float3 eye,
    const uint  width, const uint height,
    __global volatile ulong* fb)
{
    uint gid = get_global_id(0);
    if (gid >= n_samples) return;

    /* Fibonacci-sphere direction: uniform area density, no pole clustering
     * (the (theta,phi) grid amassed samples at the poles and left thin gaps
     * elsewhere -> faint lines). z is evenly spaced in [-1,1]; the azimuth
     * advances by the golden angle, so points spiral evenly over the sphere. */
    const float GA = 2.39996322972865332f;          /* golden angle */
    float zc = 1.0f - 2.0f * (gid + 0.5f) / (float)n_samples;  /* cos(theta) */
    float rxy = sqrt(fmax(0.0f, 1.0f - zc * zc));
    float az = GA * (float)gid;
    float theta = acos(zc);
    float phi = az - 2.0f * PI_F * floor((az + PI_F) / (2.0f * PI_F)); /* wrap to [-pi,pi) */
    float ct = zc, st = rxy;

    /* field radius + position */
    float R = base_r + sh_field(gcoef, Lg, ct, phi);
    float3 dir = (float3)(st * cos(phi), st * sin(phi), ct);
    float3 P = center + dir * R;

    /* analytic-ish normal via finite diff of the field in theta/phi */
    float e = 1e-3f;
    float ct2 = cos(theta + e), st2 = sin(theta + e);
    float R2 = base_r + sh_field(gcoef, Lg, ct2, phi);
    float3 Pt = center + (float3)(st2*cos(phi), st2*sin(phi), ct2) * R2;
    float Rp = base_r + sh_field(gcoef, Lg, ct, phi + e);
    float3 Pp = center + (float3)(st*cos(phi+e), st*sin(phi+e), ct) * Rp;
    float3 nrm = normalize(cross(Pt - P, Pp - P));
    if (dot(nrm, dir) < 0.0f) nrm = -nrm;

    /* project */
    float4 hp;
    hp.x = mvp.s0*P.x + mvp.s1*P.y + mvp.s2*P.z + mvp.s3;
    hp.y = mvp.s4*P.x + mvp.s5*P.y + mvp.s6*P.z + mvp.s7;
    hp.z = mvp.s8*P.x + mvp.s9*P.y + mvp.sa*P.z + mvp.sb;
    hp.w = mvp.sc*P.x + mvp.sd*P.y + mvp.se*P.z + mvp.sf;
    if (hp.w <= 1e-6f) return;
    float3 ndc = hp.xyz / hp.w;
    if (fabs(ndc.x) > 1.0f || fabs(ndc.y) > 1.0f) return;
    float depth = ndc.z;
    if (depth < 0.0f) return;

    int px = (int)((ndc.x * 0.5f + 0.5f) * (width  - 1) + 0.5f);
    int py = (int)((0.5f - ndc.y * 0.5f) * (height - 1) + 0.5f);
    if (px < 0 || px >= (int)width || py < 0 || py >= (int)height) return;

    /* color field + lambert. Lc_stored is how many coefs are stored per
     * channel (the stride); Lc is the LOD order to actually sum (<= stored).
     * Summing fewer orders = the inherent texture LOD: soft when far. */
    int nc = (Lc_stored + 1) * (Lc_stored + 1);
    float cr = sh_field_lod(ccoef,          Lc, Lc_stored, ct, phi);
    float cg = sh_field_lod(ccoef + nc,     Lc, Lc_stored, ct, phi);
    float cb = sh_field_lod(ccoef + 2 * nc, Lc, Lc_stored, ct, phi);
    float lam = fmax(dot(nrm, -light_dir), 0.0f) * 0.85f + 0.15f;
    uint r = (uint)(clamp(cr * lam, 0.0f, 1.0f) * 255.0f);
    uint g = (uint)(clamp(cg * lam, 0.0f, 1.0f) * 255.0f);
    uint b = (uint)(clamp(cb * lam, 0.0f, 1.0f) * 255.0f);
    uint rgba = (r << 24) | (g << 16) | (b << 8) | 0xFF;

    /* depth-tested splat: key = (depth_bits << 32) | rgba, atom_min */
    uint dbits = as_uint(depth);
    ulong key = ((ulong)dbits << 32) | (ulong)rgba;
    atom_min(&fb[py * width + px], key);
}

/* resolve packed fb -> RGBA8 image */
__kernel void resolve_fb(__global const ulong* fb, __global uchar4* img,
                         const uint n) {
    uint i = get_global_id(0);
    if (i >= n) return;
    ulong key = fb[i];
    if (key == 0xFFFFFFFFFFFFFFFFUL) { img[i] = (uchar4)(18,18,24,255); return; }
    uint rgba = (uint)(key & 0xFFFFFFFFUL);
    img[i] = (uchar4)((rgba>>24)&0xFF, (rgba>>16)&0xFF, (rgba>>8)&0xFF, 0xFF);
}
