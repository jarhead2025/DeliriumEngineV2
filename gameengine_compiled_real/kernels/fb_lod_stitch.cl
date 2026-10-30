// Distance / LOD-aware surface stitch for sparse equation impacts.
// - Alpha 0 = empty (no geometry)
// - Alpha = object_id → never blend different objects
// - Wider average where impacts are sparse (far / coarse LOD)
// - Tighter where hits are dense (near / fine LOD)
// Optional depth buffer: larger world gap → larger screen radius.

__kernel void fb_lod_stitch(
    __global const uchar4 *src,
    __global uchar4 *dst,
    __global const float *depth,  // linear depth per pixel; may be null-ish if depth_valid=0
    const uint W,
    const uint H,
    const int base_radius,        // min radius (near)
    const int max_radius,         // max radius (far / sparse)
    const int depth_valid,
    const float depth_scale)      // maps depth → extra radius
{
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x < 0 || y < 0 || x >= (int)W || y >= (int)H) return;
    const uint idx = (uint)y * W + (uint)x;
    uchar4 c = src[idx];
    // ALPHA IS THE EMPTY/HIT MASK. resolve_fb now writes a=0 on background
    // and a=255 on hits, so alpha alone is authoritative. The old luminance
    // fallback (r<14&&g<14&&b<20) was a hack for when resolve wrote a=255
    // everywhere: it misclassified DARK SHADED hit pixels as empty (and the
    // (18,18,24) background as a hit), so the stitch blurred object colors
    // into the background and smeared dark faces -> the "dissolving" grain.
    #define IS_EMPTY(p) ((p).w == 0)


    // Estimate local hit density in a small window to choose LOD radius
    int hit_count = 0;
    uchar local_oid = IS_EMPTY(c) ? (uchar)0 : c.w;
    const int probe = 3;
    for (int dy = -probe; dy <= probe; ++dy) {
        for (int dx = -probe; dx <= probe; ++dx) {
            int xx = x + dx, yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= (int)W || yy >= (int)H) continue;
            uchar4 s = src[(uint)yy * W + (uint)xx];
            if (IS_EMPTY(s)) continue;
            if (local_oid == 0) local_oid = s.w;
            if (s.w == local_oid) ++hit_count;
        }
    }

    // Sparse → larger radius (stitch coarse far dots); dense → small radius
    int radius = base_radius;
    // Dense near-camera: do NOT widen radius (avoids POV smear glitches)
    if (hit_count >= 12) radius = base_radius;
    else if (hit_count >= 5) radius = base_radius + (max_radius > base_radius ? 1 : 0);
    else if (hit_count >= 2) radius = (base_radius + max_radius) / 2;
    else radius = max_radius;

    if (depth_valid && depth) {
        float d = depth[idx];
        if (d > 0.f && d < 1e8f) {
            int add = (int)(d * depth_scale);
            if (add < 0) add = 0;
            if (add > max_radius) add = max_radius;
            radius = base_radius + add;
            if (radius > max_radius) radius = max_radius;
        }
    }
    if (radius < 1) radius = 1;

    if (IS_EMPTY(c)) {
        // Fill empty pixel from same-object impacts only (lateral stitch)
        if (local_oid == 0) {
            dst[idx] = (uchar4)(10, 10, 16, 0);
            return;
        }
        int sr = 0, sg = 0, sb = 0, n = 0;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                int xx = x + dx, yy = y + dy;
                if (xx < 0 || yy < 0 || xx >= (int)W || yy >= (int)H) continue;
                // circular weight: prefer nearer impacts
                float dist2 = (float)(dx * dx + dy * dy);
                if (dist2 > (float)(radius * radius)) continue;
                uchar4 s = src[(uint)yy * W + (uint)xx];
                if (IS_EMPTY(s) || s.w != local_oid) continue;
                float w = 1.f / (1.f + dist2);
                sr += (int)(s.x * w * 256.f);
                sg += (int)(s.y * w * 256.f);
                sb += (int)(s.z * w * 256.f);
                n += (int)(w * 256.f);
            }
        }
        if (n > 0)
            dst[idx] = (uchar4)((uchar)(sr / n), (uchar)(sg / n), (uchar)(sb / n), local_oid);
        else
            dst[idx] = (uchar4)(10, 10, 16, 0);
        return;
    }

    // Hit pixel: smooth with same-object neighbors (preserve object mask)
    int sr = c.x * 256, sg = c.y * 256, sb = c.z * 256, n = 256;
    const uchar oid = c.w;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int xx = x + dx, yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= (int)W || yy >= (int)H) continue;
            float dist2 = (float)(dx * dx + dy * dy);
            if (dist2 > (float)(radius * radius)) continue;
            uchar4 s = src[(uint)yy * W + (uint)xx];
            if (IS_EMPTY(s) || s.w != oid) continue;
            float w = 1.f / (1.f + 0.5f * dist2);
            sr += (int)(s.x * w * 256.f);
            sg += (int)(s.y * w * 256.f);
            sb += (int)(s.z * w * 256.f);
            n += (int)(w * 256.f);
        }
    }
    dst[idx] = (uchar4)((uchar)(sr / n), (uchar)(sg / n), (uchar)(sb / n), oid);
}
