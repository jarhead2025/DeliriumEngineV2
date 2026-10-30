// Object-masked lateral stitch: average only among same non-zero alpha.
// Alpha 0 = empty (no draw). Alpha = object_id (1..255).
// Guarantees no color bleed between different objects in the scene.

__kernel void fb_object_stitch(
    __global const uchar4 *src,
    __global uchar4 *dst,
    const uint W,
    const uint H,
    const int radius)
{
    const int x = (int)get_global_id(0);
    const int y = (int)get_global_id(1);
    if (x < 0 || y < 0 || x >= (int)W || y >= (int)H) return;

    const uint idx = (uint)y * W + (uint)x;
    uchar4 c = src[idx];
    if (c.w == 0) {
        // Empty: try fill from same-object neighbors only (lateral stitch of dots)
        int sr=0, sg=0, sb=0, n=0;
        // Prefer any non-zero object in neighborhood — but only one object id per fill
        uchar oid = 0;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                int xx = x + dx, yy = y + dy;
                if (xx < 0 || yy < 0 || xx >= (int)W || yy >= (int)H) continue;
                uchar4 s = src[(uint)yy * W + (uint)xx];
                if (s.w == 0) continue;
                if (oid == 0) oid = s.w;
                if (s.w != oid) continue; // other object: never blend
                sr += s.x; sg += s.y; sb += s.z; ++n;
            }
        }
        if (n > 0)
            dst[idx] = (uchar4)((uchar)(sr/n), (uchar)(sg/n), (uchar)(sb/n), oid);
        else
            dst[idx] = (uchar4)(10, 10, 16, 0);
        return;
    }

    // Already drawn: refine with same-object average (smooth between dots)
    int sr = c.x, sg = c.y, sb = c.z, n = 1;
    const uchar oid = c.w;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx == 0 && dy == 0) continue;
            int xx = x + dx, yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= (int)W || yy >= (int)H) continue;
            uchar4 s = src[(uint)yy * W + (uint)xx];
            if (s.w != oid) continue;
            sr += s.x; sg += s.y; sb += s.z; ++n;
        }
    }
    dst[idx] = (uchar4)((uchar)(sr/n), (uchar)(sg/n), (uchar)(sb/n), oid);
}
