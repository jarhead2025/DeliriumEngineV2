/* GPU hole fill: empty pixels take mean of lit neighbors.
 * Device-only; CPU enqueues passes then Present readback.
 */
__kernel void fb_infill_pass(
    __global const uchar4 *src,
    __global uchar4 *dst,
    const uint W,
    const uint H)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    if (x >= W || y >= H) return;
    const uint i = y * W + x;
    const uchar4 c = src[i];
    /* ALPHA IS THE HIT/EMPTY + OBJECT MASK. resolve_fb writes a=255 on hits
     * and a=0 on background (splat path); the raymarch path writes a=object_id
     * (1..255) on hits, a=0 on background. Either way a=0 is empty and a>0
     * is a hit carrying an object id. The old luminance test (r>28||g>28||b>36)
     * misclassified DARK SHADED surface pixels (rims, shadowed faces) as
     * "empty" and averaged them away -> the dark gaps / "dissolving" look. */
    if (c.w > 0) {
        dst[i] = c;
        return;
    }
    /* Empty pixel: adopt the object id of its lit neighbors and fill from
     * SAME-object neighbors only, so we never bleed one object's color into
     * an adjacent object's gap. */
    int n = 0;
    int sr = 0, sg = 0, sb = 0;
    uchar oid = 0;
    const int offs[8][2] = {
        {-1,0},{1,0},{0,-1},{0,1},{-1,-1},{1,-1},{-1,1},{1,1}
    };
    for (int k = 0; k < 8; ++k) {
        const int xx = (int)x + offs[k][0];
        const int yy = (int)y + offs[k][1];
        if (xx < 0 || yy < 0 || xx >= (int)W || yy >= (int)H) continue;
        const uchar4 npx = src[(uint)yy * W + (uint)xx];
        if (npx.w > 0) {
            if (oid == 0) oid = npx.w;
            if (npx.w == oid) { sr += npx.x; sg += npx.y; sb += npx.z; ++n; }
        }
    }
    /* One same-object neighbor is enough to fill a 1px gap (the old >=2 left
     * a checkerboard of unfilled pixels between sparse impacts). */
    if (n >= 1) {
        dst[i] = (uchar4)((uchar)(sr / n), (uchar)(sg / n), (uchar)(sb / n), oid);
    } else {
        dst[i] = c;
    }
}
