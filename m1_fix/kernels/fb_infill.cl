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
    const int lit = (c.x > 28 || c.y > 28 || c.z > 36);
    if (lit) {
        dst[i] = c;
        return;
    }
    int n = 0;
    int sr = 0, sg = 0, sb = 0;
    const int offs[8][2] = {
        {-1,0},{1,0},{0,-1},{0,1},{-1,-1},{1,-1},{-1,1},{1,1}
    };
    for (int k = 0; k < 8; ++k) {
        const int xx = (int)x + offs[k][0];
        const int yy = (int)y + offs[k][1];
        if (xx < 0 || yy < 0 || xx >= (int)W || yy >= (int)H) continue;
        const uchar4 npx = src[(uint)yy * W + (uint)xx];
        if (npx.x > 28 || npx.y > 28 || npx.z > 36) {
            sr += npx.x; sg += npx.y; sb += npx.z;
            ++n;
        }
    }
    if (n >= 2) {
        dst[i] = (uchar4)((uchar)(sr / n), (uchar)(sg / n), (uchar)(sb / n), (uchar)255);
    } else {
        dst[i] = c;
    }
}
