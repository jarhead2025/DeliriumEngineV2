/* Final box AA only — runs after shading is complete. */
__kernel void fb_aa_box2(
    __global const uchar4 *src,
    __global uchar4 *dst,
    const uint W,
    const uint H)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    if (x >= W || y >= H) return;
    int sr=0,sg=0,sb=0,n=0;
    for (int dy=-1; dy<=1; ++dy)
    for (int dx=-1; dx<=1; ++dx) {
        int xx=(int)x+dx, yy=(int)y+dy;
        if (xx<0||yy<0||xx>=(int)W||yy>=(int)H) continue;
        uchar4 c = src[(uint)yy*W+(uint)xx];
        sr+=c.x; sg+=c.y; sb+=c.z; ++n;
    }
    dst[y*W+x] = (uchar4)((uchar)(sr/n),(uchar)(sg/n),(uchar)(sb/n),(uchar)255);
}
