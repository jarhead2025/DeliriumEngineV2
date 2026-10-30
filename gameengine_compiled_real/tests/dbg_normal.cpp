#include "delirium/psf/PSFBody.h"
#include <cstdio>
#include <cmath>
using namespace DeliRium;
static void pr(const char* s, const Vec3& v){ printf("%s (%.4f, %.4f, %.4f)\n", s, v.x, v.y, v.z); }
int main(){
    auto body = PSFBody::FromFile("assets/torus.psf");
    if(!body){ printf("load fail\n"); return 1; }
    const Patch& p = body->Patches()[100];
    for (float u : {0.25f, 0.5f, 0.75f}) {
        for (float v : {0.25f, 0.5f, 0.75f}) {
            const float e = 1e-3f;
            Vec3 P, du, dv;
            p.EvaluateDerivs(u, v, P, du, dv);
            Vec3 pu1=p.Evaluate(u+e,v), pu0=p.Evaluate(u-e,v);
            Vec3 pv1=p.Evaluate(u,v+e), pv0=p.Evaluate(u,v-e);
            Vec3 fdu{pu1.x-pu0.x, pu1.y-pu0.y, pu1.z-pu0.z};
            Vec3 fdv{pv1.x-pv0.x, pv1.y-pv0.y, pv1.z-pv0.z};
            Vec3 na = p.EvaluateNormal(u,v);
            Vec3 nn{ fdu.y*fdv.z-fdu.z*fdv.y, fdu.z*fdv.x-fdu.x*fdv.z, fdu.x*fdv.y-fdu.y*fdv.x };
            float l=std::sqrt(nn.x*nn.x+nn.y*nn.y+nn.z*nn.z); nn={nn.x/l,nn.y/l,nn.z/l};
            float dot = na.x*nn.x+na.y*nn.y+na.z*nn.z;
            printf("u=%.2f v=%.2f | P(%.3f,%.3f,%.3f)\n", u, v, P.x,P.y,P.z);
            pr("  analytic dPdu", du); pr("  FD du/2e     ", {fdu.x/(2*e),fdu.y/(2*e),fdu.z/(2*e)});
            pr("  analytic dPdv", dv); pr("  FD dv/2e     ", {fdv.x/(2*e),fdv.y/(2*e),fdv.z/(2*e)});
            pr("  analytic N  ", na); pr("  FD N        ", nn);
            printf("  dot=%.4f (1-cos=%.3e)\n\n", dot, 1.0-dot);
        }
    }
    return 0;
}